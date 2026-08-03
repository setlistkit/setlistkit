# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""``slkit pull``: fetch each configured source into the raw cache.

Nothing here parses, and nothing here writes to the database. A pull is the only step that
spends somebody else's bandwidth, so this module carries the politeness: one identity on every
request, a rate limit, and a conditional re-ask that moves no bytes when an item is unchanged.
"""

from __future__ import annotations

import argparse

from ..config import require_network_identity
from ..diagnostics import ERROR, Diagnostic, DiagnosticError
from ..importers import IMPORTERS, configured
from ..store.raw_cache import RawCache
from .common import min_year

EXIT_OK = 0

# How often a long pull says it is still going. One metadata request per item at roughly a
# request a second means a first pull of a large collection runs for a quarter of an hour, and
# silence for that long is indistinguishable from a hang.
_PROGRESS_EVERY = 25

# Seconds per REQUEST, for the estimate a dry run prints. MEASURED against archive.org (94 items
# in 2m25s, then 4614 more), not computed from the configured delay: the delay is only the half
# we control, and the round trip is the other half. archive.org makes one request per item, so
# this doubles as seconds-per-item there; setlist.fm pulls ~20 setlists per request, so its dry
# run reports its own request count and this same per-request cost applies. An estimate, and
# labeled as one -- but a real one, and the difference between "4222 items" and "about two
# hours" is the whole decision.
_SECONDS_PER_REQUEST = 1.5


def source_list(value: str):
    """Parse ``--source=a,b`` into importer ids, refusing a name the registry does not hold."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in IMPORTERS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown source(s): {', '.join(unknown)}. Known: {', '.join(IMPORTERS)}")
    return names


def pull_cmd(config, args) -> int:
    """Fetch each named source into the raw cache. Writes nothing to the database.

    A source named but not configured is skipped and said aloud, because a silent skip and a
    source that returned nothing look the same afterwards.
    """
    require_network_identity(config)
    # A source named by hand is run whatever the config holds, so the importer's own diagnostic
    # gets to name the exact setting that is missing. Only the default list is filtered, because
    # there "every source" can only mean the ones this deployment opted into.
    if args.source is None:
        running = configured(config)
    else:
        running = [IMPORTERS[name] for name in args.source]
    if not running:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="no configured source to pull",
            detail="A [sources.<id>] table in the config is what opts a source in. Add one,\n"
                   "or name a source you have configured.",
        ))
    for importer in running:
        if len(running) > 1:
            print(f"\n-- {importer.display_name} --", flush=True)
        _pull_one(config, args, importer)
    return EXIT_OK


def _pull_one(config, args, importer) -> int:
    """Fetch one source. Every setting is read under that source's own id, never a fixed name."""
    cache = RawCache(config.data_root)

    # flush=True on both, because a pull of a whole collection runs for hours and Python buffers
    # stdout whenever it is not a terminal. Redirected to a file or a log, an unflushed progress
    # line arrives after the job it was reporting on has finished, which is not progress.
    def progress(done: int, total: int) -> None:
        if done % _PROGRESS_EVERY == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    def announce(batch_id: str) -> None:
        # Printed so this run is greppable in OUR logs by the same id the source sees in theirs.
        # A batch id only one side of the conversation knows is half a tracking id.
        print(f"batch {batch_id} (sent in the User-Agent of every request this run)", flush=True)

    # One path for every source: the importer names its client and what the pull is keyed on (an
    # archive collection, a MusicBrainz id). Both clients' pull() take the same keyword shape.
    result = importer.make_client(config, cache).pull(
        importer.identity(config), min_year=min_year(config, importer.id, args.min_year),
        force_rescan=args.force_rescan, dry_run=args.dry_run,
        progress=progress, announce=announce)
    if args.dry_run:
        # `planned` counts records; the wall clock is set by REQUESTS. They match for archive.org
        # (one request per item, planned_requests left None) and diverge for a paged source like
        # setlist.fm, which reports the smaller request count the estimate must be built from.
        requests = result.planned if result.planned_requests is None else result.planned_requests
        minutes = requests * _SECONDS_PER_REQUEST / 60
        across = "" if result.planned_requests is None else f" across {requests} requests"
        print(f"pull {importer.display_name} (dry run): {result.listed} listed, "
              f"{result.cached} already cached, {result.planned} would be fetched"
              f"{across} (about {minutes:.0f} min)")
        # What the dry run itself just spent. Sources that report it get an exact number; the rest
        # keep the older prose. This used to be inferred from `planned_requests`, which meant every
        # paged source was described as having fetched "the first page" -- false for a source that
        # sizes the job from config and makes no request at all.
        if result.probe_requests is not None:
            spent = result.probe_requests
            print(f"  This estimate cost {spent} request(s)." if spent else
                  "  This estimate cost no requests at all: the job's size is known without asking.")
        elif result.planned_requests is None:
            print("  No item metadata was requested. The listing itself was, because that is how\n"
                  "  this learns what is new; it is a few requests against thousands.")
        else:
            # A paged source has no separate cheap listing: the first page IS data, and fetching it
            # is how the run learns the total. So the honest disclosure names that one request.
            print("  Only the first page was requested, because that is how this learns the total.\n"
                  "  It is one request against thousands of setlists; the rest is what a real pull costs.")
    else:
        print(f"pull {importer.display_name}: {result.listed} listed, {result.fetched} fetched, "
              f"{result.cached} already cached")
    if result.missing:
        print(f"  {len(result.missing)} listed item(s) the metadata API does not have; "
              f"they are retried on the next pull:")
        for identifier in result.missing:
            print(f"    {identifier}")
    if result.failed:
        print(f"  {len(result.failed)} item(s) the source could not serve, even after retries. "
              "Nothing was cached\n  for them, so the next pull tries them again:")
        for identifier in result.failed:
            print(f"    {identifier}")
    if result.unidentified:
        # Counted rather than swallowed. An item in none of the counters is an item nobody
        # misses, and "listed" not adding up is the only signal anything went past unfetched.
        print(f"  {result.unidentified} listed item(s) carried no identifier and could not be "
              "fetched")
    if result.truncated:
        print("  warning: hit the paging backstop, so this listing is a PREFIX of the "
              "collection.\n  Ingest will look complete and will not be. Raise _MAX_PAGES.")
    return EXIT_OK

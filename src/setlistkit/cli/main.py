# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""``slkit`` command dispatch.

The one job of this module is to be the single place that catches a
:class:`DiagnosticError` and turns it into rendered stderr output plus a non-zero exit
code. Every other layer raises well-formed diagnostics and stays out of the presentation
business. Subcommands grow here phase by phase; for now the scaffold ships ``config``,
which is enough to exercise config resolution and the diagnostic renderer end to end.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import __version__
from ..catalog.lint import lint
from ..catalog.normalizer import Normalizer
from ..catalog.scaffold import starter_pack
from ..catalog.suggest import render_markdown, suggest, titles_from_cache
from ..config import load_config, require_network_identity
from ..diagnostics import ERROR, Diagnostic, DiagnosticError, render
from ..importers import IMPORTERS, configured, mirror, secondary
from ..picks.ledger import LEDGER_TABLES
from ..sources.client import SourceError
from ..store import Store
from ..store.raw_cache import RawCache
from .common import min_year, required_setting, resolve_pack_dir
from .derive import derive
from .export import export
from .ingest import ingest
from ..picks.drops import DEFAULT_DEPTH
from .predict import drops_cmd, predict, replay

EXIT_OK = 0
EXIT_DIAGNOSTIC = 2

# How often a long pull says it is still going. One metadata request per item at roughly a
# request a second means a first pull of a large collection runs for a quarter of an hour, and
# silence for that long is indistinguishable from a hang.
_PROGRESS_EVERY = 25

# Seconds per REQUEST, for the estimate a dry run prints. MEASURED against archive.org (94 items in
# 2m25s, then 4614 more), not computed from the configured delay: the delay is only the half we
# control, and the round trip is the other half. archive.org makes one request per item, so this
# doubles as seconds-per-item there; setlist.fm pulls ~20 setlists per request, so its dry run
# reports its own request count and this same per-request cost applies. An estimate, and labeled as
# one -- but a real one, and the difference between "4222 items" and "about two hours" is the whole
# decision.
_SECONDS_PER_REQUEST = 1.5

# Sources `slkit pull` knows how to fetch: the importer registry's keys, so argparse and the
# registry can never disagree about what exists. Naming them (rather than deriving choices from the
# config's [sources.*] tables) still lets argparse reject an unknown name with a usage line.
_SOURCES = tuple(IMPORTERS)


def _add_dry_run(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the show-me-what-you-would-do flag, under all three names people reach for.

    One flag, three spellings, because which one is "the" name is pure habit and getting it
    wrong on a command whose whole purpose is to be safe to try is a bad first experience. The
    dest stays ``dry_run`` whichever they type.
    """
    parser.add_argument("-n", "--dry-run", "--noop", dest="dry_run", action="store_true",
                        help=help_text)


def _add_config(sub) -> None:
    """The ``slkit config`` parser."""
    config_cmd = sub.add_parser("config", help="inspect the resolved configuration")
    config_sub = config_cmd.add_subparsers(dest="config_action")
    config_sub.add_parser("show", help="print the resolved configuration")
    config_sub.add_parser("check", help="validate the config, including network identity")


def _add_store(sub) -> None:
    """The ``slkit store`` parser."""
    store_cmd = sub.add_parser("store", help="create and inspect the state store")
    store_sub = store_cmd.add_subparsers(dest="store_action")
    store_sub.add_parser("init", help="create data_root and apply schema migrations")
    store_sub.add_parser("status", help="show schema version and table counts")


def _add_dump(sub) -> None:
    """The ``slkit dump`` parser."""
    dump_cmd = sub.add_parser("dump", help="print a plain-text view of derived state")
    dump_cmd.add_argument(
        "what", nargs="?", choices=("all", "picks"), default="all",
        help="which tables to print (default: %(default)s). `picks` is the prediction ledger, "
             "and it is the strongest case for this command in the project: those tables are the "
             "only ones here that cannot be rebuilt",
    )
    # Inclusive at both ends. A half-open range is the classic off-by-one, and this is the view
    # someone reaches for when they already suspect something is wrong -- one bug at a time.
    dump_cmd.add_argument("--since", metavar="YYYY-MM-DD",
                          help="only rows on or after this date (tables with no date axis are "
                               "printed in full, and say so)")
    dump_cmd.add_argument("--until", metavar="YYYY-MM-DD",
                          help="only rows on or before this date")


def _add_pull(sub) -> None:
    """The ``slkit pull`` parser."""
    pull_cmd = sub.add_parser("pull", help="fetch raw source data into the cache")
    pull_cmd.add_argument("source", choices=_SOURCES, help="which source to fetch from")
    pull_cmd.add_argument(
        "--force-rescan", action="store_true",
        help="re-ask for data already cached (still rate-limited, and conditional: an "
             "unchanged item answers 304 and no bytes move)",
    )
    pull_cmd.add_argument(
        "--min-year", type=int, metavar="YEAR",
        help="ignore shows played before YEAR (overrides min_year in config)",
    )
    _add_dry_run(pull_cmd, "list the collection and report what a real run would fetch, "
                           "without fetching any of it")


def _add_ingest(sub) -> None:
    """The ``slkit ingest`` parser."""
    ingest_cmd = sub.add_parser(
        "ingest", help="parse the cached raw data and publish the merged corpus")
    ingest_cmd.add_argument("source", choices=_SOURCES, nargs="?", default=_SOURCES[0],
                            help="which cached source to parse (default: %(default)s)")
    ingest_cmd.add_argument("--pack", metavar="PATH",
                            help="pack directory to parse with (overrides [catalog] pack)")
    ingest_cmd.add_argument("--min-year", type=int, metavar="YEAR",
                            help="the min_year the cache was pulled under, if not the configured "
                                 "one (a listing is cached per collection and min_year)")
    _add_dry_run(ingest_cmd, "parse, merge and report in full, but write nothing to the database")
    ingest_cmd.add_argument(
        "--force", action="store_true",
        help="publish even when the merge produced far fewer shows than are stored (see the "
             "no-shrink guard)",
    )
    ingest_cmd.add_argument(
        "--profile", metavar="PATH",
        help="write the funnel's per-decision counts to PATH as JSON (see catalog.funnel); the "
             "counts are always taken during a parse, this only decides whether they are "
             "written out",
    )


def _add_derive(sub) -> None:
    """The ``slkit derive`` parser."""
    derive_cmd = sub.add_parser(
        "derive", help="compute derived state from what ingest published")
    derive_sub = derive_cmd.add_subparsers(dest="derive_action")
    durations_cmd = derive_sub.add_parser(
        "durations", help="how long each song runs, reconciled across every tape of a night")
    durations_cmd.add_argument("--pack", metavar="PATH",
                               help="pack directory to derive with (overrides [catalog] pack)")
    _add_dry_run(durations_cmd,
                 "read, reconcile and report in full, but write nothing to the database")


def _add_predict(sub) -> None:
    """The ``slkit predict`` parser."""
    predict_cmd = sub.add_parser(
        "predict", help="rank every song's chance of being played on one night")
    predict_cmd.add_argument(
        "--asof", metavar="YYYY-MM-DD",
        help="the night to predict (default: the day after the last stored show, which is the "
             "earliest date the corpus does not already hold). Nothing dated on or after it is "
             "read, so naming a night inside the corpus gives a blinded re-prediction of it",
    )
    _add_model_flags(predict_cmd)
    predict_cmd.add_argument(
        "--set-len", type=int, metavar="N",
        help="how many songs to assume will be played (overrides set_len in [model]). A live "
             "prediction has to guess this; a replay is told it by the night it grades",
    )


def _add_replay(sub) -> None:
    """The ``slkit replay`` parser."""
    replay_cmd = sub.add_parser(
        "replay",
        help="predict every gradeable night in a window before revealing it, and grade the lot")
    # Inclusive at both ends and spelled exactly as `dump` and `export` spell them. A range that
    # means one thing in one command and something a day wider in another is a range nobody can
    # check a published number against.
    replay_cmd.add_argument("--since", metavar="YYYY-MM-DD",
                            help="grade nights on or after this date (default: the window in "
                                 "[model] replay_since_back, anchored on the last stored show)")
    replay_cmd.add_argument("--until", metavar="YYYY-MM-DD",
                            help="grade nights on or before this date")
    _add_model_flags(replay_cmd)
    _add_dry_run(replay_cmd, "predict and grade the whole window and report it, but write "
                             "nothing to the picks ledger")
    replay_cmd.add_argument(
        "--force", action="store_true",
        help="write even when the run leaves a ledger table with less than half the rows it "
             "started with (see the no-shrink guard)",
    )


def _add_drops(sub) -> None:
    """The ``slkit drops`` parser.

    The window flags are spelled and defaulted exactly as ``replay`` spells them, deliberately: a
    churn number and a recall number quoted off the same store have to be about the same nights,
    and two commands that disagree about what "the window" means make that impossible to check.
    """
    drops_parser = sub.add_parser(
        "drops",
        help="predict every gradeable night in a window from several earlier cutoffs, and watch "
             "the shortlist move")
    drops_parser.add_argument("--since", metavar="YYYY-MM-DD",
                              help="cover nights on or after this date (default: the window in "
                                   "[model] replay_since_back, anchored on the last stored show)")
    drops_parser.add_argument("--until", metavar="YYYY-MM-DD",
                              help="cover nights on or before this date")
    drops_parser.add_argument(
        "--depth", type=int, metavar="N",
        help=f"how many earlier cutoffs to predict each night from (default: {DEFAULT_DEPTH}). "
             "The walk costs nights x depth predictions, so this is what decides whether it runs "
             "in seconds or minutes",
    )
    _add_model_flags(drops_parser)
    _add_dry_run(drops_parser, "walk the whole window and report it, but write nothing to the "
                               "picks ledger")
    drops_parser.add_argument(
        "--force", action="store_true",
        help="write even when the run leaves a ledger table with less than half the rows it "
             "started with (see the no-shrink guard)",
    )


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    """``--rollouts``, on both commands that draw.

    An iteration flag rather than a setting: a lower count shows the direction of a model change
    several times faster, and it is the wrong number to leave behind in a file a nightly run
    reads. The cost of a prediction is in the scoring pass over the corpus, not the draws, so
    lowering it buys less than it looks like it should.
    """
    parser.add_argument(
        "--rollouts", type=int, metavar="N",
        help="how many times to draw the night (overrides rollouts in [model])")


def _add_export_common(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add --since/--until to one export sub-verb's parser, each carrying its own explanation.

    Mirrors ``_add_dry_run``'s idiom: one implementation, called once per sub-verb with the text
    that explains what narrowing means for THAT bundle, because the two do not mean the same
    thing (the Tape Measure recomputes statistics over the window; the Songbook floors and
    dedupes over it). NOT added to the parent ``export`` parser: argparse would then require
    ``slkit export --since ... songbook``, and ``slkit export songbook --help`` would not list the
    flag at all -- the same recursive-help complaint ``_add_dry_run`` already exists to avoid.

    Spelled and bounded exactly as ``slkit dump`` spells them -- inclusive at both ends, same
    YYYY-MM-DD, same two names. A range that means one thing in one command and something a day
    wider in another is a range nobody can check a published number against.
    """
    parser.add_argument("--since", metavar="YYYY-MM-DD",
                        help=f"only {help_text}, on or after this date")
    parser.add_argument("--until", metavar="YYYY-MM-DD",
                        help=f"only {help_text}, on or before this date")


def _add_export(sub) -> None:
    """The ``slkit export`` parser."""
    export_cmd = sub.add_parser(
        "export", help="write what was derived as one file, for something else to draw")
    export_cmd.add_argument(
        "--explain", action="store_true",
        help="print what every configured [reports.*] window resolves to, and why -- builds "
             "nothing",
    )
    export_sub = export_cmd.add_subparsers(dest="export_action")

    tapemeasure_cmd = export_sub.add_parser(
        "tapemeasure", help="song lengths, features, credits and caveats as one JSON bundle")
    tapemeasure_cmd.add_argument("--out", metavar="PATH", default="tapemeasure.json",
                                 help="where to write the bundle (default: %(default)s)")
    _add_export_common(tapemeasure_cmd,
                       "performances (song statistics are recomputed over the window, not "
                       "read whole-corpus)")
    _add_dry_run(tapemeasure_cmd, "build the bundle and report it, but write no file")

    songbook_cmd = export_sub.add_parser(
        "songbook", help="every show's songs, sorted into a vocabulary, as one JSON bundle")
    songbook_cmd.add_argument("--out", metavar="PATH", default="songbook.json",
                              help="where to write the bundle (default: %(default)s)")
    songbook_cmd.add_argument(
        "--pack", metavar="PATH",
        help="pack directory to canonicalize song names with (overrides [catalog] pack)")
    _add_export_common(songbook_cmd,
                       "shows (below-floor shows and out-of-vocabulary songs are still "
                       "counted and reported, never silently dropped)")
    _add_dry_run(songbook_cmd, "build the bundle and report it, but write no file")


def _add_pack(sub) -> None:
    """The ``slkit pack`` parser."""
    pack_cmd = sub.add_parser("pack", help="work with band packs")
    pack_sub = pack_cmd.add_subparsers(dest="pack_action")
    init_cmd = pack_sub.add_parser(
        "init", help="scaffold the starter files a valid pack needs, without clobbering any")
    init_cmd.add_argument(
        "--pack", metavar="DIR", required=True,
        help="pack directory to scaffold (created if absent)")
    init_cmd.add_argument(
        "--force", action="store_true",
        help="overwrite scaffold files that already exist (default: leave them untouched)")
    lint_cmd = pack_sub.add_parser("lint", help="validate a pack and run conformance checks")
    lint_cmd.add_argument(
        "--pack", metavar="PATH",
        help="pack directory to lint (overrides [catalog] pack in config)",
    )
    lint_cmd.add_argument(
        "--format", choices=("human", "json"), default="human",
        help="output format (default: human)",
    )
    lint_cmd.add_argument(
        "--no-corpus", action="store_true",
        help="skip the checks that read the cached corpus (dead rules, redundant rules, "
             "unreachable aliases, unknown titles)",
    )
    suggest_cmd = pack_sub.add_parser(
        "suggest-vocab",
        help="draft a pack's vocabulary/aliases from a bag of raw archive.org titles",
    )
    suggest_cmd.add_argument(
        "--from", dest="source", metavar="PATH", required=True,
        help="a JSON array of raw title strings (e.g. raw/archive_org/titles.json)",
    )
    suggest_cmd.add_argument(
        "--curated", metavar="PATH",
        help="a JSON array of song names from a source that PUBLISHES titles (setlist.fm), "
             "which are clustered with --from but win the canonical spelling. `slkit bootstrap` "
             "reads these from the configured setlist sources automatically",
    )
    suggest_cmd.add_argument(
        "--out", metavar="DIR", required=True,
        help="directory to write vocabulary.json, aliases.json and SUGGESTIONS.md into",
    )
    suggest_cmd.add_argument(
        "--cutoff", type=float, default=0.88, metavar="RATIO",
        help="difflib similarity at which two titles merge; higher is more conservative "
             "(default: 0.88)",
    )
    suggest_cmd.add_argument(
        "--force", action="store_true",
        help="overwrite existing non-empty vocabulary.json / aliases.json in --out",
    )


def _add_bootstrap(sub) -> None:
    """The ``slkit bootstrap`` parser: the whole pipeline behind one verb.

    The command takes no source argument. The config decides which sources take part.
    """
    boot = sub.add_parser(
        "bootstrap",
        help="carry an empty deployment to a reviewable pack and a published corpus in one run",
    )
    boot.add_argument(
        "--pack", metavar="DIR",
        help="pack directory to draft and scaffold into (overrides [catalog] pack in config)")
    boot.add_argument(
        "--cutoff", type=float, default=0.88, metavar="RATIO",
        help="suggest-vocab merge similarity; higher is more conservative (default: 0.88)")
    boot.add_argument(
        "--skip-pull", action="store_true",
        help="do not touch the network; draft and ingest from whatever is already cached")
    boot.add_argument(
        "--force-rescan", action="store_true",
        help="passed to the pull step: re-ask for data already cached")
    boot.add_argument(
        "--force", action="store_true",
        help="passed to the suggest step: overwrite an existing non-empty vocabulary/aliases")
    _add_dry_run(boot, "run store init, then report what a real pull would fetch, and stop "
                       "before drafting, scaffolding or ingesting anything")


def _build_parser() -> argparse.ArgumentParser:
    """Every command's parser, one call each.

    One builder per command rather than eighty lines in a row, so that adding a command is one
    function here and one entry in ``_COMMANDS`` below -- the same shape the dispatch table
    already has, and the reason it has it. Inline, each command left two or three locals behind
    for the next one to read past.
    """
    parser = argparse.ArgumentParser(
        prog="slkit",
        description="setlistkit — a setlist prediction toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"slkit {__version__}")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="path to a config file (overrides SLKIT_CONFIG and the default search order)",
    )

    sub = parser.add_subparsers(dest="command")

    _add_config(sub)
    _add_store(sub)
    _add_dump(sub)
    _add_pull(sub)
    _add_ingest(sub)
    _add_derive(sub)
    _add_predict(sub)
    _add_replay(sub)
    _add_drops(sub)
    _add_export(sub)
    _add_pack(sub)
    _add_bootstrap(sub)

    return parser


def _cmd_config(config, args) -> int:
    """`slkit config [show|check]`, defaulting to show."""
    if args.config_action == "check":
        return _cmd_config_check(config)
    return _cmd_config_show(config)


def _cmd_store(config, args) -> int:
    """`slkit store [init|status]`, defaulting to init."""
    if args.store_action == "status":
        return _cmd_store_status(config)
    return _cmd_store_init(config)


def _cmd_config_show(config) -> int:
    print(f"config file : {config.source_path}")
    print(f"data_root   : {config.data_root}")
    # Resolved the same way the working commands resolve it (relative to this config's own
    # directory, not the process cwd), so what prints here is the pack those commands will act on.
    pack = config.section("catalog").get("pack")
    if pack:
        pack_path = Path(str(pack)).expanduser()
        if not pack_path.is_absolute():
            pack_path = config.source_path.parent / pack_path
        print(f"pack        : {pack_path.resolve()}")
    else:
        print("pack        : (not set — pass --pack or set [catalog] pack)")
    print(f"user_agent  : {config.user_agent}")
    if config.user_agent_is_sentinel:
        print("            : (placeholder — network operations will refuse to run)")
    return EXIT_OK


def _cmd_config_check(config) -> int:
    require_network_identity(config)
    print(f"ok: {config.source_path} is valid and identifies itself to the network")
    return EXIT_OK


def _cmd_store_init(config) -> int:
    with Store(config.data_root) as store:
        applied = store.init()
        if applied:
            print(f"applied migrations: {', '.join(str(v) for v in applied)}")
        else:
            print("already current, nothing to apply")
        print(f"schema version {store.schema_version()} at {store.db_path}")
    return EXIT_OK


def _human_bytes(count: int) -> str:
    """A byte count at a glance. Three significant figures is all anyone reads."""
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"                                    # pragma: no cover - loop returns


def _cmd_store_status(config) -> int:
    store = Store(config.data_root)
    print(f"db          : {store.db_path}")
    print(f"raw cache   : {store.raw.root}")
    # What is actually in it, not just where it is. This is the no-network answer to "how far
    # has the pull got", and it is safe to run in a loop -- `slkit pull -n` answers it better
    # but spends search requests to do it, which makes it the wrong thing to poll with.
    for namespace, stats in store.raw.stats().items():
        print(f"  {namespace}: {stats.entries} entries, {_human_bytes(stats.bytes)}")
    # Don't open (and thereby create) the DB just to report it doesn't exist yet.
    if not store.db_path.is_file():
        print("schema      : not initialized — run `slkit store init`")
        return EXIT_OK
    with store:
        print(f"schema      : version {store.schema_version()}")
        for table, count in store.table_counts().items():
            print(f"  {table}: {count} rows")
    return EXIT_OK


def _cmd_dump(config, args) -> int:
    """Print the store as text, optionally narrowed to a date range.

    A malformed date is refused rather than compared. Dates are stored as YYYY-MM-DD strings and
    the range is a string comparison, so ``--until 2023`` sorts below every date IN 2023 and would
    hand back an empty dump of the year that was asked for -- which reads as "there is nothing
    there", the one answer this view exists to make trustworthy.
    """
    only = None if args.what == "all" else LEDGER_TABLES
    try:
        with Store(config.data_root) as store:
            print(store.dump(since=args.since, until=args.until, only=only), end="")
    except ValueError as exc:
        raise _malformed_date(exc, "print") from exc
    return EXIT_OK


def _malformed_date(exc: ValueError, verb: str) -> DiagnosticError:
    """The refusal both ranged commands give, so they cannot explain the same trap differently."""
    return DiagnosticError(Diagnostic(
        severity=ERROR,
        summary=str(exc),
        detail="Dates are stored and compared as YYYY-MM-DD text, so a shorter one does not\n"
               "mean what it looks like: `--until 2023` sorts below every date in 2023 and\n"
               f"would {verb} an empty range rather than the year you asked for.",
    ))


def _cmd_predict(config, args) -> int:
    """`slkit predict`, with the same refusal every other dated command gives."""
    try:
        return predict(config, args)
    except ValueError as exc:
        raise _malformed_date(exc, "predict from") from exc


def _cmd_replay(config, args) -> int:
    """`slkit replay`, likewise.

    Wrapped here rather than inside the walk for the reason ``_cmd_export`` gives: a malformed
    flag is a CLI fault, and the replay's job starts once the window is known to mean what it
    says. Unwrapped, ``--since "the fall tour"`` arrives as a traceback -- which is the one shape
    of output this program does not use to say no.
    """
    try:
        return replay(config, args)
    except ValueError as exc:
        raise _malformed_date(exc, "grade") from exc


def _cmd_drops(config, args) -> int:
    """`slkit drops`, with the same refusal ``replay`` gives a window it cannot read."""
    try:
        return drops_cmd(config, args)
    except ValueError as exc:
        raise _malformed_date(exc, "walk") from exc


def _cmd_export(config, args) -> int:
    """`slkit export`, with the same refusal ``dump`` gives a date it cannot compare.

    Wrapped here rather than inside the exporter because a malformed flag is a CLI fault: the
    exporter's job starts once the window is known to mean what it says.
    """
    try:
        return export(config, args)
    except ValueError as exc:
        raise _malformed_date(exc, "write") from exc


def _cmd_pull(config, args) -> int:
    """Fetch a source into the raw cache. Writes nothing to the database.

    The refusal comes first and comes from :func:`require_network_identity`, so a run with the
    placeholder ``user_agent`` stops here rather than at the first request -- there is no state
    to half-write, and the message is the same one ``slkit config check`` gives.

    Every setting is looked up under ``args.source``, never under a hard-coded table name. With
    one source those are the same string; with two, a hard-coded one makes ``slkit pull setlist_fm``
    quietly read the archive.org table and pull archive.org.
    """
    require_network_identity(config)
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
    # archive collection, a MusicBrainz id). Both clients' pull() take the same keyword shape, so
    # the branch that used to sit here is now a registry lookup.
    importer = IMPORTERS[args.source]
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


def _lint_corpus(args) -> list:
    """The cached source items to lint against, or empty when there is no usable cache.

    Best-effort by design. ``pack lint --pack PATH`` has always worked with no config at all, and
    a pack author checking their own file should not be made to configure a data_root first. When
    there is no corpus the checks that need one report themselves skipped, which is the honest
    outcome and not a failure.
    """
    if getattr(args, "no_corpus", False):
        return []
    # Absence is fine; malformed is not, and only the first is swallowed. No config file at all,
    # or a config with no source configured, are both ordinary states -- `pack lint --pack PATH`
    # has always worked with neither, and a pack author checking their own file should not have
    # to set up a data_root first. A config that EXISTS and is wrong is a different thing: a
    # quoted min_year or unreadable TOML is a fault `slkit ingest` treats as fatal, and
    # swallowing it here told the user to run a pull they had already run while hiding the
    # reason it did not take.
    try:
        config = load_config(args.config)
    except DiagnosticError:
        return []
    if not str(config.section("sources", "archive_org").get("collection") or "").strip():
        return []
    collection = required_setting(config, ("sources", "archive_org"), "collection", "")
    client = IMPORTERS["archive_org"].make_client(config, RawCache(config.data_root))
    cached = client.cached_items(collection, min_year=min_year(config, "archive_org", None))
    if cached.truncated or cached.absent:
        # The same facts `slkit ingest` refuses to publish over. A lint run against a fragment of
        # the corpus reports most of a pack dead and ranks its unknown titles against a fraction
        # of the real play counts -- and every one of those findings recommends deletion.
        print(f"warning: the cached corpus is incomplete ({len(cached.absent)} listed item(s) "
              f"unreadable"
              f"{', listing truncated' if cached.truncated else ''}).\n"
              "  Corpus-aware findings below are measured against what is cached, not against "
              "the collection.\n  Finish `slkit pull archive_org` before deleting anything on "
              "their say-so.", file=sys.stderr)
    return cached.items


def _cmd_pack_lint(args) -> int:
    """Lint the resolved pack, reporting findings as human text or JSON with an honest code."""
    pack_dir = resolve_pack_dir(getattr(args, "pack", None), args.config)
    try:
        diagnostics = lint(pack_dir, _lint_corpus(args))
    except DiagnosticError as exc:
        diagnostics = [exc.diagnostic]     # a structural failure is itself the one finding

    if getattr(args, "format", "human") == "json":
        print(json.dumps([diag.to_dict() for diag in diagnostics], indent=2))
    else:
        for diag in diagnostics:
            print(render(diag))
            print()
        errors = sum(1 for diag in diagnostics if diag.is_error)
        print(f"{pack_dir}: {errors} error(s), {len(diagnostics) - errors} other finding(s)")

    return EXIT_DIAGNOSTIC if any(diag.is_error for diag in diagnostics) else EXIT_OK


def _cmd_pack(args) -> int:
    """`slkit pack [init|lint|suggest-vocab]`, defaulting to lint (the original behavior)."""
    action = getattr(args, "pack_action", None)
    if action == "init":
        return _cmd_pack_init(args)
    if action == "suggest-vocab":
        return _cmd_pack_suggest(args)
    return _cmd_pack_lint(args)


def _cmd_pack_init(args) -> int:
    """Drop the starter files a valid pack needs into --pack, never clobbering existing ones.

    Writes only files that are absent, so it composes with suggest-vocab in either order: run it
    first for an empty skeleton suggest-vocab then fills, or run it after to add the pack.json and
    policy files a suggest-vocab run leaves out -- and in that second case it leaves the
    already-written vocabulary.json and aliases.json exactly as they are. ``--force`` overwrites.
    """
    pack_dir = Path(args.pack).expanduser()
    files = starter_pack(pack_dir.resolve().name or "pack")
    pack_dir.mkdir(parents=True, exist_ok=True)

    created, skipped = [], []
    for name, content in files.items():
        path = pack_dir / name
        if path.exists() and not args.force:
            skipped.append(name)
            continue
        path.write_text(content, encoding="utf-8")
        created.append(name)

    if created:
        print(f"scaffolded {pack_dir}:")
        for name in created:
            print(f"  wrote {name}")
    if skipped:
        print(f"left {len(skipped)} existing file(s) untouched: {', '.join(skipped)}")
        print("(pass --force to overwrite them)")
    if not created:
        print(f"{pack_dir} already has every starter file; nothing to do")
    else:
        print("\nEdit pack.json (add a band_name), fill vocabulary.json, then `slkit pack lint`.")
    return EXIT_OK


def _nonempty_pack_file(path: Path) -> bool:
    """True when ``path`` holds a pack file with real content already -- the thing --force guards.

    An empty ``[]`` / ``{}`` is treated as absent: a fresh pack directory that was `git init`ed with
    placeholder files should not need --force to fill in. Anything unparseable is treated as real,
    because refusing to clobber something we cannot read is the safe direction."""
    if not path.is_file():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return True


def _read_titles(source: Path) -> list[str]:
    """A JSON array of title strings from ``source``, or a diagnostic naming what was wrong.

    Shared by ``--from`` and ``--curated`` so the two read the same shape and fail the same way.
    """
    try:
        titles = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary=f"could not read titles from {source}",
            detail=f"{exc}\n\nExpected a JSON array of title strings, such as the "
                   f"titles.json produced by the dirty-stripper jq filter.",
        )) from exc
    if not isinstance(titles, list) or not all(isinstance(t, str) for t in titles):
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary=f"{source} is not a JSON array of strings",
            detail="This command reads a flat list of raw title strings and clusters them.",
        ))
    return titles


def _cmd_pack_suggest(args) -> int:
    """Draft vocabulary.json, aliases.json and SUGGESTIONS.md from a bag of raw titles."""
    titles = _read_titles(Path(args.source).expanduser())
    curated = _read_titles(Path(args.curated).expanduser()) if args.curated else []
    suggestion = suggest(titles, cutoff=args.cutoff, curated=curated)
    out_dir = Path(args.out).expanduser()
    _write_suggestion(suggestion, out_dir, force=args.force)
    if (out_dir / "pack.json").is_file():
        print("\nReview SUGGESTIONS.md, trim by hand, then `slkit pack lint`.")
    else:
        print(f"\nReview SUGGESTIONS.md and trim by hand. This is not a complete pack yet: run\n"
              f"  slkit pack init --pack {out_dir}\nto add pack.json and the policy files, "
              f"then `slkit pack lint`.")
    return EXIT_OK


def _write_suggestion(suggestion, out_dir: Path, *, force: bool) -> None:
    """Write vocabulary.json, aliases.json and SUGGESTIONS.md into ``out_dir``.

    Shared by ``pack suggest-vocab`` and ``bootstrap`` so the refuse-to-clobber rule and the
    summary line are defined once. A non-empty vocabulary/aliases already present is refused
    unless ``force``; SUGGESTIONS.md is always safe to rewrite (it is only ever read by a human).
    """
    vocab_path = out_dir / "vocabulary.json"
    aliases_path = out_dir / "aliases.json"
    sugg_path = out_dir / "SUGGESTIONS.md"

    if not force:
        clashing = [p for p in (vocab_path, aliases_path) if _nonempty_pack_file(p)]
        if clashing:
            names = ", ".join(str(p) for p in clashing)
            raise DiagnosticError(Diagnostic(
                severity=ERROR,
                summary="refusing to overwrite existing pack files",
                detail=f"Already present and non-empty:\n    {names}\n\n"
                       f"Pass --force to overwrite, or point the output at an empty directory. "
                       f"(SUGGESTIONS.md is always rewritten.)",
            ))

    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(
        json.dumps(suggestion.vocabulary(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    aliases_path.write_text(
        json.dumps(suggestion.aliases(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sugg_path.write_text(render_markdown(suggestion), encoding="utf-8")

    folded = sum(c.size for c in suggestion.songs) - len(suggestion.songs)
    print(f"{suggestion.raw_count} raw titles -> {len(suggestion.songs)} song candidates "
          f"({folded} variants folded in), {len(suggestion.non_songs)} probable non-songs")
    # Reported separately because the two numbers answer different questions: how many names a
    # setlist source got to SPELL, and how many songs would not be in the draft at all without it.
    if suggestion.curated:
        print(f"  {len(suggestion.curated)} named by a setlist source "
              f"({len(suggestion.backfilled)} of them on no tape at all)")
    for path in (vocab_path, aliases_path, sugg_path):
        print(f"  wrote {path}")


_BOOTSTRAP_STEPS = 7


def _step(number: int, title: str) -> None:
    """Print one banner for one bootstrap step.

    The banners divide a run into marked phases. Without them the output is one wall of text.
    """
    print(f"\n=== step {number}/{_BOOTSTRAP_STEPS}: {title} ===", flush=True)


def _bootstrap_titles(config) -> list[str]:
    """Return each distinct track title the mirror source holds in the cache.

    This function reads the cache that ``ingest`` reads. It sends the titles through the same
    parser. Bootstrap thus needs no external tool between the pull and the draft.

    Return an empty list when the config declares no mirror source. Tape titles are one input to
    the draft, and a deployment with no tapes has none of them. The caller reports the absence.
    """
    importer = mirror(config)
    if importer is None:
        return []
    client = importer.make_client(config, RawCache(config.data_root))
    cached = client.cached_items(importer.identity(config),
                                 min_year=min_year(config, importer.id, None))
    return titles_from_cache(cached.items)


def _curated_titles(config) -> list[str]:
    """Return each distinct song name that the configured setlist sources report as played.

    The draft once read the mirror alone. The dictionary that judges every later stage thus came
    from taper prose, while the source that publishes real song titles sat pulled and unread.
    This function corrects that. See :func:`suggest` for what the names change.

    Read each source through its own ``candidates`` function, not off the raw blobs. The parsers
    already flag ``non_song`` entries. setlist.fm's ``tape`` marker names a song played over the
    PA, and the flag keeps it out. No rule here must know what a tape marker is.

    The loop covers the configured sources only. A registered source that this operator does not
    have contributes nothing and reports nothing.

    The normalizer is empty on purpose. No pack exists yet at draft time, because the pack is
    what this drafts. Names thus arrive as each source spells them, and the clustering does the
    rest.
    """
    cache = RawCache(config.data_root)
    empty = Normalizer(vocabulary=[])
    names: set[str] = set()
    have = {importer.id for importer in configured(config)}
    for importer in (imp for imp in secondary() if imp.id in have):
        for record in importer.candidates(config, cache, normalizer=empty,
                                          min_year=min_year(config, importer.id, None)) or ():
            for entries in list(record.get("sets") or ()) + [record.get("encore") or ()]:
                for entry in entries:
                    if not entry.get("non_song") and entry.get("song"):
                        names.add(str(entry["song"]).strip())
    return sorted(names, key=str.casefold)


def _cmd_bootstrap(config, args) -> int:
    """Run the whole pipeline behind one verb: store, pull, draft, scaffold, lint, ingest, derive.

    The pipeline initializes itself. Every parameter comes from the resolved config. A new
    deployment thus goes from an empty directory to a draft pack and a published corpus in one
    run.

    The config decides which sources take part. A ``[sources.<id>]`` table is the opt-in, and the
    operator writes one table for each source they have. This command names no source itself. A
    source added to the registry later works here when the operator configures it.

    Each step is cheap to repeat. Store init and pack init make the same result every time. A
    pull reads its own cache for free. The draft step refuses to overwrite a vocabulary that a
    person started to edit.
    """
    pack_dir = resolve_pack_dir(getattr(args, "pack", None), args.config)
    sources = configured(config)

    _step(1, "store init — create data_root and apply the schema")
    _cmd_store_init(config)

    if args.skip_pull:
        _step(2, "pull — SKIPPED (--skip-pull); drafting from the existing cache")
    else:
        _step(2, "pull — fetch raw data into the cache, one configured source at a time")
        for importer in sources:
            print(f"\n-- {importer.display_name} --", flush=True)
            _cmd_pull(config, argparse.Namespace(
                source=importer.id, min_year=None,
                force_rescan=args.force_rescan, dry_run=args.dry_run))

    if args.dry_run:
        print("\ndry run: stopping before draft, scaffold and ingest. "
              "Re-run without --dry-run to build.")
        return EXIT_OK

    _step(3, f"suggest-vocab — draft a vocabulary into {pack_dir}")
    # Two inputs, and a deployment can have either one alone. Tape titles come from the mirror
    # source. Published song names come from the setlist sources. A band that nobody taped has
    # only the names, and those are the better input: a person typed them on purpose.
    titles = _bootstrap_titles(config)
    curated = _curated_titles(config)
    print(f"  {len(titles)} cached tape title(s); {len(curated)} published song name(s)")
    if not mirror(config):
        print("  no mirror source is configured, so this draft reads no tape titles")
    if not titles and not curated:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="the cache holds nothing to draft a vocabulary from",
            detail="Each configured source cached no usable data under this config. Check that\n"
                   "your config declares a [sources.<id>] table for each source you have. Then\n"
                   "run `slkit pull` (or re-run bootstrap without --skip-pull) and retry.\n\n"
                   f"Sources configured now: {', '.join(imp.id for imp in sources) or '(none)'}",
        ))
    _write_suggestion(suggest(titles, cutoff=args.cutoff, curated=curated),
                      pack_dir, force=args.force)

    _step(4, "pack init — scaffold the remaining pack files, clobbering none")
    _cmd_pack_init(argparse.Namespace(pack=str(pack_dir), force=False, config=args.config))

    _step(5, "pack lint — validate the drafted pack")
    if _cmd_pack_lint(argparse.Namespace(
            pack=str(pack_dir), format="human", no_corpus=False, config=args.config)) != EXIT_OK:
        print("  (lint reported findings — expected for a fresh draft; they are a worklist, "
              "not a failure)")

    _step(6, "ingest — parse the cache and publish the merged corpus")
    # No source is named here. Ingest reads the mirror the config declares, and it folds each
    # configured setlist source on top of it.
    ingest(config, argparse.Namespace(
        source=None, pack=str(pack_dir), min_year=None,
        dry_run=False, force=False, profile=None, config=args.config))

    _step(7, "derive durations — reconcile each song's length across every tape of a night")
    derive(config, argparse.Namespace(
        derive_action="durations", pack=str(pack_dir), dry_run=False, config=args.config))

    print(f"\nbootstrap complete. The corpus is ingested and durations are derived — "
          f"`slkit export`\nwill now write a bundle to report from. Review "
          f"{pack_dir / 'SUGGESTIONS.md'}, trim vocabulary.json\nby hand, then re-run "
          f"`slkit ingest` and `slkit derive durations` to apply your edits.")
    return EXIT_OK


# Command name -> handler. A table rather than an if-chain, so adding a subcommand is one
# entry here and one parser above, and the dispatcher itself stops growing a branch per phase.
_COMMANDS = {
    "config": _cmd_config,
    "store": _cmd_store,
    "pull": _cmd_pull,
    "ingest": ingest,
    "derive": derive,
    "predict": _cmd_predict,
    "replay": _cmd_replay,
    "drops": _cmd_drops,
    "export": _cmd_export,
    "dump": _cmd_dump,
    "bootstrap": _cmd_bootstrap,
}


def _run(args) -> int:
    """Dispatch a parsed command. Assumes a command argparse already accepted."""
    if args.command == "pack":
        return _cmd_pack(args)               # resolves config lazily, only if --pack is absent
    return _COMMANDS[args.command](load_config(args.config), args)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises past this boundary."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_OK

    try:
        return _run(args)
    except DiagnosticError as exc:
        print(render(exc.diagnostic), file=sys.stderr)
        return EXIT_DIAGNOSTIC
    except SourceError as exc:
        # An upstream having a bad day is an ordinary Tuesday, not a bug in this program, so it
        # renders like every other failure instead of arriving as a traceback. Nothing is
        # half-written: payloads are cached one at a time, and the next run resumes from there.
        print(render(Diagnostic(
            severity=ERROR,
            summary=f"the source could not be read: {exc}",
            detail="Nothing was lost -- whatever was fetched before this is cached, and the\n"
                   "next pull carries on from there. If it persists, the source is likely down\n"
                   "or serving an error page; wait rather than retrying in a loop.",
        )), file=sys.stderr)
        return EXIT_DIAGNOSTIC


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

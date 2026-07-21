# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""``slkit ingest``: parse the cached raw payloads, merge them, and publish the corpus.

The parsing and the merging are the catalog's, and this module calls them without deciding
anything they decide. What lives HERE is the three things Phase 3 deliberately kept out of
``catalog/`` because they are IO, and one that is reporting:

1. **The no-shrink guard.** A merge that produces less than half the shows already stored is
   almost never a corpus that got smaller; it is an upstream that went missing. Both of the old
   repo's scripts had this guard and neither ``parse`` nor ``merge`` does, because neither of them
   knows there is a previous snapshot. Refusing costs one re-run. Not refusing overwrites years of
   work with the output of a bad afternoon, and nothing anywhere says it happened.

2. **Reading the cache honestly.** ``cached_items`` reports what it could not read, and this is
   where that becomes something a person sees. An ingest over a half-finished pull publishes a
   corpus missing a third of the shows, and every number in the report looks fine.

3. **Saying why a night is missing.** Three rules can refuse an item and all three leave the same
   trace downstream, which is none: the date is not in the corpus, exactly as if nobody had ever
   taped it. The parser reports which rule refused what, and the reason for a dropped date is read
   back out of the pack that asked for the drop, so "this show is absent" becomes "this show is
   absent BECAUSE".

4. **The stats block.** Winners by source, new dates, dates that vanished, and where a date
   changed which source it trusts. The point of a source flip is that it is usually right and
   occasionally the first sign something broke, and neither is visible from the corpus alone.
"""

from __future__ import annotations

import textwrap
from collections import Counter
from dataclasses import dataclass

from ..catalog.merge import merge_shows, override_disagreements
from ..catalog.pack import load_pack
from ..catalog.parse import (DROPPED_DATE, NO_DATE, NOT_THIS_BAND, count_songs,
                             parse_archive_items)
from ..diagnostics import ERROR, Diagnostic, DiagnosticError
from ..sources.archive_org import ArchiveOrgClient
from ..store import Store
from ..store.raw_cache import RawCache
from .common import min_year, required_setting, resolve_pack_dir

EXIT_OK = 0

# Below this fraction of the stored corpus, a merge is refused as an accident rather than
# published as a result. Ported from the old repo, which arrived at it the hard way.
NO_SHRINK_FRAC = 0.5

# A drop reason in a pack runs to a paragraph, and three of them is a wall of text nobody reads
# past. The first sentence says which show it was; the file says the rest.
_REASON_WIDTH = 96

_SKIP_LABELS = {
    NOT_THIS_BAND: "title names a different band (a side project's tape)",
    NO_DATE: "no date we can believe",
}


@dataclass(frozen=True)
class _Totals:
    """The two nouns the corpus is measured in.

    Both, because they fail independently and only one of them is what the corpus is FOR. Shows
    catch an upstream that went missing. Songs catch the thing shows cannot see at all: a pack
    edit that widens a junk or gear fragment deletes a song from EVERY night at once, and the
    date count, the source counts, the new-date list and the removed-date list all come out
    byte-identical to the run before it. Six songs left a six-show corpus in testing and the
    report was the same report.
    """

    shows: int = 0
    songs: int = 0

    def __str__(self) -> str:
        return f"{self.shows} shows / {self.songs} songs"


@dataclass(frozen=True)
class _Previous:
    """What was in the corpus before this run, for the parts of the report that are a diff."""

    totals: _Totals
    source_by_date: dict[str, str]


def ingest(config, args) -> int:
    """Parse the cache, merge, report, guard, publish. Returns a process exit code.

    Reporting comes BEFORE the guard on purpose. The guard's own advice is to go and look at what
    changed, and the run that just computed it is the only thing that knows -- so it says so
    first, and refuses second. A refusal with the evidence withheld sends someone to re-derive by
    hand what was on screen a moment ago.
    """
    pack = load_pack(resolve_pack_dir(getattr(args, "pack", None), args.config))
    collection = required_setting(
        config, ("sources", args.source), "collection",
        f"An ingest reads what `slkit pull {args.source}` cached, and needs the same collection.")

    cache = RawCache(config.data_root)
    cached = ArchiveOrgClient(config, cache).cached_items(
        collection, min_year=min_year(config, args.source, args.min_year))
    _report_cache(cached, collection, args.source)

    parsed = parse_archive_items(cached.items, normalizer=pack.normalizer,
                                 policy=pack.archive_policy())
    merged = merge_shows(parsed.shows, overrides=pack.corpus.overrides, policy=pack.merge_policy())
    produced = _Totals(shows=len(merged.shows),
                       songs=sum(count_songs(show["sets"], show["encore"])
                                 for show in merged.shows))

    with Store(config.data_root) as store:
        store.init()
        previous = _Previous(totals=_Totals(store.show_count(), store.song_count()),
                             source_by_date=store.show_sources())
        _report_merge(merged, previous, produced, pack, len(cached.items))
        _report_skipped(parsed.skipped, pack, len(cached.items))
        # --dry-run is exempt: the guard protects the stored corpus, and a dry run cannot reach
        # it. Refusing anyway would make the one command that is safe to run while diagnosing a
        # shrink the one command that will not tell you about it.
        if args.dry_run:
            _warn_shrink(produced, previous.totals, force=args.force)
            print("dry run: nothing written")
            return EXIT_OK
        _guard_no_shrink(produced, previous.totals, force=args.force)
        store.replace_shows(merged.shows)
    return EXIT_OK


def _shrink(produced: _Totals, previous: _Totals) -> str | None:
    """Which noun collapsed, as a sentence, or ``None`` if neither did.

    Either one is enough. A run that keeps every date and empties every setlist has not held
    steady, and neither has one that keeps the songs of half the nights.
    """
    # No special case for a first run. An empty corpus makes the floor zero, and nothing is below
    # zero -- the arithmetic already says "nothing stored means nothing to measure against". An
    # explicit `if before` here would be a branch no input can distinguish, which is a branch no
    # test can hold to account.
    for noun, now, before in (("shows", produced.shows, previous.shows),
                              ("songs", produced.songs, previous.songs)):
        if now < NO_SHRINK_FRAC * before:
            return f"{noun} fell from {before} to {now}"
    return None


def _guard_no_shrink(produced: _Totals, previous: _Totals, *, force: bool) -> None:
    """Refuse to replace a good corpus with a much smaller one.

    Measured against what is already stored, not against anything within this run, because the
    failure being caught is upstream: a source that answered with nothing, a cache half deleted,
    a pack edit that turned the band filter on everything or widened a fragment onto a real song.
    Each of those produces a small, clean, entirely wrong corpus in which every other number
    looks reasonable.

    Note the floor is per-run and not cumulative: 100 -> 51 -> 26 clears it every time. The
    removed-date list above is what makes that visible.
    """
    collapsed = _shrink(produced, previous)
    if collapsed is None or force:
        return
    raise DiagnosticError(Diagnostic(
        severity=ERROR,
        summary=f"refusing to publish: {collapsed}",
        detail=f"That is under {NO_SHRINK_FRAC:.0%} of what is stored, which is almost always an\n"
               "upstream that went missing rather than a corpus that got smaller. The stored\n"
               "corpus is untouched, and the report above says what this run would have done.\n\n"
               "Check the cache first (`slkit pull` re-lists without re-fetching), and check\n"
               "whether a pack edit widened the band filter, a junk or gear fragment, or added a\n"
               "drop date. A fragment that reaches a real title deletes it from every show at\n"
               "once, which is what the song count is here to catch.\n\n"
               "Use --dry-run to see the full report without writing, and --force to publish\n"
               "a shrink you meant.",
    ))


def _warn_shrink(produced: _Totals, previous: _Totals, *, force: bool) -> None:
    """Say what the guard WOULD have refused, on a run that cannot trigger it."""
    collapsed = _shrink(produced, previous)
    if collapsed is not None and not force:
        print(f"  warning: a real run would refuse this ({collapsed}); --force publishes it")


def _report_cache(cached, collection: str, source: str) -> None:
    """What the cache had, and everything about it that should stop someone trusting the run.

    An empty listing is refused rather than published. A cached page that is valid JSON of the
    wrong shape -- an archive.org error document, most likely -- parses to zero docs, and every
    check below would have passed it: a page WAS found, so ``pages`` is not zero, and ``expected``
    is ``None`` because no ``numFound`` was readable, which switches off the only count there is.
    The result was an empty corpus published with exit 0.
    """
    if cached.pages == 0 or cached.listed == 0:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary=f"no usable cached listing for {source} collection {collection!r}",
            detail=f"Nothing to ingest. Run `slkit pull {source}` first.\n\n"
                   "If you have pulled, two things produce this. A listing is cached per\n"
                   "(collection, min_year), so an ingest with a different min_year than the pull\n"
                   "looks in a place nothing was ever written. And a cached page that is valid\n"
                   "JSON of the wrong shape -- an error document the source returned with status\n"
                   "200 -- reads as a listing that names nothing. `slkit pull --force-rescan`\n"
                   "replaces it.",
        ))
    print(f"ingest {source}: {len(cached.items)} cached items")
    if cached.expected is None:
        # Not cosmetic. `expected` is the only completeness check that counts anything, and it is
        # absent exactly when the listing payload was damaged -- so its absence is itself news.
        print("  note: the cached listing recorded no item count, so this run cannot tell "
              "whether it is complete")
    elif cached.listed != cached.expected:
        print(f"  warning: the cached listing names {cached.listed} items but promised "
              f"{cached.expected}; the pull did not finish")
    if cached.unidentified:
        print(f"  {cached.unidentified} cached listing doc(s) carry no identifier, so nothing "
              "could ever be fetched for them")
    if cached.absent:
        print(f"  {len(cached.absent)} listed item(s) have no readable cached metadata, so their "
              "shows are NOT in this corpus:")
        for identifier in cached.absent:
            print(f"    {identifier}")
        print("  An ordinary pull will not re-fetch these: it skips an identifier whose payload\n"
              "  file exists, and a truncated write exists. Use `--force-rescan` to replace them.")
    if cached.truncated:
        print("  warning: the cached listing hit the paging backstop, so it is a PREFIX of the\n"
              "  collection. This corpus will look complete and will not be.")


def _report_merge(merged, previous: _Previous, produced: _Totals, pack, n_items: int) -> None:
    """Winners by source, what changed against the stored corpus, and the override review."""
    winners = Counter(show["source"] for show in merged.shows)
    # Songs alongside shows, because the run that quietly empties every setlist keeps every date.
    print(f"  {produced} from {n_items} items (was {previous.totals}); "
          f"winners by source: {dict(sorted(winners.items()))}")

    now = {show["date"]: show["source"] for show in merged.shows}
    added = sorted(set(now) - set(previous.source_by_date))
    removed = sorted(set(previous.source_by_date) - set(now))
    flips = [(date, previous.source_by_date[date], now[date])
             for date in sorted(set(previous.source_by_date) & set(now))
             if previous.source_by_date[date] != now[date]]

    # Added is abbreviated; removed never is. A first ingest adds nine hundred dates and printing
    # them scrolls every warning above off the screen, whereas a date LEAVING is the loudest
    # thing this report can say and the no-shrink guard only catches it in bulk.
    print(f"  +{len(added)} new date(s)" + (f": {_abbreviated(added)}" if added else ""))
    if removed:
        print(f"  -{len(removed)} removed date(s): {removed}")
    if flips:
        print(f"  {len(flips)} date(s) changed source:")
        for date, was, now_source in flips:
            print(f"    {date}: {was} -> {now_source}")

    if merged.applied:
        print(f"  {len(merged.applied)} override(s) applied: {merged.applied}")
    if merged.refused:
        # Two files in the same pack disagreeing about one night. Someone sat and listened to
        # write that override, and without this it does nothing and says nothing.
        print(f"  {len(merged.refused)} override(s) NOT applied, because the date is in "
              f"drop_dates: {list(merged.refused)}")
        print("    A dropped date stays dropped however carefully it was written down. Remove it "
              "from\n    drop_dates if the override is what you now want.")
    _report_disagreements(merged, pack)


def _abbreviated(dates: list[str], limit: int = 12) -> str:
    """A date list, cut short with a count of what was cut, never silently truncated."""
    if len(dates) <= limit:
        return str(dates)
    return f"{dates[:limit]} and {len(dates) - limit} more"


def _report_disagreements(merged, pack) -> None:
    """Sources carrying a real song an override does not have.

    An override always wins, so nothing else will ever tell us it went stale or was wrong to
    begin with. This is the only signal there is, which is why it prints in full.
    """
    disagreements = override_disagreements(merged.candidates, pack.corpus.overrides, pack.normalizer)
    if not disagreements:
        return
    print(f"  override review ({len(disagreements)}):")
    for item in disagreements:
        print(f"    {item['date']}: the override has {item['n_override']} songs; source "
              f"{item['source']!r} ({item['identifier']})")
        print(f"      carries {len(item['missing'])} song(s) the override lacks: "
              f"{item['missing']}")
        print("      -> the override still won. Re-check by ear, then update or delete it.")


def _report_skipped(skipped, pack, n_items: int) -> None:
    """Every item the parser refused, grouped by the rule that refused it.

    A dropped date is reported with the reason the pack gave for dropping it. That reason is the
    only thing a later reader can check the call against, and it is a long way from the run that
    acts on it -- so it is repeated here rather than left in a file nobody opens.
    """
    if not skipped:
        return
    by_reason = Counter(item.reason for item in skipped)
    print(f"  refused {len(skipped)} of {n_items} item(s):")
    # Iterate the reasons OBSERVED, not the label table, so a reason nobody wrote a label for
    # still gets a line. A module whose whole thesis is "say why a night is missing" must not
    # answer a new refusal with silence, and this table grows when the second source lands.
    for reason, count in sorted(by_reason.items()):
        if reason == DROPPED_DATE:
            continue                       # reported below, with the pack's reason for each date
        label = _SKIP_LABELS.get(reason, f"{reason} (no label for this reason yet)")
        print(f"    {count}: {label}")
        # Named, not just counted. This is the refusal that is a JUDGEMENT about a real band --
        # a taper who inverted the article gets their whole night thrown away -- so it has to
        # leave behind something to grep the cache for.
        for item in skipped:
            if item.reason == reason:
                print(f"      {item.identifier or '(no identifier)'}")
    dropped = sorted({item.date for item in skipped if item.reason == DROPPED_DATE})
    if not dropped:
        return
    print(f"    {by_reason[DROPPED_DATE]} on {len(dropped)} dropped date(s):")
    for date in dropped:
        # Filtered on the reason as well as the date. Only DROPPED_DATE records a date today,
        # but the field is general, and the day another rule fills it this count would quietly
        # absorb refusals that had nothing to do with the drop.
        tapes = sum(1 for item in skipped if item.reason == DROPPED_DATE and item.date == date)
        why = textwrap.shorten(str(pack.corpus.drop_dates.get(date) or "no reason recorded"),
                               width=_REASON_WIDTH, placeholder=" [...]")
        print(f"      {date} ({tapes} tape(s)): {why}")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""``slkit export``: hand what was derived to something that draws it.

This is the boundary of the project. Everything above it decides things; this writes one file and
stops. What reads that file -- a web page, a notebook, someone's spreadsheet -- imports nothing
from setlistkit and is not expected to. That separation is the whole reason the bundle exists: the
implementation this replaces had a dashboard builder that reached back into the pipeline's working
directory for seven files and a raw cache, and so it could only ever run on the machine that ran
the pipeline.

LIKE ``derive``, IT READS THE STORE AND NEVER THE CACHE, for the same reason and with the same test
holding it: the raw cache is gitignored, so anything reading it works on a workstation and comes
back empty in production.

The rule that matters is not "reads only tables derive wrote" -- the Songbook bundle disproves
that on its first day, computed ENTIRELY at export time from the corpus, with no ``derive`` pass
behind it at all. The rule is: NEVER PUBLISH A NUMBER THAT DISAGREES WITH A STORED ONE.
``features.py`` already lived under that rule before the Songbook did -- there has never been a
features table, and nothing in ``derive`` computes one -- so a windowed Tape Measure's structural
profiles were always computed here, from the stored shows, the one thing this module has always
computed rather than read. The ranged Tape Measure's statistics live under the same rule:
recomputed, yes, but by calling the same function ``derive`` itself calls
(:func:`~setlistkit.catalog.lengths.song_stats`), so a ranged bundle and the stored table can
differ in population but never in method -- see :func:`_stats_for`.

The Songbook has no stored table to differ from at all -- there is no play-frequency number
``derive`` has ever published -- so for it the rule above is a design constraint, not something
any unit test can check: it PASSES whether or not the bundle is right. What actually holds a
Songbook export accountable is the golden file locking its shape and the intersection parity gate
checking its numbers against a second, independent pipeline (see
``scripts/songbook_parity.py``). Do not point at this docstring later as evidence that anything
here was tested; it wasn't, and by its nature it can't be.
"""

from __future__ import annotations

from datetime import date

from ..catalog import songbook
from ..catalog import window as report_window
from ..catalog.features import song_features
from ..catalog.lengths import from_row, song_stats
from ..catalog.pack import load_pack
from ..catalog.tapemeasure import Concluded, SCHEMA, bundle
from ..store import Store, daterange
from . import exportio
from .common import resolve_pack_dir

EXIT_OK = 0
EXIT_NOTHING = 1


def export(config, args) -> int:
    """`slkit export tapemeasure|songbook`, defaulting to tapemeasure; or `--explain`, which
    builds nothing.

    Defaulting rather than requiring the noun, the way `derive` does. There are two bundles today.
    `--explain` is checked first because it must win regardless of which sub-verb (or none) was
    also given -- it never builds anything, so no sub-verb dispatch should run ahead of it.
    """
    if getattr(args, "explain", False):
        return _explain(config, args)
    if getattr(args, "export_action", None) == "songbook":
        return _songbook(config, args)
    return _tapemeasure(config, args)


def _explain(config, _args) -> int:
    """``slkit export --explain``: what every configured report's window resolves to, and why.

    Printed before anything is built. A compact literal syntax people half-remember is printf
    all over again -- the fix is not a better paragraph in a doc, it is the tool stating what it
    understood before it does anything (design doc: "--explain, which matters more than the
    grammar"). EVERY report under `[reports.*]` is explained in one pass, not just whichever one
    this run happens to build, because the failure this exists for is a wrong unit discovered on
    the WRONG report months later.
    """
    reports = config.section("reports")
    with Store(config.data_root) as store:
        store.init()
        span = store.corpus.first_and_last()
        if span is None:
            print("no shows are stored -- nothing to explain")
            return EXIT_NOTHING
        first, last = span
        print(f"anchor: last_show = {last}   "
              f"(latest show in corpus, {store.corpus.show_count()} shows held)\n")
        if not reports:
            print("no [reports.*] configured -- nothing else to explain")
            return EXIT_OK
        for name in sorted(reports):
            _explain_one(store, config, name, first, last)
    return EXIT_OK


def _explain_one(store, config, name: str, first: str, last: str) -> None:
    """One report's window: its literals restated in words, its resolved dates, its show count.

    The page-level header above prints ``anchor: last_show = {last}`` exactly once, because
    `last_show` is the default every report inherits unless it says otherwise. A report that
    overrides its anchor (an explicit date, or `first_show`) resolves everything below against a
    DIFFERENT date, so its own anchor is restated here whenever it disagrees with that header --
    otherwise the header's anchor and this block's resolved dates read as consistent when they
    are not, and the report's real anchor would be visible nowhere unless it happened to also
    turn up inside a clamp note (which does not fire for every offset, and says nothing when it
    does not).
    """
    spec = report_window.window_spec_from_config(config, name)
    if spec is None:
        print(f"{name}\n  (no [reports.{name}.window] configured)\n")
        return
    resolved = report_window.resolve_explained(spec, first=first, last=last)
    since, until = resolved.as_dates()
    print(name)
    if resolved.anchor != last:
        print(f"  anchor: {_anchor_words(spec.anchor, resolved.anchor)}")
    since_key = "since_back" if spec.since_back is not None else "since_from"
    _explain_endpoint(since_key, spec.since_back or spec.since_from, resolved.since)
    until_key = "until_back" if spec.until_back is not None else "until"
    _explain_endpoint(until_key, spec.until_back, resolved.until)
    shows = store.corpus.shows(since=since, until=until)
    days = (date.fromisoformat(until) - date.fromisoformat(since)).days + 1
    print(f"  window: {since} .. {until}  inclusive  ·  {days} days  ·  "
          f"{len(shows)} shows\n")


def _anchor_words(configured: str, resolved: str) -> str:
    """How to state a report's own anchor: `"first_show = 2020-03-14"`, or just the date itself.

    `configured` is `spec.anchor` -- either the keyword `last_show`/`first_show`, or an explicit
    `YYYY-MM-DD` literal (see `catalog.window.WindowSpec`). A keyword is restated next to what it
    resolved to, the same shape the page-level header already uses (`anchor: last_show = ...`);
    a literal date IS its own resolution, so repeating it twice (`anchor: 2025-03-31 =
    2025-03-31`) would say nothing a single date does not already say.
    """
    if configured in ("first_show", "last_show"):
        return f"{configured} = {resolved}"
    return resolved


def _explain_endpoint(key: str, literal: str | None, endpoint) -> None:
    """One line of ``--explain`` output for one endpoint, plus its clamp note if one fired.

    Silent about clamping the rest of the time -- a note that prints unconditionally would bury
    the one time it actually matters, which is the same "printed only when it looks bad has no
    baseline" problem the tape measure's own `_report_gaps` avoids by always printing its counts.
    Here the right default is the opposite: say nothing unless something happened, because a
    clamp is the ANOMALY, not the baseline.
    """
    shown = f'"{literal}"' if literal is not None else "(omitted)"
    print(f"  {key:<10} = {shown:<9} ->  {endpoint.value}   {endpoint.words}")
    if endpoint.clamp_note:
        print(f"  note: {endpoint.clamp_note}")


def _window_for(config, report_name: str, args,
                span: tuple[str, str] | None) -> tuple[str | None, str | None]:
    """One report's effective window: an explicit flag beats ``[reports.<report_name>.window]``.

    Shared by ``_tapemeasure`` and ``_songbook`` so the two sub-verbs cannot drift into resolving
    windows two different ways. ``span`` is the corpus's earliest/latest stored show date
    (``store.corpus.first_and_last()``), or ``None`` for an empty corpus -- there is no
    `last_show`/`first_show` to anchor a configured window against yet, and the caller's own
    "nothing stored" message fires immediately after this returns regardless of what it returns,
    so falling back to bare flag-checking (no config lookup at all) is enough here, rather than
    duplicating ``_explain``'s explicit "nothing to explain" message a second time.
    """
    since_flag = getattr(args, "since", None)
    until_flag = getattr(args, "until", None)
    if span is None:
        return (daterange.check_date(since_flag, "--since"),
                daterange.check_date(until_flag, "--until"))
    first, last = span
    return exportio.resolve_window(config, report_name, since_flag=since_flag,
                                   until_flag=until_flag, first=first, last=last)


def _tapemeasure(config, args) -> int:
    """Read the last derivation out of the store and write it as one bundle.

    ``--since``/``--until`` narrow every section to one inclusive window, statistics included --
    an explicit flag beats a configured ``[reports.tapemeasure.window]`` per endpoint (see
    :func:`_window_for`), and a run with neither stays the open window this command has always
    defaulted to. See :func:`_stats_for` for why the statistics are the one thing here that is
    computed rather than read, and why that is not the recomputation this module's docstring
    refuses to do.
    """
    with Store(config.data_root) as store:
        store.init()
        since, until = _window_for(config, "tapemeasure", args, store.corpus.first_and_last())
        window = {"since": since, "until": until}
        performances = store.durations.performances(**window)
        if not performances:
            print(_nothing_message(since, until))
            return EXIT_NOTHING
        concluded = Concluded(
            performances=performances,
            stats=_stats_for(performances, ranged=since is not None or until is not None,
                             stored=store.durations.song_length_stats),
            review=store.durations.review(**window),
            abandoned=store.durations.abandoned(**window),
            edges=store.durations.edges(**window),
        )
        shows = store.corpus.shows(**window)
        payload = bundle(concluded, song_features(shows),
                         store.tapes.uploader_counts(**window),
                         corpus_shows=len(shows),
                         recordings_read=store.tapes.recording_count(**window),
                         since=since, until=until)
    _report(payload)
    if getattr(args, "dry_run", False):
        print("dry run: nothing written")
        return EXIT_OK
    out = exportio.write_bundle(payload, getattr(args, "out", None) or "tapemeasure.json")
    print(f"  wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")
    return EXIT_OK


def _stats_for(performances: list[dict], *, ranged: bool, stored) -> list[dict]:
    """The per-song statistics for what is actually in this bundle.

    Unranged, these are READ from the table ``derive`` wrote, and this module's promise holds
    exactly as stated: an export cannot publish a number that disagrees with the stored one.

    Ranged, there is no such table to read. A song's statistics are computed over every year it
    was played -- ``store.db`` keeps ``song_length_stats`` out of its date map for this very
    reason -- so the stored median is not a narrower answer to the same question, it is the answer
    to a different one. Publishing it beside a windowed performance list would put two populations
    in one file under one heading, which is the failure this recomputation exists to avoid rather
    than an exception carved out of the rule.

    What makes that safe is that nothing is reimplemented. The rows are rebuilt into the type they
    were stored from and handed to :func:`~setlistkit.catalog.lengths.song_stats`, the same
    function ``derive`` calls, so a ranged bundle and the stored table can differ in population but
    never in method. Ordered to match what the store hands back, because the bundle is diffed
    between runs by a person.
    """
    if not ranged:
        return stored()
    stats = song_stats([from_row(row) for row in performances])
    return [vars(stat) for stat in sorted(stats, key=lambda s: (-s.median_seconds, s.song))]


def _nothing_message(since: str | None, until: str | None) -> str:
    """Why an empty bundle was refused, distinguishing an empty store from an empty window.

    "No durations are stored" sent someone to re-run ``derive`` when the real answer was that they
    asked for a year the band did not tour, and derive would have cheerfully rewritten a correct
    table to prove it.
    """
    if since is None and until is None:
        return "nothing to export: no durations are stored. Run `slkit derive durations` first."
    return (f"nothing to export: no performances fall in {daterange.label(since, until)}. "
            "The store may hold others outside it -- `slkit store status` counts them all.")


def _nothing_songbook(since: str | None, until: str | None) -> str:
    """Why an empty bundle was refused before the show floor even ran."""
    if since is None and until is None:
        return "nothing to export: no shows are stored. Run `slkit ingest` first."
    return (f"nothing to export: no shows fall in {daterange.label(since, until)}. "
            "The store may hold others outside it -- `slkit store status` counts them all.")


def _nothing_songbook_below_floor(n_shows_seen: int, since: str | None,
                                  until: str | None) -> str:
    """Why an empty bundle was refused AFTER the show floor ran -- a different emptiness from
    :func:`_nothing_songbook`, which fires before the floor gets a look at anything."""
    where = "the whole corpus" if since is None and until is None else daterange.label(since, until)
    return (f"nothing to export: all {n_shows_seen} show(s) in {where} fall below the "
            f"{songbook.MIN_SHOW_SONGS}-entry floor (protection against truncated parses). "
            "Nothing was written.")


def _report_songbook(payload: dict) -> None:
    """What went in the bundle, in the terms the page will show it in."""
    generated = payload["generated"]
    print(f"export songbook ({payload['schema']})")
    window = generated["window"]
    if window["since"] is not None or window["until"] is not None:
        print(f"  window {daterange.label(window['since'], window['until'])}")
    print(f"  {generated['n_shows']} show(s), {generated['first']} to {generated['last']}")
    print(f"  {len(payload['vocab'])} distinct name(s) in this window "
          f"({generated['catalog']} in the pack's own vocabulary)")
    if payload["unknown"]:
        print(f"  {len(payload['unknown'])} of them are not in the pack -- kept and flagged, "
              "not dropped")
    if generated["below_floor"]:
        print(f"  {generated['below_floor']} show(s) below the {songbook.MIN_SHOW_SONGS}-entry "
              "floor, excluded")
    if generated["deduped"]:
        print(f"  {generated['deduped']} repeat play(s) within a show, collapsed to one index "
              "each")


def _songbook(config, args) -> int:
    """`slkit export songbook`: every show's songs, over one inclusive window.

    Unlike `_tapemeasure`, nothing here is read from a `derive`-written table -- there is none.
    The bundle is a pure function of the stored corpus and the pack's vocabulary (see
    `catalog.songbook.bundle`), so this handler's only jobs are: resolve the window (an explicit
    flag beats a configured ``[reports.songbook.window]`` per endpoint -- see :func:`_window_for`),
    load the pack, read the corpus, and hand both to the pure function untouched.
    """
    pack = load_pack(resolve_pack_dir(getattr(args, "pack", None), args.config))
    with Store(config.data_root) as store:
        store.init()
        since, until = _window_for(config, "songbook", args, store.corpus.first_and_last())
        shows = store.corpus.shows(since=since, until=until)
        if not shows:
            print(_nothing_songbook(since, until))
            return EXIT_NOTHING
        payload = songbook.bundle(shows, normalizer=pack.normalizer, since=since, until=until,
                                  fingerprint=exportio.fingerprint(store))
    if payload["generated"]["n_shows"] == 0:
        print(_nothing_songbook_below_floor(len(shows), since, until))
        return EXIT_NOTHING
    _report_songbook(payload)
    if getattr(args, "dry_run", False):
        print("dry run: nothing written")
        return EXIT_OK
    out = exportio.write_bundle(payload, getattr(args, "out", None) or "songbook.json")
    print(f"  wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")
    return EXIT_OK


def _report(payload: dict) -> None:
    """What went in the bundle, in the terms the page will show it in."""
    totals, generated = payload["totals"], payload["generated"]
    span = generated["date_range"] or ["?", "?"]
    window = generated["window"]
    print(f"export tapemeasure ({SCHEMA})")
    if window["since"] is not None or window["until"] is not None:
        # Printed before the counts, not after, because it is the thing that explains them. A
        # window that silently dropped nine tenths of the corpus should not be discovered by
        # noticing the totals look small.
        print(f"  window {daterange.label(window['since'], window['until'])} "
              "(statistics recomputed over it)")
    print(f"  {totals['performances']} performance(s) over {totals['nights']} night(s), "
          f"{span[0]} to {span[1]}")
    print(f"  {totals['songs']} song(s) with statistics "
          f"({totals['songs_at_n3']} at n>=3), {totals['hours']} hours of music")
    print(f"  {len(payload['credits'])} taper(s) credited from "
          f"{generated['recordings_read']} stored tape(s)")
    _report_gaps(payload, totals)


def _report_gaps(payload: dict, totals: dict) -> None:
    """The parts of the bundle a reader should know are incomplete before they draw it.

    Printed every time rather than only when they look bad. A number that appears only when
    something is wrong is a number nobody has a baseline for.
    """
    unprofiled = sum(1 for song in payload["songs"] if song["features"] is None)
    alone, total = totals["single_tape_performances"], totals["performances"]
    print(f"  {alone} performance(s) rest on a single taper ({100 * alone / total:.1f}%)")
    if unprofiled:
        # A song with lengths but no profile was timed off a tape whose night the corpus does not
        # list it in. It is a disagreement between the two halves, not a rounding error.
        print(f"  {unprofiled} song(s) have lengths but no structural profile")
    if totals["withheld"]:
        print(f"  held back from the per-song pools: {totals['withheld']}")
    print(f"  {totals['tapes_queued_for_review']} tape(s) queued for review, "
          f"{totals['tapes_abandoned']} abandoned")

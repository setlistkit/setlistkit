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
holding it. It reads only tables ``derive`` wrote, plus the corpus, so an export is a view of the
last derivation and cannot quietly recompute a number that disagrees with the stored one.

The one thing it computes rather than reads is the structural features, because they are a function
of the setlists alone and there is no features table to read them from. They come from the STORED
shows, so this stays true either way.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..catalog.features import song_features
from ..catalog.lengths import from_row, song_stats
from ..catalog.tapemeasure import Concluded, SCHEMA, bundle
from ..store import Store, daterange

EXIT_OK = 0
EXIT_NOTHING = 1

# Written with a trailing newline and stable key order so the file is diffable and a change to the
# data reads as a change to the data. Indented for the same reason: this is a file people open.
_JSON = {"indent": 2, "sort_keys": False, "ensure_ascii": False, "default": str}


def export(config, args) -> int:
    """`slkit export tapemeasure`, defaulting to tapemeasure.

    Defaulting rather than requiring the noun, the way ``derive`` does. There is one bundle today.
    """
    return _tapemeasure(config, args)


def _tapemeasure(config, args) -> int:
    """Read the last derivation out of the store and write it as one bundle.

    ``--since``/``--until`` narrow every section to one inclusive window, statistics included. See
    :func:`_stats_for` for why the statistics are the one thing here that is computed rather than
    read, and why that is not the recomputation this module's docstring refuses to do.
    """
    since = daterange.check_date(getattr(args, "since", None), "--since")
    until = daterange.check_date(getattr(args, "until", None), "--until")
    window = {"since": since, "until": until}
    with Store(config.data_root) as store:
        store.init()
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
    out = _write(payload, getattr(args, "out", None) or "tapemeasure.json")
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


def _write(payload: dict, out: str) -> Path:
    """Serialize the bundle to ``out``, creating its directory.

    Written whole and replaced whole. A consumer polling this file should see the previous bundle
    or the next one, never four megabytes of a bundle that is still being written -- which is what
    a reader hitting a partial file gets, and it fails as a JSON parse error somewhere far from
    here.
    """
    path = Path(out).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + ".partial")
    scratch.write_text(json.dumps(payload, **_JSON) + "\n", encoding="utf-8")
    scratch.replace(path)
    return path


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

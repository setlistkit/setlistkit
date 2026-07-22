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
from ..catalog.tapemeasure import Concluded, SCHEMA, bundle
from ..store import Store

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
    """Read the last derivation out of the store and write it as one bundle."""
    with Store(config.data_root) as store:
        store.init()
        concluded = Concluded(
            performances=store.durations.performances(),
            stats=store.durations.song_length_stats(),
            review=store.durations.review(),
            abandoned=store.durations.abandoned(),
            edges=store.durations.edges(),
        )
        if not concluded.performances:
            print("nothing to export: no durations are stored. "
                  "Run `slkit derive durations` first.")
            return EXIT_NOTHING
        shows = store.corpus.shows()
        payload = bundle(concluded, song_features(shows), store.tapes.uploader_counts(),
                         corpus_shows=len(shows),
                         recordings_read=store.tapes.recording_count())
    _report(payload)
    if getattr(args, "dry_run", False):
        print("dry run: nothing written")
        return EXIT_OK
    out = _write(payload, getattr(args, "out", None) or "tapemeasure.json")
    print(f"  wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")
    return EXIT_OK


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
    print(f"export tapemeasure ({SCHEMA})")
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

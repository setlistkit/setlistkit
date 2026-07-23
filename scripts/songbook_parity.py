#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The Songbook's Slice 1 gate: does setlistkit's per-song count agree with the POC's, where both
corpora hold the same night?

Not a pytest test, and deliberately not one: the assertion is measured against a second, real
codebase's real output (the POC's `site/songbook/index.html`, which embeds its corpus in a
`<script id="data" type="application/json">` block) and against a real setlistkit export, neither
of which any unit fixture can stand in for. See the design document's "Verification" section for
why the naive version of this gate ("restrict to 2020+ and diff") cannot fail and is therefore not
a gate.

Usage:

    pyenv/bin/python scripts/songbook_parity.py \\
        --poc /path/to/famoe.ly/site/songbook/index.html \\
        --bundle songbook.json

Exit code is 0 only if every date held by BOTH corpora agrees exactly on every song's `plays`.
Everything else -- the size of all three date sets, and a per-date, per-song accounting of where
they disagree outside the intersection -- is printed, never asserted on, because outside the
intersection a mismatch is EXPECTED and the report exists to say WHY, not to fail on it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_DATA_BLOCK = re.compile(r'<script id="data" type="application/json">(.*?)</script>', re.S)


def _load_poc(path: Path) -> dict:
    """The POC's embedded corpus: `{"meta":..., "vocab":[...], "shows":[{"d":..., "s":[...]}]}`."""
    text = path.read_text(encoding="utf-8")
    match = _DATA_BLOCK.search(text)
    if not match:
        raise SystemExit(f'{path}: no <script id="data"> block found -- is this the right file?')
    return json.loads(match.group(1))


def _load_bundle(path: Path) -> dict:
    """A `slkit export songbook` bundle, schema-checked before anything reads its contents."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not str(data.get("schema", "")).startswith("setlistkit.songbook/"):
        raise SystemExit(f"{path}: schema is {data.get('schema')!r}, not a songbook bundle")
    return data


def _plays_by_date(vocab: list, shows: list) -> dict[str, Counter]:
    """date -> Counter(song name -> 1) for every song that date's show list names.

    A show either contains a song or it doesn't (both bundles already dedupe within a night), so
    the count either side contributes per date is always 0 or 1 -- summing this across dates, done
    by :func:`_total_plays`, is exactly what "plays" means in the design document's sense: shows
    containing the song, never performances of it. ``set()`` over each show's indices is what
    holds that guarantee even against a show list that has NOT deduped its indices -- neither
    bundle is expected to send one, but a count that silently doubled on a stray duplicate would
    be a worse failure than a defensive dedupe that never fires.
    """
    return {show["d"]: Counter(vocab[i] for i in set(show["s"])) for show in shows}


def _total_plays(by_date: dict[str, Counter]) -> Counter:
    total: Counter = Counter()
    for counts in by_date.values():
        total.update(counts.keys())          # each date contributes at most one play per song
    return total


def _report_date_sets(poc_by_date: dict, slk_by_date: dict) -> tuple[set, set, set]:
    """Print the three date-set sizes and flag the one direction that should never happen.

    Returns ``(both, poc_only, slk_only)`` for :func:`_report_intersection` and
    :func:`_report_outside_intersection` to divide the rest of the report over.
    """
    poc_dates, slk_dates = set(poc_by_date), set(slk_by_date)
    both = poc_dates & slk_dates
    poc_only, slk_only = poc_dates - slk_dates, slk_dates - poc_dates
    print(f"dates in both corpora   : {len(both)}")
    print(f"dates only in the POC   : {len(poc_only)}")
    print(f"dates only in setlistkit: {len(slk_only)}")
    if slk_only:
        # The design document flags this explicitly: if it happens, "missing sources" is the
        # wrong story and something else needs explaining -- setlistkit, with one source against
        # the POC's three, is not supposed to hold a night the POC does not.
        print("  ** setlistkit holds a night the POC does not -- investigate before trusting "
              "'missing sources' as the explanation for anything below **")
    return both, poc_only, slk_only


def _report_intersection(poc_by_date: dict, slk_by_date: dict,
                         both: set) -> list[tuple[str, int, int]]:
    """Print the PASS/FAIL intersection check; return the mismatches found (empty on PASS)."""
    poc_totals = _total_plays({d: poc_by_date[d] for d in both})
    slk_totals = _total_plays({d: slk_by_date[d] for d in both})
    songs = sorted(set(poc_totals) | set(slk_totals))
    mismatches = [(song, poc_totals[song], slk_totals[song]) for song in songs
                  if poc_totals[song] != slk_totals[song]]
    print(f"\nintersection check ({len(both)} shared dates, {len(songs)} songs seen on them):")
    if mismatches:
        print(f"  FAIL: {len(mismatches)} song(s) disagree on shared dates -- missing sources "
              "cannot explain this, both sides hold every one of these nights")
        for song, poc_n, slk_n in mismatches:
            print(f"    {song}: POC={poc_n} setlistkit={slk_n}")
    else:
        print("  PASS: every song's play count agrees exactly on every shared date")
    return mismatches


def _report_outside_intersection(poc_by_date: dict, slk_by_date: dict, poc_only: set,
                                 slk_only: set) -> None:
    """Print the descriptive, non-blocking counts for what each side holds the other doesn't."""
    print("\noutside the intersection (descriptive, not pass/fail):")
    poc_out = _total_plays({d: poc_by_date[d] for d in poc_only})
    slk_out = _total_plays({d: slk_by_date[d] for d in slk_only})
    print(f"  {sum(poc_out.values())} POC play(s) fall on a date setlistkit does not hold")
    print(f"  {sum(slk_out.values())} setlistkit play(s) fall on a date the POC does not hold")


def main(argv: list[str] | None = None) -> int:
    """Run the gate. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poc", required=True, type=Path, metavar="index.html",
                        help="the POC Songbook's built page, carrying its embedded corpus")
    parser.add_argument("--bundle", required=True, type=Path, metavar="songbook.json",
                        help="a setlistkit `slkit export songbook` bundle")
    args = parser.parse_args(argv)

    poc = _load_poc(args.poc)
    bundle = _load_bundle(args.bundle)
    poc_by_date = _plays_by_date(poc["vocab"], poc["shows"])
    slk_by_date = _plays_by_date(bundle["vocab"], bundle["shows"])

    both, poc_only, slk_only = _report_date_sets(poc_by_date, slk_by_date)
    mismatches = _report_intersection(poc_by_date, slk_by_date, both)
    _report_outside_intersection(poc_by_date, slk_by_date, poc_only, slk_only)

    return 1 if mismatches or slk_only else 0


if __name__ == "__main__":
    sys.exit(main())

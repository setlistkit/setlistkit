#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Which rung of ``canonicalize`` answers each raw title, and is the fuzzy one safe?

FILLING A VOCABULARY ARMS A RUNG THAT WAS PREVIOUSLY DEAD. ``_canonicalize`` tries alias, then
exact normalized match, then ``difflib`` at cutoff 0.9, then falls through to the cleaned display
form. With an empty vocabulary the fuzzy rung has nothing to match against and can never fire, so
a pack going from 0 to 376 names does not only fix misses -- it switches on a rule that can now
fold one real song onto another's name. That is the Hi & Lo failure: a song with a play history,
invisible, filed under a neighbor.

So every fuzzy verdict is printed in full for a human to read, rather than counted. A tally of
"12 fuzzy matches" is not evidence of anything; the twelve pairs are.

The near-miss rung (``s3.band``, scored between 0.80 and the cutoff) is printed too. Those are the
titles that did NOT merge but came close, which is where the next alias belongs if any of them is
really the same song. Printing them is the only way the cutoff is a decision rather than a
default.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import sys

# A script run from a checkout, not an installed package: the path fix has to precede the
# import it enables, so both linters are told the order is deliberate.
# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from setlistkit.catalog.normalizer import normalize  # noqa: E402
from setlistkit.catalog.pack import load_pack        # noqa: E402


def _titles(raw_root: str, artist_slug: str) -> tuple[collections.Counter, collections.Counter]:
    """Every raw title either source will hand the normalizer, split by who flagged it.

    Both sources, because a rung that is safe for one can be unsafe for the other: phish.net's
    spellings ARE the vocabulary and cannot fuzzy-match anything, so a check run against it alone
    would report a clean sweep and prove nothing.

    Returned as two counters rather than one. phish.in marks its soundcheck material with
    ``exclude_from_stats``, and those titles carry a "(Check) " prefix that fuzzy-matches the real
    song underneath -- 45 of them. Pooled with everything else they bury the finding that matters
    under a pile of merges the parser already never sees. Reported apart, not dropped: an audit
    that hides a rung's traffic cannot say the rung is safe.
    """
    live: collections.Counter = collections.Counter()
    flagged: collections.Counter = collections.Counter()
    net = os.path.join(raw_root, "phish_net", "blob")
    for name in sorted(os.listdir(net)):
        with open(os.path.join(net, name), encoding="utf-8") as handle:
            for row in json.load(handle).get("data", []):
                if isinstance(row, dict) and row.get("artist_slug") == artist_slug:
                    if row.get("song"):
                        live[row["song"]] += 1
    inn = os.path.join(raw_root, "phish_in", "blob")
    for name in sorted(os.listdir(inn)):
        with open(os.path.join(inn, name), encoding="utf-8") as handle:
            for track in json.load(handle).get("tracks") or []:
                if track.get("title"):
                    sink = flagged if track.get("exclude_from_stats") else live
                    sink[track["title"]] += 1
    return live, flagged


# A report-printing main: the locals ARE the report, and naming each one is what makes the
# output legible. Splitting it into helpers would move lines without moving meaning.
# pylint: disable=too-many-locals,too-many-branches
def main() -> int:
    """Classify every raw title by the rung that answered it, and print the risky ones."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--artist", default="phish")
    args = parser.parse_args()

    pack = load_pack(args.pack)
    normalizer = pack.normalizer
    _, norm_to_canon = normalizer.build_vocab()
    aliases = normalizer.aliases()
    keys = list(norm_to_canon)

    live, flagged = _titles(args.raw, args.artist)
    rungs: collections.Counter = collections.Counter()
    fuzzy, near, unknown = [], [], []
    for title, plays in live.items():
        norm = normalize(title)
        if not norm:
            rungs["empty"] += plays
            continue
        if norm in aliases:
            rungs["alias"] += plays
            continue
        if norm in norm_to_canon:
            rungs["exact"] += plays
            continue
        match = difflib.get_close_matches(norm, keys, n=1, cutoff=0.9)
        if match:
            rungs["fuzzy"] += plays
            fuzzy.append((plays, title, norm_to_canon[match[0]],
                          difflib.SequenceMatcher(None, norm, match[0]).ratio()))
            continue
        close = difflib.get_close_matches(norm, keys, n=1, cutoff=0.80)
        if close:
            rungs["near miss"] += plays
            near.append((plays, title, norm_to_canon[close[0]],
                         difflib.SequenceMatcher(None, norm, close[0]).ratio()))
        else:
            rungs["unlisted"] += plays
            unknown.append((plays, title, normalizer.is_non_song(title)))

    total = sum(rungs.values())
    print(f"{total} plays across both sources, plus {sum(flagged.values())} the source flags "
          f"itself ({len(flagged)} titles, not scored below)")
    for rung in ("alias", "exact", "fuzzy", "near miss", "unlisted", "empty"):
        if rungs[rung]:
            print(f"  {rung:<10} {rungs[rung]:>6}  ({rungs[rung] / total * 100:5.1f}%)")

    # No verdict is printed beside these. Whether a merge is right is band knowledge, and a
    # machine-written "looks wrong" next to a correct fold trains a reader to skim the list --
    # which is the one thing that would make printing it worse than counting it.
    print(f"\nFUZZY -- every one of these is a merge nobody wrote down ({len(fuzzy)}):")
    for plays, title, canon, ratio in sorted(fuzzy, reverse=True):
        print(f"  {plays:>4}  {title!r} -> {canon!r}  ({ratio:.3f})")
    if not fuzzy:
        print("  none. Every title was answered by an exact key or fell through unmerged.")

    print(f"\nNEAR MISSES -- scored 0.80..0.90, merged with nothing ({len(near)}):")
    for plays, title, canon, ratio in sorted(near, reverse=True)[:20]:
        print(f"  {plays:>4}  {title!r} ~ {canon!r}  ({ratio:.3f})")
    if not near:
        print("  none.")

    print(f"\nUNLISTED -- kept under their own name, nothing lost ({len(unknown)}):")
    for plays, title, tagged in sorted(unknown, reverse=True):
        print(f"  {plays:>4}  {title!r}{'  [tagged non-song]' if tagged else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

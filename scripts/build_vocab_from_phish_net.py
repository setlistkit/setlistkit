#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Draft a pack's vocabulary from a cached phish.net pull, and report what it cannot answer.

``slkit pack draft`` exists and is the wrong tool here. It carves a vocabulary out of a
bag of archive.org FLAC titles -- a stranger's filenames, full of track numbers and taper sigils --
by fuzzy-clustering near-duplicates for a human to trim. That work is unnecessary against
phish.net, which ships a stable ``songid`` beside every ``song``. The identity question
a draft has to guess at is already answered by the source, so guessing at it again can only
lose.

WHY phish.net AND NOT phish.in. The merge is configured so a phish.net record wins any date it
holds, because phish.net is the only source carrying segue marks. So phish.net's spelling is
already what the corpus stores, and taking canon from anywhere else would leave every stored row
disagreeing with its own dictionary. phish.in then folds in for free: ``normalize()`` lowercases,
so "Limb By Limb" and "Limb by Limb" are one key and need no alias.

WHAT THIS REFUSES TO DECIDE. Two things, both reported and neither written:

* **Titles phish.in has and phish.net does not.** Some are stage business ("Banter", "Intro"),
  which belong in ``classifiers.json`` as non-songs; one is a performance phish.net files under
  another name. Which is which is band knowledge, so they are printed as a worklist rather than
  guessed into the vocabulary.
* **Medley tracks** -- "Alumni Blues > Letter to Jimmy Page > Alumni Blues" is one audio file
  holding a run of songs. Splitting one duration across three songs invents the boundaries, which
  is the guessing the durations chain exists to refuse. They are counted and listed.

Not a pytest test: the output is a file to review and a worklist to act on.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

# A script run from a checkout, not an installed package: the path fix has to precede the
# import it enables, so both linters are told the order is deliberate.
# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from setlistkit.catalog.normalizer import Normalizer, normalize  # noqa: E402


# What phish.net puts in `slug` when a row names something its catalog has no entry for. The
# marker, not the id number behind it: the number is theirs to change, the marker is what it means.
_NO_CATALOG_ID = "custom"


def _rows(blob_dir: str, artist_slug: str):
    """Every cached phish.net setlist row belonging to one artist.

    The ``showyear`` endpoint has no artist dimension -- a year payload carries Trey solo, Mike
    Gordon and the rest alongside Phish -- so the filter is not optional. Without it the
    vocabulary would teach the pack that a TAB original is a Phish song.
    """
    for name in sorted(os.listdir(blob_dir)):
        with open(os.path.join(blob_dir, name), encoding="utf-8") as handle:
            payload = json.load(handle)
        for row in payload.get("data", []):
            if isinstance(row, dict) and row.get("artist_slug") == artist_slug:
                yield row


def _canon_by_id(blob_dir: str, artist_slug: str):
    """``songid`` -> the spelling phish.net used most, plus the raw tally behind each one.

    ONE SONGID IS NOT A SONG. phish.net files anything with no catalog entry -- a one-off cover, a
    guest's tune, a named medley -- under a single catch-all id, and marks it by setting ``slug``
    to the literal ``custom``. Over 2009-2026 that id carries 826 rows and dozens of distinct
    titles. Treated as an id like any other it would elect one spelling as the canon for all of
    them, and the collision check below would (correctly) refuse to write anything at all.

    So the sentinel is keyed on the SOURCE'S OWN MARKER rather than on the id number: ``slug`` is
    what phish.net sets to say "this row has no catalog identity", and the number behind it is an
    implementation detail that can change. Those rows fall back to their title as their identity,
    which is the only thing they have.

    Under ``artist_slug = phish`` this is 8 rows of 14,289, and 6 of the 8 are ordinary songs that
    also hold a real id elsewhere -- so they fold into their own entry by title and cost nothing.
    """
    spellings: dict[object, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in _rows(blob_dir, artist_slug):
        song, songid = row.get("song"), row.get("songid")
        if not song:
            continue
        if row.get("slug") == _NO_CATALOG_ID:
            spellings[("custom", normalize(song))][song] += 1
        elif songid is not None:
            spellings[songid][song] += 1
    # sorted() before max() so a tie resolves the same way on every run. Set and dict iteration
    # order over equal counts is not something to leave to chance in a file that gets committed.
    canon = {sid: max(sorted(counter), key=counter.__getitem__)
             for sid, counter in spellings.items()}
    return _fold_uncataloged(canon), spellings


def _fold_uncataloged(canon: dict) -> dict:
    """Drop any title-keyed entry that a real song id already answers for.

    A song can appear BOTH ways: filed under its id on most nights and under the catch-all on the
    one night somebody entered it by hand. Kept apart those are two vocabulary entries for one
    song, which is the same corpus damage a collision causes, arrived at from the other direction.
    Measured over 2009-2026: 6 of the 8 title-keyed entries are this.

    The real id always wins, and that is not arbitrary -- it is the entry the other several
    thousand plays of that song are already filed under. Only a title nothing else claims survives
    as its own entry, which is the case the fallback exists for.
    """
    known = {normalize(song) for sid, song in canon.items() if not isinstance(sid, tuple)}
    return {sid: song for sid, song in canon.items()
            if not isinstance(sid, tuple) or normalize(song) not in known}


def _collisions(canon_by_id: dict) -> dict:
    """Normalized keys that two DIFFERENT song ids both claim.

    The only failure here that loses data. A collision means the vocabulary would fold two real
    songs into one, and every play of the loser would be filed under the winner -- a song with a
    play history, invisible. Reported so it is never discovered later as a suspiciously popular
    title.
    """
    by_key = collections.defaultdict(set)
    for songid, song in canon_by_id.items():
        by_key[normalize(song)].add((songid, song))
    return {key: members for key, members in by_key.items() if len(members) > 1}


def _phish_in_gaps(blob_dir: str, vocabulary: list[str]):
    """Every phish.in title the drafted vocabulary does not already answer.

    Counted against the tracks the source does NOT flag itself: ``exclude_from_stats`` covers
    soundcheck and pre-show material, and a title that only ever appears there is not a gap in a
    vocabulary of performed songs.
    """
    normalizer = Normalizer(vocabulary=vocabulary)
    known = {normalize(name) for name in vocabulary}
    hit = collections.Counter()
    missed = collections.Counter()
    for name in sorted(os.listdir(blob_dir)):
        with open(os.path.join(blob_dir, name), encoding="utf-8") as handle:
            show = json.load(handle)
        for track in show.get("tracks") or []:
            title = track.get("title")
            if not title or track.get("exclude_from_stats"):
                continue
            canon, _ = normalizer.canonicalize(title)
            if normalize(canon or title) in known:
                hit[title] += 1
            else:
                missed[title] += 1
    return hit, missed


# A report-printing main: the locals ARE the report, and naming each one is what makes the
# output legible. Splitting it into helpers would move lines without moving meaning.
# pylint: disable=too-many-locals
def main() -> int:
    """Draft the vocabulary, write it, and print everything it declined to decide."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", required=True,
                        help="the pack's raw/ directory, holding phish_net/ and phish_in/")
    parser.add_argument("--artist", default="phish", help="artist_slug to keep (default: phish)")
    parser.add_argument("--out", help="where to write vocabulary.json; omit for a dry run")
    args = parser.parse_args()

    net = os.path.join(args.raw, "phish_net", "blob")
    canon_by_id, spellings = _canon_by_id(net, args.artist)
    plays = sum(sum(counter.values()) for counter in spellings.values())
    print(f"phish.net: {len(canon_by_id)} song ids over {plays} plays by {args.artist!r}")

    split = {sid: dict(counter) for sid, counter in spellings.items() if len(counter) > 1}
    print(f"  ids carrying more than one spelling: {len(split)}")
    for songid, counter in list(split.items())[:20]:
        print(f"    {songid}: {counter}   -> {canon_by_id[songid]!r}")

    collisions = _collisions(canon_by_id)
    print(f"  normalized-key collisions between ids: {len(collisions)}")
    for key, members in collisions.items():
        print(f"    {key!r}: {sorted(members, key=repr)}")
    if collisions:
        print("  REFUSING to write: a collision folds two real songs into one.")
        return 1

    vocabulary = sorted(set(canon_by_id.values()))
    hit, missed = _phish_in_gaps(os.path.join(args.raw, "phish_in", "blob"), vocabulary)
    total = sum(hit.values()) + sum(missed.values())
    print(f"\nphish.in: {total} tracks the source does not flag itself")
    print(f"  answered by this vocabulary: {sum(hit.values())} "
          f"({sum(hit.values()) / total * 100:.1f}%)")
    print(f"  unanswered: {sum(missed.values())} over {len(missed)} distinct titles --")
    for title, count in missed.most_common():
        kind = "medley" if ">" in title else "review"
        print(f"    {count:>3}  [{kind}]  {title!r}")

    if not args.out:
        print("\ndry run; pass --out to write")
        return 0
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(vocabulary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"\nwrote {len(vocabulary)} names to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The phish.in parser: raw cached show payloads into merge show-records.

The smallest of the three parsers, because phish.in hands back a show that is already a show: one
object per night, an ordered ``tracks`` array, each track carrying its own ``set_name``. There is
no prose to mine and no flat table to reassemble.

What it does NOT carry is the reason it ranks below phish.net in the merge: **there is no segue
field.** Not "empty", absent -- the API has no notion of one song running into the next. Every
entry here is written ``segue: False``, and that is a statement about our knowledge rather than
about the night. phish.net is where the ``>`` and ``->`` marks come from, and the merge is
configured so a phish.net record wins a date it also holds.

Two things about ``set_name`` that a naive mapping gets wrong, both measured against the live
2020-2026 cache (4,652 tracks):

**It is not only sets.** Observed values are ``Set 1``, ``Set 2``, ``Set 3``, ``Encore`` -- and
also ``Soundcheck`` (46 tracks) and ``Pre-Show`` (14). Fold those into the sets and they become
plays, inflating a song's totals with material nobody performed to a room.

**``exclude_from_stats`` already partitions them exactly.** All 60 Soundcheck and Pre-Show tracks
carry it; every one of the 4,592 real set and encore tracks does not. So this parser flags
non-songs from that FIELD rather than by matching set names -- the upstream's own judgement, kept
current by someone who is watching, instead of a blocklist here that goes stale the first time
they add a category. It is the same field that solves the stub-track problem archive.org needed a
duration threshold and a granularity comparison to approximate.

Durations are deliberately not read here. A merge record has no slot for them, and the track
lengths are the reason a separate durations pipeline exists; this parser produces setlists.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from .normalizer import Normalizer
from .parse import count_songs, fold_sections, keep, number, text

# "Set 1" / "Set 2" / "Set 3" -- the only labels that are a numbered set. Anything else is either
# the encore or one of the non-performance sections handled below.
_SET_NAME = re.compile(r"^\s*set\s+(\d+)\s*$", re.IGNORECASE)


def _section(set_name: str) -> tuple[int, int]:
    """Sort key placing a track's section: numbered sets, then the encore, then everything else.

    The trailing group is Soundcheck and Pre-Show. They sort LAST despite happening first, and
    that is the deliberate part: a set's index carries meaning downstream -- ``features.py`` reads
    index 0 as the show's opening set -- so letting a soundcheck take that slot would rewrite what
    "opener" means for every night that had one. They are still present, and every entry in them
    is flagged ``non_song``, so nothing is lost and nothing is counted.
    """
    match = _SET_NAME.match(set_name)
    if match:
        return (0, int(match.group(1)))
    if set_name.strip().lower().startswith("encore"):
        return (1, 0)
    return (2, 0)


def _entry(track: Mapping, normalizer: Normalizer) -> dict | None:
    """One track into a ``{song, segue, non_song}`` entry, or ``None`` if it names no song.

    ``segue`` is always ``False``: phish.in has no segue field, so claiming one either way would
    be inventing data. ``non_song`` comes from ``exclude_from_stats``, which the source maintains
    and which exactly covers the soundcheck and pre-show material.
    """
    name = text(track.get("title"))
    if not name:
        return None
    canon, _ = normalizer.canonicalize(name)
    if not canon:
        return None
    non_song = bool(track.get("exclude_from_stats")) or normalizer.is_non_song(canon)
    return {"song": canon, "segue": False, "non_song": non_song}


def _record(show: Mapping, normalizer: Normalizer) -> dict | None:
    """One phish.in show payload into a merge show-record, or ``None`` without a usable date.

    Tracks are grouped by section and ordered by ``position`` within it. Unlike phish.net's
    show-wide counter, ``position`` here is already per-show and monotonic, so the sort is a
    guarantee rather than a correction -- but it is done anyway, because "the array happened to
    arrive in order" is not something a parser should depend on.
    """
    date = text(show.get("date"))
    if not date:
        return None
    by_section: dict[tuple[int, int], list[Mapping]] = {}
    for track in show.get("tracks", []):
        if isinstance(track, Mapping):
            by_section.setdefault(_section(text(track.get("set_name"))), []).append(track)

    sets, encore = fold_sections(
        (key[0] == 1,                         # section 1 is the encore; 0 is a set, 2 is the rest
         [e for e in (_entry(t, normalizer)
                      for t in sorted(by_section[key], key=lambda t: number(t.get("position")))) if e])
        for key in sorted(by_section))
    if not sets and not encore:
        return None
    return {"date": date, "year": date[:4], "sets": sets, "encore": encore,
            "n_songs": count_songs(sets, encore), "source": "phish_in",
            "identifier": f"phish_in-{text(show.get('id')) or date}"}


def parse_phish_in(shows: Iterable[Mapping], *, normalizer: Normalizer,
                   min_year: int | None = None) -> list[dict]:
    """Cached phish.in show payloads into merge show-record candidates.

    Shows the catalog lists but holds no audio for are parsed like any other: phish.in carries
    their setlists too (``audio_status: "missing"``), which is exactly why it is worth consulting
    for nights nobody taped. Sorted by ``(date, identifier)`` so two runs agree.
    """
    out = [rec for rec in (_record(show, normalizer) for show in shows if isinstance(show, Mapping))
           if keep(rec, min_year)]
    return sorted(out, key=lambda rec: (rec["date"], rec["identifier"]))

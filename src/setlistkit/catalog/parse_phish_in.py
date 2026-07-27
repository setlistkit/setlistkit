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

DURATIONS DO NOT BELONG IN A MERGE RECORD, and they are not put in one. A show-record has no slot
for a track length and inventing one would push a measurement through a pipeline built to
reconcile setlists. They come out through :func:`phish_in_tapes` instead, as MIRROR rows -- the
same shape archive.org's tapes take -- so the durations chain reads phish.in through exactly the
machinery it reads everybody through. See that function for why re-attributing them here would
have been the more tempting and more expensive answer.
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


def _track_row(index: int, track: Mapping) -> dict:
    """One phish.in track as a mirror ``duration_tracks`` row.

    ``name`` is synthesized to look like a taper's filename because that is what the recognition
    reads, and phish.in has no filenames -- it has an object with a title on it. Writing the title
    into a plausible filename is not a lie about the source: ``title`` beside it holds the same
    string unaltered, and the two columns exist precisely so a synthesized name and a stated one
    can never be confused for each other.

    ``length_raw`` keeps the source's own milliseconds verbatim and ``seconds`` is NULL when there
    is nothing to convert, which is the mirror's existing rule -- a length that could not be read
    stays diagnosable from the row instead of needing a re-pull. phish.in returns ``duration: 0``
    on the nights it lists but holds no audio for, and those are exactly the rows this leaves
    NULL: a zero-second track is not a measurement of a song that took no time.
    """
    milliseconds = number(track.get("duration"))
    title = text(track.get("title"))
    return {"idx": index, "name": f"{index + 1:02d} {title}.mp3", "title": title,
            "length_raw": str(track.get("duration") if track.get("duration") is not None else ""),
            "seconds": (milliseconds / 1000.0) if milliseconds else None}


def _tape(show: Mapping) -> dict | None:
    """One phish.in show as a mirror recording, or ``None`` when there is no recording.

    A NIGHT PHISH.IN LISTS IS NOT A NIGHT PHISH.IN HAS AUDIO FOR, and the difference has to be
    made here because it is invisible afterwards. The catalog carries the setlist either way, and
    an audio-less show still arrives with a full ``tracks`` array -- every track present, every
    ``duration`` zero. Mirrored as-is that is a row asserting we hold a tape of a night nobody
    taped, indistinguishable in the table from a real one, and every count of "recordings held"
    downstream is inflated by it. Measured on the 2020-2026 cache: 14 of 249.

    So a show is a recording when at least one track has a length. ``audio_status`` says the same
    thing -- it is exactly ``missing`` on those 14 and ``complete`` on the other 235 -- and this
    keys on the durations rather than on the label because the durations are what a mirror row is
    FOR. A row carrying no measurement is not a recording whatever it is labeled.

    The setlist is untouched by any of this: :func:`_record` parses those nights like any other,
    which is the point of consulting a catalog that covers shows nobody has audio for.

    Soundcheck and pre-show tracks are KEPT, unlike in :func:`_record` where they are flagged
    non-song. A mirror is a record of what the recording contains, and the recording contains
    them; dropping them here would make the stored track count disagree with the audio it claims
    to mirror. They simply match nothing in the setlist and are reported as unclaimed, which is
    the same thing that happens to a taper's tuning track.
    """
    date = text(show.get("date"))
    tracks = [t for t in show.get("tracks", []) if isinstance(t, Mapping)]
    if not date or not tracks or not any(number(t.get("duration")) for t in tracks):
        return None
    ordered = sorted(tracks, key=lambda t: number(t.get("position")))
    # The identifier is spelled exactly as :func:`_record` spells it, so a night's tape and its
    # setlist name the same thing. `uploader` is the archive's own name and not a person, which
    # is the honest answer: every one of these transfers is credited to phish.in itself, and a
    # taper-credit list that invented a name here would be crediting nobody.
    return {"identifier": f"phish_in-{text(show.get('id')) or date}", "date": date,
            "uploader": "phish.in", "audio_format": "MP3", "source": "phish_in",
            "duration_tracks": [_track_row(index, track)
                                for index, track in enumerate(ordered)]}


def phish_in_tapes(shows: Iterable[Mapping], *, min_year: int | None = None) -> list[dict]:
    """Cached phish.in show payloads as mirror recordings, sorted by date.

    ONE RECORDING PER NIGHT, and that is the difference from archive.org that matters most. The
    mirror normally holds several tapes of a night precisely so the durations chain has something
    to vote with; here there is one official transfer and no second opinion. Every performance
    timed from these comes back ``n_tapes=1``, ``spread_seconds=0``, and any page publishing them
    has to say so rather than let a single uncorroborated measurement read like a settled one.

    WHY THESE GO THROUGH THE RECOGNITION AT ALL, when phish.in already states which song each
    track is. Because that is not the hard question. The durations arrive from phish.in and the
    SLOTS belong to phish.net, which outranks it in the merge and therefore defines the setlist
    of every date they share. Attaching a duration means joining two sources' setlists, and
    :mod:`setlistkit.catalog.durations` already does that -- partial recordings, short-name
    collisions, every call it could not make recorded as an edge. Shortcutting it would be a
    second implementation of the join, answering to nothing, with its own failures landing
    somewhere other than the review queue.
    """
    out = [tape for tape in (_tape(show) for show in shows if isinstance(show, Mapping)) if tape]
    if min_year is not None:
        out = [tape for tape in out if tape["date"][:4].isdigit()
               and int(tape["date"][:4]) >= min_year]
    return sorted(out, key=lambda tape: (tape["date"], tape["identifier"]))


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

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Structural properties of each song, derived from setlists we already have.

No network and no new data: every fact here is already latent in the stored corpus, we just
never asked for it. Where a song *sits* in a night is a property of the song, not of the
night. Bearsong closes sets. Timmy Tucker opens them. Rebubula segues out of things. A
rotation model that treats all thirty slots in a show as interchangeable will happily predict
four set-closers and no opener.

Everything is a rate in [0,1] with an ``n_plays`` beside it. A 1.00 encore rate on n=1 means
nothing, and a consumer needs the n to know that. The six rates live together in ``Rates``
because they share that denominator; ``mean_slot`` does not share it and so does not join them.

Computed over whatever ``shows`` the caller hands in -- this module has no opinion about a window
and takes none. The version this was ported from used 2023-01-01 onward unconditionally, on the
argument that a song's 2019 habits are not evidence about 2026, which was reasonable when the
corpus started at 2023 and stopped being reasonable once it ran back to 1992. Baking a window in
here would have made that argument permanent for the wrong reason.

So the choice moved outward: an unwindowed caller gets whole-corpus features, and the ranged Tape
Measure export (``cli/export.py``) already hands this a windowed ``shows`` and gets windowed
features back, correctly -- the Songbook does the same. ``first_seen``, ``last_seen`` and
``n_plays`` are still emitted so a windowed caller can tell how much history a rate rests on, and
doing THAT here -- rather than silently narrowing -- is what keeps the opinion out of the catalog
layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from collections import Counter, defaultdict
from dataclasses import dataclass

from .normalizer import clean_song

_TOP_PARTNERS = 3


@dataclass(frozen=True)
class Rates:
    """How often a song took each structural role. All six are over the same ``n_plays``.

    Grouped because they share that denominator, which is the thing that makes them
    comparable to each other and to nothing else in the record. ``mean_slot`` deliberately
    stays outside: it averages slot observations, and a set of one contributes none, so its
    denominator is not ``n_plays`` and never was.
    """

    opener: float
    set_closer: float
    show_closer: float
    encore: float
    segue_out: float
    segue_in: float


@dataclass(frozen=True)
class SongFeature:
    """One song's structural profile.

    Every field is immutable, including ``top_partners``, so a feature can be sorted, hashed
    and put in a set. There will be one of these per song in the corpus and consumers will
    want to compare them; a mutable member would defeat the freeze quietly rather than loudly.
    """

    song: str
    n_plays: int
    rates: Rates
    mean_slot: float
    top_partners: tuple[tuple[str, int], ...]
    first_seen: str
    last_seen: str


class _Tally:
    """The running counts for every song, filled in one pass over the shows.

    A class rather than eleven parallel dicts threaded through three functions: pylint counts
    locals, and more to the point a reader should not have to check that eleven containers are
    all keyed the same way.
    """

    def __init__(self) -> None:
        self.plays: Counter = Counter()
        self.opener: Counter = Counter()
        self.set_closer: Counter = Counter()
        self.show_closer: Counter = Counter()
        self.encore: Counter = Counter()
        self.segue_out: Counter = Counter()
        self.segue_in: Counter = Counter()
        self.partners: defaultdict = defaultdict(Counter)
        self.slots: defaultdict = defaultdict(list)
        self.dates: defaultdict = defaultdict(list)

    def play(self, song: str, date: str) -> None:
        """Record one performance of ``song`` on ``date``."""
        self.plays[song] += 1
        self.dates[song].append(date)

    def segue(self, song: str, nxt: str) -> None:
        """Record that ``song`` ran into ``nxt`` without a stop."""
        self.segue_out[song] += 1
        self.partners[song][nxt] += 1
        self.segue_in[nxt] += 1


def _songs_only(entries: Iterable[Mapping]) -> list[Mapping]:
    """The entries that are music.

    The corpus stores what happened, which includes what was not a song: a set break, a drum
    segment, an untitled jam. Those are real records of the night and they are deliberately
    kept -- and they are not songs, so they take no slot in a song's profile. Counted, "break"
    would be the strongest set-closer this band has ever had.
    """
    return [entry for entry in entries if not entry.get("non_song")]


def _name(entry: Mapping) -> str:
    """What this entry's song is called, under the one name the whole project files it by.

    Canonicalised HERE and not only in the length chain, because a profile keyed by the raw
    setlist string and a length keyed by the cleaned one are two records of the same song that
    nothing can join. That is not hypothetical -- it is how "Hey, It's Christmas" ended up with
    lengths under one spelling and a structural profile under another.
    """
    return clean_song(entry["song"])


def _walk_set(tally: _Tally, songs: list[Mapping], date: str, *, is_last: bool) -> None:
    """Fold one set into the tally."""
    count = len(songs)
    for position, entry in enumerate(songs):
        song = _name(entry)
        tally.play(song, date)
        # A set of one has no first-to-last axis to place anything on, so it contributes no
        # slot observation rather than a fabricated 0.0.
        if count > 1:
            tally.slots[song].append(position / (count - 1))
        if position == 0:
            tally.opener[song] += 1
        if position == count - 1:
            tally.set_closer[song] += 1
            if is_last:
                tally.show_closer[song] += 1
        if entry.get("segue") and position + 1 < count:
            tally.segue(song, _name(songs[position + 1]))


def _walk_encore(tally: _Tally, songs: list[Mapping], date: str) -> None:
    """Fold the encore into the tally. An encore song is not an opener or a set closer."""
    for position, entry in enumerate(songs):
        song = _name(entry)
        tally.play(song, date)
        tally.encore[song] += 1
        if entry.get("segue") and position + 1 < len(songs):
            tally.segue(song, _name(songs[position + 1]))


def _feature_of(tally: _Tally, song: str, plays: int) -> SongFeature:
    """One song's row, once the whole corpus has been folded in."""
    slots = tally.slots.get(song) or [0.5]
    return SongFeature(
        song=song,
        n_plays=plays,
        rates=Rates(
            opener=round(tally.opener[song] / plays, 3),
            set_closer=round(tally.set_closer[song] / plays, 3),
            show_closer=round(tally.show_closer[song] / plays, 3),
            encore=round(tally.encore[song] / plays, 3),
            segue_out=round(tally.segue_out[song] / plays, 3),
            segue_in=round(tally.segue_in[song] / plays, 3),
        ),
        mean_slot=round(sum(slots) / len(slots), 3),
        top_partners=tuple(tally.partners[song].most_common(_TOP_PARTNERS)),
        first_seen=min(tally.dates[song]),
        last_seen=max(tally.dates[song]),
    )


def song_features(shows: Iterable[Mapping]) -> list[SongFeature]:
    """Every song's structural profile, most-played first.

    ``shows`` is the shape :func:`setlistkit.store.corpus.shows` returns.

    Ties on play count break on the song name, so two runs over the same corpus agree and so
    does anything computed downstream.
    """
    tally = _Tally()
    for show in shows:
        date = str(show.get("date") or "")
        sets = [_songs_only(one) for one in (show.get("sets") or [])]
        encore = _songs_only(show.get("encore") or [])
        # With an encore, the night ends there, so no song in the final SET closed the show.
        # Without one, the last song of the last non-empty set did.
        last = max((index for index, one in enumerate(sets) if one), default=-1)
        for index, songs in enumerate(sets):
            _walk_set(tally, songs, date, is_last=index == last and not encore)
        _walk_encore(tally, encore, date)

    out = [_feature_of(tally, song, plays) for song, plays in tally.plays.items()]
    return sorted(out, key=lambda feature: (-feature.n_plays, feature.song))

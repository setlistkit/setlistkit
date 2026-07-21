# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Per-year play counts: what a rotation model believes before it knows anything else.

How often the band reached for each song in a given year, counted straight off the stored
corpus. A model asking "what is the chance they open with this tonight" starts here and then
adjusts for recency, rotation and whatever else it knows.

WHY THIS IS OURS AND NOT SOMEBODY ELSE'S
This replaces a per-year artifact built from a third-party setlist API, retired on licensing
grounds. Every handoff since carried "the model is blocked on base rates" forward as a fact and
nobody had tested it. Measured against the retired artifact over the only seven years it ever
covered, our corpus maps 100% of its song names, and rescaled to share of setlist the two agree
to a mean absolute error of 0.0011-0.0017 for 2023 onward.

The premise turned out to be backwards. The other source ran 13.3 songs per show against our
14.3-15.2, every year: user-entered setlists are partial, because a person typed in what they
remembered. A tape captures the whole night. We were not approximating the better source, we
were blocked on the worse one.

WHAT "SHOWS" MEANS HERE
Nights we hold a tape for, which is not the same as nights the band played -- coverage against
the retired artifact ran 69-81%. This does not distort a rate, because the numerator and the
denominator scale together, and that is exactly why the agreement above holds. It does mean an
absolute count from here is not comparable to an absolute count from anywhere else, and no
consumer should treat ``shows`` as "how many shows happened".

COUNTING, PRECISELY
``songs`` counts PLAYS, not nights-containing, matching the artifact this replaces. A song
played in both sets counts twice, so a rate built from these can exceed 1.0 per show. That is a
real difference from counting each night once and a consumer that blends this with a
shows-containing rate is blending two different measurements.

A show with no readable songs still counts in ``shows``. Dropping it would leave the numerator
untouched and shrink the denominator, quietly inflating every rate in that year.

NO WINDOW AND NO WEIGHTING
Every year in the corpus comes back, all thirty-four of them. Windowing and era weighting are
model opinions with real consequences -- and they bite harder than they look, since at the
recency half-life the ported model was tuned with, everything before 2020 carries under a
hundredth of one percent of the total weight. That is a decision for the layer that owns it,
made against a backtest. This layer reports what the corpus says and is equally useful to
someone who never touches prediction.

Acoustic nights and alter-ego nights are counted like any other. A consumer that should not
pool them filters with :func:`setlistkit.catalog.show_types` first.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

_YEAR_RE = re.compile(r"\d{4}")


@dataclass(frozen=True)
class YearRate:
    """One year's counts, and nothing that could disagree with them.

    ``songs`` is a tuple of pairs rather than a dict so the record is really immutable and can
    be hashed -- a dict inside a frozen dataclass can be mutated straight through the freeze.
    Call ``dict(row.songs)`` where a mapping is wanted.
    """

    year: str
    shows: int
    songs: tuple[tuple[str, int], ...]

    @property
    def distinct_songs(self) -> int:
        """How many different songs were played. Derived, never stored.

        The artifact this replaces published this beside the counts, where a hand-edit could
        make the two disagree and nothing would say so. It is ``len(songs)`` and it stays that.
        """
        return len(self.songs)


def _songs_of(show: Mapping) -> Iterator[str]:
    """Every song played that night, sets then encore, in order, non-songs dropped.

    The corpus stores set breaks, drum segments, tuning and banter as real records of the
    night. They are not repertoire: counted as songs, "break" would be the most-played item in
    the band's history.
    """
    for one_set in show.get("sets") or []:
        for entry in one_set:
            if not entry.get("non_song"):
                yield str(entry["song"])
    for entry in show.get("encore") or []:
        if not entry.get("non_song"):
            yield str(entry["song"])


def base_rates(shows: Iterable[Mapping]) -> list[YearRate]:
    """Per-year play counts, oldest year first.

    ``shows`` is the shape :func:`setlistkit.store.corpus.shows` returns. A show whose date
    carries no four-digit year is skipped -- it cannot be filed under one.

    Within a year, songs come back most-played first with ties broken on the name, so two runs
    over the same corpus agree and so does everything computed downstream.
    """
    played: dict[str, Counter] = {}
    nights: Counter = Counter()
    for show in shows:
        year = str(show.get("date") or "")[:4]
        if not _YEAR_RE.fullmatch(year):
            continue
        nights[year] += 1
        played.setdefault(year, Counter()).update(_songs_of(show))
    return [YearRate(year=year, shows=nights[year],
                     songs=tuple(sorted(played[year].items(),
                                        key=lambda pair: (-pair[1], pair[0]))))
            for year in sorted(nights)]

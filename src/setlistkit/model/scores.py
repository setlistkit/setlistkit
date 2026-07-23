# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Score every song the band knows, as of a date, using only what was true before it.

Four signals multiply into one number:

    base        how often the song has been played, weighted toward recent shows
    due boost   how overdue it is against its own typical gap
    cooldown    how recently it last played, which suppresses rather than boosts
    era weight  a flat discount on shows before a threshold year

The as-of date is the whole design. Every function here takes one and refuses to look past it,
because a scorer that can see the night it is predicting produces an excellent number that
means nothing, and nothing in the output says so. :mod:`setlistkit.model.backtest` walks it
forward one show at a time and that only works if this is honest.

WHAT WAS DELIBERATELY NOT PORTED
The version this comes from blended two base rates: one from per-show archive data and one
from a third-party per-year artifact, mixed by an ``sfm_blend`` knob at 0.3. That artifact is
retired on licensing grounds, and :mod:`setlistkit.catalog.baserates` shows our own corpus
reproduces it to a mean share error of 0.0011-0.0017. So the blend is gone, not set to zero:

  * the per-year path only ever existed because that source published year aggregates. We hold
    per-show data, which is strictly more information -- an exact date beats a year midpoint.
  * the two sides counted differently. The year artifact counted PLAYS and the archive side
    counts NIGHTS-CONTAINING, so the blend was mixing two measurements that disagree for any
    song played twice in a night. Deleting it removes that quietly.
  * it carried a ``sfm_cutoff_year`` leak guard that existed only because year aggregates leak
    a show's own year into its prediction. Per-show data makes a plain date filter sufficient,
    so the guard is deleted rather than ported. One fewer thing to get wrong.

WHAT THE CONSTANTS ARE WORTH, MEASURED HERE
The defaults arrived from a pipeline fitted against a corpus starting in 2020 with the blend at
0.3. Both of those are now false, so they were re-measured with :mod:`setlistkit.model.backtest`
over the most recent 80 gradeable shows (2025-03-13 to 2026-07-18), walk-forward:

    naive all-time frequency                0.141
    recency-weighted frequency only         0.186
    this scorer, inherited defaults         0.252 +/-0.013

So the machinery above recency is worth about +0.066, and recency alone is worth +0.045 over
counting. Beyond that the knobs sort into three groups, and the sorting is the useful part:

  * EARNS ITS PLACE. The cooldown, decisively: 0.252 with it against 0.201 without, roughly
    four standard errors. It is the largest single lever here and the least obvious one.
  * ARGUABLE. The recency half-life. Out of sample 400 days scores 0.249 against 200's 0.233,
    and in sample the order reverses -- which is the overfitting trap this scorer's ancestor
    fell into, reproduced on our data. About one standard error, so the default stays at 200
    until somebody decides otherwise on purpose.
  * DOES NOTHING MEASURABLE. ``era_weight_before`` reads 0.248 / 0.249 / 0.247 at 0.25 / 0.5 /
    1.0 -- flat. It is a second recency mechanism and recency already dominates. ``gap_cap``,
    ``min_show_songs`` and ``lookback_years`` are flat too, and ``gap_weight`` scores the same
    at 0.0 as at 0.35, so the due boost is currently decoration. None of them is deleted,
    because flat over 80 recent shows is not the same as useless, but nothing should be tuned
    on them and no result should be attributed to them.

Two limits on all of the above. The evaluation window is recent shows only, so none of this
grades the deep corpus. And at a 200-day half-life everything before 2020 carries under a
hundredth of one percent of the total weight, which means 87% of the corpus cannot reach this
scorer at all -- a fact about the configuration, not about the corpus.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

# Shows-since-last-play -> score multiplier. The band essentially never repeats a song on
# consecutive nights, yet a just-played song otherwise ranks UP: its gap is zero so the due
# boost is neutral, and its base is at peak recency. Without this the top pick is regularly
# last night's closer. Index is the gap; anything past the end is off cooldown at 1.0.
_COOLDOWN = (0.12, 0.55, 0.82, 0.93)


@dataclass(frozen=True)
class ScoreConfig:
    """The knobs, with what each one measured out of sample noted above the module docstring.

    Unchanged from the values that arrived, deliberately. The one candidate for a change is
    ``recency_halflife_days`` at 400, which beats 200 by about a standard error out of sample
    and loses to it in sample. Adopting inside the noise is how the previous pipeline drifted
    1.4 sigma worse, so it stays until a person moves it on purpose.

    ``cooldown`` is indexed by shows-since-last-play and an empty tuple turns it off.
    ``lookback_years`` of None uses the whole corpus, which measured better than any cut.
    """

    recency_halflife_days: float = 200.0
    era_full_from_year: int = 2023
    era_weight_before: float = 0.5
    gap_weight: float = 0.35
    gap_cap: float = 3.0
    cooldown: tuple[float, ...] = field(default=_COOLDOWN)
    min_show_songs: int = 5
    lookback_years: int | None = None


@dataclass(frozen=True)
class SongScore:
    """One song's standing as of the as-of date.

    ``weighted_plays`` is evidence, not a play count: it is the recency- and era-weighted sum,
    so a song played four times last month can outweigh one played twenty times in 1998. Kept
    beside the score because a consumer ranking on ``score`` alone cannot tell a well-evidenced
    song from a thin one.
    """

    song: str
    score: float
    base: float
    gap_ratio: float
    cooldown: float
    weighted_plays: float
    shows_since: int | None


def _as_date(value: str | dt.date) -> dt.date | None:
    """An ISO date, or None if it is not one.

    The corpus really does contain impossible dates -- a transposed `2008-31-08` among them --
    because the parser validated the shape and not the calendar. Returning None lets a scorer
    skip a bad row rather than refuse to run at all.
    """
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _songs_of(show: Mapping) -> list[str]:
    """Every song played that night, non-songs dropped.

    Set breaks, drum segments and banter are stored deliberately and are not repertoire. The
    prior pipeline filtered these in one consumer and not in the predictor, which left 79
    pieces of stage business sitting in the play history as live prediction candidates.
    """
    out = [str(entry["song"]) for one_set in show.get("sets") or [] for entry in one_set
           if not entry.get("non_song")]
    out += [str(entry["song"]) for entry in show.get("encore") or []
            if not entry.get("non_song")]
    return out


def _era(year: int, config: ScoreConfig) -> float:
    """Flat discount on shows before the threshold year."""
    return 1.0 if year >= config.era_full_from_year else config.era_weight_before


def _cooldown(gap: int | None, config: ScoreConfig) -> float:
    """Suppression for a song played ``gap`` shows ago. Never played means never suppressed."""
    if gap is None or gap >= len(config.cooldown):
        return 1.0
    return config.cooldown[gap]


def _training_shows(shows: Iterable[Mapping], config: ScoreConfig,
                    asof: dt.date) -> list[tuple[dt.date, list[str]]]:
    """The shows that may inform a prediction for ``asof``, oldest first.

    Strictly before the as-of date: a show on the night being predicted is the answer, not
    evidence. The song floor rejects truncated parses rather than short shows -- a set cut
    to five songs by a lightning call is a complete record of what the band played.
    """
    floor = None
    if config.lookback_years:
        floor = asof - dt.timedelta(days=int(round(365.25 * config.lookback_years)))
    out = []
    for show in shows:
        date = _as_date(show.get("date") or "")
        if date is None or date >= asof or (floor is not None and date < floor):
            continue
        songs = _songs_of(show)
        if len(songs) < config.min_show_songs:
            continue
        out.append((date, songs))
    out.sort(key=lambda pair: pair[0])
    return out


def _base_rates(past: Sequence[tuple[dt.date, list[str]]], config: ScoreConfig,
                asof: dt.date) -> tuple[dict[str, float], dict[str, float]]:
    """Weighted nights-containing over weighted nights, plus the raw weighted evidence.

    Nights-containing, not plays: a song played twice in one night is one night's worth of
    evidence that the band reaches for it, and counting it twice would make a rate that can
    exceed 1.0 and no longer reads as a probability.
    """
    weighted: defaultdict[str, float] = defaultdict(float)
    total = 0.0
    for date, songs in past:
        weight = (_era(date.year, config)
                  * math.pow(0.5, (asof - date).days / config.recency_halflife_days))
        total += weight
        for song in set(songs):
            weighted[song] += weight
    if not total:
        return {}, dict(weighted)
    return {song: value / total for song, value in weighted.items()}, dict(weighted)


def rotation(past: Sequence[tuple[dt.date, list[str]]]) -> tuple[dict[str, int],
                                                                 dict[str, float]]:
    """Shows since each song last played, and its typical gap between plays.

    Counted in shows rather than days on purpose: the band's rotation moves when they play,
    not when the calendar does, and a three-month winter off should not make everything
    overdue at once.

    PUBLIC AS OF SLICE 4, AND NOT YET CALLED BY ANYTHING THE SONGBOOK SHIPS. The Songbook
    computes gap and mean-gap itself, client-side, in JavaScript
    (``famoe.ly/bin/songbook/logic.js``'s ``dueRatio()``, called from ``aggregateWindow()``) --
    see :func:`overdue_ratio` for the one piece of that computation this module shares with
    it. This function is made
    public here as groundwork for the Scorecard port instead, which would consume it directly
    rather than writing a fourth copy of the same shows-since-last-play arithmetic. The
    Scorecard itself is out of scope for the Songbook
    (docs/plans/2026-07-22-songbook-design.md, "Out of scope") -- its existence as a public
    function is not evidence that port has happened.
    """
    last_at: dict[str, int] = {}
    intervals: defaultdict[str, list[int]] = defaultdict(list)
    for index, (_, songs) in enumerate(past):
        for song in set(songs):
            if song in last_at:
                intervals[song].append(index - last_at[song])
            last_at[song] = index
    total = len(past)
    since = {song: total - 1 - index for song, index in last_at.items()}
    typical = {song: (sum(gaps) / len(gaps) if gaps else float(total))
               for song, gaps in ((s, intervals[s]) for s in last_at)}
    return since, typical


# The ratio when it cannot be computed normally -- mean_gap is falsy, which in practice only
# happens for an empty window (a song with exactly one play still gets a mean_gap of the
# window length itself, never zero; see rotation() above). Settled in slice 2a
# (docs/plans/2026-07-22-songbook-design.md) as the one number this scorer and the Songbook's
# browser JS are both written to use, replacing two people's independent guesses (0.0 in the
# Scorecard's ancestor, 1.0 here) with one decided value. Named so it has exactly one place to
# change, and so tests/unit/test_rotation_parity.py has one thing to import instead of a
# second typed-in-by-hand literal.
OVERDUE_FALLBACK_RATIO = 1.0


def overdue_ratio(gap: int | None, mean_gap: float) -> float:
    """How many "typical gaps" overdue a song is, right now.

    This is the whole of the computation that used to exist three times, in two languages --
    the Scorecard's POC ancestor, the Songbook's POC ancestor, and here. Extracted to its own
    function so this module has exactly one definition of it, published in
    setlistkit.catalog.songbook's module docstring and pinned against a copy of the Songbook's
    own JS in tests/unit/test_rotation_parity.py.

    ``gap`` is shows since the song last played, or ``None`` if it never has. ``mean_gap`` is
    its typical interval between plays -- already carrying its OWN fallback to the window
    length for a song with exactly one play (see rotation()), which is identical across every
    implementation and is not the fallback this function exists for. This function's fallback
    fires only when ``mean_gap`` itself is falsy, which is the empty-window edge case.
    """
    return (gap / mean_gap) if (gap is not None and mean_gap) else OVERDUE_FALLBACK_RATIO


def _score_one(song: str, rate: float,
               rotation_maps: tuple[dict[str, int], dict[str, float]],
               weighted: Mapping[str, float], config: ScoreConfig) -> SongScore:
    """One song's row: its rate, adjusted for how overdue it is and how recently it played.

    The due boost is capped because an uncapped one hands the top of the list permanently to
    whatever was played once in 1994 and never again.
    """
    since, typical = rotation_maps
    gap = since.get(song)
    mean_gap = typical.get(song) or 0.0
    ratio = overdue_ratio(gap, mean_gap)
    due = 1.0 + config.gap_weight * max(0.0, min(ratio, config.gap_cap) - 1.0)
    cool = _cooldown(gap, config)
    return SongScore(song=song, score=rate * due * cool, base=rate, gap_ratio=ratio,
                     cooldown=cool, weighted_plays=weighted.get(song, 0.0), shows_since=gap)


def song_scores(shows: Iterable[Mapping], config: ScoreConfig,
                asof: str | dt.date) -> list[SongScore]:
    """Every candidate song scored as of ``asof``, best first, ties broken on the song name.

    ``shows`` is the shape :func:`setlistkit.store.corpus.shows` returns. Nothing dated on or
    after ``asof`` is read.
    """
    when = _as_date(asof)
    if when is None:
        raise ValueError(f"as-of date is not a date: {asof!r}")
    past = _training_shows(shows, config, when)
    base, weighted = _base_rates(past, config, when)
    rotation_maps = rotation(past)
    out = [_score_one(song, rate, rotation_maps, weighted, config)
           for song, rate in base.items()]
    return sorted(out, key=lambda row: (-row.score, row.song))

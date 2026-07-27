# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""How much better than chance -- and better than the dumb answer, which is the harder question.

A hit rate on its own says nothing. Forty percent of a setlist is either remarkable or embarrassing
depending entirely on what it is being compared against, and the comparison is where almost all the
dishonesty in model reporting lives. So this module builds the comparison explicitly, as a LADDER,
and reports every rung:

1. **Chance.** Draw ``picks`` songs uniformly from the band's active repertoire. The overlap with
   tonight's setlist is HYPERGEOMETRIC -- drawing without replacement from a pool where some items
   are "successes" -- so its mean and variance have closed forms and nothing needs simulating.
2. **Popularity.** Always name the ``picks`` songs played on the most nights so far. No recency, no
   model, no thought. This is the real competitor.
3. **The model.**

THE FIRST RUNG IS NEARLY WORTHLESS AND IS HERE ANYWAY. Uniform chance assumes every song is equally
likely, which is flatly false -- and the size of the gap between rungs 1 and 2 is exactly the
measure of how false. A model that beats chance by 4x while beating popularity by 1.4x has mostly
demonstrated that some songs are more popular than others, which nobody disputed. Reporting the 4x
alone is the standard way to make a model look better than it is, so both numbers are computed
together and the report carries both or neither.

WHY THE POOL DEFINITION IS A RANGE AND NOT A NUMBER. "The active repertoire" has no true value: a
song last played four years ago is neither clearly in nor clearly out. It matters, and it matters
in OPPOSITE DIRECTIONS for the two comparisons -- a wider pool makes chance look worse (flattering
the model) while raising the bar popularity clears (unflattering). There is no single span that is
conservative for both, so :func:`measure` takes several and the caller is expected to publish the
range. A single span picked after seeing the results is the thing this design exists to prevent.

SEPARATELY, AND MORE INFORMATIVE: the model publishes CONFIDENCES, not just a set of names, and a
set-overlap metric throws that away. :func:`brier_decomposition` scores the probabilities as
probabilities, and splits the result the way Murphy (1973) does, because "how good is this
forecast" is really two questions that a single number cannot separate:

* **Reliability** -- when it says 30%, does it happen 30% of the time? This is calibration, and it
  is the axis :class:`~setlistkit.picks.ledger.Outcome` already reports.
* **Resolution** -- does it say different things about different nights? A forecast that always
  states the base rate is perfectly reliable and completely useless. Only resolution catches that.

Brier = reliability - resolution + uncertainty, exactly, and the test suite asserts the identity
rather than trusting the algebra.
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ..model.backtest import gradeable
from ..model.rollout import SongChance

# How far back a song can have been played and still count as part of the active repertoire. Three
# spans rather than one, for the reason the module docstring gives: no single span is the
# conservative choice for both comparisons, so the honest presentation is the range they span.
DEFAULT_SPANS = (2, 3, 5)

# Forecast bins for the Murphy decomposition. Ten is conventional and the identity holds for any
# partition, so this trades resolution-of-the-resolution-term against bins too sparse to mean
# anything -- it is not a tuned value.
DEFAULT_BINS = 10


@dataclass(frozen=True)
class Chance:
    """What drawing blind would have got, summed over every night graded.

    ``sd`` is the standard deviation of that total, which is what turns a difference into a
    distance. It is the square root of a SUM of per-night variances and not a per-night figure:
    each night is an independent draw, so the variances add.
    """

    expected: float
    sd: float
    pool_median: int

    def sigma(self, hits: int) -> float:
        """How many standard deviations ``hits`` sits above blind drawing.

        Worth stating plainly at the call site: with a couple of thousand songs on the line this
        number is enormous for any model that works at all, and it answers a question nobody
        seriously asks. It rules out coincidence. It says nothing about whether the model is good.
        """
        return (hits - self.expected) / self.sd if self.sd else 0.0


@dataclass(frozen=True)
class Rung:
    """One repertoire span, and what every rung of the ladder scored under it."""

    years: int
    played: int
    chance: Chance
    popularity_hits: int
    model_hits: int

    @property
    def lift_over_chance(self) -> float:
        """How many times blind drawing the model is worth. The flattering number."""
        return self.model_hits / self.chance.expected if self.chance.expected else 0.0

    @property
    def lift_over_popularity(self) -> float:
        """How many times "name the hits" the model is worth. The number that costs something."""
        return self.model_hits / self.popularity_hits if self.popularity_hits else 0.0

    @property
    def skill(self) -> float:
        """Share of popularity's misses the model recovered.

        The meteorologists' skill score, ``(model - baseline) / (perfect - baseline)``, and the
        single most honest summary available: it is bounded above by 1.0, it is negative for a
        model that loses to the baseline, and it cannot be inflated by an easy denominator the way
        a raw hit rate can.
        """
        headroom = self.played - self.popularity_hits
        return (self.model_hits - self.popularity_hits) / headroom if headroom else 0.0


@dataclass(frozen=True)
class Brier:
    """A probabilistic forecast scored as probabilities, split into its parts.

    Every component is in units of squared probability and only compares against another Brier
    over the SAME events. ``score`` is the mean squared error; lower is better.

    THE IDENTITY DOES NOT CLOSE ON ``score``, AND THAT IS NOT A ROUNDING ERROR. Murphy's
    decomposition is exact for the BINNED forecast -- the one where every member of a bin is
    replaced by that bin's mean. Real forecasts vary inside a bin, and that variation is invisible
    to the three terms. It is reported as ``within_bin`` rather than absorbed into one of them or
    hidden behind a loose tolerance, because it is a real quantity: it is how much the binning
    itself threw away, and a large value means the bins are too coarse to describe this forecast.

        score == reliability - resolution + uncertainty + within_bin

    ``binned`` is the first three terms, which is the quantity the decomposition actually explains.
    Quote ``score``; read the components.
    """

    score: float
    reliability: float
    resolution: float
    uncertainty: float
    within_bin: float
    n: int

    @property
    def binned(self) -> float:
        """The Brier score of the binned forecast: exactly what the three terms decompose."""
        return self.reliability - self.resolution + self.uncertainty

    def skill_against(self, other: "Brier") -> float:
        """Brier Skill Score: the share of ``other``'s error this forecast removes.

        1.0 is perfect, 0.0 is no better than ``other``, and negative means worse -- which is a
        real and common outcome for a model measured against a well-chosen baseline, and the
        reason this returns a signed number rather than an absolute improvement.
        """
        return 1.0 - self.score / other.score if other.score else 0.0


def repertoire(nights: Sequence[tuple[str, set[str]]], asof: str, years: int) -> set[str]:
    """Every song played in the ``years`` before ``asof``. The pool a blind draw draws from.

    Not the whole songbook. A band's catalog contains things shelved a decade ago, and counting
    them as drawable would understate what chance achieves -- which would overstate the model by
    exactly the same amount.
    """
    cut = _shift_years(asof, years)
    songs: set[str] = set()
    for night, played in nights:
        if cut <= night < asof:
            songs |= played
    return songs


def chance_overlap(pool_size: int, drawable: int, picks: int) -> tuple[float, float]:
    """Mean and variance of the overlap when ``picks`` are drawn blind from ``pool_size``.

    The hypergeometric, which is the right null here rather than the binomial: a shortlist cannot
    name the same song twice, so the draw is without replacement.

    ``drawable`` is how many of tonight's songs are IN the pool, not how many were played. A song
    outside the repertoire cannot be drawn by any strategy, and charging chance for missing it
    would make the null weaker than it really is -- the one direction that flatters the model.
    """
    if pool_size <= 1 or pool_size <= picks or not drawable:
        return 0.0, 0.0
    rate = drawable / pool_size
    mean = picks * rate
    variance = picks * rate * (1 - rate) * (pool_size - picks) / (pool_size - 1)
    return mean, variance


def perfect_night_odds(pool_size: int, drawable: int, picks: int) -> float:
    """P(a blind draw of ``picks`` contains every one of ``drawable`` songs).

    The "well, technically it COULD get them all" case, answered with a number instead of a shrug.
    Computed through log-gamma because the factorials involved overflow long before the answer
    does -- the result is routinely smaller than 1e-30 and is still perfectly representable.
    """
    if drawable > picks or pool_size < picks or not drawable:
        return 0.0
    return math.exp(_log_choose(picks, drawable) - _log_choose(pool_size, drawable))


def popularity_picks(nights: Sequence[tuple[str, set[str]]], asof: str, picks: int,
                     years: int) -> set[str]:
    """The ``picks`` songs played on the most nights in the ``years`` before ``asof``.

    Counted by NIGHTS CONTAINING rather than by total performances, matching
    :func:`~setlistkit.model.backtest.naive_baseline`, so the two baselines in this codebase mean
    the same thing by "popular". Ties break on the song name, which is arbitrary and stable --
    arbitrary matters less than reproducible here, since a tie at rank 30 would otherwise make the
    baseline depend on dict ordering.
    """
    cut = _shift_years(asof, years)
    seen: Counter = Counter()
    for night, played in nights:
        if cut <= night < asof:
            seen.update(played)
    ranked = sorted(seen, key=lambda song: (-seen[song], song))
    return set(ranked[:picks])


def brier_decomposition(pairs: Sequence[tuple[float, int]], *,
                        bins: int = DEFAULT_BINS) -> Brier:
    """Score (forecast, outcome) pairs and split the result into Murphy's three terms.

    ``pairs`` is every (probability, happened) the forecast committed to -- including the songs it
    gave no probability at all, which enter as ``(0.0, 1)`` when they played. Leaving those out
    would score the model only on the songs it chose to talk about, which is marking your own
    homework.
    """
    if not pairs:
        return Brier(score=0.0, reliability=0.0, resolution=0.0, uncertainty=0.0,
                     within_bin=0.0, n=0)
    total = len(pairs)
    base = sum(outcome for _p, outcome in pairs) / total
    score = sum((forecast - outcome) ** 2 for forecast, outcome in pairs) / total

    grouped: dict[int, list[tuple[float, int]]] = {}
    for forecast, outcome in pairs:
        # min() keeps a forecast of exactly 1.0 in the last bin rather than in a bin of its own.
        grouped.setdefault(min(int(forecast * bins), bins - 1), []).append((forecast, outcome))

    reliability = resolution = 0.0
    for members in grouped.values():
        weight = len(members) / total
        mean_forecast = statistics.fmean(forecast for forecast, _o in members)
        mean_outcome = statistics.fmean(outcome for _f, outcome in members)
        reliability += weight * (mean_forecast - mean_outcome) ** 2
        resolution += weight * (mean_outcome - base) ** 2
    uncertainty = base * (1 - base)
    return Brier(score=score, reliability=reliability, resolution=resolution,
                 uncertainty=uncertainty,
                 within_bin=score - (reliability - resolution + uncertainty), n=total)


def _rung(nights: Sequence[tuple[str, set[str]]],
          graded: Sequence[tuple[str, Sequence[SongChance], set[str]]],
          years: int, picks: int) -> Rung:
    """One repertoire span's worth of the ladder, totalled across every graded night."""
    pools = [repertoire(nights, night, years) for night, _chances, _played in graded]
    overlaps = [chance_overlap(len(pool), len(played & pool), picks)
                for pool, (_night, _chances, played) in zip(pools, graded)]
    return Rung(
        years=years,
        played=sum(len(played) for _night, _chances, played in graded),
        chance=Chance(
            expected=sum(mean for mean, _var in overlaps),
            sd=math.sqrt(sum(var for _mean, var in overlaps)),
            pool_median=int(statistics.median(len(pool) for pool in pools)) if pools else 0),
        popularity_hits=sum(len(played & popularity_picks(nights, night, picks, years))
                            for night, _chances, played in graded),
        model_hits=sum(len({chance.song for chance in chances} & played)
                       for _night, chances, played in graded))


def measure(shows: Iterable[Mapping], graded: Sequence[tuple[str, Sequence[SongChance], set[str]]],
            *, picks: int, spans: Sequence[int] = DEFAULT_SPANS) -> tuple[Rung, ...]:
    """Walk the ladder for every repertoire span.

    ``graded`` is (night, what the model finally said, what actually played) -- the caller's own
    walk, passed in rather than recomputed, so this cannot silently grade a different set of
    nights than the page reports on.
    """
    nights = gradeable(shows, 1)
    return tuple(_rung(nights, graded, years, picks) for years in spans)


@dataclass(frozen=True)
class Forecasts:
    """Three answers to the same question, over exactly the same song-nights.

    Separating ``strict`` from ``backed_off`` is the whole point, because a shortlist conflates two
    different claims and only one of them is the model's actual work:

    * ``strict`` -- the shortlist read literally: these songs at these odds, EVERYTHING ELSE AT
      ZERO. A hard zero is the strongest statement a forecast can make, and a top-30 list over a
      repertoire of hundreds makes it hundreds of times a night.
    * ``backed_off`` -- the same ranking, with unlisted songs falling back to their long-run rate
      instead of to zero. This is the charitable reading: the model spoke about 30 songs and
      declined to comment on the rest.
    * ``climate`` -- every song at its long-run rate. Nobody's model.

    If ``strict`` loses to ``climate`` while ``backed_off`` beats it, the ranking is worth
    something and the TRUNCATION is what costs -- which is a fixable problem with the report, not
    a broken model. Collapsing these into one number would hide exactly that distinction.
    """

    strict: list[tuple[float, int]]
    backed_off: list[tuple[float, int]]
    climate: list[tuple[float, int]]


def scored_pairs(shows: Iterable[Mapping],
                 graded: Sequence[tuple[str, Sequence[SongChance], set[str]]], *,
                 years: int) -> Forecasts:
    """Every song in every night's repertoire, scored three ways. See :class:`Forecasts`.

    All three answer for the same songs, which is the only way two Brier scores can be compared.
    The universe per night is the repertoire PLUS anything that actually played, so a song none of
    them saw coming counts against all three equally.
    """
    nights = gradeable(shows, 1)
    out = Forecasts(strict=[], backed_off=[], climate=[])
    for night, chances, played in graded:
        _score_night(out, nights, night, years=years, played=played,
                     stated={chance.song: chance.confidence for chance in chances})
    return out


def _score_night(out: Forecasts, nights: Sequence[tuple[str, set[str]]], night: str, *,
                 years: int, played: set[str], stated: Mapping[str, float]) -> None:
    """Append one night's song-nights to all three forecasts, in lockstep."""
    rates = _play_rates(nights, night, years)
    for song in repertoire(nights, night, years) | played:
        happened = 1 if song in played else 0
        rate = rates.get(song, 0.0)
        out.strict.append((stated.get(song, 0.0), happened))
        out.backed_off.append((stated.get(song, rate), happened))
        out.climate.append((rate, happened))


def _play_rates(nights: Sequence[tuple[str, set[str]]], asof: str,
                years: int) -> dict[str, float]:
    """Share of the last ``years`` of nights each song turned up on: the climatology forecast.

    The probabilistic twin of :func:`popularity_picks`. A forecast that names no songs but states
    each song's long-run rate is the honest do-nothing baseline, and it is a hard one to beat on
    Brier because it is very nearly perfectly calibrated by construction.
    """
    cut = _shift_years(asof, years)
    seen: Counter = Counter()
    total = 0
    for night, played in nights:
        if cut <= night < asof:
            seen.update(played)
            total += 1
    return {song: count / total for song, count in seen.items()} if total else {}


def _shift_years(date: str, years: int) -> str:
    """``date`` moved back ``years``, as an ISO string. Feb 29 lands on Feb 28."""
    when = dt.date.fromisoformat(date)
    try:
        return when.replace(year=when.year - years).isoformat()
    except ValueError:
        return when.replace(year=when.year - years, day=28).isoformat()


def _log_choose(n: int, k: int) -> float:
    """log(n choose k), via log-gamma so nothing overflows on the way to a tiny answer."""
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

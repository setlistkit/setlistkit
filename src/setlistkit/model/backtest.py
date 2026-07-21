# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Walk the scorer forward one show at a time and grade it against what was actually played.

For each show in an evaluation window, score every song using only shows dated before it, take
the top ``m`` where ``m`` is how many songs the band actually played that night, and report the
share that were right. Top-m recall, and honest by construction because
:func:`setlistkit.model.scores.song_scores` will not read past its as-of date.

This exists because the scorer's constants are inherited and unvalidated. They were fitted
against a corpus starting in 2020 with a third-party base rate blended in at 0.3; the corpus now
runs to 1992 and the blend is gone. Numbers carried across that change are not evidence of
anything, and this module is how they get re-earned.

TUNING IS A TRAP AND THE SHAPE OF THIS MODULE SAYS SO
The pipeline this comes from auto-tuned nightly on the most recent window and adopted whatever
scored best. That is an in-sample criterion, and an out-of-sample probe caught it: the adopted
config scored about 1.4 sigma WORSE on held-out shows than the values it replaced. Widening the
grid would not have helped, because the search was not overfitting the grid, it was overfitting
the window.

So :func:`holdout` is the only function here that picks a config, it searches strictly on the
older half of a span and reports its score on the newer half, and it returns that config rather
than writing it anywhere. Adopting a number is a human decision made against an out-of-sample
figure, not something a nightly job does quietly.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from .scores import ScoreConfig, _as_date, _songs_of, song_scores


@dataclass(frozen=True)
class BacktestResult:
    """What the scorer scored, and how sure we are of it.

    ``std_error`` is the standard error of the per-show mean. A hit rate quoted without it
    invites comparing two configs that differ by a third of a show.
    """

    n_shows: int
    hit_rate: float
    std_error: float
    through: str


@dataclass(frozen=True)
class HoldoutResult:
    """A search on older shows, graded on newer ones it never saw.

    ``in_sample`` is the winning config's score on the shows it was chosen on and is optimistic
    by construction. ``out_of_sample`` is the number worth quoting.
    """

    train_shows: int
    held_shows: int
    in_sample: float
    out_of_sample: float
    std_error: float
    best: dict[str, object]


def _gradeable(shows: Iterable[Mapping], min_songs: int) -> list[tuple[str, set[str]]]:
    """(date, songs played) for every show complete enough to grade, oldest first.

    A night recorded as two songs grades the taper's notes, not the model.
    """
    out = []
    for show in shows:
        date = _as_date(show.get("date") or "")
        if date is None:
            continue
        songs = set(_songs_of(show))
        if len(songs) >= min_songs:
            out.append((date.isoformat(), songs))
    out.sort(key=lambda pair: pair[0])
    return out


def _window(shows: Iterable[Mapping], count: int,
            min_songs: int) -> list[tuple[str, set[str]]]:
    """The most recent ``count`` gradeable shows."""
    return _gradeable(shows, min_songs)[-count:] if count > 0 else []


def hit_rates(shows: Iterable[Mapping], tests: Sequence[tuple[str, set[str]]],
              config: ScoreConfig) -> list[float]:
    """Top-m recall for each test show, using only shows dated before it.

    ``m`` is how many songs were actually played, so the scorer is asked for exactly as many
    guesses as there were answers. Predicting more would inflate recall for free.
    """
    corpus = list(shows)
    out = []
    for date, actual in tests:
        ranked = song_scores(corpus, config, asof=date)
        picked = {row.song for row in ranked[:len(actual)]}
        out.append(len(picked & actual) / len(actual))
    return out


def _summarise(rates: Sequence[float], through: str) -> BacktestResult:
    if not rates:
        return BacktestResult(n_shows=0, hit_rate=0.0, std_error=0.0, through=through)
    error = statistics.pstdev(rates) / math.sqrt(len(rates)) if len(rates) > 1 else 0.0
    return BacktestResult(n_shows=len(rates), hit_rate=sum(rates) / len(rates),
                          std_error=error, through=through)


def backtest(shows: Iterable[Mapping], config: ScoreConfig, *, count: int = 24,
             min_songs: int = 8) -> BacktestResult:
    """Walk ``config`` forward over the most recent ``count`` gradeable shows."""
    corpus = list(shows)
    tests = _window(corpus, count, min_songs)
    through = tests[-1][0] if tests else ""
    return _summarise(hit_rates(corpus, tests, config), through)


def naive_baseline(shows: Iterable[Mapping],
                   tests: Sequence[Mapping | tuple[str, set[str]]]) -> float:
    """Top-m by all-time raw frequency: no recency, no gap, no era.

    The floor any amount of machinery should clear. It flatters the model, because it withholds
    recency too -- which is why :func:`recency_baseline` exists beside it.
    """
    corpus = _gradeable(shows, 1)
    graded = _coerce(tests)
    rates = []
    for date, actual in graded:
        freq: Counter = Counter()
        for other, songs in corpus:
            if other < date:
                freq.update(set(songs))
        picked = {song for song, _ in freq.most_common(len(actual))}
        rates.append(len(picked & actual) / len(actual))
    return sum(rates) / len(rates) if rates else 0.0


def recency_baseline(shows: Iterable[Mapping],
                     tests: Sequence[Mapping | tuple[str, set[str]]],
                     config: ScoreConfig) -> float:
    """Top-m by recency-weighted frequency alone: no gap, no era, no cooldown.

    The fair comparison. Recency is the single dominant signal, so the question worth asking is
    not "does the model beat a coin" but "does the rest of the machinery earn its place on top
    of recency", and only this baseline answers it.
    """
    corpus = _gradeable(shows, config.min_show_songs)
    rates = []
    for date, actual in _coerce(tests):
        weighted = _weighted_frequency(corpus, date, config.recency_halflife_days)
        picked = {song for song, _ in weighted.most_common(len(actual))}
        rates.append(len(picked & actual) / len(actual))
    return sum(rates) / len(rates) if rates else 0.0


def _weighted_frequency(corpus: Sequence[tuple[str, set[str]]], date: str,
                        halflife: float) -> Counter:
    """Recency-weighted nights-containing for every song, using only shows before ``date``."""
    asof = _as_date(date)
    weighted: Counter = Counter()
    for other, songs in corpus:
        when = _as_date(other)
        if when is None or asof is None or when >= asof:
            continue
        weight = math.pow(0.5, (asof - when).days / halflife)
        for song in set(songs):
            weighted[song] += weight
    return weighted


def _coerce(tests: Sequence[Mapping | tuple[str, set[str]]]) -> list[tuple[str, set[str]]]:
    """Accept either raw shows or the (date, songs) pairs the window produces."""
    out = []
    for test in tests:
        if isinstance(test, tuple):
            out.append(test)
            continue
        date = _as_date(test.get("date") or "")
        if date is not None:
            out.append((date.isoformat(), set(_songs_of(test))))
    return out


def holdout(shows: Iterable[Mapping], config: ScoreConfig,
            grid: Mapping[str, Sequence], *, window: int = 80,
            min_songs: int = 8) -> HoldoutResult:
    """Grid-search on the older half of a span, then score the winner on the newer half.

    ``grid`` maps :class:`ScoreConfig` field names to the values to try. The returned config is
    a recommendation and nothing here writes it anywhere: adopting it is a decision made by a
    person looking at ``out_of_sample``, which is the only number in the result that was not
    chosen to look good.
    """
    corpus = list(shows)
    tests = _window(corpus, window, min_songs)
    half = len(tests) // 2
    train, held = tests[:half], tests[half:]
    best, best_score = _search(corpus, train, config, grid)
    rates = hit_rates(corpus, held, replace(config, **best) if best else config)
    graded = _summarise(rates, held[-1][0] if held else "")
    return HoldoutResult(train_shows=len(train), held_shows=len(held),
                         in_sample=max(best_score, 0.0), out_of_sample=graded.hit_rate,
                         std_error=graded.std_error, best=best)


def _search(corpus: Sequence[Mapping], train: Sequence[tuple[str, set[str]]],
            config: ScoreConfig,
            grid: Mapping[str, Sequence]) -> tuple[dict[str, object], float]:
    """The best config in ``grid`` on ``train``, and its score there.

    Kept separate from :func:`holdout` so there is exactly one place the search can see shows,
    and it takes ``train`` rather than the whole span. A search that can reach the held-out half
    is the failure this module exists to rule out, and it should take an edit to cause, not an
    oversight.
    """
    keys = list(grid)
    best: dict[str, object] = {}
    best_score = -1.0
    for combo in _combinations([grid[key] for key in keys]):
        candidate = dict(zip(keys, combo))
        rates = hit_rates(corpus, train, replace(config, **candidate))
        score = sum(rates) / len(rates) if rates else 0.0
        if score > best_score:
            best_score, best = score, candidate
    return best, best_score


def _combinations(values: Sequence[Sequence]) -> Iterable[tuple]:
    """Every combination across ``values``; one empty combination when there is nothing to try."""
    out: list[tuple] = [()]
    for options in values:
        out = [combo + (option,) for combo in out for option in options]
    return out

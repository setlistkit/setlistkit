# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""How likely is each song tonight? Draw the night a few hundred times and count.

:mod:`setlistkit.model.scores` produces a RANKING, and a ranking carries no probabilities. That is
the whole limit on what :mod:`setlistkit.model.backtest` can report: a hit rate, and nothing to
compare it against. Rank 4 and rank 40 are both "predicted" and there is no number saying the
first was meant more seriously than the second.

Rollouts fix that. Draw ``set_len`` songs weighted by score, a few hundred times, and the share of
draws a song appears in IS its stated confidence. Three numbers follow that could not be computed
before:

    recall_at_n        hits / m                 how much it got right
    expected_hits      sum of confidences       how much it CLAIMED it would get right
    calibration_ratio  hits / expected          whether those two agree

The third is the one worth having, and it is the reason rollouts are not decoration. A model can
be right rarely and honest about it; that is a useful model. A model that is right rarely while
claiming it will be right often is broken in a way no hit rate can show. Calibration near 1.0 is
what "starting to look healthy" means.

Single night only. Simulating a whole tour means feeding each simulated night back into the
history so the next night's recency and cooldown see it, which is a different piece of work (the
forward sim) and is not this.

The cost is in the scoring, not the drawing: :func:`~setlistkit.model.scores.song_scores` walks
the whole corpus once per night, and the draws are a few hundred passes over a list of fifty.
Lowering ``rollouts`` to iterate faster barely helps; it is the right knob for noise and the wrong
one for speed.
"""

from __future__ import annotations

import datetime as dt
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .scores import ScoreConfig, song_scores
from .setlists import PredictConfig, draw_set

# How much deeper than the set the draw pool reaches. Four is what the version this came from
# used: wide enough that an unlikely song can still turn up, narrow enough that the thousandth-
# ranked song is not quietly given a lottery ticket every night. Not a config knob, because it is
# not a threshold on anything measured -- it is the shape of the draw itself.
_POOL_MULTIPLE = 4


@dataclass(frozen=True)
class SongChance:
    """One song's stated chance of turning up on the night, and where that puts it.

    ``rank`` is 1-based and is a property of THIS night's ranking, not of the song. Stored beside
    the confidence rather than left to the reader's enumerate() because the ledger keys on it:
    ``(iteration, night, rank)`` is what makes a recorded prediction addressable after the fact.
    """

    rank: int
    song: str
    confidence: float


def rollout_night(shows: Iterable[Mapping], score_config: ScoreConfig,
                  predict_config: PredictConfig, *, asof: str | dt.date,
                  set_len: int) -> list[SongChance]:
    """Every song's chance of being played on ``asof``, best first, truncated to ``top_n``.

    Honest by construction: ``song_scores`` refuses to read anything dated on or after ``asof``,
    so nothing here has to filter the corpus and nothing here is able to widen what the scorer
    would look at. ``shows`` can be the whole corpus, including nights after the one being
    predicted, and the answer does not change -- which is the strongest leak test available and
    the one the suite runs.

    ``set_len`` is how many songs one draw takes. The replay passes the number actually played
    that night, for the same reason ``backtest.hit_rates`` does: asking for more guesses than
    there were answers inflates recall for free.

    Deterministic. One generator, seeded from the config, for the whole call -- so the same seed
    and the same corpus produce the same list, twice, and a replay can be re-run and compared.
    """
    ranked = song_scores(shows, score_config, asof=asof)
    # At least `top_n` deep, whatever the set length. The pool bounds what can be RECORDED, not
    # just what can be drawn: a five-song night against a 4x pool could only ever report twenty
    # songs, and the missing ten would read as ten songs with no chance rather than ten songs
    # nobody asked about -- understating expected_hits and flattering the calibration ratio.
    width = max(set_len * _POOL_MULTIPLE, predict_config.top_n)
    pool = [row.song for row in ranked[:width]]
    weights = {row.song: row.score for row in ranked[:width]}
    # nosec B311 -- a reproducible simulation, seeded on purpose. Nothing here is a secret.
    rng = random.Random(predict_config.seed)  # nosec B311
    counts: Counter = Counter()
    for _ in range(predict_config.rollouts):
        # set(), not the list: a draw is without replacement so it cannot repeat, but counting
        # appearances rather than draws is what makes the number a probability of the song
        # turning up at all -- which is what it is compared against when the night is graded.
        counts.update(set(draw_set(pool, weights, set_len, rng)))
    top = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:predict_config.top_n]
    return [SongChance(rank=rank, song=song, confidence=count / predict_config.rollouts)
            for rank, (song, count) in enumerate(top, start=1)]

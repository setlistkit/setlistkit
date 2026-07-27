# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""One night, predicted whole: what is likely, and what that looks like as a setlist.

The composition point, and the only place the two halves meet. :mod:`setlistkit.model.rollout`
says how likely each song is and is the half that gets GRADED; :mod:`setlistkit.model.setlists`
says what order they would come in and is presentation only. Keeping them in separate modules and
joining them here is what stops the ordering from quietly getting into a metric.

Everything in a :class:`Prediction` is a tuple, so two predictions compare with ``==``. That is
not tidiness either: the leak test in the suite runs this twice, against a corpus that includes
the night being predicted and one that stops before it, and asserts the two results are identical.
One equality check covers every path at once -- scores, draws, transitions, ordering, encore --
which is worth more than four narrower assertions that each cover one.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .rollout import SongChance, rollout_night
from .scores import ScoreConfig, song_scores
from .setlists import Entry, PredictConfig, order_set, pick_encore, transitions


@dataclass(frozen=True)
class Prediction:
    """One night's whole prediction: the graded half, the shown half, and what produced them.

    ``set_len`` is carried because it is not derivable from ``setlist`` -- a corpus with fewer
    songs than the set asks for produces a short one -- and because it is the single number a
    replay hands the predictor about the night it is predicting. Recording it beside the answer
    is what makes that visible later rather than something to reconstruct.
    """

    asof: str
    set_len: int
    chances: tuple[SongChance, ...]
    setlist: tuple[Entry, ...]
    encore: tuple[Entry, ...]


def predict_night(shows: Iterable[Mapping], score_config: ScoreConfig,
                  predict_config: PredictConfig, *, asof: str | dt.date,
                  set_len: int | None = None) -> Prediction:
    """Predict the night of ``asof``, reading only what was true before it.

    ``set_len`` defaults to the configured one. A replay overrides it with the count actually
    played that night; a live prediction cannot, and has to live with the default until there is
    a set-length prior worth trusting.

    The shown setlist is the top ``set_len`` songs BY CONFIDENCE rather than a single draw. A draw
    is one plausible night out of five hundred and reads as a strange one about as often as not;
    the confidence ranking is the night the model actually means. The encore is picked from what
    the set did not use, so the two never collide.

    The corpus is scored twice -- once inside the rollout, once here for the encore's base rates --
    which is one extra pass per call. Worth it: it is what keeps ``rollout_night`` the only thing
    that decides what its scorer may see, rather than trusting a caller to hand it honest scores.
    """
    corpus = list(shows)
    length = predict_config.set_len if set_len is None else set_len
    chances = rollout_night(corpus, score_config, predict_config, asof=asof, set_len=length)
    trans = transitions(corpus, asof=asof)
    scores = song_scores(corpus, score_config, asof=asof)
    chosen = [chance.song for chance in chances[:length]]
    setlist = order_set(chosen, trans, predict_config)
    encore = pick_encore(scores, trans, predict_config, set(chosen))
    return Prediction(asof=trans.asof, set_len=length, chances=tuple(chances),
                      setlist=tuple(setlist), encore=tuple(encore))

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Model: prediction, backtest, tuning, and forward simulation.

May import ``catalog``; must not import ``picks`` or ``report``.

:mod:`~setlistkit.model.scores` ranks every song as of a date and refuses to read past it.
:mod:`~setlistkit.model.backtest` walks that forward and grades it, which is the only reason to
believe any number the scorer produces -- its constants were inherited from a pipeline fitted
against a six-year corpus and are not validated for this one.

:mod:`~setlistkit.model.rollout` turns the ranking into stated probabilities by drawing the night
a few hundred times, which is what makes calibration measurable at all.
:mod:`~setlistkit.model.setlists` lays a chosen bag of songs out in an order, and is presentation
only -- nothing scores it. :mod:`~setlistkit.model.predict` is where the two meet.
"""

from .backtest import BacktestResult, HoldoutResult, backtest, gradeable, holdout, hit_rates
from .backtest import naive_baseline, recency_baseline
from .predict import Prediction, predict_night
from .rollout import SongChance, rollout_night
from .scores import OVERDUE_FALLBACK_RATIO, ScoreConfig, SongScore, overdue_ratio, rotation
from .scores import song_scores
from .setlists import Entry, PredictConfig, Transitions, draw_set, order_set, pick_encore
from .setlists import transitions
from .settings import DEFAULT_REPLAY_SPAN, predict_config, replay_window

__all__ = ["BacktestResult", "DEFAULT_REPLAY_SPAN", "Entry", "HoldoutResult",
           "OVERDUE_FALLBACK_RATIO", "PredictConfig", "Prediction", "ScoreConfig", "SongChance",
           "SongScore", "Transitions", "backtest", "draw_set", "gradeable", "hit_rates",
           "holdout", "naive_baseline", "order_set", "overdue_ratio", "pick_encore",
           "predict_config", "predict_night", "recency_baseline", "replay_window",
           "rollout_night", "rotation", "song_scores", "transitions"]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Model: prediction, backtest, tuning, and forward simulation.

May import ``catalog``; must not import ``picks`` or ``report``.

:mod:`~setlistkit.model.scores` ranks every song as of a date and refuses to read past it.
:mod:`~setlistkit.model.backtest` walks that forward and grades it, which is the only reason to
believe any number the scorer produces -- its constants were inherited from a pipeline fitted
against a six-year corpus and are not validated for this one.
"""

from .backtest import BacktestResult, HoldoutResult, backtest, holdout, hit_rates
from .backtest import naive_baseline, recency_baseline
from .scores import OVERDUE_FALLBACK_RATIO, ScoreConfig, SongScore, overdue_ratio, rotation
from .scores import song_scores

__all__ = ["BacktestResult", "HoldoutResult", "OVERDUE_FALLBACK_RATIO", "ScoreConfig",
           "SongScore", "backtest", "hit_rates", "holdout", "naive_baseline",
           "overdue_ratio", "recency_baseline", "rotation", "song_scores"]

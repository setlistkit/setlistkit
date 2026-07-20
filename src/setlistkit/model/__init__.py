"""Model: prediction, backtest, tuning, and forward simulation.

May import ``catalog``; must not import ``picks`` or ``report``. The model math ports across
essentially unchanged from the prior pipeline along with its tests — the expensive,
hard-won part — while everything structural around it is written fresh. (Populated in a
later phase.)
"""

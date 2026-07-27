# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Picks: the prediction ledger, scoring, and history.

May import ``catalog`` and ``model``; must not import ``report``. The ledger keeps a
committed plain-text view alongside the SQLite state, because reviewable diffs of derived
state have caught real bugs -- that view is ``slkit dump picks``, and this is the strongest
case for it in the project: the picks tables are the only ones here that cannot be rebuilt.

:mod:`~setlistkit.picks.ledger` stores what was predicted and grades it.
:mod:`~setlistkit.picks.replay` is the driver that walks the corpus one night at a time behind
a leak boundary, which is how the facility gets exercised end to end with no schedule source.
"""

from .ledger import LEDGER_TABLES, Outcome, grade, record, score_outcome
from .replay import NightResult, ReplayReport, ReplaySpec, replay

__all__ = ["LEDGER_TABLES", "NightResult", "Outcome", "ReplayReport", "ReplaySpec", "grade",
           "record", "replay", "score_outcome"]

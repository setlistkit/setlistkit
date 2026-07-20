# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Picks: the prediction ledger, scoring, and history.

May import ``catalog`` and ``model``; must not import ``report``. The ledger keeps a
committed plain-text view alongside the SQLite state, because reviewable diffs of derived
state have caught real bugs. (Populated in a later phase.)
"""

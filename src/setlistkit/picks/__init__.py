"""Picks: the prediction ledger, scoring, and history.

May import ``catalog`` and ``model``; must not import ``report``. The ledger keeps a
committed plain-text view alongside the SQLite state, because reviewable diffs of derived
state have caught real bugs. (Populated in a later phase.)
"""

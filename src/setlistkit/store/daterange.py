# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""An inclusive date window, and the one place its SQL is built.

Dates are stored as YYYY-MM-DD strings, which sort lexicographically, so a range is a plain string
comparison and needs no parsing. That equivalence is the whole reason this is cheap, and it holds
only for the full shape: ``2023`` sorts below every date IN 2023, so an unchecked ``--until 2023``
answers a question nobody asked and answers it with silence. Checked here, once.

This module exists because the window is now asked for by two commands that had no reason to meet.
``dump`` narrows a table view; ``export`` narrows a bundle and recomputes statistics over what is
left. Two spellings of "on or after" is two places for the inclusive/exclusive decision to be made
differently, and the one that got it wrong would look right -- an off-by-one at a range edge is a
missing show, not a crash.
"""

from __future__ import annotations

import re

# The full shape, anchored. See the module docstring for why a prefix is refused rather than
# helpfully widened: a caller who writes `2023` and means the year is wrong in a way that returns
# rows, and guessing which of the two they meant is how a range quietly stops being auditable.
_DATE_SHAPE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def check_date(value: str | None, flag: str) -> str | None:
    """Reject a date that is not YYYY-MM-DD, rather than silently answering the wrong question."""
    if value is None:
        return None
    if not _DATE_SHAPE.match(value):
        raise ValueError(f"{flag} wants a YYYY-MM-DD date, not {value!r}")
    return value


def clause(expr: str, since: str | None, until: str | None) -> tuple[str, tuple]:
    """The range test over ``expr``, as ``(sql, params)``. Empty when no bound was given.

    Inclusive at both ends. A half-open range is the classic off-by-one, and these are the views
    people reach for when they already suspect something is wrong -- an end date meaning "up to but
    not including" is a second thing to be wrong about while debugging the first.

    ``expr`` is SQL the caller wrote, never anything a user typed: a column name for a table that
    carries its own date, or a correlated subquery for one that borrows its parent's. The values
    are bound.
    """
    if since is None and until is None:
        return "", ()
    tests, params = [], []
    if since is not None:
        tests.append(f"{expr} >= ?")
        params.append(since)
    if until is not None:
        tests.append(f"{expr} <= ?")
        params.append(until)
    return " WHERE " + " AND ".join(tests), tuple(params)


def label(since: str | None, until: str | None) -> str:
    """How a window reads to a person, in whichever of its three forms was asked for."""
    if since is not None and until is not None:
        return f"{since}..{until}"
    return f"from {since}" if since is not None else f"through {until}"

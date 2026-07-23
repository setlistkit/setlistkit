# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Report windows: an anchor plus up to two offsets, resolved into two inclusive dates.

A report's date window used to be typed as two literal dates on every rebuild -- fine for one run
by hand, and how a page quietly ends up covering the wrong span for six months once three reports
share a rebuild script. This module is the fix: a window is expressed relative to an ANCHOR
(`last_show` by default, so it stays a pure function of the store -- see the "No `today`" rule in
docs/plans/2026-07-22-songbook-design.md), plus up to two offsets from it.

The offset grammar accepted here is a STRICT SUBSET of ISO 8601 durations, written out as an ABNF
grammar in the design doc rather than left as "some ISO 8601 durations", which is not a
specification anyone could implement against or test against. The two traps it exists to close:

- Full ISO 8601 has a `T` production before time-of-day units, and `M` after `T` means MINUTES.
  `PT18M` is a valid ISO 8601 duration meaning eighteen minutes; it is very easy to type by
  accident when eighteen MONTHS was meant, and it silently produces a window a few minutes wide.
  This grammar has no `T` production at all, so `PT18M` is a parse error rather than a bug.
- ISO 8601 orders its units Y-M-D; `P6M3Y` (3 years, 6 months, spelled backwards) is not valid
  ISO 8601 either, but a naive regex of "some digits, then a unit letter, repeated" would happily
  accept it. The grammar below enforces the ordering by nesting the optional groups, the same
  trick RFC 5545 uses for its own duration grammar.

Calendar arithmetic (subtracting "1 month" from a date) is delegated to `isodate`, which returns
an `isodate.Duration` for anything carrying Y or M and a plain `datetime.timedelta` for anything
that is only D or W -- because a fixed number of days is exact and a fixed number of calendar
months is not (see the CLAMPING section of this module and the design doc's "The clamping rule").
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import isodate

# The full grammar from docs/plans/2026-07-22-songbook-design.md, "The offset grammar":
#
#   offset     = "P" ( dur-date / dur-week )
#   dur-date   = dur-year / dur-month / dur-day
#   dur-year   = 1*DIGIT "Y" [ dur-month ]
#   dur-month  = 1*DIGIT "M" [ dur-day ]
#   dur-day    = 1*DIGIT "D"
#   dur-week   = 1*DIGIT "W"
#
# Anchored start-to-end (`^...$`) so a trailing fragment can never sneak a rejected shape past a
# prefix match. The four alternatives are, in order: year (optionally followed by month,
# optionally followed by day), month (optionally followed by day), day alone, week alone.
_OFFSET = re.compile(
    r"^P("
    r"\d+Y(\d+M(\d+D)?)?"
    r"|\d+M(\d+D)?"
    r"|\d+D"
    r"|\d+W"
    r")$"
)


class WindowError(ValueError):
    """An offset, anchor, or `[reports.<name>.window]` stanza that does not parse or resolve.

    A `ValueError` subclass rather than a bare one so a caller that already catches `ValueError`
    (as `store/daterange.py`'s callers do for a malformed `--since`/`--until`) keeps working
    unchanged, while a caller that wants to be specific about which kind of date problem it hit
    can still catch `WindowError` alone.
    """


def parse_offset(text: str):
    """A validated offset literal as an `isodate` duration, ready for date arithmetic.

    Validates against the grammar FIRST, before handing off to `isodate.parse_duration` --
    `isodate` parses full ISO 8601, which is strictly more permissive than what this project
    accepts (it would happily parse `PT18M` as eighteen minutes), so skipping the local check
    would silently widen the accepted grammar to whatever `isodate` happens to support.
    """
    if not _OFFSET.match(text):
        raise WindowError(
            f"{text!r} is not a valid report window offset -- want e.g. P18M, P3Y6M, P30D, P2W "
            "(see the offset grammar in docs/plans/2026-07-22-songbook-design.md)"
        )
    return isodate.parse_duration(text)


def _apply_offset(anchor: date, text: str) -> tuple[date, str | None]:
    """``anchor - offset``, plus a human-readable note when clamping fired.

    Subtracting a calendar month from March 31st has no honest answer -- February has no 31st --
    and `isodate.Duration`'s date arithmetic gives the same answer `dateutil.relativedelta` does:
    take the last valid day of the target month (see the module docstring's CLAMPING section).
    That is a real information loss, not arithmetic, so this function reports it rather than
    letting a caller discover it by comparing days itself. A day/week offset (`isodate` returns a
    plain `timedelta` for those) is exact and can never clamp, so the note is always `None` there.
    """
    offset = parse_offset(text)
    result = anchor - offset
    if isinstance(offset, timedelta) or result.day == anchor.day:
        return result, None
    return result, (f"{anchor.isoformat()} - {text} clamped to {result.isoformat()} "
                    f"({result.strftime('%B')} has no {anchor.day}{_ordinal(anchor.day)})")


def _ordinal(n: int) -> str:
    """"31" -> "st", for the clamp note's "February has no 31st"."""
    if 11 <= n % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

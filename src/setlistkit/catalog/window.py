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
from dataclasses import dataclass
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


# design doc: "since_from = 'year' # start of the anchor's year"
_TRUNCATE_UNITS = ("year", "quarter", "month")

_UNIT_NAME = {"Y": "year", "M": "month", "D": "day"}

# A literal YYYY-MM-DD anchor, checked the same way store/daterange.py checks --since/--until:
# a prefix like "2023" sorts below every date IN 2023 and would answer a question nobody asked.
_ANCHOR_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class WindowSpec:
    """One `[reports.<name>.window]` stanza, validated in shape but not yet resolved to dates.

    `since_back`/`until_back` are offset literals (Task 2's grammar); `since_from` is the one
    truncation this project accepts ("year"/"quarter"/"month") as an alternative way to spell a
    window's start. Exactly one of `since_back`/`since_from` must be set -- see `resolve()`.
    `until_back` is optional: omitted, `until` is the anchor itself, which is what lets the
    simplest report ("everything back to N months ago") skip a key entirely.
    """

    anchor: str = "last_show"
    since_back: str | None = None
    until_back: str | None = None
    since_from: str | None = None


@dataclass(frozen=True)
class ResolvedEndpoint:
    """One resolved date, plus how it was produced -- what `--explain` restates in words."""

    value: str                    # YYYY-MM-DD
    words: str                    # "18 calendar months before the anchor", "the anchor itself"
    clamp_note: str | None = None


@dataclass(frozen=True)
class ResolvedWindow:
    """A window's two endpoints, each carrying its own explanation."""

    since: ResolvedEndpoint
    until: ResolvedEndpoint

    def as_dates(self) -> tuple[str, str]:
        """What `resolve()` returns -- the plain answer, with the explanation dropped."""
        return self.since.value, self.until.value


def resolve_explained(spec: WindowSpec, *, first: str, last: str) -> ResolvedWindow:
    """A window's two endpoints, each resolved INDEPENDENTLY from the anchor, with its own words.

    `first`/`last` are the corpus's earliest/latest stored show dates (YYYY-MM-DD), needed only
    when `spec.anchor` is `last_show`/`first_show`. Never chained: `until` is computed straight
    from `anchor`, never from the already-resolved `since` -- see the module docstring's CLAMPING
    section and the design doc's "resolve every endpoint independently from the anchor."

    This is the one true resolution path; `resolve()` is a thin wrapper over it, so the plain
    answer and the `--explain` answer can never drift apart from each other.
    """
    if spec.since_back is not None and spec.since_from is not None:
        raise WindowError("a window stanza may set since_back or since_from, not both")
    if spec.since_back is None and spec.since_from is None:
        raise WindowError("a window stanza needs since_back or since_from")

    anchor_date = _resolve_anchor(spec.anchor, first=first, last=last)

    if spec.since_back is not None:
        value, clamp = _apply_offset(anchor_date, spec.since_back)
        since = ResolvedEndpoint(value.isoformat(), _describe_offset(spec.since_back), clamp)
    else:
        value = _truncate(anchor_date, spec.since_from)
        since = ResolvedEndpoint(value.isoformat(), f"start of the anchor's {spec.since_from}")

    if spec.until_back is not None:
        value, clamp = _apply_offset(anchor_date, spec.until_back)
        until = ResolvedEndpoint(value.isoformat(), _describe_offset(spec.until_back), clamp)
    else:
        until = ResolvedEndpoint(anchor_date.isoformat(), "the anchor itself")

    return ResolvedWindow(since=since, until=until)


def resolve(spec: WindowSpec, *, first: str, last: str) -> tuple[str, str]:
    """The window's two endpoints as plain `(since, until)` strings, both inclusive.

    See `resolve_explained()` for the version that also says HOW each endpoint was produced --
    that is what `--explain` prints.
    """
    return resolve_explained(spec, first=first, last=last).as_dates()


def _resolve_anchor(anchor: str, *, first: str, last: str) -> date:
    """`last_show`/`first_show`/an explicit date, as a `datetime.date`.

    `today` is deliberately not a branch here: the design doc refuses it explicitly, because an
    anchor tied to wall-clock time makes the resolved window depend on WHEN a report is built
    rather than only on what the store holds, and two runs over an identical store could then
    disagree. Falling through to the "not a valid anchor" error is what rejects it -- `"today"`
    matches neither keyword and does not match `_ANCHOR_DATE`, so no special case is needed to
    reject it, but a test pins the rejection explicitly anyway, since the doc calls it out by name.
    """
    if anchor == "last_show":
        return date.fromisoformat(last)
    if anchor == "first_show":
        return date.fromisoformat(first)
    if _ANCHOR_DATE.match(anchor):
        return date.fromisoformat(anchor)
    raise WindowError(
        f"{anchor!r} is not a valid window anchor -- want last_show, first_show, or YYYY-MM-DD. "
        "'today' is deliberately not accepted: it would make a window depend on wall-clock time "
        "instead of only on the store, and two runs over the same data could then disagree."
    )


def _truncate(anchor: date, unit: str) -> date:
    """The start of `anchor`'s year/quarter/month -- what `since_from` spells."""
    if unit == "year":
        return anchor.replace(month=1, day=1)
    if unit == "quarter":
        return anchor.replace(month=3 * ((anchor.month - 1) // 3) + 1, day=1)
    if unit == "month":
        return anchor.replace(day=1)
    raise WindowError(
        f"{unit!r} is not a valid since_from (want one of {', '.join(_TRUNCATE_UNITS)})"
    )


def _describe_offset(text: str) -> str:
    """`"P18M"` -> `"18 calendar months before the anchor"`. Precondition: already validated."""
    if text.endswith("W"):
        count = text[1:-1]
        return f"{count} calendar week{'' if count == '1' else 's'} before the anchor"
    parts = [f"{count} calendar {_UNIT_NAME[unit]}{'' if count == '1' else 's'}"
             for count, unit in re.findall(r"(\d+)([YMD])", text)]
    return " and ".join(parts) + " before the anchor"

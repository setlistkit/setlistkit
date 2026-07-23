# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `catalog.window`: the offset grammar, clamping arithmetic, anchor resolution, and
`[reports.<name>.window]` parsing.

The grammar table below is not illustrative -- it IS the spec. "Some ISO 8601 durations" is not a
rule anyone could implement against; this table, lifted directly from the design doc's ABNF
section, is.
"""

from datetime import date

import pytest

from setlistkit.catalog.window import WindowError, _apply_offset, parse_offset

# design doc: docs/plans/2026-07-22-songbook-design.md, "The offset grammar"
ACCEPT = ["P18M", "P3Y6M", "P2W", "P1Y", "P1Y6M", "P30D", "P3Y"]
REJECT = [
    "PT18M",   # the single most common mistake this grammar exists to catch: minutes, not P+M
    "P6M3Y",   # wrong order -- ISO's nesting only allows Y before M before D
    "P1W2D",   # weeks are a sibling alternative, never combined with other units
    "18M",     # no leading "P"
    "P",       # no digits at all
    "",        # empty string
    "P1H",     # hours are not in this strict subset
    "P-1M",    # no sign production -- direction lives in the key name, not the literal
]


@pytest.mark.parametrize("text", ACCEPT)
def test_accepted_offsets_parse(text):
    parse_offset(text)   # must not raise


@pytest.mark.parametrize("text", REJECT)
def test_rejected_offsets_raise_window_error(text):
    with pytest.raises(WindowError):
        parse_offset(text)


def test_a_rejected_offset_names_itself_in_the_error():
    with pytest.raises(WindowError, match="PT18M"):
        parse_offset("PT18M")


# design doc: docs/plans/2026-07-22-songbook-design.md, "The clamping rule"
CLAMPED = [
    (date(2026, 3, 31), "P1M", date(2026, 2, 28)),   # February has no 31st
    (date(2026, 3, 30), "P1M", date(2026, 2, 28)),   # nor a 30th
    (date(2026, 3, 29), "P1M", date(2026, 2, 28)),   # nor a 29th, in a non-leap year
    (date(2024, 2, 29), "P1Y", date(2023, 2, 28)),   # leap day, non-leap year
]


@pytest.mark.parametrize("anchor, offset, expected", CLAMPED)
def test_subtracting_a_calendar_offset_clamps_to_the_last_valid_day(anchor, offset, expected):
    result, note = _apply_offset(anchor, offset)
    assert result == expected
    assert note is not None   # every case in this table DOES clamp


def test_the_exact_clamp_note_wording_from_the_design_doc():
    _result, note = _apply_offset(date(2026, 3, 31), "P1M")
    assert note == "2026-03-31 - P1M clamped to 2026-02-28 (February has no 31st)"


def test_a_day_or_week_offset_never_clamps():
    """P30D/P2W arrive as a plain timedelta -- exact arithmetic, so clamping cannot apply.

    2026-03-31 minus 30 days is 2026-03-01 (31 - 30 = 1st of the same month the anchor
    started in) -- the plan draft's expected value here (2026-01-30) does not hold up
    arithmetically and was corrected during implementation; see the slice report.
    """
    result, note = _apply_offset(date(2026, 3, 31), "P30D")
    assert result == date(2026, 3, 1)
    assert note is None


def test_clamping_does_not_round_trip():
    """Pinned so nobody "fixes" this later. `2026-02-28 - P1M -> 2026-01-28`, matching the design
    doc's own example -- NOT back to `2026-01-31`, even though `2026-01-31` is a valid date and
    is where an earlier `-P1M` step (see Task 4's independence test) came from."""
    once, _ = _apply_offset(date(2026, 2, 28), "P1M")
    assert once == date(2026, 1, 28)

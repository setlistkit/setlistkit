# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `catalog.window`: the offset grammar, clamping arithmetic, anchor resolution, and
`[reports.<name>.window]` parsing.

The grammar table below is not illustrative -- it IS the spec. "Some ISO 8601 durations" is not a
rule anyone could implement against; this table, lifted directly from the design doc's ABNF
section, is.
"""

from datetime import date
from pathlib import Path

import pytest

from setlistkit.catalog.window import (WindowError, WindowSpec, _apply_offset, parse_offset,
                                       resolve, resolve_explained, window_spec_from_config)
from setlistkit.config import Config

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


def test_anchor_last_show_uses_the_latest_stored_date():
    since, until = resolve(WindowSpec(since_back="P1M"), first="2020-01-01", last="2026-06-14")
    assert until == "2026-06-14"
    assert since == "2026-05-14"


def test_anchor_first_show_uses_the_earliest_stored_date():
    spec = WindowSpec(anchor="first_show", since_from="year")
    since, until = resolve(spec, first="2020-03-14", last="2026-06-14")
    assert until == "2020-03-14"
    assert since == "2020-01-01"


def test_anchor_accepts_an_explicit_literal_date():
    spec = WindowSpec(anchor="2023-01-01", since_back="P30D")
    since, until = resolve(spec, first="2020-01-01", last="2026-01-01")
    assert until == "2023-01-01"
    assert since == "2022-12-02"


def test_today_is_not_an_accepted_anchor():
    """Deliberate, per the design doc: an anchor must be a pure function of the stored data, or
    two runs over the same store could disagree."""
    spec = WindowSpec(anchor="today", since_back="P1M")
    with pytest.raises(WindowError, match="today"):
        resolve(spec, first="2020-01-01", last="2026-01-01")


def test_an_unrecognized_anchor_is_rejected():
    spec = WindowSpec(anchor="not-a-date", since_back="P1M")
    with pytest.raises(WindowError):
        resolve(spec, first="2020-01-01", last="2026-01-01")


def test_until_defaults_to_the_anchor_itself_when_until_back_is_omitted():
    spec = WindowSpec(since_back="P18M")
    _since, until = resolve(spec, first="2020-01-01", last="2026-06-14")
    assert until == "2026-06-14"


def test_a_stanza_needs_since_back_or_since_from():
    with pytest.raises(WindowError):
        resolve(WindowSpec(), first="2020-01-01", last="2026-01-01")


def test_since_back_and_since_from_together_is_a_config_error():
    spec = WindowSpec(since_back="P1M", since_from="year")
    with pytest.raises(WindowError):
        resolve(spec, first="2020-01-01", last="2026-01-01")


def test_endpoints_resolve_independently_from_the_anchor_never_chained():
    """A test that would FAIL if `until` were computed by extending FROM `since` instead of
    independently from the anchor.

    `since_back="P1M"` from anchor `2026-03-31` clamps to `2026-02-28` (February has no 31st), so
    `since`'s day-of-month is no longer 31 -- three days were lost to the clamp. A chained
    implementation computing `until = since - P2M`-worth-of-remaining-offset from that
    ALREADY-CLAMPED date would land on `2026-01-28` (see Task 3's non-invertibility test:
    `2026-02-28 - P1M -> 2026-01-28`). Resolving `until` independently, straight from the
    original anchor (`2026-03-31 - P2M`), lands on `2026-01-31` instead, because January has a
    31st and the anchor's own day survives when nothing forces a clamp on THAT subtraction. The
    two answers disagree -- which is what makes this test meaningful rather than passing no
    matter which way it was implemented. See the design doc's "resolve every endpoint
    independently from the anchor, and never chain offsets."
    """
    spec = WindowSpec(anchor="2026-03-31", since_back="P1M", until_back="P2M")
    since, until = resolve(spec, first="1990-01-01", last="2030-01-01")
    assert since == "2026-02-28"
    assert until == "2026-01-31"   # NOT "2026-01-28", which chaining through `since` would give


def test_resolve_explained_carries_the_words_and_clamp_note_resolve_does_not():
    spec = WindowSpec(since_back="P18M")
    resolved = resolve_explained(spec, first="2020-01-01", last="2026-06-14")
    assert resolved.since.value == "2024-12-14"
    assert resolved.since.words == "18 calendar months before the anchor"
    assert resolved.until.value == "2026-06-14"
    assert resolved.until.words == "the anchor itself"
    assert resolved.as_dates() == ("2024-12-14", "2026-06-14")


def test_resolve_explained_reports_a_since_from_truncation_in_words():
    spec = WindowSpec(since_from="year")
    resolved = resolve_explained(spec, first="2020-01-01", last="2026-06-14")
    assert resolved.since.value == "2026-01-01"
    assert resolved.since.words == "start of the anchor's year"


def _config(reports=None):
    return Config(data_root=Path("/unused"), user_agent="x", source_path=Path("/unused.toml"),
                  raw={"reports": reports or {}})


def test_a_report_with_no_window_stanza_returns_none():
    assert window_spec_from_config(_config(), "songbook") is None


def test_a_configured_window_becomes_a_windowspec():
    config = _config({"songbook": {"window": {"since_back": "P18M"}}})
    spec = window_spec_from_config(config, "songbook")
    assert spec == WindowSpec(anchor="last_show", since_back="P18M")


def test_anchor_defaults_to_last_show_when_omitted():
    config = _config({"ytd": {"window": {"since_from": "year"}}})
    spec = window_spec_from_config(config, "ytd")
    assert spec.anchor == "last_show"


def test_an_explicit_anchor_is_carried_through():
    config = _config({"r": {"window": {"anchor": "2023-01-01", "since_back": "P1M"}}})
    assert window_spec_from_config(config, "r").anchor == "2023-01-01"


def test_an_unknown_key_in_a_window_stanza_is_a_config_error():
    config = _config({"songbook": {"window": {"since_back": "P18M", "oops": "typo"}}})
    with pytest.raises(WindowError, match="oops"):
        window_spec_from_config(config, "songbook")


def test_only_the_named_report_is_read():
    """A malformed sibling report's window must not break reading THIS report's window -- each
    report's config errors are that report's problem, discovered when it is actually resolved."""
    config = _config({
        "songbook": {"window": {"since_back": "P18M"}},
        "broken": {"window": {"nonsense_key": True}},
    })
    assert window_spec_from_config(config, "songbook").since_back == "P18M"

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `catalog.window`: the offset grammar, clamping arithmetic, anchor resolution, and
`[reports.<name>.window]` parsing.

The grammar table below is not illustrative -- it IS the spec. "Some ISO 8601 durations" is not a
rule anyone could implement against; this table, lifted directly from the design doc's ABNF
section, is.
"""

import pytest

from setlistkit.catalog.window import WindowError, parse_offset

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

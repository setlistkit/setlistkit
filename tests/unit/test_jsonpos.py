# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for the position-tracking JSON scanner.

The scanner exists so a schema error at a JSON path (``/non_song/12/why``) resolves to a
line and column, which json.load knows on a syntax error but jsonschema throws away. These
lock the path->position map and the parse itself against the stdlib json module.
"""

import json

import pytest

from setlistkit.catalog.jsonpos import JSONPosError, Pos, parse


def test_parses_the_same_values_as_stdlib_json():
    text = '{"a": 1, "b": [true, null, "x"], "c": {"d": -2.5}}'
    data, _ = parse(text)
    assert data == json.loads(text)


def test_root_scalar():
    data, pos = parse('  "hello"  ')
    assert data == "hello"
    assert pos[()] == Pos(line=1, col=3, length=7)   # opening quote through closing


def test_object_member_positions():
    text = '{\n  "pattern": "reprise",\n  "why": ""\n}'
    _, pos = parse(text)
    # the string value "reprise" starts at its opening quote on line 2
    assert pos[("pattern",)] == Pos(line=2, col=14, length=9)
    # the empty "why" value: two-character token, so a two-caret underline
    assert pos[("why",)] == Pos(line=3, col=10, length=2)


def test_array_index_positions():
    text = '["setbreak",\n "reprise"]'
    _, pos = parse(text)
    assert pos[(0,)] == Pos(line=1, col=2, length=10)
    assert pos[(1,)] == Pos(line=2, col=2, length=9)


def test_nested_path():
    text = '{"non_song": [\n  {"pattern": "x", "why": "because"}\n]}'
    _, pos = parse(text)
    assert pos[("non_song", 0, "why")].line == 2
    # the container gets a position too, anchored at its opening bracket
    assert pos[("non_song",)].length == 1
    assert pos[("non_song", 0)].length == 1


def test_string_escapes_and_unicode():
    data, _ = parse(r'{"k": "a\"b\nü"}')
    assert data == {"k": 'a"b\nü'}


def test_numbers_all_forms():
    data, _ = parse('[0, -1, 2.5, 1e3, -3.2e-2, 42]')
    assert data == [0, -1, 2.5, 1e3, -3.2e-2, 42]


def test_line_col_after_newlines():
    text = '{\n  "a": 1,\n  "k": 2\n}'
    _, pos = parse(text)
    assert pos[("k",)].line == 3
    assert pos[("k",)].col == 8    # after '  "k": '


def test_trailing_garbage_is_an_error():
    with pytest.raises(JSONPosError):
        parse('{"a": 1} extra')


def test_unterminated_string_is_an_error():
    with pytest.raises(JSONPosError):
        parse('"no end')


def test_error_carries_a_position():
    try:
        parse('{\n  "a": nope\n}')
    except JSONPosError as err:
        assert err.line == 2
        assert err.col >= 1
    else:
        pytest.fail("expected JSONPosError")


def test_empty_input_is_an_error():
    with pytest.raises(JSONPosError):
        parse('   ')


def test_duplicate_key_keeps_last_like_stdlib():
    data, pos = parse('{"a": 1, "a": 2}')
    assert data == {"a": 2}
    # position map tracks the surviving (last) value
    assert pos[("a",)].col == 15


@pytest.mark.parametrize("bad", ["1.2.3", "--1", "1e", "01", "00", "1.", "1..2", "-"])
def test_malformed_numbers_are_rejected_like_stdlib(bad):
    """the number grammar is enforced, not left to float()/int() to quietly coerce."""
    with pytest.raises(JSONPosError):
        parse(bad)
    with pytest.raises(ValueError):
        json.loads(bad)   # stdlib rejects the same inputs


def test_raw_control_char_in_string_is_rejected():
    with pytest.raises(JSONPosError):
        parse('"a\tb"')          # a literal tab must be escaped as \t
    with pytest.raises(ValueError):
        json.loads('"a\tb"')


def test_u_pair_combines_like_stdlib():
    escaped = '"\\ud834\\udd1e"'   # G-clef written as a \u pair (a high half + a low half)
    data, _ = parse(escaped)
    assert data == json.loads(escaped)
    assert data == "\U0001d11e"


def test_lone_high_half_is_returned_like_stdlib():
    data, _ = parse(r'"\ud834 tail"')
    assert data == json.loads(r'"\ud834 tail"')

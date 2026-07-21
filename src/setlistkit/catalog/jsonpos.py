# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""A JSON parser that remembers where every value came from.

``json.load`` reports a line and column when the *syntax* is wrong, but throws that
information away for a valid parse. Schema validation is where we actually need it:
``jsonschema`` hands back a JSON Pointer like ``/non_song/12/why`` with no source position,
which leaves a user counting array elements by hand. So packs are parsed through this
instead, which returns the data plus a map from each value's path to its ``(line, col)``.
JSON is simple enough that this is a small scanner, not a dependency.

The path is a tuple of object keys and array indices, matching what ``jsonschema`` puts in
an error's ``absolute_path``: ``("non_song", 12, "why")``. Scalars record the full span of
their token (so ``""`` underlines as ``^^``); objects and arrays record just their opening
bracket, because a container-level error (a missing required key) wants a caret on the
brace, not a rule under the whole block.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

import jsonschema

from ..diagnostics import ERROR, Diagnostic, DiagnosticError

_WS = " \t\n\r"
_NUMBER_CHARS = "0123456789+-.eE"          # a number's body; _NUMBER_RE does the real validating
# The JSON number grammar, so "01", "1.", "1e", "--1" and "1.2.3" are rejected the way
# stdlib json rejects them rather than being quietly coerced by float()/int().
_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")
_HEX = "0123456789abcdefABCDEF"

# JSON string escapes, minus \u which is handled separately.
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
            "n": "\n", "r": "\r", "t": "\t"}


class Pos(NamedTuple):
    """Where a value sits in the source. All 1-based; ``length`` is the caret width."""

    line: int
    col: int
    length: int


class Positions(dict):
    """Value positions by path, with the object-key positions carried beside them.

    The mapping itself is path -> :class:`Pos` for VALUES, because that is what nearly every
    diagnostic is about: a schema error names a value, and :func:`position_for` walks up this
    mapping to fall back on a containing brace. ``key_positions`` holds the other half, for the
    handful of findings whose subject is the key itself.

    That distinction is not academic. An unreachable alias is a claim about the key -- nobody
    writes ``tambo`` -- and anchoring its caret on the value put "never used" under
    ``Tambourine``, a name tapers have written 581 times. A caret pointing one span too far
    right does not read as a misplaced caret; it reads as the tool asserting something false.

    A ``dict`` subclass rather than a third return value from :func:`parse`, so every existing
    ``positions.get(path)`` and ``path in positions`` keeps working untouched. The attribute is
    ``key_positions`` and not ``keys`` because ``dict.keys`` is already a method.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.key_positions: dict[tuple, Pos] = {}


def diagnostic_at(severity: str, summary: str, *, file: Path | None, source: str | None,
                  pos: Pos | None, detail: str | None = None,
                  caption: str = "") -> Diagnostic:
    """A :class:`~setlistkit.diagnostics.Diagnostic` anchored at ``pos`` in ``file``.

    One place rather than one per caller, and the reason is the degrading: a finding whose
    position could not be recovered still has to name its file and render without a caret, and
    one whose file is not known either still has to render at all. Those fallbacks are easy to
    get subtly wrong, and a copy that gets one wrong reports an error pointing at nothing --
    which is the failure this whole module exists to prevent.
    """
    return Diagnostic(
        severity=severity,
        summary=summary,
        path=str(file) if file is not None else None,
        line=pos.line if pos else None,
        col=pos.col if pos else None,
        length=pos.length if pos else 1,
        caret_caption=caption,
        detail=detail,
        source=source,
    )


def position_for(path: tuple, positions: Mapping[tuple, Pos]) -> Pos | None:
    """The recorded position for ``path``, walking up to the nearest recorded ancestor.

    A ``required``/``additionalProperties`` error names a key the scanner never recorded a
    value for, so we fall back to the containing object's brace rather than losing the caret.
    """
    while path and path not in positions:
        path = path[:-1]
    return positions.get(path) or positions.get(())


@dataclass(frozen=True)
class JSONSource:
    """A JSON file, its text, and where every value in it came from.

    The three travel together because they are only useful together: a position with no text
    renders no code frame, and text with no positions renders no caret. Passing them as three
    parameters is how one of them ends up missing at one call site and the error there quietly
    degrades to a filename, which reads like the position simply was not recoverable.

    Every field is optional, and an empty ``JSONSource()`` is a legitimate value meaning "this
    data did not come from a file I can point at" -- a caller holding a parsed dict rather than
    a path. That is the same code path, degraded, rather than a second one.
    """

    file: Path | None = None
    text: str | None = None
    positions: Mapping[tuple, Pos] = field(default_factory=dict)

    def fail(self, summary: str, *, detail: str | None = None, caption: str = "",
             at: tuple = ()) -> NoReturn:
        """Raise a :class:`DiagnosticError` anchored at the value ``at`` in this file."""
        raise DiagnosticError(self.diagnostic(summary, detail=detail, caption=caption, at=at))

    def diagnostic(self, summary: str, *, severity: str = ERROR, detail: str | None = None,
                   caption: str = "", at: tuple = ()) -> Diagnostic:
        """A diagnostic anchored at the value ``at`` in this file."""
        return diagnostic_at(severity, summary, file=self.file, source=self.text,
                             pos=position_for(at, self.positions), detail=detail,
                             caption=caption)


def load_json(file: Path, schema: dict) -> tuple[Any, JSONSource]:
    """Read, position-scan, and schema-validate one JSON file against ``schema``.

    Lives here rather than in the pack loader because it is not about packs: it is the three
    steps that turn a file on disk into data plus somewhere to point a caret, and every JSON
    file setlistkit reads wants all three. The alternative was a second copy of the
    schema-error-to-caret mapping for the override file, which is the shape this catalog was
    rebuilt to remove.

    Raises :class:`DiagnosticError` for unreadable JSON or a schema violation, in both cases
    with a caret on the offending value.
    """
    text = file.read_text(encoding="utf-8")
    try:
        data, positions = parse(text)
    except JSONPosError as err:
        # Strip the "(line N, column M)" tail: it is already in the location line and the caret.
        JSONSource(file, text, {(): Pos(err.line, err.col, 1)}).fail(
            str(err).split(" (line", maxsplit=1)[0], detail="the file is not valid JSON.")
    source = JSONSource(file, text, positions)
    error = _best_error(list(jsonschema.Draft202012Validator(schema).iter_errors(data)))
    if error is not None:
        source.fail(error.message, detail="the file does not match the schema.",
                    at=tuple(error.absolute_path))
    return data, source


def _best_error(errors: list):
    """The one schema error worth showing, preferring the one that names the actual mistake.

    ``best_match`` is a general heuristic and it gets the commonest case here backwards. A typo'd
    key raises TWO errors at the same path -- ``additionalProperties`` ("'resaon' was unexpected")
    and ``required`` ("'reason' is a required property") -- and ``best_match`` picks the second.
    That one describes the CONSEQUENCE: it tells an author their file is missing a reason while
    the reason sits three characters away, spelled wrong, which reads like the tool cannot see it.
    The unexpected-key error names the mistake itself, so it wins whenever there is one.
    """
    if not errors:
        return None
    unexpected = [err for err in errors if err.validator == "additionalProperties"]
    return unexpected[0] if unexpected else jsonschema.exceptions.best_match(errors)


class JSONPosError(ValueError):
    """A malformed-JSON error that knows its own line and column.

    ``pack.py`` turns this into a :class:`~setlistkit.diagnostics.Diagnostic` so a broken
    pack file reports like every other setlistkit error, caret and all.
    """

    def __init__(self, message: str, line: int, col: int) -> None:
        self.line = line
        self.col = col
        super().__init__(f"{message} (line {line}, column {col})")


def _line_col(text: str, index: int) -> tuple[int, int]:
    """1-based line and column of a character offset."""
    line = text.count("\n", 0, index) + 1
    col = index - text.rfind("\n", 0, index)   # rfind returns -1 before the first line
    return line, col


class _Scanner:
    """Recursive-descent over the text, recording a Pos per value as it goes."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.i = 0
        self.positions = Positions()

    def fail(self, message: str, index: int | None = None) -> NoReturn:
        """Raise a positioned error at ``index`` (default: the current offset)."""
        line, col = _line_col(self.text, self.i if index is None else index)
        raise JSONPosError(message, line, col)

    def skip_ws(self) -> None:
        """Advance past insignificant whitespace."""
        while self.i < len(self.text) and self.text[self.i] in _WS:
            self.i += 1

    def parse(self) -> tuple[Any, dict[tuple, Pos]]:
        """Parse the whole text as a single JSON document; reject trailing data."""
        self.skip_ws()
        if self.i >= len(self.text):
            self.fail("empty input")
        value = self.value(())
        self.skip_ws()
        if self.i != len(self.text):
            self.fail("trailing data after the top-level value")
        return value, self.positions

    def value(self, path: tuple) -> Any:
        """Parse one JSON value at ``path`` and record where it started."""
        char = self.text[self.i]
        start = self.i
        if char == "{":
            result = self.object(path)
            self._record(path, start, 1)          # caret on the '{', not the whole block
        elif char == "[":
            result = self.array(path)
            self._record(path, start, 1)
        elif char == '"':
            result = self.string()
            self._record(path, start, self.i - start)
        elif char == "-" or char.isdigit():
            result = self.number()
            self._record(path, start, self.i - start)
        elif self.text.startswith("true", self.i):
            result, self.i = True, self.i + 4
            self._record(path, start, 4)
        elif self.text.startswith("false", self.i):
            result, self.i = False, self.i + 5
            self._record(path, start, 5)
        elif self.text.startswith("null", self.i):
            result, self.i = None, self.i + 4
            self._record(path, start, 4)
        else:
            self.fail(f"unexpected character {char!r}")
        return result

    def _pos(self, start: int, length: int) -> Pos:
        line, col = _line_col(self.text, start)
        return Pos(line, col, length)

    def _record(self, path: tuple, start: int, length: int) -> None:
        self.positions[path] = self._pos(start, length)

    def _record_key(self, path: tuple, start: int, length: int) -> None:
        self.positions.key_positions[path] = self._pos(start, length)

    def object(self, path: tuple) -> dict:
        """Parse a ``{...}`` object, recording each member value under ``path + (key,)``."""
        obj: dict = {}
        self.i += 1                                # consume '{'
        self.skip_ws()
        if self._peek() == "}":
            self.i += 1
            return obj
        while True:
            self.skip_ws()
            if self._peek() != '"':
                self.fail("expected a string key")
            key_start = self.i
            key = self.string()
            # Recorded before the value is parsed, and overwritten by a repeated key exactly as
            # the value is, so the key and value positions always describe the same member.
            self._record_key(path + (key,), key_start, self.i - key_start)
            self.skip_ws()
            if self._peek() != ":":
                self.fail("expected ':' after object key")
            self.i += 1
            self.skip_ws()
            if self.i >= len(self.text):
                self.fail("expected a value after ':'")
            obj[key] = self.value(path + (key,))   # a repeated key overwrites, as stdlib does
            self.skip_ws()
            nxt = self._peek()
            if nxt == ",":
                self.i += 1
                continue
            if nxt == "}":
                self.i += 1
                return obj
            self.fail("expected ',' or '}' in object")

    def array(self, path: tuple) -> list:
        """Parse a ``[...]`` array, recording each element under ``path + (index,)``."""
        arr: list = []
        self.i += 1                                # consume '['
        self.skip_ws()
        if self._peek() == "]":
            self.i += 1
            return arr
        index = 0
        while True:
            self.skip_ws()
            if self.i >= len(self.text):
                self.fail("expected a value in array")
            arr.append(self.value(path + (index,)))
            index += 1
            self.skip_ws()
            nxt = self._peek()
            if nxt == ",":
                self.i += 1
                continue
            if nxt == "]":
                self.i += 1
                return arr
            self.fail("expected ',' or ']' in array")

    def string(self) -> str:
        """Parse a ``"..."`` string, decoding escapes; the caller records its position."""
        self.i += 1                                # consume opening quote
        out: list[str] = []
        while True:
            if self.i >= len(self.text):
                self.fail("unterminated string")
            char = self.text[self.i]
            if char == '"':
                self.i += 1
                return "".join(out)
            if char == "\\":
                out.append(self._escape())
                continue
            if ord(char) < 0x20:
                self.fail("control character in string")   # tabs/newlines must be escaped
            out.append(char)
            self.i += 1

    def _escape(self) -> str:
        self.i += 1                                # consume the backslash
        if self.i >= len(self.text):
            self.fail("unterminated escape")
        code = self.text[self.i]
        if code == "u":
            return self._unicode_escape()
        if code not in _ESCAPES:
            self.fail(f"invalid escape \\{code}")
        self.i += 1
        return _ESCAPES[code]

    def _unicode_escape(self) -> str:
        """Decode a ``\\uXXXX`` escape, joining a ``\\u`` pair into one astral char.

        A codepoint above U+FFFF is written as two ``\\u`` escapes: a high half in
        0xD800-0xDBFF followed by a low half in 0xDC00-0xDFFF. Join them when the pair is
        there; a lone high half comes back as-is, matching stdlib json.
        """
        value = self._hex4()
        if 0xD800 <= value <= 0xDBFF and self.text[self.i:self.i + 2] == "\\u":
            self.i += 1                            # step onto the second escape's 'u' for _hex4
            low_half = self._hex4()
            if 0xDC00 <= low_half <= 0xDFFF:
                return chr(0x10000 + ((value - 0xD800) << 10) + (low_half - 0xDC00))
            self.fail("invalid low half in the \\u pair")
        return chr(value)

    def _hex4(self) -> int:
        """Consume exactly four hex digits after a ``\\u`` and return their value."""
        hexits = self.text[self.i + 1:self.i + 5]
        if len(hexits) != 4 or any(h not in _HEX for h in hexits):
            self.fail("invalid \\u escape")
        self.i += 5
        return int(hexits, 16)

    def number(self) -> int | float:
        """Parse a numeric token; a ``.``, ``e`` or ``E`` makes it a float, else an int."""
        start = self.i
        while self.i < len(self.text) and self.text[self.i] in _NUMBER_CHARS:
            self.i += 1
        token = self.text[start:self.i]
        if not _NUMBER_RE.match(token):
            self.fail(f"invalid number {token!r}", start)
        return float(token) if any(c in token for c in ".eE") else int(token)

    def _peek(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""


def parse(text: str) -> tuple[Any, Positions]:
    """Parse ``text`` as JSON, returning ``(data, positions)``.

    ``positions`` maps each value's path tuple to its :class:`Pos`, and carries the matching
    object-key spans on :attr:`Positions.key_positions`. Raises :class:`JSONPosError` (carrying
    a line and column) on malformed input.
    """
    return _Scanner(text).parse()

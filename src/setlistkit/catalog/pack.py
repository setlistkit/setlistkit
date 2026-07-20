# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Load a band pack off disk into a data-driven :class:`Normalizer`.

A pack is a directory of JSON governed by a JSON Schema. This module reads it, validates
the shape (jsonschema), enforces the domain rules a schema cannot state, compiles the
non-song patterns, and hands back a :class:`Pack` whose ``.normalizer`` has the policy hooks
populated from the files. Every failure is a :class:`~setlistkit.diagnostics.Diagnostic`
with a caret, because "your pack is broken" is worth saying precisely.

Three layers do the validating, in order:

1. **Shape** — jsonschema. Types, required fields, the "bare string OR {pattern, why}" union
   for non-song rules, ``additionalProperties: false`` to catch a typo'd key. jsonschema
   reports a JSON Pointer with no source position, so :mod:`~setlistkit.catalog.jsonpos`
   maps the pointer to a line and column.
2. **Domain** — hand-rolled, because "a free-floating pattern must justify itself" is about
   what the pattern *means*, not its type. A bare substring that can reach into a title (no
   ``^``/``$`` anchor) must ship as an object carrying a ``why``; an object whose ``why`` is
   blank is the same failure from the other side. This is the spec's headline diagnostic.
3. **Compile** — the patterns become real ``re.Pattern`` objects here, so a bad regex is
   caught at load with a caret pointing *inside* the pattern string at the offending char,
   not at runtime three phases later.

Corpus-aware checks (a rule that matches nothing, an alias target missing from the
vocabulary) live in ``slkit pack lint``, not here: they need the cached corpus, and a pack
is structurally valid without them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import jsonschema

from ..diagnostics import ERROR, Diagnostic, DiagnosticError
from .jsonpos import JSONPosError, Pos, parse
from .normalizer import Normalizer, normalize

PACK_FILE = "pack.json"
VOCABULARY_FILE = "vocabulary.json"
ALIASES_FILE = "aliases.json"
CLASSIFIERS_FILE = "classifiers.json"
PROTECTED_FILE = "protected.json"

# --- schemas (the "shape" layer) ----------------------------------------------------------
# Deliberately loose where a domain check does the work: the non-song object requires a `why`
# to be *present* but not non-empty, because an empty `why` earns a better message and a
# caret on the value than jsonschema's generic "does not match" ever could.

_PACK_SCHEMA = {
    "type": "object",
    "required": ["name", "version"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "version": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
}

_VOCABULARY_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
    "uniqueItems": True,
}

_ALIASES_SCHEMA = {
    "type": "object",
    "propertyNames": {"minLength": 1},
    "additionalProperties": {"type": "string", "minLength": 1},
}

_CLASSIFIERS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "non_song": {
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string", "minLength": 1},
                    {
                        "type": "object",
                        "required": ["pattern", "why"],
                        "additionalProperties": False,
                        "properties": {
                            "pattern": {"type": "string", "minLength": 1},
                            "why": {"type": "string"},
                            "must_not_match": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                ]
            },
        },
    },
}

_PROTECTED_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
    "uniqueItems": True,
}


@dataclass(frozen=True)
class Rule:
    """One compiled non-song classifier plus the metadata lint needs.

    ``why`` is ``None`` for a bare-string (anchored) rule and a non-empty string for an
    object rule. ``anchored`` is what decides whether a bare string was allowed to skip its
    justification. ``pos`` locates the pattern in the source for later diagnostics.
    """

    pattern: str
    compiled: re.Pattern
    anchored: bool
    why: str | None = None
    must_not_match: tuple[str, ...] = ()
    pos: Pos | None = None


@dataclass(frozen=True)
class Pack:
    """A loaded, validated band pack and the normalizer it builds."""

    name: str
    version: str
    sources: tuple[str, ...]
    vocabulary: tuple[str, ...]
    aliases: dict[str, str]
    protected: tuple[str, ...]
    rules: tuple[Rule, ...]
    normalizer: Normalizer = field(compare=False)


class _PackNormalizer(Normalizer):
    """A :class:`Normalizer` whose policy hooks are filled from pack data."""

    def __init__(self, vocabulary, aliases, patterns, protected) -> None:
        super().__init__(vocabulary)
        self._pack_aliases = aliases
        self._pack_patterns = patterns
        self._pack_protected = protected

    def aliases(self) -> dict[str, str]:
        return self._pack_aliases

    def non_song_patterns(self) -> list[re.Pattern]:
        return self._pack_patterns

    def protected_titles(self) -> set[str]:
        return self._pack_protected


def _anchored(pattern: str) -> bool:
    """A pattern is safely narrow only if it pins an end and hides no free-floating branch.

    An anchored rule (``^foo``, ``foo$``, ``^foo$``) cannot reach into the middle of a title,
    so it needs no justification; a free-floating substring can, so it must carry a ``why``.
    A ``|`` alternation can smuggle a free-floating branch past a leading anchor -- ``^intro|jam``
    starts with ``^`` yet its ``jam`` branch matches "Pajamas" -- so any alternation is treated
    as not-safely-anchored and has to justify itself. Well-formed packs ship each alternative as
    its own entry anyway, so this only bites the pattern that was trying to cheat.
    """
    if "|" in pattern:
        return False
    return pattern.startswith("^") or pattern.endswith("$")


def _position_for(path: tuple, positions: dict[tuple, Pos]) -> Pos | None:
    """The recorded position for ``path``, walking up to the nearest recorded ancestor.

    A ``required``/``additionalProperties`` error names a key the scanner never recorded a
    value for, so we fall back to the containing object's brace rather than losing the caret.
    """
    while path and path not in positions:
        path = path[:-1]
    return positions.get(path) or positions.get(())


def _fail(file: Path, summary: str, source: str, pos: Pos | None,
          *, detail: str | None = None, caption: str = "") -> NoReturn:
    """Raise a DiagnosticError anchored at ``pos`` (or path-only when position is unknown)."""
    diag = Diagnostic(
        severity=ERROR,
        summary=summary,
        path=str(file),
        line=pos.line if pos else None,
        col=pos.col if pos else None,
        length=pos.length if pos else 1,
        caret_caption=caption,
        detail=detail,
        source=source,
    )
    raise DiagnosticError(diag)


def _load_json(file: Path, schema: dict):
    """Read, position-scan, and schema-validate one pack file. Returns (data, positions, source)."""
    source = file.read_text(encoding="utf-8")
    try:
        data, positions = parse(source)
    except JSONPosError as err:
        _fail(file, str(err).split(" (line", maxsplit=1)[0], source, Pos(err.line, err.col, 1),
              detail="the file is not valid JSON.")
    validator = jsonschema.Draft202012Validator(schema)
    error = jsonschema.exceptions.best_match(validator.iter_errors(data))
    if error is not None:
        pos = _position_for(tuple(error.absolute_path), positions)
        _fail(file, error.message, source, pos, detail="the pack does not match the schema.")
    return data, positions, source


def _compile_rules(entries: list, positions: dict[tuple, Pos], file: Path, source: str) -> tuple:
    """Turn schema-valid ``non_song`` entries into compiled Rules, enforcing the domain rule.

    Two ways a pattern fails to justify itself, one message: a free-floating bare string that
    should have been an object, and an object whose ``why`` is blank.
    """
    rules: list[Rule] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            pattern, why, must_not = entry, None, ()
            pattern_pos = positions.get(("non_song", index))
            if not _anchored(pattern):
                _fail(file, "free-floating pattern must justify itself", source, pattern_pos,
                      detail=_justify_detail(pattern), caption="no reason given")
        else:
            pattern = entry["pattern"]
            why = entry["why"]
            must_not = tuple(entry.get("must_not_match", ()))
            pattern_pos = positions.get(("non_song", index, "pattern"))
            if not why.strip():
                _fail(file, "free-floating pattern must justify itself", source,
                      positions.get(("non_song", index, "why")),
                      detail=_justify_detail(pattern), caption="empty")
        rules.append(Rule(
            pattern=pattern,
            compiled=_compile(pattern, pattern_pos, file, source),
            anchored=_anchored(pattern),
            why=why,
            must_not_match=must_not,
            pos=pattern_pos,
        ))
    return tuple(rules)


def _justify_detail(pattern: str) -> str:
    return (f'"{pattern}" matches anywhere in a title, so it can delete a real song.\n'
            f"Either anchor it (^{pattern}$) or give a reason and a counter-example.")


def _compile(pattern: str, pattern_pos: Pos | None, file: Path, source: str) -> re.Pattern:
    """Compile a pattern, pointing a caret *inside* the string at a bad char on failure."""
    try:
        return re.compile(pattern)
    except re.error as err:
        pos = None
        if pattern_pos is not None:
            offset = err.pos if err.pos is not None else 0
            pos = Pos(pattern_pos.line, _decoded_col(source, pattern_pos, offset), 1)
        _fail(file, f"invalid regex: {err.msg}", source, pos,
              detail="fix the pattern so it compiles.")


def _decoded_col(source: str, pattern_pos: Pos, decoded_offset: int) -> int:
    """Source column (1-based) of the decoded-string char at ``decoded_offset``.

    ``re.error.pos`` indexes the *decoded* pattern, but a caret has to land in the *source*,
    and a JSON escape (``\\d`` is two source chars for one decoded char, ``\\uXXXX`` is six)
    makes the two diverge. So walk the raw source token from just past its opening quote,
    counting one decoded char per literal or escape, until we reach the engine's offset.
    """
    line = source.splitlines()[pattern_pos.line - 1]
    index = pattern_pos.col              # 0-based index of the first content char (past the quote)
    decoded = 0
    while decoded < decoded_offset and index < len(line):
        if line[index] == "\\":
            index += 6 if line[index + 1:index + 2] == "u" else 2
        else:
            index += 1
        decoded += 1
    return index + 1                     # back to a 1-based column


def load_pack(pack_dir) -> Pack:
    """Load and validate the pack in ``pack_dir``, returning a :class:`Pack`.

    Requires ``pack.json`` and ``vocabulary.json``; ``aliases.json``, ``classifiers.json``
    and ``protected.json`` are optional and default empty, so a minimal pack (a dictionary
    and nothing else) is legal. Raises :class:`DiagnosticError` on any malformed or invalid
    file, and a plain diagnostic when a required file is missing.
    """
    pack_dir = Path(pack_dir)
    identity, _, _ = _load_json(_require(pack_dir, PACK_FILE), _PACK_SCHEMA)
    vocabulary, _, _ = _load_json(_require(pack_dir, VOCABULARY_FILE), _VOCABULARY_SCHEMA)

    aliases = {}
    if (pack_dir / ALIASES_FILE).exists():
        aliases, _, _ = _load_json(pack_dir / ALIASES_FILE, _ALIASES_SCHEMA)

    protected: list = []
    if (pack_dir / PROTECTED_FILE).exists():
        protected, _, _ = _load_json(pack_dir / PROTECTED_FILE, _PROTECTED_SCHEMA)

    rules: tuple = ()
    classifiers_path = pack_dir / CLASSIFIERS_FILE
    if classifiers_path.exists():
        data, positions, source = _load_json(classifiers_path, _CLASSIFIERS_SCHEMA)
        rules = _compile_rules(data.get("non_song", []), positions, classifiers_path, source)

    normalizer = _PackNormalizer(
        vocabulary=list(vocabulary),
        # aliases() is contracted as normalized-key -> canonical, and canonicalize looks up
        # normalize(raw). A pack author writes a readable key ("The Rec Chem"); normalizing it
        # here is what makes that key actually match instead of silently never firing.
        aliases={normalize(key): value for key, value in aliases.items()},
        patterns=[rule.compiled for rule in rules],
        protected=set(protected),
    )
    return Pack(
        name=identity["name"],
        version=identity["version"],
        sources=tuple(identity.get("sources", ())),
        vocabulary=tuple(vocabulary),
        aliases=dict(aliases),
        protected=tuple(protected),
        rules=rules,
        normalizer=normalizer,
    )


def _require(pack_dir: Path, name: str) -> Path:
    """Return ``pack_dir/name`` or raise a diagnostic naming the missing required file."""
    path = pack_dir / name
    if not path.exists():
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary=f"pack is missing a required file: {name}",
            path=str(path),
            detail=f"every pack needs {PACK_FILE} and {VOCABULARY_FILE}.",
        ))
    return path

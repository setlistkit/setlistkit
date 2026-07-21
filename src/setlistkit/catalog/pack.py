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

``corpus.json`` -- what this band's tapes are known to get wrong, plus the filter residue only
its tapers write -- runs the same three layers with one extra domain rule of its own: every
entry states why it is there. A drop date with no reason is indistinguishable from a typo, and
the reason is the only thing a later reader can check the call against.

Corpus-aware checks (a rule that matches nothing, an alias target missing from the
vocabulary) live in ``slkit pack lint``, not here: they need the cached corpus, and a pack
is structurally valid without them.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import jsonschema

from ..diagnostics import ERROR, Diagnostic, DiagnosticError
from .jsonpos import JSONPosError, Pos, diagnostic_at, parse
from .merge import COMPLETE_FRAC, DEFAULT_RANKS, MergePolicy
from .normalizer import Normalizer, normalize
from .parse import ArchivePolicy, fragment_pattern, title_band_filter

PACK_FILE = "pack.json"
VOCABULARY_FILE = "vocabulary.json"
ALIASES_FILE = "aliases.json"
CLASSIFIERS_FILE = "classifiers.json"
PROTECTED_FILE = "protected.json"
CORPUS_FILE = "corpus.json"

# A date the rest of the toolkit will accept. Enforced in the schema because a typo'd date is
# the silent kind of wrong: "2025-13-01" never equals any show's date, so the drop or the
# correction it was written for simply never happens and nothing says so.
#
# The month and day are range-bounded rather than \d{2}, because the whole point is to reject
# the example in the sentence above and `^\d{4}-\d{2}-\d{2}$` accepts it. And the tail anchor is
# \Z, not $: jsonschema matches with Python's re, where `$` also matches just before a trailing
# newline, so "2025-06-14\n" was a schema-valid date. That one does not merely fail to fire --
# _show_date accepts it too, so an override carrying it puts a date with a newline in it into
# the corpus, where it joins to nothing for the rest of time.
#
# Day 31 is allowed in every month. Catching 2025-02-31 wants a calendar, not a regex, and the
# failure it would prevent is the loud kind: a date that matches no show gets noticed.
_DATE = r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\Z"

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
        # Not the same string as `name`, and the difference is load-bearing. `name` identifies
        # the pack ("moe"); `band_name` is what the band calls itself ("moe."), which is what
        # a taper types and what archive.org puts in an item title. Optional: a pack that never
        # meets a rival band's tape does not need it.
        "band_name": {"type": "string", "minLength": 1},
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

# A corpus filter fragment. No bare-string form, unlike a classifier: these are folded into an
# alternation and bounded by lookarounds, so EVERY one of them is free-floating by construction
# and there is no anchored shape that could earn its way out of explaining itself.
_FRAGMENTS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["pattern", "why"],
        "additionalProperties": False,
        "properties": {
            "pattern": {"type": "string", "minLength": 1},
            "why": {"type": "string"},
            "must_not_match": {"type": "array", "items": {"type": "string"}},
        },
    },
}

_CORPUS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "drop_dates": {
            "type": "object",
            "propertyNames": {"pattern": _DATE},
            "additionalProperties": {"type": "string"},
        },
        "date_overrides": {
            "type": "object",
            "propertyNames": {"minLength": 1},
            "additionalProperties": {
                "type": "object",
                "required": ["date", "why"],
                "additionalProperties": False,
                "properties": {
                    "date": {"type": "string", "pattern": _DATE},
                    "why": {"type": "string"},
                },
            },
        },
        "junk_patterns": _FRAGMENTS_SCHEMA,
        "gear_patterns": _FRAGMENTS_SCHEMA,
    },
}


def _matches(pattern: re.Pattern[str], text: str) -> bool:
    """``pattern.search(text)``, as a bool.

    A free function and not an inline call, for one dull reason: pylint cannot resolve
    ``.search`` on a ``re.Pattern`` reached through ``self``, though it resolves the same call
    on a parameter perfectly well. The alternative is a suppression, and the rule here is fix
    in code rather than turn the check off.
    """
    return pattern.search(text) is not None


@dataclass(frozen=True)
class Rule:
    """One compiled non-song classifier plus the metadata lint needs.

    ``why`` is ``None`` for a bare-string (anchored) rule and a non-empty string for an
    object rule. ``anchored`` is what decides whether a bare string was allowed to skip its
    justification. ``pos`` locates the pattern in the source for later diagnostics.

    ``wrapped`` says which of two things ``compiled`` is, and it exists because getting that
    wrong fails SILENTLY in both directions. A classifier rule compiles bare and is matched
    against ``squash(title)``; a corpus fragment compiles inside the alternation bounds the
    filters apply and is matched against the title as written. Feed a corpus rule down the
    classifier path and its ``\\s+`` can never match, because squash has already removed the
    space -- no error, just a check that silently never fires. Feed a classifier rule down the
    corpus path and it gets bounded twice. Both produce a clean run and a wrong answer, so the
    pairing is recorded rather than remembered.

    ``anchored`` is meaningful only when ``wrapped`` is False. A corpus fragment is folded into
    an alternation regardless, and the schema demands its ``why`` regardless, so nothing about
    it turns on whether the author happened to write a ``^``.
    """

    pattern: str
    compiled: re.Pattern[str]
    anchored: bool
    why: str | None = None
    must_not_match: tuple[str, ...] = ()
    pos: Pos | None = None
    wrapped: bool = False

    def reaches(self, title: str, squash: Callable[[str], str]) -> bool:
        """Would this rule match ``title``, in the form the runtime actually hands it?

        The two forms are not interchangeable and the rule is the only thing that knows which
        one it is, so it decides here rather than at each call site. A caller that got to pick
        would be picking silently: both wrong pairings run clean and simply never fire.
        """
        return _matches(self.compiled, title if self.wrapped else squash(title))


@dataclass(frozen=True)
class CorpusPolicy:
    """What this band's corpus is known to get wrong, and the residue only its tapers write.

    Everything here is a correction or an exclusion, and every entry states why it earned its
    place. None of it is derivable: an uploader can type any date they like, and a well-formed
    lie is indistinguishable from the truth to every parser downstream. The reasons are not
    decoration -- they are the only thing that lets a later reader check the call.

    ``junk`` and ``gear`` are compiled the way the filters apply them, so what lint holds
    against a vocabulary is what the parser will actually run.
    """

    # hash=False, matching ArchivePolicy and MergePolicy: a dict is unhashable and frozen=True
    # synthesises __hash__ from every compared field. Equality still reads them.
    drop_dates: Mapping[str, str] = field(default_factory=dict, hash=False)
    date_overrides: Mapping[str, Mapping[str, str]] = field(default_factory=dict, hash=False)
    junk: tuple[Rule, ...] = ()
    gear: tuple[Rule, ...] = ()


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
    corpus: CorpusPolicy
    band_name: str | None
    normalizer: Normalizer = field(compare=False)

    def merge_policy(self, ranks: Mapping[str, int] | None = None,
                     complete_frac: float = COMPLETE_FRAC) -> MergePolicy:
        """The merge policy this pack implies.

        Exists because ``drop_dates`` is the one corpus fact TWO layers consume, and the merge
        needs it for a reason of its own: refusing a date in the archive parser was not enough,
        because the other sources carry the same night and the merge quietly picked one of those
        copies up instead. The show came back. A caller who builds :class:`ArchivePolicy` from
        the pack and then hand-writes a :class:`MergePolicy` has recreated exactly that bug, so
        both come from here.

        ``ranks`` and ``complete_frac`` are config, not pack data, and are passed through.
        """
        return MergePolicy(
            ranks=dict(ranks) if ranks is not None else dict(DEFAULT_RANKS),
            complete_frac=complete_frac,
            drop_dates=frozenset(self.corpus.drop_dates),
        )

    def archive_policy(self) -> ArchivePolicy:
        """The archive parser's policy, assembled from this pack.

        The one place these six inputs get put together. Assembling them at each call site is
        how a toolkit ends up with four answers to one question -- which is the failure this
        catalog was rebuilt to remove, so it is not a failure worth reintroducing for the sake
        of a constructor call. See :meth:`merge_policy` for the drop-dates half.
        """
        return ArchivePolicy(
            drop_dates=frozenset(self.corpus.drop_dates),
            date_overrides={ident: entry["date"]
                            for ident, entry in self.corpus.date_overrides.items()},
            # No band name means no band filter. Refusing to guess is the whole point: an
            # unreadable title is not evidence that a show is fake, and neither is a silent one.
            band_filter=title_band_filter(self.band_name) if self.band_name else None,
            junk_patterns=tuple(rule.pattern for rule in self.corpus.junk),
            gear_patterns=tuple(rule.pattern for rule in self.corpus.gear),
            band_name=self.band_name,
        )


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
    raise DiagnosticError(diagnostic_at(ERROR, summary, file=file, source=source, pos=pos,
                                        detail=detail, caption=caption))


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
        rules.append(_rule(pattern, why, must_not, pattern_pos, file=file, source=source))
    return tuple(rules)


def _rule(pattern: str, why: str | None, must_not: tuple[str, ...], pattern_pos: Pos | None,
          *, file: Path, source: str, wrap: bool = False) -> Rule:
    """One validated :class:`Rule`. ``wrap`` compiles it the way the corpus filters apply it.

    The raw pattern is compiled either way, and first, even when the wrapped form is what gets
    kept: ``re.error.pos`` indexes the pattern the engine was handed, so a caret computed from a
    *wrapped* pattern lands several characters to the right of the character that is wrong.
    """
    compiled = _compile(pattern, pattern_pos, file, source)
    return Rule(
        pattern=pattern,
        compiled=fragment_pattern(pattern) if wrap else compiled,
        anchored=_anchored(pattern),
        why=why,
        must_not_match=must_not,
        pos=pattern_pos,
        wrapped=wrap,
    )


def _compile_fragments(entries: list, positions: dict[tuple, Pos], file: Path, source: str,
                       *, key: str) -> tuple[Rule, ...]:
    """Turn one corpus pattern list into compiled Rules, each carrying its stated reason.

    No bare-string form to allow for: the schema requires an object, because a fragment is
    free-floating by construction and cannot anchor its way out of explaining itself.
    """
    rules: list[Rule] = []
    for index, entry in enumerate(entries):
        if not entry["why"].strip():
            _fail(file, f"{key} entry must say what it is", source,
                  positions.get((key, index, "why")),
                  detail=(f'"{entry["pattern"]}" is folded into a filter that DROPS every entry\n'
                          "it matches, without recording that anything was there. Say what it is,\n"
                          "so a later reader can tell junk from a song nobody had written yet."),
                  caption="empty")
        rules.append(_rule(entry["pattern"], entry["why"],
                           tuple(entry.get("must_not_match", ())),
                           positions.get((key, index, "pattern")),
                           file=file, source=source, wrap=True))
    return tuple(rules)


def _load_corpus(pack_dir: Path) -> CorpusPolicy:
    """Load the optional ``corpus.json``, enforcing that every correction states its evidence."""
    file = pack_dir / CORPUS_FILE
    if not file.exists():
        return CorpusPolicy()
    data, positions, source = _load_json(file, _CORPUS_SCHEMA)

    drops = data.get("drop_dates", {})
    for date, reason in drops.items():
        if not reason.strip():
            _fail(file, f"drop_dates {date!r} does not say why", source,
                  positions.get(("drop_dates", date)),
                  detail="Dropping a date throws away every song played that night, including\n"
                         "the ordinary ones. Say what makes the show no evidence about the band.",
                  caption="empty")

    overrides = data.get("date_overrides", {})
    for identifier, entry in overrides.items():
        if not entry["why"].strip():
            _fail(file, f"date_overrides {identifier!r} does not say why", source,
                  positions.get(("date_overrides", identifier, "why")),
                  detail="An override moves a show to a different night on your say-so, and a\n"
                         "wrong one invents a show that never happened while burying a real one.\n"
                         "State the evidence: artwork, upload date, the taper's own description.",
                  caption="empty")

    return CorpusPolicy(
        drop_dates=dict(drops),
        date_overrides={ident: dict(entry) for ident, entry in overrides.items()},
        junk=_compile_fragments(data.get("junk_patterns", []), positions, file, source,
                                key="junk_patterns"),
        gear=_compile_fragments(data.get("gear_patterns", []), positions, file, source,
                                key="gear_patterns"),
    )


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

    Requires ``pack.json`` and ``vocabulary.json``; ``aliases.json``, ``classifiers.json``,
    ``protected.json`` and ``corpus.json`` are optional and default empty, so a minimal pack
    (a dictionary and nothing else) is legal. Raises :class:`DiagnosticError` on any malformed
    or invalid file, and a plain diagnostic when a required file is missing.
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
        corpus=_load_corpus(pack_dir),
        band_name=identity.get("band_name"),
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

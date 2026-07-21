# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Conformance checks for a loaded pack: the ones that map to bugs the old repo hit.

Loading a pack (see :mod:`~setlistkit.catalog.pack`) proves it is well-*formed* -- valid
JSON, valid schema, every free-floating rule justified, every regex compiles. This module
proves it is well-*behaved*: that no rule silently deletes a real song and no alias silently
drops one. These are the checks the spec makes core, not optional, because the conformance
suite is what stops an adopter from losing songs they didn't know they had.

Three checks run here, each a hard error:

- **A rule matches a protected title.** A protected title is always a song. A classifier that
  matches one would delete it if the runtime guard ever came off, so the rule is wrong: narrow
  it. This is the ``ATL``/``NYC`` deletion, caught before it happens.
- **A rule matches its own ``must_not_match``.** The rule contradicts the counter-example its
  author wrote to pin it down. One of the two is wrong.
- **An alias target is absent from the vocabulary.** The vocabulary is the dictionary; an alias
  pointing at a name that isn't in it is the "Hi & Lo" silent-loss bug, where a real song's
  plays scatter across a name nothing can ever join to.

The corpus-aware checks the spec also lists (a rule that matches nothing, a rule subsumed by
another, a canonicalization that reaches nothing) need the cached corpus, which does not exist
until ingest. Those are reported as a skipped note rather than silently omitted.
"""

from __future__ import annotations

from pathlib import Path

from ..diagnostics import ERROR, NOTE, Diagnostic
from .jsonpos import Pos, parse
from .pack import ALIASES_FILE, CLASSIFIERS_FILE, Pack, Rule, load_pack


def lint(pack_dir) -> list[Diagnostic]:
    """Load the pack in ``pack_dir`` and return every conformance finding.

    Loading enforces structure and raises :class:`~setlistkit.diagnostics.DiagnosticError` on
    a malformed pack; that is the caller's to catch. Once loaded, the behavioural checks here
    never raise -- they accumulate diagnostics so a CI run sees all of them at once.
    """
    pack_dir = Path(pack_dir)
    pack = load_pack(pack_dir)

    diagnostics: list[Diagnostic] = []
    diagnostics.extend(_rule_findings(pack, pack_dir))
    diagnostics.extend(_alias_findings(pack, pack_dir))
    diagnostics.append(Diagnostic(
        severity=NOTE,
        summary="corpus-aware checks skipped (no cached corpus yet)",
        detail="Checks for dead rules and unreachable canonicalizations need the cached\n"
               "corpus, which exists once ingest runs. Re-lint after that to catch them.",
    ))
    return diagnostics


def _rule_findings(pack: Pack, pack_dir: Path) -> list[Diagnostic]:
    """Every classifier checked against the protected titles and its own must_not_match."""
    source = _read_optional(pack_dir / CLASSIFIERS_FILE)
    squash = pack.normalizer.squash
    out: list[Diagnostic] = []
    for rule in pack.rules:
        for title in pack.protected:
            if rule.compiled.search(squash(title)):
                out.append(_rule_diag(
                    pack_dir, rule, source,
                    summary=f"rule /{rule.pattern}/ matches protected title {title!r}",
                    caption=f"deletes {title!r}",
                    detail=f"{title!r} is a protected title, always a song. Narrow this rule\n"
                           "so it cannot reach it -- anchor it, or make the pattern more specific.",
                ))
        for example in rule.must_not_match:
            if rule.compiled.search(squash(example)):
                out.append(_rule_diag(
                    pack_dir, rule, source,
                    summary=f"rule /{rule.pattern}/ matches its own must_not_match {example!r}",
                    caption=f"hits {example!r}",
                    detail=f"The rule was given {example!r} as a song it must NOT match, and it\n"
                           "matches it anyway. Fix the pattern or drop the counter-example.",
                ))
    return out


def _alias_findings(pack: Pack, pack_dir: Path) -> list[Diagnostic]:
    """Every alias whose target is not a vocabulary entry (the Hi & Lo silent-loss bug)."""
    if not pack.aliases:
        return []
    vocabulary = set(pack.vocabulary)
    source = _read_optional(pack_dir / ALIASES_FILE)
    positions: dict[tuple, Pos] = {}
    if source:
        _, positions = parse(source)
    out: list[Diagnostic] = []
    for key, target in pack.aliases.items():
        if target not in vocabulary:
            pos = positions.get((key,))
            out.append(Diagnostic(
                severity=ERROR,
                summary=f"alias target {target!r} is absent from the vocabulary",
                path=str(pack_dir / ALIASES_FILE),
                line=pos.line if pos else None,
                col=pos.col if pos else None,
                length=pos.length if pos else 1,
                caret_caption="not in vocabulary",
                source=source,
                detail=f"The alias {key!r} -> {target!r} points at a name the vocabulary does\n"
                       "not contain. Plays for the song scatter across a name nothing joins to.\n"
                       "Add the target to vocabulary.json.",
            ))
    return out


def _rule_diag(pack_dir: Path, rule: Rule, source: str | None, *,
               summary: str, caption: str, detail: str) -> Diagnostic:
    """A diagnostic anchored at a classifier rule's pattern in classifiers.json."""
    pos = rule.pos
    return Diagnostic(
        severity=ERROR,
        summary=summary,
        path=str(pack_dir / CLASSIFIERS_FILE),
        line=pos.line if pos else None,
        col=pos.col if pos else None,
        length=pos.length if pos else 1,
        caret_caption=caption,
        source=source,
        detail=detail,
    )


def _read_optional(path: Path) -> str | None:
    """The file text, or ``None`` if it isn't there (an optional pack file)."""
    return path.read_text(encoding="utf-8") if path.is_file() else None

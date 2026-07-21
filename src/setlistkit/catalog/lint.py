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

One warning, which is deliberately not an error:

- **A corpus fragment reaches a title the band actually plays.** Held against the vocabulary,
  the aliases and the protected titles together, since all three are the pack declaring "this
  is a song". It cannot delete the title -- ``parse._claimed`` gates every rule that drops --
  so this is not fatal. It is reported because a fragment wide enough to reach a real title is
  almost always wider than its author intended, and the guard only protects the titles that are
  in the pack *today*.

The corpus-aware checks the spec also lists (a rule that matches nothing, a rule subsumed by
another, a canonicalization that reaches nothing) need the cached corpus, which does not exist
until ingest. Those are reported as a skipped note rather than silently omitted.
"""

from __future__ import annotations

from pathlib import Path

from ..diagnostics import ERROR, NOTE, WARNING, Diagnostic
from .jsonpos import Pos, diagnostic_at, parse
from .pack import ALIASES_FILE, CLASSIFIERS_FILE, CORPUS_FILE, Pack, Rule, load_pack


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
    diagnostics.extend(_corpus_findings(pack, pack_dir))
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
            if rule.reaches(title, squash):
                out.append(_rule_diag(
                    pack_dir / CLASSIFIERS_FILE, rule, source,
                    summary=f"rule /{rule.pattern}/ matches protected title {title!r}",
                    caption=f"deletes {title!r}",
                    detail=f"{title!r} is a protected title, always a song. Narrow this rule\n"
                           "so it cannot reach it -- anchor it, or make the pattern more specific.",
                ))
        for example in rule.must_not_match:
            if rule.reaches(example, squash):
                out.append(_rule_diag(
                    pack_dir / CLASSIFIERS_FILE, rule, source,
                    summary=f"rule /{rule.pattern}/ matches its own must_not_match {example!r}",
                    caption=f"hits {example!r}",
                    detail=f"The rule was given {example!r} as a song it must NOT match, and it\n"
                           "matches it anyway. Fix the pattern or drop the counter-example.",
                ))
    return out


def _corpus_findings(pack: Pack, pack_dir: Path) -> list[Diagnostic]:
    """Every corpus filter fragment held against the titles it should not be reaching.

    The rules are compiled the way the filters apply them -- grouped, bounded by non-word
    lookarounds -- so what runs here is what will run during a parse, not an approximation of
    it. Matched against the titles as WRITTEN, because that is the form the parser hands these
    filters: a taper's token, cleaned but not squashed.

    A vocabulary collision is a WARNING, not an error, and the distinction is honest rather
    than lenient: ``parse._claimed`` makes it inert. What stays an error is a fragment that
    contradicts its own ``must_not_match``, because that is the author disagreeing with
    themselves and no guard can decide which half they meant.
    """
    source = _read_optional(pack_dir / CORPUS_FILE)
    squash = pack.normalizer.squash
    # dict.fromkeys, not a set: a title listed in two of these files is one song and earns one
    # finding, and the report stays in the order they were written rather than in whatever
    # order the hashes came out. All three are the pack declaring "this is a song".
    claimed = dict.fromkeys((*pack.protected, *pack.vocabulary, *pack.aliases))
    out: list[Diagnostic] = []
    for rule in (*pack.corpus.junk, *pack.corpus.gear):
        for title in claimed:
            if rule.reaches(title, squash):
                out.append(_rule_diag(
                    pack_dir / CORPUS_FILE, rule, source, severity=WARNING,
                    summary=f"pattern /{rule.pattern}/ matches {title!r}, a song this band plays",
                    caption=f"reaches {title!r}",
                    detail=f"Not fatal: the parser asks the vocabulary, the aliases and the\n"
                           "protected titles before it applies any rule that drops, so this\n"
                           f"cannot delete {title!r}. It is reported because a fragment that\n"
                           "reaches a real title is almost always broader than its author meant,\n"
                           "and the next title added to the vocabulary may not be so lucky.",
                ))
        for example in rule.must_not_match:
            if rule.reaches(example, squash):
                out.append(_rule_diag(
                    pack_dir / CORPUS_FILE, rule, source,
                    summary=f"pattern /{rule.pattern}/ matches its own must_not_match {example!r}",
                    caption=f"hits {example!r}",
                    detail=f"The fragment was given {example!r} as a title it must NOT match, and\n"
                           "it matches it anyway. Fix the pattern or drop the counter-example.",
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
            out.append(diagnostic_at(
                ERROR, f"alias target {target!r} is absent from the vocabulary",
                file=pack_dir / ALIASES_FILE, source=source, pos=positions.get((key,)),
                caption="not in vocabulary",
                detail=f"The alias {key!r} -> {target!r} points at a name the vocabulary does\n"
                       "not contain. Plays for the song scatter across a name nothing joins to.\n"
                       "Add the target to vocabulary.json.",
            ))
    return out


def _rule_diag(file: Path, rule: Rule, source: str | None, *, summary: str, caption: str,
               detail: str, severity: str = ERROR) -> Diagnostic:
    """A diagnostic anchored at a rule's pattern in the pack file it was written in."""
    return diagnostic_at(severity, summary, file=file, source=source, pos=rule.pos,
                         caption=caption, detail=detail)


def _read_optional(path: Path) -> str | None:
    """The file text, or ``None`` if it isn't there (an optional pack file)."""
    return path.read_text(encoding="utf-8") if path.is_file() else None

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

The corpus-aware checks need a corpus, and they need the RIGHT one. Pass ``items`` -- the raw
cached source items -- and this module parses them to build a
:class:`~setlistkit.catalog.parse.Census` of every token the parser met. Without them the
corpus-aware checks are reported as a skipped note rather than silently omitted.

**Why the census and not the stored shows.** A junk or gear rule DROPS what it matches, so by the
time a show reaches the database every token those rules removed is gone from it. A check that
asked "does this pattern match anything in the corpus" against the stored shows would find nothing
for every rule, on every pack, forever -- and would report all of them dead while running
perfectly clean. That is the same shape as a filter test written above the set header: it does not
fail, it just never fires. The parser records what it actually saw, and that is what these checks
are held against.

Four corpus-aware findings, all warnings, because none of them is a broken pack:

- **A rule matches nothing.** Dead weight, and the pack cannot know it without a corpus.
- **A rule is subsumed by another.** Everything it catches, another rule already caught.
- **An alias key reaches nothing.** Nobody has ever written that spelling; the alias is a guess.
- **A frequent title the pack does not know.** The "Hi & Lo" bug from the other side: not an
  alias pointing nowhere, but a real song nothing points AT. Its plays scatter across a name that
  joins to nothing, which is silent and permanent and looks exactly like a song nobody played.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from ..diagnostics import ERROR, NOTE, WARNING, Diagnostic
from .jsonpos import Pos, diagnostic_at, parse
from .pack import ALIASES_FILE, CLASSIFIERS_FILE, CORPUS_FILE, Pack, Rule, load_pack
from .parse import parse_archive_items


# A title has to turn up at least this often before "the pack does not know it" is worth a
# finding. Below it the long tail is mostly one-off taper noise -- a typo, a jam nobody named,
# an annotation that slipped a filter -- and reporting all 1,800 of those buries the twenty that
# are real songs. Frequency is the only signal available here that separates the two.
_UNKNOWN_TITLE_FLOOR = 8

# How many unknown titles to report. The list is ordered by how often each was played, so the cut
# always falls on the least important end, and the finding says how many it did not show.
_UNKNOWN_TITLE_LIMIT = 40


def lint(pack_dir, items: Iterable[Mapping] = ()) -> list[Diagnostic]:
    """Load the pack in ``pack_dir`` and return every conformance finding.

    Loading enforces structure and raises :class:`~setlistkit.diagnostics.DiagnosticError` on
    a malformed pack; that is the caller's to catch. Once loaded, the behavioural checks here
    never raise -- they accumulate diagnostics so a CI run sees all of them at once.

    ``items`` are raw source items, as :meth:`ArchiveOrgClient.cached_items` returns them. Given
    them, this parses them with the pack's own policy to learn what the parser actually met, and
    runs the corpus-aware checks. Parsing rather than reading the stored shows is the whole point
    -- see the module docstring.
    """
    pack_dir = Path(pack_dir)
    pack = load_pack(pack_dir)

    diagnostics: list[Diagnostic] = []
    diagnostics.extend(_rule_findings(pack, pack_dir))
    diagnostics.extend(_corpus_findings(pack, pack_dir))
    diagnostics.extend(_alias_findings(pack, pack_dir))

    items = list(items)
    if not items:
        diagnostics.append(Diagnostic(
            severity=NOTE,
            summary="corpus-aware checks skipped (no cached corpus yet)",
            detail="Checks for dead rules, redundant rules, unreachable aliases and titles the\n"
                   "pack does not know need the cached corpus, which exists once `slkit pull`\n"
                   "has run. Re-lint after that to catch them.",
        ))
        return diagnostics

    census = parse_archive_items(items, normalizer=pack.normalizer,
                                 policy=pack.archive_policy()).census
    diagnostics.extend(_dead_rule_findings(pack, pack_dir, census))
    diagnostics.extend(_subsumed_rule_findings(pack, pack_dir, census))
    diagnostics.extend(_unreachable_alias_findings(pack, pack_dir, census))
    diagnostics.extend(_unknown_title_findings(pack, census))
    return diagnostics


def _all_rules(pack: Pack) -> list[tuple[Rule, Path]]:
    """Every rule in the pack, paired with the file it was written in."""
    return ([(rule, Path(CLASSIFIERS_FILE)) for rule in pack.rules]
            + [(rule, Path(CORPUS_FILE)) for rule in (*pack.corpus.junk, *pack.corpus.gear)])


def _reached(rule: Rule, census) -> set[str]:
    """Every token this rule actually matched, as the parser recorded it.

    Read, not re-derived. Asking "would this pattern match this token" here means reconstructing
    what the runtime fed it, and the first version of this got that wrong in the way the module
    docstring warns about: a classifier is matched against the squashed CANONICAL name, not the
    raw token, so ``^setbreak$`` was reported dead while it was tagging every
    "Set Break [crowd noise]" in the corpus -- and the finding told the author to delete it.

    Reading the record also makes the pipeline's gates free. A gear rule the vocabulary guard
    stops from ever firing, or a classifier shadowed by ``protected.json``, simply never appears
    in ``fired`` -- which is the truth, and something no amount of re-matching would have found.
    """
    return set(census.fired.get(rule.pattern, ()))


def _dead_rule_findings(pack: Pack, pack_dir: Path, census) -> list[Diagnostic]:
    """Rules that match nothing the parser has ever seen.

    A pack author cannot discover this alone: a rule that matches nothing also breaks nothing, so
    it survives every test and every review. It costs a line of a file and a little doubt about
    which rules are load-bearing, and the doubt is the expensive part -- the next person to read
    the pack cannot tell the dead ones from the ones holding the corpus together.
    """
    out: list[Diagnostic] = []
    for rule, filename in _all_rules(pack):
        if _reached(rule, census):
            continue
        source = _read_optional(pack_dir / filename)
        out.append(_rule_diag(
            pack_dir / filename, rule, source, severity=WARNING,
            summary=f"rule /{rule.pattern}/ matches nothing in {len(census.seen)} known titles",
            caption="never fires",
            detail="Nothing in the whole corpus reaches this rule, so it is not doing any work.\n"
                   "That may be right -- it may be defending against something that has not\n"
                   "happened yet -- but if it came from an older tool and nobody can say what it\n"
                   "was for, delete it. A rule nobody can justify and nobody can observe is a\n"
                   "rule that only costs doubt about which of its neighbours matter.",
        ))
    return out


def _subsumed_rule_findings(pack: Pack, pack_dir: Path, census) -> list[Diagnostic]:
    """Rules whose every match is already caught by another rule.

    Reported at most once per rule, against the first rule that covers it, because a family of
    three overlapping patterns would otherwise produce a finding for every pair and read like a
    much bigger problem than it is.
    """
    rules = _all_rules(pack)
    reached = {id(rule): _reached(rule, census) for rule, _ in rules}
    out: list[Diagnostic] = []
    for index, (rule, filename) in enumerate(rules):
        mine = reached[id(rule)]
        if not mine:
            continue                    # already reported as dead; every empty set subsumes
        for other_index, (other, _) in enumerate(rules):
            if other is rule or not mine <= reached[id(other)]:
                continue
            # Mutual containment means they match exactly the same tokens, so reporting both
            # halves would describe one problem twice. Keep the one written FIRST, by position
            # rather than by comparing pattern strings -- a lexicographic tie-break reads as
            # source order and is not, and it cannot break a tie between two rules whose patterns
            # are identical, which is the most literal redundancy a pack can contain.
            if reached[id(other)] <= mine and index < other_index:
                continue
            source = _read_optional(pack_dir / filename)
            out.append(_rule_diag(
                pack_dir / filename, rule, source, severity=WARNING,
                summary=f"rule /{rule.pattern}/ is fully covered by /{other.pattern}/",
                caption="redundant",
                detail=f"Every one of the {len(mine)} title(s) this matches is already matched\n"
                       f"by /{other.pattern}/, so removing it changes nothing about the corpus.\n"
                       "Two rules doing one job is two places to look when the job is done wrong.",
            ))
            break
    return out


def _unreachable_alias_findings(pack: Pack, pack_dir: Path, census) -> list[Diagnostic]:
    """Alias keys no taper has ever written.

    An alias is a claim about how people misspell a song. This is the corpus disagreeing: nobody
    spells it that way, so the entry is a guess. Harmless on its own -- and worth knowing, because
    a guessed alias is often a guessed CANONICAL name too, and that one is not harmless.
    """
    normalize = pack.normalizer.normalize
    seen_keys = {normalize(token) for token in census.seen}
    source = _read_optional(pack_dir / ALIASES_FILE)
    positions: dict[tuple, Pos] = {}
    if source:
        _, positions = parse(source)
    out: list[Diagnostic] = []
    for key, target in pack.aliases.items():
        if normalize(key) in seen_keys:
            continue
        out.append(diagnostic_at(
            WARNING, f"alias {key!r} matches nothing in the corpus",
            file=pack_dir / ALIASES_FILE, source=source, pos=positions.get((key,)),
            caption="never used",
            detail=f"No source has ever written this spelling, so the mapping to {target!r} has\n"
                   "never once fired. Not harmful by itself. Worth a look anyway: an alias key\n"
                   "nobody writes is often paired with a canonical name nobody writes either.",
        ))
    return out


def _spelt(variants: Counter) -> str:
    """The commonest spelling of one unknown title, naming the rest when there are any.

    The variants are the argument, not decoration. One song written three ways is three keys
    nothing joins, and seeing "Sticks and Stones (also: Sticks & Stones)" tells a pack author both
    that the song is missing AND that it needs an alias when they add it.
    """
    ranked = [name for name, _ in variants.most_common()]
    if len(ranked) == 1:
        return ranked[0]
    return f"{ranked[0]} (also: {', '.join(ranked[1:4])})"


def _unknown_title_findings(pack: Pack, census) -> list[Diagnostic]:
    """Titles the corpus keeps producing that the pack has never heard of.

    The "Hi & Lo" bug approached from the other side. That one is an alias pointing at a name the
    vocabulary lacks; this is a name the vocabulary lacks with nothing pointing at it at all. Both
    end the same way -- a real song's plays scatter across a key that joins to nothing -- and this
    direction is the one no amount of reading the pack can reveal, because the evidence is
    entirely in the corpus.

    Only titles the parser KEPT are counted. Something a rule dropped is not an unknown song, it
    is a rule doing its job.

    Counted by NORMALIZED key, not by the spelling that reached the corpus. An unknown title has
    no canonical form to collapse onto -- ``canonicalize`` falls through to the cleaned display
    text -- so "Sticks and Stones" and "Sticks & Stones" arrive as two different names for one
    song. Counting those separately would report twice as many unknowns as exist and halve the
    play count of each, which is backwards on both axes: the whole finding is ranked by frequency
    because frequency is what says which one to fix first. The spellings are shown together, since
    seeing the variants IS the argument for adding the song.
    """
    normalize = pack.normalizer.normalize
    _, norm_to_canon = pack.normalizer.build_vocab()
    plays: Counter = Counter()
    spellings: dict[str, Counter] = {}
    for token, count in census.seen.items():
        if token in census.dropped or token in census.tagged:
            continue
        canon, _ = pack.normalizer.canonicalize(token)
        key = normalize(canon) if canon else ""
        if not key or key in norm_to_canon:
            continue
        plays[key] += count
        spellings.setdefault(key, Counter())[canon] += count
    unknown = Counter({_spelt(spellings[key]): n for key, n in plays.items()})
    frequent = [(name, n) for name, n in unknown.most_common() if n >= _UNKNOWN_TITLE_FLOOR]
    if not frequent:
        return []
    shown = frequent[:_UNKNOWN_TITLE_LIMIT]
    lines = "\n".join(f"  {n:5d}  {name}" for name, n in shown)
    more = (f"\n  ...and {len(frequent) - len(shown)} more above the threshold"
            if len(frequent) > len(shown) else "")
    return [Diagnostic(
        severity=WARNING,
        summary=f"{len(frequent)} title(s) played {_UNKNOWN_TITLE_FLOOR}+ times are not in the "
                f"vocabulary",
        detail="Each of these was kept as a song and canonicalized to itself, because nothing in\n"
               "the pack recognised it. Every play scatters across that spelling instead of\n"
               "joining a song, which is silent, permanent, and indistinguishable from a song\n"
               "nobody played.\n\n"
               "They are not all songs. Expect three kinds: real titles missing from\n"
               "vocabulary.json, spelling variants that want an alias, and structural noise that\n"
               "wants a classifier or a drop rule. Sorted by how often each was played, because\n"
               "that is the order in which getting it wrong costs the most.\n\n"
               f"{lines}{more}",
    )]


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

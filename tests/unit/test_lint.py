# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for pack conformance linting and the `slkit pack lint` command.

The three hard-error checks each map to a bug the old repo hit: a rule that would delete a
protected song, a rule that contradicts its own counter-example, and an alias pointing at a
name the dictionary doesn't have (the Hi & Lo silent loss). A clean pack reports only the
skipped-corpus note and exits zero.
"""

import json
from pathlib import Path

from setlistkit.catalog import lint
from setlistkit.cli.main import EXIT_DIAGNOSTIC, EXIT_OK, main
from setlistkit.diagnostics import render

EXAMPLE_PACK = Path(__file__).resolve().parents[2] / "examples" / "packs" / "example"
IDENTITY = '{"name": "t", "version": "1.0.0"}'


def _write_pack(tmp_path, **files):
    pack = tmp_path / "pack"
    pack.mkdir()
    for name, text in files.items():
        (pack / name).write_text(text, encoding="utf-8")
    return pack


def _errors(diagnostics):
    return [diag for diag in diagnostics if diag.is_error]


# --- the lint checks ----------------------------------------------------------------------

def test_clean_pack_has_no_errors(tmp_path):
    pack = _write_pack(tmp_path, **{"pack.json": IDENTITY, "vocabulary.json": '["Aurora"]'})
    diagnostics = lint(pack)
    assert _errors(diagnostics) == []
    # it still reports the deferred corpus checks rather than silently omitting them
    assert any("corpus-aware" in diag.summary for diag in diagnostics)


def test_shipped_example_pack_lints_clean():
    assert _errors(lint(EXAMPLE_PACK)) == []


def test_moe_pack_lints_clean_if_present():
    moe = Path(__file__).resolve().parents[3] / "moe-pack"
    if not (moe / "pack.json").is_file():
        return  # the sibling pack repo isn't always checked out; not this suite's job to require it
    assert _errors(lint(moe)) == []


def test_rule_matching_protected_title_is_an_error(tmp_path):
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["Jamboree"]',
        "classifiers.json": '{"non_song": [{"pattern": "jam", "why": "an unnamed improv"}]}',
        "protected.json": '["Jamboree"]'})
    errors = _errors(lint(pack))
    assert len(errors) == 1
    assert "protected title" in errors[0].summary and "Jamboree" in errors[0].summary
    assert "^" in render(errors[0])          # points at the rule in classifiers.json


def test_rule_matching_its_own_must_not_match_is_an_error(tmp_path):
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["Jamboree"]',
        "classifiers.json": '{"non_song": [{"pattern": "jam", "why": "x", '
                            '"must_not_match": ["Jamboree"]}]}'})
    errors = _errors(lint(pack))
    assert len(errors) == 1
    assert "must_not_match" in errors[0].summary


def test_alias_target_absent_from_vocabulary_is_an_error(tmp_path):
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["Real Song"]',
        "aliases.json": '{"foo": "Ghost Song"}'})
    errors = _errors(lint(pack))
    assert len(errors) == 1
    assert "Ghost Song" in errors[0].summary
    assert errors[0].line is not None        # anchored into aliases.json for a caret


def test_all_findings_accumulate_rather_than_short_circuiting(tmp_path):
    """a pack with several distinct faults reports every one, not just the first."""
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["Jamboree"]',
        "classifiers.json": '{"non_song": [{"pattern": "jam", "why": "an improv"}]}',
        "protected.json": '["Jamboree"]',
        "aliases.json": '{"foo": "Ghost Song"}'})
    errors = _errors(lint(pack))
    assert len(errors) == 2                   # protected-title hit AND absent alias target
    summaries = " ".join(diag.summary for diag in errors)
    assert "protected title" in summaries and "Ghost Song" in summaries


# --- the CLI command ----------------------------------------------------------------------

def test_cli_lint_clean_pack_exits_ok(capsys):
    assert main(["pack", "lint", "--pack", str(EXAMPLE_PACK)]) == EXIT_OK
    assert "0 error(s)" in capsys.readouterr().out


def test_cli_lint_broken_pack_exits_nonzero(tmp_path, capsys):
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["Real Song"]',
        "aliases.json": '{"foo": "Ghost Song"}'})
    assert main(["pack", "lint", "--pack", str(pack)]) == EXIT_DIAGNOSTIC


def test_cli_lint_json_format(capsys):
    assert main(["pack", "lint", "--pack", str(EXAMPLE_PACK), "--format", "json"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list) and data[0]["severity"] in {"note", "warning", "error"}


def test_cli_lint_resolves_pack_from_config(tmp_path, capsys):
    cfg = tmp_path / "slkit.toml"
    cfg.write_text(
        'data_root = "data"\nuser_agent = "test (me@example.com)"\n'
        f'[catalog]\npack = "{EXAMPLE_PACK}"\n', encoding="utf-8")
    assert main(["--config", str(cfg), "pack", "lint"]) == EXIT_OK


def test_cli_lint_without_pack_or_config_entry_is_an_error(tmp_path, capsys):
    cfg = tmp_path / "slkit.toml"
    cfg.write_text('data_root = "data"\nuser_agent = "test (me@example.com)"\n', encoding="utf-8")
    assert main(["--config", str(cfg), "pack", "lint"]) == EXIT_DIAGNOSTIC
    assert "no pack" in capsys.readouterr().err


def test_cli_lint_folds_a_structural_failure_into_an_error(tmp_path, capsys):
    """load_pack raises on a malformed pack; the command catches it and exits non-zero.

    A free-floating pattern that won't compile trips regex validation inside load_pack, which
    lint() lets propagate -- the CLI is what turns it into a reported finding."""
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["Aurora"]',
        "classifiers.json": '{"non_song": [{"pattern": "(unclosed", "why": "x"}]}'})
    assert main(["pack", "lint", "--pack", str(pack)]) == EXIT_DIAGNOSTIC
    assert "invalid regex" in capsys.readouterr().out


# --- the corpus filters ---------------------------------------------------------------------

def _warnings(diagnostics):
    return [diag for diag in diagnostics if diag.severity == "warning"]


def test_a_corpus_fragment_that_reaches_a_real_song_warns_but_does_not_fail(tmp_path):
    """not an error, because parse._claimed makes it inert -- and saying "error" about
    something that cannot happen teaches an author to stop reading the errors.

    Still worth reporting: a fragment wide enough to reach a title is wider than its author
    meant, and the guard only covers the titles that are in the pack today.
    """
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["Plane Crash", "Aurora"]',
        "corpus.json": '{"junk_patterns": [{"pattern": "crash", "why": "a taper note"}]}'})
    diagnostics = lint(pack)
    assert _errors(diagnostics) == []
    warnings = _warnings(diagnostics)
    assert len(warnings) == 1
    assert "matches 'Plane Crash'" in warnings[0].summary
    assert "corpus.json" in warnings[0].path
    assert "^" in render(warnings[0])


def test_a_corpus_fragment_is_held_against_protected_titles_and_alias_keys_too(tmp_path):
    """all three files are the pack declaring "this is a song", so all three are checked.

    The alias key matters most: every DROP rule runs before canonicalize, so what a fragment
    actually meets is the taper's spelling, not the canonical name.
    """
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["Aurora"]', "protected.json": '["ATL"]',
        "aliases.json": '{"rora": "Aurora"}',
        "corpus.json": '{"junk_patterns": [{"pattern": "atl", "why": "x"},'
                       ' {"pattern": "rora", "why": "y"}]}'})
    summaries = " ".join(diag.summary for diag in _warnings(lint(pack)))
    assert "'ATL'" in summaries and "'rora'" in summaries
    assert _errors(lint(pack)) == []


def test_a_junk_pattern_bounded_away_from_a_song_is_clean(tmp_path):
    """the boundary is real, not decorative: 'home team' must not reach 'Homeward Bound'."""
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["Homeward Bound", "Letter Home"]',
        "corpus.json": '{"junk_patterns": [{"pattern": "home\\\\s+team", "why": "ported"}]}'})
    assert _errors(lint(pack)) == []


def test_gear_and_junk_fragments_are_held_to_the_same_standard(tmp_path):
    """both are guarded and both warn. They used to be treated differently here, on the theory
    that gear deferred to the vocabulary and junk did not -- which was true, and was the bug."""
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["Wave", "Aurora"]',
        "corpus.json": '{"gear_patterns": [{"pattern": "wave", "why": "a tape format"}]}'})
    assert _errors(lint(pack)) == []
    assert len(_warnings(lint(pack))) == 1


def test_a_corpus_fragment_that_hits_its_own_counter_example_is_an_error(tmp_path):
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["Aurora"]',
        "corpus.json": '{"junk_patterns": [{"pattern": "home", "why": "x",'
                       ' "must_not_match": ["Home Team"]}]}'})
    errors = _errors(lint(pack))
    assert len(errors) == 1
    assert "must_not_match 'Home Team'" in errors[0].summary


def test_a_gear_fragment_that_hits_its_own_counter_example_is_an_error(tmp_path):
    """gear gets no vocabulary check, but its author's own counter-example still binds it."""
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["Aurora"]',
        "corpus.json": '{"gear_patterns": [{"pattern": "cf", "why": "x",'
                       ' "must_not_match": ["CF"]}]}'})
    assert len(_errors(lint(pack))) == 1


def test_a_title_in_both_protected_and_vocabulary_earns_one_finding(tmp_path):
    """one song, one finding -- listing a title twice in the pack is not two problems."""
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": '["ATL", "Aurora"]',
        "protected.json": '["ATL"]',
        "corpus.json": '{"junk_patterns": [{"pattern": "atl", "why": "a taper note"}]}'})
    assert len(_warnings(lint(pack))) == 1


# --- corpus-aware checks --------------------------------------------------------------------
#
# These need a corpus, and specifically they need the tokens the PARSER MET rather than the shows
# that reached the database. A junk or gear rule drops what it matches, so anything held against
# the stored shows would find nothing for every rule, on every pack, forever -- and report them
# all dead while running perfectly clean.

VOCAB = '["Aurora", "Wormhole", "Jamboree"]'


def _item(description, identifier="t1", date="2025-07-04"):
    return {"identifier": identifier, "date": date,
            "title": f"The Band Live at Northlands on {date}",
            "description": description}


def _summaries(diagnostics):
    return [diag.summary for diag in diagnostics]


def test_a_rule_that_matches_nothing_in_the_corpus_is_reported(tmp_path):
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "classifiers.json": '{"non_song": ["^setbreak$", "^neveroccurs$"]}'})
    found = lint(pack, [_item("Set 1:\n01. Aurora\n02. Setbreak\n03. Wormhole\n")])
    assert any("/^neveroccurs$/ matches nothing" in s for s in _summaries(found))
    # ...and the rule that DID fire is not reported
    assert not any("/^setbreak$/ matches nothing" in s for s in _summaries(found))


def test_a_junk_rule_is_judged_on_what_it_dropped_not_on_what_survived(tmp_path):
    """The trap this whole design exists to avoid.

    A junk rule DROPS its matches, so they never reach the corpus. Held against the stored shows
    this rule looks dead; held against what the parser met, it is plainly working.
    """
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "corpus.json": json.dumps({"junk_patterns": [
            {"pattern": "bootleg\\s+notice", "why": "a taper's boilerplate"}]})})
    found = lint(pack, [_item("Set 1:\n01. Aurora\n02. Bootleg Notice\n03. Wormhole\n")])
    assert not any("matches nothing" in s for s in _summaries(found))


def test_a_rule_fully_covered_by_a_wider_one_names_the_narrow_one(tmp_path):
    """`set` also reaches "Sunset", so `^setbreak$` is a strict subset of it and is the redundant
    one. Direction matters: naming the wider rule would tell an author to delete the useful one."""
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "classifiers.json": '{"non_song": ["^setbreak$", {"pattern": "set",'
                            ' "why": "wider, catches everything the anchored one does"}]}'})
    found = lint(pack, [_item("Set 1:\n01. Aurora\n02. Setbreak\n03. Sunset\n")])
    covered = [s for s in _summaries(found) if "fully covered by" in s]
    assert covered == ["rule /^setbreak$/ is fully covered by /set/"]


def test_two_rules_matching_exactly_the_same_titles_are_reported_once(tmp_path):
    """Interchangeable in this corpus, so which one is "redundant" is arbitrary -- but saying it
    twice describes one problem as two."""
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "classifiers.json": '{"non_song": ["^setbreak$", {"pattern": "setbreak",'
                            ' "why": "the same thing, unanchored"}]}'})
    found = lint(pack, [_item("Set 1:\n01. Aurora\n02. Setbreak\n03. Wormhole\n")])
    assert len([s for s in _summaries(found) if "fully covered by" in s]) == 1


def test_an_alias_nobody_writes_is_reported_and_one_in_use_is_not(tmp_path):
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "aliases.json": '{"the hole": "Wormhole", "nobody writes this": "Aurora"}'})
    found = lint(pack, [_item("Set 1:\n01. Aurora\n02. The Hole\n")])
    assert any("alias 'nobody writes this' matches nothing" in s for s in _summaries(found))
    assert not any("alias 'the hole'" in s for s in _summaries(found))


def test_the_unused_alias_caret_lands_on_the_key_not_the_target(tmp_path):
    """The finding is about the key, so the caret has to be.

    Anchoring it on the value reads as a claim that the TARGET is the unwritten spelling, which
    is false and is the one word in the line that tapers demonstrably do write. Reported from the
    real pack: 'tambo' is never written, 'Tambourine' appears 581 times, and the caret sat on
    'Tambourine' under the caption "never used".
    """
    source = '{"the hole": "Wormhole", "nobody writes this": "Aurora"}'
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB, "aliases.json": source})
    found = lint(pack, [_item("Set 1:\n01. Aurora\n02. The Hole\n")])
    unused = [d for d in found if "matches nothing in the corpus" in d.summary]
    assert len(unused) == 1
    assert source[unused[0].col - 1:][:unused[0].length] == '"nobody writes this"'


def test_a_title_the_pack_does_not_know_is_reported_with_its_play_count(tmp_path):
    pack = _write_pack(tmp_path, **{"pack.json": IDENTITY, "vocabulary.json": VOCAB})
    items = [_item("Set 1:\n01. Aurora\n02. Mystery Song\n", identifier=f"t{n}",
                   date=f"2025-07-{n:02d}") for n in range(1, 11)]
    found = lint(pack, items)
    unknown = [d for d in found if "not in the vocabulary" in d.summary]
    assert len(unknown) == 1
    assert "Mystery Song" in unknown[0].detail
    assert "Aurora" not in unknown[0].detail            # in the vocabulary, so not a finding


def test_spelling_variants_of_one_unknown_title_are_counted_as_one_song(tmp_path):
    """Counted by normalized key, not by the spelling that reached the corpus.

    An unknown title has no canonical form to collapse onto, so canonicalize falls through to the
    display text and one song arrives under several names. Counting those separately reports twice
    as many unknowns as exist and halves the play count of each -- backwards on both axes, since
    the finding is ranked by frequency precisely to say which one to fix first.
    """
    pack = _write_pack(tmp_path, **{"pack.json": IDENTITY, "vocabulary.json": VOCAB})
    spellings = ["Sticks and Stones", "Sticks & Stones", "sticks and stones"]
    items = [_item(f"Set 1:\n01. Aurora\n02. {spellings[n % 3]}\n", identifier=f"t{n}",
                   date=f"2025-07-{n + 1:02d}") for n in range(9)]
    found = lint(pack, items)
    detail = next(d for d in found if "not in the vocabulary" in d.summary).detail
    assert "9  Sticks and Stones (also:" in detail      # one song, nine plays, not three of three
    assert "1 title(s)" in next(d.summary for d in found if "not in the vocabulary" in d.summary)


def test_a_dropped_token_is_not_an_unknown_song(tmp_path):
    """Something a rule removed is not a song the pack is missing, it is a rule doing its job."""
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "corpus.json": json.dumps({"junk_patterns": [
            {"pattern": "bootleg\\s+notice", "why": "a taper's boilerplate"}]})})
    items = [_item("Set 1:\n01. Aurora\n02. Bootleg Notice\n", identifier=f"t{n}",
                   date=f"2025-07-{n + 1:02d}") for n in range(10)]
    found = lint(pack, items)
    assert not any("not in the vocabulary" in d.summary for d in found)


def test_a_tagged_non_song_is_not_an_unknown_song(tmp_path):
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "classifiers.json": '{"non_song": ["^setbreak$"]}'})
    items = [_item("Set 1:\n01. Aurora\n02. Setbreak\n", identifier=f"t{n}",
                   date=f"2025-07-{n + 1:02d}") for n in range(10)]
    found = lint(pack, items)
    assert not any("not in the vocabulary" in d.summary for d in found)


def test_a_one_off_unknown_title_is_below_the_reporting_floor(tmp_path):
    """The long tail is mostly typos and unnamed jams. Reporting all of it buries the real ones."""
    pack = _write_pack(tmp_path, **{"pack.json": IDENTITY, "vocabulary.json": VOCAB})
    found = lint(pack, [_item("Set 1:\n01. Aurora\n02. Played Once Ever\n")])
    assert not any("not in the vocabulary" in d.summary for d in found)


def test_corpus_checks_report_themselves_skipped_without_a_corpus(tmp_path):
    pack = _write_pack(tmp_path, **{"pack.json": IDENTITY, "vocabulary.json": VOCAB})
    assert any("corpus-aware checks skipped" in s for s in _summaries(lint(pack)))
    assert not any("corpus-aware checks skipped" in s
                   for s in _summaries(lint(pack, [_item("Set 1:\n01. Aurora\n")])))


def test_corpus_findings_are_warnings_not_errors(tmp_path):
    """None of them is a broken pack, and a lint that fails CI over a dead rule gets ignored."""
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "classifiers.json": '{"non_song": ["^neveroccurs$"]}',
        "aliases.json": '{"nobody writes this": "Aurora"}'})
    found = lint(pack, [_item("Set 1:\n01. Aurora\n")])
    assert len(_errors(found)) == 0
    assert len(found) >= 2


def test_a_classifier_that_fires_on_an_annotated_title_is_not_reported_dead(tmp_path):
    """The bug this module was built to prevent, committed by this module.

    A classifier is matched against the squashed CANONICAL name, not the raw token. Re-deriving
    the question in lint asked it of the wrong string: `^setbreak$` tagged every
    "Set Break [crowd noise]" in the real corpus and was reported dead, and the finding said to
    delete it. Deleting it turns "Set Break" into a song.
    """
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "classifiers.json": '{"non_song": ["^setbreak$", "^neveroccurs$"]}'})
    found = lint(pack, [_item("Set 1:\n01. Aurora\n02. Set Break [crowd noise]\n")])
    assert not any("/^setbreak$/ matches nothing" in s for s in _summaries(found))
    # ...and the genuinely dead one beside it is still caught, so this is not just "the check
    # got deleted".
    assert any("/^neveroccurs$/ matches nothing" in s for s in _summaries(found))


def test_a_classifier_shadowed_by_a_protected_title_is_reported_dead(tmp_path):
    """is_non_song checks is_protected FIRST, so this rule can never fire on anything, ever.

    Re-matching the pattern would report it alive; reading what actually fired reports the truth.
    """
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "protected.json": '["Aurora"]',
        "classifiers.json": '{"non_song": ["^aurora$"]}'})
    found = lint(pack, [_item("Set 1:\n01. Aurora\n02. Wormhole\n")])
    assert any("/^aurora$/ matches nothing" in s for s in _summaries(found))


def test_two_identical_patterns_are_reported_as_redundant(tmp_path):
    """The most literal redundancy a pack can contain, and a lexicographic tie-break could not
    see it: pattern <= pattern is true in both directions, so both halves skipped."""
    pack = _write_pack(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "classifiers.json": '{"non_song": ["^setbreak$", "^setbreak$"]}'})
    found = lint(pack, [_item("Set 1:\n01. Aurora\n02. Setbreak\n")])
    assert len([s for s in _summaries(found) if "fully covered by" in s]) == 1


def test_the_rule_written_first_is_the_one_kept(tmp_path):
    """Source order, not pattern spelling. A lexicographic tie-break reads as source order and
    is not, so which rule an author was told to delete depended on the alphabet."""
    both = '{"non_song": [{"pattern": "%s", "why": "x"}, {"pattern": "%s", "why": "y"}]}'
    for first, second in (("setbreak", "setbrea"), ("setbrea", "setbreak")):
        (tmp_path / first).mkdir()
        pack = _write_pack(tmp_path / first, **{
            "pack.json": IDENTITY, "vocabulary.json": VOCAB,
            "classifiers.json": both % (first, second)})
        found = lint(pack, [_item("Set 1:\n01. Aurora\n02. Setbreak\n")])
        covered = [s for s in _summaries(found) if "fully covered by" in s]
        assert covered == [f"rule /{second}/ is fully covered by /{first}/"], covered


def test_a_weak_description_parse_does_not_count_its_tracklist_twice(tmp_path):
    """The tracklist attempt only runs when the description parse is thin, and its census is
    folded in only if it WINS. Both usually cover the same show, so counting both put every
    token of every weak-parse item in twice -- moving titles across the reporting floor and
    reordering a ranking whose entire job is to say which one to fix first."""
    pack = _write_pack(tmp_path, **{"pack.json": IDENTITY, "vocabulary.json": VOCAB})
    items = []
    for n in range(9):
        item = _item("Set 1:\n01. Mystery Song\n", identifier=f"t{n}", date=f"2025-07-{n + 1:02d}")
        item["tracks"] = [{"title": "Mystery Song"}]
        items.append(item)
    found = lint(pack, items)
    detail = next(d for d in found if "not in the vocabulary" in d.summary).detail
    assert "    9  Mystery Song" in detail            # nine items, nine plays, not eighteen

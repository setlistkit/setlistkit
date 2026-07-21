# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for loading a band pack into a data-driven Normalizer.

Covers the three validating layers (schema shape, the free-floating-justification domain
rule, regex compilation), the required-file and malformed-JSON failures, and that the built
normalizer actually applies the pack's aliases, patterns and protected titles. Every failure
is checked for a rendered caret, because a pack error that can't point at the problem is the
thing this whole subsystem exists to avoid.
"""

import json
from pathlib import Path

import pytest

from setlistkit.catalog.pack import Pack, load_pack
from setlistkit.catalog.parse import parse_archive_items
from setlistkit.diagnostics import DiagnosticError, render

EXAMPLE_PACK = Path(__file__).resolve().parents[2] / "examples" / "packs" / "example"


def _pack(tmp_path, **files):
    """Write the given {basename: text} into a fresh pack dir and return it."""
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


IDENTITY = '{"name": "moe", "version": "1.0.0", "sources": ["setlist.fm"]}'
VOCAB = '["Recreational Chemistry", "Lazarus", "St. Augustine"]'


def _load_or_render(tmp_path, **files):
    """Load a pack, re-raising any DiagnosticError as (summary, rendered) for assertions."""
    try:
        return load_pack(_pack(tmp_path, **files))
    except DiagnosticError as err:
        raise AssertionError(render(err.diagnostic)) from err


# --- happy paths --------------------------------------------------------------------------

def test_minimal_pack_needs_only_identity_and_vocabulary(tmp_path):
    pack = _load_or_render(tmp_path, **{"pack.json": IDENTITY, "vocabulary.json": VOCAB})
    assert isinstance(pack, Pack)
    assert pack.name == "moe"
    assert pack.version == "1.0.0"
    assert pack.sources == ("setlist.fm",)
    assert pack.aliases == {}
    assert pack.protected == ()
    assert pack.rules == ()
    # the normalizer stands up and canonicalizes against the vocabulary
    assert pack.normalizer.canonicalize("recreational chemistry")[0] == "Recreational Chemistry"


def test_full_pack_populates_every_hook(tmp_path):
    pack = _load_or_render(
        tmp_path,
        **{
            "pack.json": IDENTITY,
            "vocabulary.json": VOCAB,
            "aliases.json": '{"rec chem": "Recreational Chemistry"}',
            "protected.json": '["ATL", "NYC"]',
            "classifiers.json": '{"non_song": ["^setbreak$", '
                                '{"pattern": "nounc", "why": "Al announcements"}]}',
        },
    )
    norm = pack.normalizer
    assert norm.canonicalize("Rec Chem")[0] == "Recreational Chemistry"   # alias hook
    assert norm.is_non_song("Setbreak") is True                          # anchored bare rule
    assert norm.is_non_song("Al.nouncements") is True                    # object rule
    assert norm.is_non_song("NYC") is False                              # protected hook wins
    assert norm.is_non_song("Recreational Chemistry") is False


def test_rule_metadata_is_preserved_for_lint(tmp_path):
    pack = _load_or_render(
        tmp_path,
        **{
            "pack.json": IDENTITY,
            "vocabulary.json": VOCAB,
            "classifiers.json": '{"non_song": ['
                                '"intros$", '
                                '{"pattern": "nounc", "why": "announcements", '
                                '"must_not_match": ["Announcer\'s Song"]}]}',
        },
    )
    bare, obj = pack.rules
    assert bare.pattern == "intros$" and bare.anchored is True and bare.why is None
    assert obj.pattern == "nounc" and obj.anchored is False
    assert obj.why == "announcements"
    assert obj.must_not_match == ("Announcer's Song",)


# --- missing required files ---------------------------------------------------------------

def test_missing_pack_json_is_a_diagnostic(tmp_path):
    with pytest.raises(DiagnosticError) as excinfo:
        load_pack(_pack(tmp_path, **{"vocabulary.json": VOCAB}))
    assert "missing a required file: pack.json" in excinfo.value.diagnostic.summary


def test_missing_vocabulary_is_a_diagnostic(tmp_path):
    with pytest.raises(DiagnosticError) as excinfo:
        load_pack(_pack(tmp_path, **{"pack.json": IDENTITY}))
    assert "vocabulary.json" in excinfo.value.diagnostic.summary


# --- malformed / invalid ------------------------------------------------------------------

def test_malformed_json_reports_a_position(tmp_path):
    with pytest.raises(DiagnosticError) as excinfo:
        load_pack(_pack(tmp_path, **{"pack.json": '{"name": "moe" "version": "1"}',
                                     "vocabulary.json": VOCAB}))
    diag = excinfo.value.diagnostic
    assert diag.line is not None and diag.col is not None
    assert "^" in render(diag)


def test_schema_violation_missing_required_field(tmp_path):
    with pytest.raises(DiagnosticError) as excinfo:
        load_pack(_pack(tmp_path, **{"pack.json": '{"name": "moe"}', "vocabulary.json": VOCAB}))
    assert "version" in excinfo.value.diagnostic.summary


def test_schema_violation_unknown_key_is_caught(tmp_path):
    with pytest.raises(DiagnosticError) as excinfo:
        load_pack(_pack(tmp_path, **{
            "pack.json": '{"name": "moe", "version": "1", "nmae": "typo"}',
            "vocabulary.json": VOCAB}))
    # additionalProperties:false turns a typo'd key into an error rather than a silent drop
    assert excinfo.value.diagnostic.is_error


def test_duplicate_vocabulary_entries_rejected(tmp_path):
    with pytest.raises(DiagnosticError):
        load_pack(_pack(tmp_path, **{"pack.json": IDENTITY,
                                     "vocabulary.json": '["Lazarus", "Lazarus"]'}))


# --- the domain rule: a free-floating pattern must justify itself -------------------------

def test_free_floating_bare_string_must_justify(tmp_path):
    with pytest.raises(DiagnosticError) as excinfo:
        load_pack(_pack(tmp_path, **{
            "pack.json": IDENTITY, "vocabulary.json": VOCAB,
            "classifiers.json": '{"non_song": ["reprise"]}'}))   # free-floating, no why
    diag = excinfo.value.diagnostic
    assert diag.summary == "free-floating pattern must justify itself"
    assert "^" in render(diag)


def test_object_with_empty_why_must_justify(tmp_path):
    text = '{\n  "non_song": [\n    { "pattern": "reprise",\n      "why": "" }\n  ]\n}'
    with pytest.raises(DiagnosticError) as excinfo:
        load_pack(_pack(tmp_path, **{
            "pack.json": IDENTITY, "vocabulary.json": VOCAB, "classifiers.json": text}))
    diag = excinfo.value.diagnostic
    assert diag.summary == "free-floating pattern must justify itself"
    assert diag.caret_caption == "empty"
    frame = render(diag)
    assert '"why": ""' in frame and "^^ empty" in frame   # caret lands on the empty value


def test_anchored_bare_string_needs_no_justification(tmp_path):
    pack = _load_or_render(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "classifiers.json": '{"non_song": ["^setbreak$", "intros$", "^intro"]}'})
    assert [r.pattern for r in pack.rules] == ["^setbreak$", "intros$", "^intro"]


# --- regex compilation --------------------------------------------------------------------

def test_shipped_example_pack_loads_and_behaves(tmp_path):
    """the example pack we ship as a template must itself load clean and demonstrate the hooks."""
    pack = load_pack(EXAMPLE_PACK)
    assert pack.name == "example"
    # alias resolves a normalized-away key ("The Long One" -> "long one")
    assert pack.normalizer.canonicalize("Long One")[0] == "The Long One"
    # anchored bare rule and the free-floating object rule both classify
    assert pack.normalizer.is_non_song("Setbreak") is True
    assert pack.normalizer.is_non_song("Soundcheck Jam") is True
    # the object rule's must_not_match holds: "soundcheck" leaves "Sound Asleep" alone
    assert pack.normalizer.is_non_song("Sound Asleep") is False
    # a real song is a real song, and the protected title survives
    assert pack.normalizer.is_non_song("Aurora") is False
    assert pack.normalizer.is_non_song("Jamboree") is False


def test_bad_regex_points_inside_the_pattern(tmp_path):
    with pytest.raises(DiagnosticError) as excinfo:
        load_pack(_pack(tmp_path, **{
            "pack.json": IDENTITY, "vocabulary.json": VOCAB,
            "classifiers.json": '{"non_song": [{"pattern": "^(unclosed", "why": "x"}]}'}))
    diag = excinfo.value.diagnostic
    assert "invalid regex" in diag.summary
    assert diag.line is not None and diag.col is not None


def test_bad_regex_caret_accounts_for_json_escapes(tmp_path):
    """the caret lands on the real offending char even when escapes shift source columns."""
    # on disk the pattern is "\\d(" which decodes to \d( ; the unbalanced paren is the fault.
    text = '{"non_song": [{"pattern": "\\\\d(", "why": "x"}]}'
    with pytest.raises(DiagnosticError) as excinfo:
        load_pack(_pack(tmp_path, **{
            "pack.json": IDENTITY, "vocabulary.json": VOCAB, "classifiers.json": text}))
    diag = excinfo.value.diagnostic
    char_under_caret = text.splitlines()[diag.line - 1][diag.col - 1]
    assert char_under_caret == "("      # not the backslash two columns to its left


def test_free_floating_alternation_branch_must_justify(tmp_path):
    """a leading ^ does not make '^intro|jam' safe: the jam branch is free-floating."""
    with pytest.raises(DiagnosticError) as excinfo:
        load_pack(_pack(tmp_path, **{
            "pack.json": IDENTITY, "vocabulary.json": VOCAB,
            "classifiers.json": '{"non_song": ["^intro|jam"]}'}))
    assert excinfo.value.diagnostic.summary == "free-floating pattern must justify itself"


def test_alternation_with_a_why_is_accepted_and_flagged_unanchored(tmp_path):
    pack = _load_or_render(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "classifiers.json": '{"non_song": [{"pattern": "^intro|jam", "why": "spelled out"}]}'})
    assert pack.rules[0].anchored is False


def test_alias_key_is_normalized_on_load(tmp_path):
    """a readable alias key ('The Rec Chem') matches, instead of silently never firing."""
    pack = _load_or_render(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "aliases.json": '{"The Rec Chem": "Recreational Chemistry"}'})
    assert pack.normalizer.canonicalize("The Rec Chem")[0] == "Recreational Chemistry"
    # the raw authored key is preserved on the Pack for diff-review and lint
    assert pack.aliases == {"The Rec Chem": "Recreational Chemistry"}


# --- corpus.json: the corrections and the residue -------------------------------------------

CORPUS = """{
  "drop_dates": {"2025-10-31": "costume show, the songs were bits"},
  "date_overrides": {"band2024-06-14": {"date": "2025-06-14",
                                        "why": "the description says June 2025"}},
  "junk_patterns": [{"pattern": "umphrey", "why": "a band they cover"}],
  "gear_patterns": [{"pattern": "zoomf\\\\d", "why": "a recorder this scene writes"}]
}"""


def test_corpus_file_is_optional(tmp_path):
    pack = _load_or_render(tmp_path, **{"pack.json": IDENTITY, "vocabulary.json": VOCAB})
    assert pack.corpus.drop_dates == {}
    assert pack.corpus.date_overrides == {}
    assert pack.corpus.junk == () and pack.corpus.gear == ()
    assert pack.band_name is None


def test_corpus_file_loads_every_key(tmp_path):
    pack = _load_or_render(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB, "corpus.json": CORPUS})
    assert pack.corpus.drop_dates == {"2025-10-31": "costume show, the songs were bits"}
    assert pack.corpus.date_overrides["band2024-06-14"]["date"] == "2025-06-14"
    assert [rule.pattern for rule in pack.corpus.junk] == ["umphrey"]
    assert [rule.pattern for rule in pack.corpus.gear] == [r"zoomf\d"]


def test_a_corpus_fragment_is_compiled_the_way_the_filter_applies_it(tmp_path):
    """Rule.compiled is the bounded form, not the bare fragment.

    This is what makes a lint check believable: a check that approximated the filter could
    report a collision the parser will never have, or miss one it will.
    """
    pack = _load_or_render(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "corpus.json": '{"junk_patterns": [{"pattern": "home", "why": "x"}]}'})
    compiled = pack.corpus.junk[0].compiled
    assert compiled.search("Home Team")            # bounded on both sides, so a word matches
    assert not compiled.search("Homeward Bound")   # ...and a longer word does not


def test_band_name_is_read_from_pack_json(tmp_path):
    pack = _load_or_render(tmp_path, **{
        "pack.json": '{"name": "moe", "version": "1", "band_name": "moe."}',
        "vocabulary.json": VOCAB})
    assert pack.band_name == "moe."


# --- corpus.json: every entry has to say why ------------------------------------------------

def _corpus_failure(tmp_path, corpus: str):
    """Load a pack with the given corpus.json, returning the diagnostic it raised."""
    with pytest.raises(DiagnosticError) as excinfo:
        load_pack(_pack(tmp_path, **{
            "pack.json": IDENTITY, "vocabulary.json": VOCAB, "corpus.json": corpus}))
    return excinfo.value.diagnostic


def test_a_drop_date_with_no_reason_is_rejected(tmp_path):
    diag = _corpus_failure(tmp_path, '{"drop_dates": {"2025-10-31": "  "}}')
    assert diag.summary == "drop_dates '2025-10-31' does not say why"
    assert "^" in render(diag)


def test_a_date_override_with_no_evidence_is_rejected(tmp_path):
    diag = _corpus_failure(
        tmp_path, '{"date_overrides": {"x1": {"date": "2025-06-14", "why": ""}}}')
    assert diag.summary == "date_overrides 'x1' does not say why"
    assert "^" in render(diag)


def test_a_junk_fragment_with_no_reason_is_rejected(tmp_path):
    diag = _corpus_failure(tmp_path, '{"junk_patterns": [{"pattern": "team", "why": ""}]}')
    assert diag.summary == "junk_patterns entry must say what it is"


def test_a_gear_fragment_with_no_reason_is_rejected(tmp_path):
    diag = _corpus_failure(tmp_path, '{"gear_patterns": [{"pattern": "kcy", "why": ""}]}')
    assert diag.summary == "gear_patterns entry must say what it is"


def test_a_corpus_fragment_has_no_bare_string_form(tmp_path):
    """unlike a classifier: every fragment is free-floating, so none can anchor its way out."""
    diag = _corpus_failure(tmp_path, '{"junk_patterns": ["^umphrey$"]}')
    assert "not of type 'object'" in diag.summary


def test_a_malformed_drop_date_is_rejected_at_the_schema(tmp_path):
    """a typo'd date is the silent kind of wrong: it simply never equals any show's date."""
    diag = _corpus_failure(tmp_path, '{"drop_dates": {"2025-13-01x": "typo"}}')
    assert "does not match" in diag.summary


def test_a_malformed_override_date_is_rejected_at_the_schema(tmp_path):
    diag = _corpus_failure(
        tmp_path, '{"date_overrides": {"x1": {"date": "June 14 2025", "why": "evidence"}}}')
    assert "does not match" in diag.summary


def test_a_broken_corpus_regex_points_inside_the_pattern(tmp_path):
    text = '{"junk_patterns": [{"pattern": "reb(", "why": "x"}]}'
    diag = _corpus_failure(tmp_path, text)
    assert diag.summary.startswith("invalid regex")
    assert text.splitlines()[diag.line - 1][diag.col - 1] == "("


# --- the policy the pack implies ------------------------------------------------------------

def test_archive_policy_is_assembled_from_the_pack(tmp_path):
    pack = _load_or_render(tmp_path, **{
        "pack.json": '{"name": "moe", "version": "1", "band_name": "moe."}',
        "vocabulary.json": VOCAB, "corpus.json": CORPUS})
    policy = pack.archive_policy()
    assert policy.drop_dates == frozenset({"2025-10-31"})
    # the override flattens to identifier -> date; the evidence stays on the pack
    assert policy.date_overrides == {"band2024-06-14": "2025-06-14"}
    assert policy.junk_patterns == ("umphrey",)
    assert policy.gear_patterns == (r"zoomf\d",)
    assert policy.band_name == "moe."
    assert policy.band_filter({"title": "moe. Live at Northlands on 2026-06-14"}) is True
    assert policy.band_filter({"title": "bob. Live at Ophelia's on 2024-11-07"}) is False


def test_a_pack_with_no_band_name_runs_no_band_filter(tmp_path):
    """refusing to guess: an unreadable title is not evidence that a show is fake, and an
    absent band name is not evidence about anything at all."""
    pack = _load_or_render(tmp_path, **{"pack.json": IDENTITY, "vocabulary.json": VOCAB})
    assert pack.archive_policy().band_filter is None


def test_the_example_pack_drives_the_parser_end_to_end():
    """The whole seam on the pack that ships: every corpus key doing its job at once.

    Each of these is a separate mechanism, and each one used to be a hardcoded fact inside the
    parser. Together they are the answer to "what does a pack actually buy you" -- so they are
    asserted against the real shipped files rather than a fixture, which also means the example
    pack cannot rot without a test noticing.
    """
    pack = load_pack(EXAMPLE_PACK)
    policy = pack.archive_policy()
    items = [
        # a side project's tape, in the same collection: rejected on the title alone
        {"identifier": "other2025-01-01", "date": "2025-01-01",
         "title": "Some Other Band Live at The Fillmore on 2025-01-01",
         "description": "Set 1:\n01. Aurora\n02. Wormhole\n"},
        # a dropped date: the costume show, refused outright
        {"identifier": "example2025-10-31", "date": "2025-10-31",
         "title": "The Example Live at The Fillmore on 2025-10-31",
         "description": "Set 1:\n01. Aurora\n02. Wormhole\n"},
        # the mis-dated item, moved to the night it actually happened
        {"identifier": "example2024-06-14", "date": "2024-06-14",
         "title": "The Example Live at Northlands on 2024-06-14",
         # a lineage line split across a segue marker: the pack's gear word catches the left
         # half, setlistkit's own catches the right, which is the split working as intended
         "description": ("Set 1:\n01. Aurora\n02. ZoomF8 > MacBook\n"
                         "03. Aurora Borealis Band\n04. Setbreak\n05. Wormhole\n")},
    ]
    records = parse_archive_items(items, normalizer=pack.normalizer, policy=policy)

    assert [record["date"] for record in records] == ["2025-06-14"]
    entries = [(entry["song"], entry["non_song"]) for entry in records[0]["sets"][0]]
    assert entries == [("Aurora", False),      # the vocabulary keeps it
                       ("Setbreak", True),     # a classifier TAGS it -- recorded, not counted
                       ("Wormhole", False)]    # gear and the cover artist are gone without trace
    assert records[0]["n_songs"] == 2


# --- the date shape: a typo'd date is the silent kind of wrong -------------------------------

@pytest.mark.parametrize("bad", [
    "2025-13-01",      # the example the comment on _DATE cites, which \d{2} accepted
    "2025-00-01",
    "2025-06-32",
    "0000-00-00",
    "2025-06-14\n",    # `$` matches before a trailing newline in Python's re
])
def test_a_malformed_drop_date_is_rejected(tmp_path, bad):
    diag = _corpus_failure(tmp_path, json.dumps({"drop_dates": {bad: "a typo"}}))
    assert "does not match" in diag.summary


@pytest.mark.parametrize("bad", ["2025-13-01", "2025-06-14\n", "June 14 2025", "2025-06-14T00:00"])
def test_a_malformed_override_date_is_rejected(tmp_path, bad):
    diag = _corpus_failure(
        tmp_path, json.dumps({"date_overrides": {"x1": {"date": bad, "why": "evidence"}}}))
    assert "does not match" in diag.summary


def test_a_real_date_still_loads(tmp_path):
    """the bound is on the shape, not on the calendar: day 31 is legal in every month, because
    catching 2025-02-31 wants a calendar and the date that matches no show gets noticed."""
    pack = _load_or_render(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "corpus.json": '{"drop_dates": {"2025-01-31": "x", "2025-12-01": "y"}}'})
    assert sorted(pack.corpus.drop_dates) == ["2025-01-31", "2025-12-01"]


# --- the merge half of drop_dates ------------------------------------------------------------

def test_merge_policy_carries_the_drop_dates_too(tmp_path):
    """refusing a date in the archive parser alone was not enough: the other sources carry the
    same night, and the merge picked one of those copies up instead."""
    pack = _load_or_render(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB, "corpus.json": CORPUS})
    assert pack.merge_policy().drop_dates == pack.archive_policy().drop_dates
    assert pack.merge_policy().drop_dates == frozenset({"2025-10-31"})
    # ranks and completeness are config, not pack data, and pass straight through
    assert pack.merge_policy(ranks={"description": 9}, complete_frac=0.5).ranks == {
        "description": 9}


# --- Rule.compiled means two things, so Rule decides which -----------------------------------

def test_a_rule_matches_against_the_form_the_runtime_hands_it(tmp_path):
    """crossing the two forms fails silently in both directions, so the rule picks, not the
    caller. A corpus fragment's `\\s+` can never match squashed text -- the space is gone."""
    pack = _load_or_render(tmp_path, **{
        "pack.json": IDENTITY, "vocabulary.json": VOCAB,
        "classifiers.json": '{"non_song": [{"pattern": "drum solo", "why": "x"}]}',
        "corpus.json": '{"junk_patterns": [{"pattern": "home\\\\s+team", "why": "y"}]}'})
    squash = pack.normalizer.squash
    corpus_rule, classifier_rule = pack.corpus.junk[0], pack.rules[0]

    assert corpus_rule.wrapped is True and classifier_rule.wrapped is False
    assert corpus_rule.reaches("Home Team", squash)          # sees the title as written
    assert not corpus_rule.reaches("Homeward Bound", squash)
    # the classifier is matched against squashed text, which is why its pattern has no space
    assert classifier_rule.reaches("Drum Solo", squash) is False
    assert pack.normalizer.is_non_song("Drumsolo") is False

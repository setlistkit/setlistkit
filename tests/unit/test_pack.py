# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for loading a band pack into a data-driven Normalizer.

Covers the three validating layers (schema shape, the free-floating-justification domain
rule, regex compilation), the required-file and malformed-JSON failures, and that the built
normalizer actually applies the pack's aliases, patterns and protected titles. Every failure
is checked for a rendered caret, because a pack error that can't point at the problem is the
thing this whole subsystem exists to avoid.
"""

import pytest

from setlistkit.catalog.pack import Pack, load_pack
from setlistkit.diagnostics import DiagnosticError, render


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
            "protected.json": '["ATL", "TLH", "NYC"]',
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

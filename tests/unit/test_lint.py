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

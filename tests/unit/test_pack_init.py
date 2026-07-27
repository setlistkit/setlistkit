# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `slkit pack init` -- scaffolding a valid pack without clobbering.

The property that matters is composition with suggest-vocab in EITHER order: init then suggest, or
suggest then init. In both, lint must end up happy, and in the second, init must not touch the
vocabulary suggest-vocab already wrote.
"""

import json

from setlistkit.catalog.pack import load_pack
from setlistkit.catalog.scaffold import starter_pack
from setlistkit.cli.main import EXIT_OK, main


def test_starter_pack_is_a_valid_minimal_pack(tmp_path):
    """Every scaffolded file written out, then load_pack accepts the directory."""
    for name, content in starter_pack("um").items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    pack = load_pack(tmp_path)          # raises DiagnosticError if the scaffold is invalid
    assert pack.name == "um"
    assert pack.vocabulary == ()        # empty dictionary is legal


def test_starter_pack_json_omits_band_name():
    """band_name is deliberately absent -- no guess means no bogus band filter."""
    identity = json.loads(starter_pack("um")["pack.json"])
    assert "band_name" not in identity
    assert identity["name"] == "um"
    assert identity["version"]


def test_init_scaffolds_an_empty_dir(tmp_path, capsys):
    dest = tmp_path / "um-pack"
    assert main(["pack", "init", "--pack", str(dest)]) == EXIT_OK
    for name in ("pack.json", "vocabulary.json", "aliases.json", "protected.json",
                 "classifiers.json", "README.md"):
        assert (dest / name).is_file()
    assert json.loads((dest / "pack.json").read_text())["name"] == "um-pack"
    assert "scaffolded" in capsys.readouterr().out


def test_init_never_clobbers_by_default(tmp_path, capsys):
    dest = tmp_path / "pack"
    dest.mkdir()
    (dest / "vocabulary.json").write_text('["Real Song"]', encoding="utf-8")
    assert main(["pack", "init", "--pack", str(dest)]) == EXIT_OK
    # the hand-written vocabulary survives untouched...
    assert json.loads((dest / "vocabulary.json").read_text()) == ["Real Song"]
    # ...and the missing files got filled in
    assert (dest / "pack.json").is_file()
    out = capsys.readouterr().out
    assert "untouched" in out and "vocabulary.json" in out


def test_init_force_overwrites(tmp_path):
    dest = tmp_path / "pack"
    dest.mkdir()
    (dest / "pack.json").write_text('{"stale": true}', encoding="utf-8")
    assert main(["pack", "init", "--pack", str(dest), "--force"]) == EXIT_OK
    assert json.loads((dest / "pack.json").read_text())["name"] == "pack"


def test_pack_init_is_idempotent(tmp_path, capsys):
    dest = tmp_path / "pack"
    assert main(["pack", "init", "--pack", str(dest)]) == EXIT_OK
    capsys.readouterr()
    assert main(["pack", "init", "--pack", str(dest)]) == EXIT_OK
    assert "nothing to do" in capsys.readouterr().out


def test_suggest_then_init_then_lint(tmp_path):
    """The reverse workflow: suggest-vocab writes a real vocabulary, init fills the rest without
    disturbing it, and the result lints clean."""
    src = tmp_path / "titles.json"
    src.write_text(json.dumps(
        ["01 1348", "1348", "Belive The Lie", "Believe The Lie", "(Outro)", "Ophelia"]),
        encoding="utf-8")
    dest = tmp_path / "pack"

    assert main(["pack", "suggest-vocab", "--from", str(src), "--out", str(dest)]) == EXIT_OK
    vocab_before = json.loads((dest / "vocabulary.json").read_text())
    assert vocab_before                                 # suggest wrote real content

    assert main(["pack", "init", "--pack", str(dest)]) == EXIT_OK
    assert json.loads((dest / "vocabulary.json").read_text()) == vocab_before   # untouched

    load_pack(dest)                                     # a valid pack now, missing files and all


def test_init_then_suggest_no_force_needed(tmp_path):
    """The forward workflow: init drops an empty [] vocabulary, and suggest-vocab may fill it
    without --force because an empty placeholder is not treated as real content."""
    src = tmp_path / "titles.json"
    src.write_text(json.dumps(["Ophelia", "01 Ophelia"]), encoding="utf-8")
    dest = tmp_path / "pack"

    assert main(["pack", "init", "--pack", str(dest)]) == EXIT_OK
    assert main(["pack", "suggest-vocab", "--from", str(src), "--out", str(dest)]) == EXIT_OK
    assert "Ophelia" in json.loads((dest / "vocabulary.json").read_text())

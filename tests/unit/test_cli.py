# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the slkit CLI: the one place diagnostics become exit codes."""

import pytest

from setlistkit.cli.main import EXIT_DIAGNOSTIC, EXIT_OK, main

GOOD = 'data_root = "data"\nuser_agent = "famoe.ly nightly (me@example.com)"\n'
SENTINEL = 'data_root = "data"\nuser_agent = "CHANGE-ME (setlistkit; you@example.com)"\n'


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_no_command_prints_help(capsys):
    assert main([]) == EXIT_OK
    assert "usage: slkit" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == EXIT_OK
    assert "slkit" in capsys.readouterr().out


def test_config_show(tmp_path, capsys):
    target = _write(tmp_path / "slkit.toml", GOOD)
    assert main(["--config", str(target), "show", "config"]) == EXIT_OK
    out = capsys.readouterr().out
    assert str(target) in out
    assert "user_agent" in out


def test_config_show_resolves_pack_against_config_dir(tmp_path, capsys):
    # A relative [catalog] pack must anchor at the config file's directory, not the process cwd --
    # the same rule the working commands follow -- so what `config show` prints is the pack they
    # will act on.
    target = _write(tmp_path / "slkit.toml", GOOD + '[catalog]\npack = "mypack"\n')
    assert main(["--config", str(target), "show", "config"]) == EXIT_OK
    out = capsys.readouterr().out
    assert f"pack        : {(tmp_path / 'mypack').resolve()}" in out


def test_config_show_pack_not_set(tmp_path, capsys):
    target = _write(tmp_path / "slkit.toml", GOOD)  # no [catalog] pack
    assert main(["--config", str(target), "show", "config"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "pack        : (not set" in out


def test_config_check_ok(tmp_path, capsys):
    target = _write(tmp_path / "slkit.toml", GOOD)
    assert main(["--config", str(target), "check"]) == EXIT_OK
    assert "identifies itself" in capsys.readouterr().out


def test_config_check_sentinel_aborts(tmp_path, capsys):
    target = _write(tmp_path / "slkit.toml", SENTINEL)
    assert main(["--config", str(target), "check"]) == EXIT_DIAGNOSTIC
    err = capsys.readouterr().err
    assert "error:" in err
    assert "user_agent" in err


def test_config_show_flags_sentinel(tmp_path, capsys):
    target = _write(tmp_path / "slkit.toml", SENTINEL)
    assert main(["--config", str(target), "show", "config"]) == EXIT_OK
    assert "placeholder" in capsys.readouterr().out


def test_missing_config_is_diagnostic(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "ghost.toml"), "show", "config"]) == EXIT_DIAGNOSTIC
    assert "config file not found" in capsys.readouterr().err

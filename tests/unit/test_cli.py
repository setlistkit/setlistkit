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
    assert main(["--config", str(target), "config", "show"]) == EXIT_OK
    out = capsys.readouterr().out
    assert str(target) in out
    assert "user_agent" in out


def test_config_check_ok(tmp_path, capsys):
    target = _write(tmp_path / "slkit.toml", GOOD)
    assert main(["--config", str(target), "config", "check"]) == EXIT_OK
    assert "identifies itself" in capsys.readouterr().out


def test_config_check_sentinel_aborts(tmp_path, capsys):
    target = _write(tmp_path / "slkit.toml", SENTINEL)
    assert main(["--config", str(target), "config", "check"]) == EXIT_DIAGNOSTIC
    err = capsys.readouterr().err
    assert "error:" in err
    assert "user_agent" in err


def test_config_show_flags_sentinel(tmp_path, capsys):
    target = _write(tmp_path / "slkit.toml", SENTINEL)
    assert main(["--config", str(target), "config", "show"]) == EXIT_OK
    assert "placeholder" in capsys.readouterr().out


def test_missing_config_is_diagnostic(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "ghost.toml"), "config", "show"]) == EXIT_DIAGNOSTIC
    assert "config file not found" in capsys.readouterr().err

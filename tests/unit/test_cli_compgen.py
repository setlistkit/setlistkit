# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `slkit compgen`, which writes a shell completion script.

The script is generated from the parser, not from a list kept by hand, so the thing worth testing
is that the generation still happens and still covers the tree. A hand-kept completion list is
exactly what this command exists to avoid: it goes stale one command at a time, and nothing fails.
"""

import pytest
import shtab

from setlistkit.cli.main import EXIT_DIAGNOSTIC, EXIT_OK, main


@pytest.mark.parametrize("shell", shtab.SUPPORTED_SHELLS)
def test_compgen_writes_a_script_for_every_shell_the_library_supports(shell, capsys):
    """Asked of shtab rather than of a list here, so a shell it adds is covered without an edit."""
    assert main(["compgen", shell]) == EXIT_OK
    assert capsys.readouterr().out.strip()


def test_compgen_reports_a_shell_the_library_cannot_write(capsys):
    """An unsupported shell renders as a diagnostic, not a traceback."""
    assert main(["compgen", "fish"]) == EXIT_DIAGNOSTIC
    assert "fish" in capsys.readouterr().err


def test_compgen_names_the_deepest_command_in_the_tree(capsys):
    """A sub-sub-command reaches the script, so the walk is not stopping at depth one."""
    assert main(["compgen", "bash"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "suggest-vocab" in out
    assert "tapemeasure" in out


def test_compgen_reads_the_shell_from_the_environment(monkeypatch, capsys):
    """No argument: the default comes from $SHELL, so `eval $(slkit compgen)` works unaided."""
    monkeypatch.setenv("SHELL", "/bin/zsh")
    assert main(["compgen"]) == EXIT_OK
    assert "#compdef" in capsys.readouterr().out


def test_compgen_falls_back_when_the_environment_names_no_shell(monkeypatch, capsys):
    """An empty $SHELL still produces a script rather than an error."""
    monkeypatch.setenv("SHELL", "")
    assert main(["compgen"]) == EXIT_OK
    assert capsys.readouterr().out.strip()


def test_compgen_needs_no_config(tmp_path, monkeypatch, capsys):
    """The command reads the parser only. A deployment with no config still completes."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLKIT_CONFIG", raising=False)
    assert main(["compgen", "bash"]) == EXIT_OK
    assert capsys.readouterr().out.strip()

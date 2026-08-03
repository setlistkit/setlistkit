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


def test_compgen_lists_each_command_by_its_one_line_summary(capsys):
    """A completion menu gets the short line, not the paragraph `--help` prints.

    shtab reads ``description`` and never ``help``, and it does not wrap, so a paragraph reaches
    the menu cut mid-word.
    """
    assert main(["compgen", "zsh"]) == EXIT_OK
    out = capsys.readouterr().out

    assert '"bootstrap:carry an empty deployment' in out
    assert "The run creates the store" not in out


def test_compgen_leaves_the_real_parser_alone(capsys):
    """The swap happens on a throwaway parser, so `--help` still prints the paragraph."""
    main(["compgen", "zsh"])
    capsys.readouterr()

    with pytest.raises(SystemExit):
        main(["bootstrap", "--help"])
    # Joined, because help wraps and the phrase spans a line break.
    assert "The run creates the store" in " ".join(capsys.readouterr().out.split())


def test_compgen_completes_a_path_argument_with_filenames(capsys):
    """`slkit --config <TAB>` offers files. Without this the shell offers a command name."""
    assert main(["compgen", "zsh"]) == EXIT_OK
    assert ':config:_files"' in capsys.readouterr().out


# Quoted at the end, because `_files` is a prefix of `_files -/` and would match either.
@pytest.mark.parametrize("metavar,spec", [("PATH", '_files"'), ("DIR", '_files -/"')])
def test_compgen_reads_the_metavar_to_decide_what_a_path_argument_takes(metavar, spec, capsys):
    """PATH offers files and DIR offers directories, from the metavar already written."""
    assert main(["compgen", "zsh"]) == EXIT_OK
    assert spec in capsys.readouterr().out, f"no {metavar} argument completes with {spec}"


def test_compgen_reports_a_shell_the_library_cannot_write(capsys):
    """An unsupported shell renders as a diagnostic, not a traceback."""
    assert main(["compgen", "fish"]) == EXIT_DIAGNOSTIC
    assert "fish" in capsys.readouterr().err


def test_compgen_names_the_deepest_command_in_the_tree(capsys):
    """A sub-sub-command reaches the script, so the walk is not stopping at depth one."""
    assert main(["compgen", "bash"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "missing" in out
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

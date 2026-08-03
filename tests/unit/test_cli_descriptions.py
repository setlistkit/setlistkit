# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Every command explains itself before it lists its flags.

``help=`` is the one line a command gets in its PARENT's list. ``description=`` is the leader
paragraph the command prints for its own ``--help``. A parser that sets only the first shows a
usage line and a wall of flags, which tells a reader what they may type and never what the
command does.

The test walks the real parser tree, so a command added later is covered without an edit here.
"""

import argparse

import pytest

from setlistkit.cli.main import _build_parser

# Short enough to read, long enough to say something. A one-line description is the failure this
# test exists to catch, wearing a different hat.
MIN_WORDS = 12


def _branches(parser, path="slkit"):
    """Every (path, parser, summary) triple under ``parser``, depth first.

    ``summary`` is the line the PARENT prints beside the command's name. argparse keeps it on the
    parent's pseudo-action, not on the child, so it is read from there.
    """
    for action in parser._actions:  # pylint: disable=protected-access
        if not isinstance(action, argparse._SubParsersAction):  # pylint: disable=protected-access
            continue
        summaries = {pseudo.dest: pseudo.help for pseudo in action._choices_actions}  # pylint: disable=protected-access
        for name, child in action.choices.items():
            yield f"{path} {name}", child, summaries.get(name)
            yield from _branches(child, f"{path} {name}")


ALL_BRANCHES = list(_branches(_build_parser()))
IDS = [path for path, _, _ in ALL_BRANCHES]


def test_the_tree_is_not_empty():
    """A walk that finds nothing would pass every test below without checking anything."""
    assert len(ALL_BRANCHES) > 10


@pytest.mark.parametrize("path,parser,summary", ALL_BRANCHES, ids=IDS)
def test_every_command_has_a_leader_paragraph(path, parser, summary):
    """The command says what it does, in prose, above its flags."""
    del summary
    description = (parser.description or "").strip()
    assert description, f"`{path} --help` prints no description"
    assert len(description.split()) >= MIN_WORDS, f"`{path}` describes itself too thinly"


@pytest.mark.parametrize("path,parser,summary", ALL_BRANCHES, ids=IDS)
def test_every_command_keeps_its_one_line_summary(path, parser, summary):
    """The parent's list still names the command. A description does not replace that line."""
    del parser
    assert (summary or "").strip(), f"`{path}` is unlabeled in its parent's command list"

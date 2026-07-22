# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
r"""The suite must not read the environment it happens to be run from.

Python 3.14 colorizes argparse help, and can_colorize() honors FORCE_COLOR, so anything that
exports it turns the "usage: slkit" test_cli asserts on into

    \x1b[1;34musage: \x1b[0m\x1b[1;35mslkit\x1b[0m [\x1b[32m-h\x1b[0m]

Nothing in a normal shell sets it, which is why this passed by hand for months. Agent harnesses
and CI both set it routinely, and `make check` only survived because whoever wrote the command
line remembered to prefix NO_COLOR=1 FORCE_COLOR=. That is a fix in the caller, and it stops
working the moment someone runs pytest directly.

Pinned here, the guarantee travels with the tests instead. The environment is read when the help
is formatted, not when the parser is built, so patching per test is enough.
"""

import pytest


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """Neutralise every ambient signal that changes what the CLI renders."""
    monkeypatch.setenv("NO_COLOR", "1")
    for leaked in ("FORCE_COLOR", "CLICOLOR", "CLICOLOR_FORCE", "PYTHON_COLORS"):
        monkeypatch.delenv(leaked, raising=False)

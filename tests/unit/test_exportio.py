# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `exportio.resolve_window`: which of a flag or a configured window wins, per
endpoint, with no store and no TOML file -- a bare `Config` built directly is enough, since
`resolve_window` only ever reads `config.section("reports", ...)`.
"""

from pathlib import Path

import pytest

from setlistkit.cli.exportio import resolve_window
from setlistkit.config import Config


def _config(reports=None):
    return Config(data_root=Path("/unused"), user_agent="x", source_path=Path("/unused.toml"),
                  raw={"reports": reports or {}})


def test_flags_win_over_a_configured_window_per_endpoint():
    config = _config({"songbook": {"window": {"since_back": "P1M"}}})
    since, until = resolve_window(config, "songbook", since_flag=None, until_flag="2020-06-01",
                                  first="2020-01-01", last="2020-07-01")
    # since comes from config (anchor 2020-07-01, minus a calendar month); until comes from the
    # flag, overriding what the config would otherwise leave as "the anchor itself".
    assert since == "2020-06-01"
    assert until == "2020-06-01"


def test_flags_are_used_when_no_window_is_configured():
    config = _config()
    since, until = resolve_window(config, "songbook", since_flag="2020-01-01",
                                  until_flag="2020-02-01", first="2020-01-01", last="2020-07-01")
    assert (since, until) == ("2020-01-01", "2020-02-01")


def test_no_flags_and_no_config_resolves_to_an_open_window():
    """The window this CLI has always had by default: no `--since`/`--until` means unbounded."""
    config = _config()
    result = resolve_window(config, "songbook", since_flag=None, until_flag=None,
                            first="2020-01-01", last="2020-07-01")
    assert result == (None, None)


def test_a_configured_window_is_used_when_no_flags_are_given():
    config = _config({"songbook": {"window": {"since_from": "year"}}})
    since, until = resolve_window(config, "songbook", since_flag=None, until_flag=None,
                                  first="2020-01-01", last="2020-07-01")
    assert (since, until) == ("2020-01-01", "2020-07-01")


def test_a_malformed_flag_date_is_refused():
    """Same check `--since`/`--until` have always gone through -- a config-aware window does not
    get to relax it."""
    config = _config()
    with pytest.raises(ValueError, match="--until"):
        resolve_window(config, "songbook", since_flag=None, until_flag="2020",
                       first="2020-01-01", last="2020-07-01")

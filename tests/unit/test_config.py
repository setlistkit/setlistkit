# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for config resolution, parsing, data_root handling, and network identity."""

import os
from pathlib import Path

import pytest

from setlistkit.config import (
    SENTINEL_USER_AGENT,
    Config,
    load_config,
    require_network_identity,
    resolve_config_path,
)
from setlistkit.diagnostics import DiagnosticError

GOOD = 'data_root = "data"\nuser_agent = "famoe.ly nightly (me@example.com)"\n'


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- resolution order -------------------------------------------------------

def test_explicit_path_wins(tmp_path):
    target = _write(tmp_path / "custom.toml", GOOD)
    # A competing local file must lose to the explicit path.
    _write(tmp_path / "slkit.toml", GOOD)
    cfg = load_config(str(target), env={"SLKIT_CONFIG": str(tmp_path / "slkit.toml")},
                      cwd=tmp_path)
    assert cfg.source_path == target.resolve()


def test_env_var_used_when_no_explicit(tmp_path):
    target = _write(tmp_path / "env.toml", GOOD)
    cfg = load_config(env={"SLKIT_CONFIG": str(target)}, cwd=tmp_path)
    assert cfg.source_path == target.resolve()


def test_env_var_beats_local(tmp_path):
    env_file = _write(tmp_path / "env.toml", GOOD)
    _write(tmp_path / "slkit.toml", GOOD)  # would win if env were ignored
    cfg = load_config(env={"SLKIT_CONFIG": str(env_file)}, cwd=tmp_path)
    assert cfg.source_path == env_file.resolve()


def test_local_slkit_toml(tmp_path):
    target = _write(tmp_path / "slkit.toml", GOOD)
    cfg = load_config(env={}, cwd=tmp_path)
    assert cfg.source_path == target.resolve()


def test_xdg_fallback(tmp_path):
    xdg = tmp_path / "xdg"
    target = _write(xdg / "setlistkit" / "config.toml", GOOD)
    cfg = load_config(env={"XDG_CONFIG_HOME": str(xdg)}, cwd=tmp_path / "empty")
    assert cfg.source_path == target.resolve()


def test_local_beats_xdg(tmp_path):
    local = _write(tmp_path / "slkit.toml", GOOD)
    _write(tmp_path / "xdg" / "setlistkit" / "config.toml", GOOD)
    cfg = load_config(env={"XDG_CONFIG_HOME": str(tmp_path / "xdg")}, cwd=tmp_path)
    assert cfg.source_path == local.resolve()


# --- error paths ------------------------------------------------------------

def test_no_config_found(tmp_path):
    with pytest.raises(DiagnosticError) as excinfo:
        load_config(env={"XDG_CONFIG_HOME": str(tmp_path / "nope")},
                    cwd=tmp_path / "empty")
    assert "no configuration file found" in excinfo.value.diagnostic.summary


def test_explicit_missing_is_specific_error(tmp_path):
    with pytest.raises(DiagnosticError) as excinfo:
        resolve_config_path(str(tmp_path / "ghost.toml"), env={}, cwd=tmp_path)
    assert "config file not found" in excinfo.value.diagnostic.summary


def test_malformed_toml(tmp_path):
    target = _write(tmp_path / "slkit.toml", "data_root = \nnope")
    with pytest.raises(DiagnosticError) as excinfo:
        load_config(env={}, cwd=tmp_path)
    assert "not valid TOML" in excinfo.value.diagnostic.summary


def test_missing_required_keys(tmp_path):
    _write(tmp_path / "slkit.toml", 'user_agent = "x (a@b.c)"\n')  # no data_root
    with pytest.raises(DiagnosticError) as excinfo:
        load_config(env={}, cwd=tmp_path)
    assert "data_root" in excinfo.value.diagnostic.summary


def test_empty_data_root_rejected(tmp_path):
    _write(tmp_path / "slkit.toml", 'data_root = "  "\nuser_agent = "x (a@b.c)"\n')
    with pytest.raises(DiagnosticError) as excinfo:
        load_config(env={}, cwd=tmp_path)
    assert "data_root is empty" in excinfo.value.diagnostic.summary


# --- data_root resolution ---------------------------------------------------

def test_relative_data_root_anchored_at_config_dir(tmp_path):
    # data_root = "data" is relative; it must anchor at the config file's directory, not
    # the process cwd. Load by explicit path and point cwd elsewhere to prove that.
    target = _write(tmp_path / "slkit.toml", GOOD)  # data_root = "data"
    cfg = load_config(str(target), env={}, cwd=tmp_path / "elsewhere")
    assert cfg.data_root == (tmp_path / "data").resolve()


def test_tilde_data_root_expands_home(tmp_path):
    unique = "~/sktest_home_marker_9137"
    _write(tmp_path / "slkit.toml",
           f'data_root = "{unique}"\nuser_agent = "x (a@b.c)"\n')
    cfg = load_config(env={}, cwd=tmp_path)
    assert cfg.data_root == Path(os.path.expanduser(unique)).resolve()


def test_absolute_data_root_preserved(tmp_path):
    abs_root = tmp_path / "abs_data"
    _write(tmp_path / "slkit.toml",
           f'data_root = "{abs_root}"\nuser_agent = "x (a@b.c)"\n')
    cfg = load_config(env={}, cwd=tmp_path)
    assert cfg.data_root == abs_root.resolve()


# --- sections and identity --------------------------------------------------

def test_section_access(tmp_path):
    _write(tmp_path / "slkit.toml", GOOD + "[sources.setlistfm]\nmbid = \"abc\"\n")
    cfg = load_config(env={}, cwd=tmp_path)
    assert cfg.section("sources", "setlistfm") == {"mbid": "abc"}
    assert cfg.section("does", "not", "exist") == {}


def test_sentinel_detection():
    cfg = Config(data_root=Path("/x"), user_agent=SENTINEL_USER_AGENT,
                 source_path=Path("/x/slkit.toml"))
    assert cfg.user_agent_is_sentinel is True


def test_require_network_identity_aborts_on_sentinel():
    cfg = Config(data_root=Path("/x"), user_agent=SENTINEL_USER_AGENT,
                 source_path=Path("/x/slkit.toml"))
    with pytest.raises(DiagnosticError) as excinfo:
        require_network_identity(cfg)
    assert "user_agent" in excinfo.value.diagnostic.summary


def test_require_network_identity_aborts_on_empty():
    cfg = Config(data_root=Path("/x"), user_agent="   ",
                 source_path=Path("/x/slkit.toml"))
    with pytest.raises(DiagnosticError) as excinfo:
        require_network_identity(cfg)
    assert "empty" in excinfo.value.diagnostic.summary


def test_require_network_identity_passes_when_set():
    cfg = Config(data_root=Path("/x"), user_agent="famoe.ly (me@example.com)",
                 source_path=Path("/x/slkit.toml"))
    assert require_network_identity(cfg) is None

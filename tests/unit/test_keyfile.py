# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the api-key-file permission check."""

import stat

import pytest

from setlistkit.diagnostics import DiagnosticError
from setlistkit.sources.keyfile import read_api_key


def _write_key(tmp_path, text, mode):
    path = tmp_path / "setlistfm.api_key"
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def test_reads_a_locked_down_key(tmp_path):
    path = _write_key(tmp_path, "s3cret\n", 0o600)
    # test that a 0600 file returns the key with surrounding whitespace stripped
    assert read_api_key(path) == "s3cret"


def test_group_readable_is_allowed(tmp_path):
    path = _write_key(tmp_path, "s3cret", 0o640)
    # group access is advisory, not refused — only the world bit is a hard failure
    assert read_api_key(path) == "s3cret"


def test_world_readable_is_refused(tmp_path):
    path = _write_key(tmp_path, "s3cret", 0o644)
    with pytest.raises(DiagnosticError) as caught:
        read_api_key(path)
    diag = caught.value.diagnostic
    assert diag.is_error
    assert "world-readable" in diag.summary
    # the fix names the file and the exact chmod, so the message is actionable
    assert f"chmod 600 {path}" in diag.detail


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(DiagnosticError) as caught:
        read_api_key(tmp_path / "nope.api_key")
    assert "not found" in caught.value.diagnostic.summary


def test_empty_file_is_refused(tmp_path):
    path = _write_key(tmp_path, "   \n", 0o600)
    with pytest.raises(DiagnosticError) as caught:
        read_api_key(path)
    assert "empty" in caught.value.diagnostic.summary


def test_expands_user_home(tmp_path, monkeypatch):
    _write_key(tmp_path, "fromhome", 0o600)
    monkeypatch.setenv("HOME", str(tmp_path))
    # test that a ~-relative path resolves against HOME rather than being read literally
    assert read_api_key("~/setlistfm.api_key") == "fromhome"


def test_world_readable_check_ignores_owner_and_group_bits(tmp_path):
    # a fully-open 0666 still trips only on the world-read bit, and it must trip
    path = _write_key(tmp_path, "s3cret", stat.S_IRUSR | stat.S_IWUSR | stat.S_IROTH)
    with pytest.raises(DiagnosticError):
        read_api_key(path)

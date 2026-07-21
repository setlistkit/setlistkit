# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the store and dump CLI commands."""

from setlistkit.cli.main import EXIT_OK, main
from setlistkit.store.migrations import MIGRATIONS

LATEST_SCHEMA = max(m.version for m in MIGRATIONS)

CONFIG = 'data_root = "state"\nuser_agent = "x (a@b.c)"\n'


def _cfg(tmp_path):
    path = tmp_path / "slkit.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return str(path)


def test_store_init_then_status(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    assert main(["--config", cfg, "store", "init"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "applied migrations: 1, 2" in out
    assert f"schema version {LATEST_SCHEMA}" in out

    assert main(["--config", cfg, "store", "status"]) == EXIT_OK
    out = capsys.readouterr().out
    assert f"version {LATEST_SCHEMA}" in out
    assert f"schema_migrations: {LATEST_SCHEMA} rows" in out


def test_store_status_reports_what_is_in_the_raw_cache(tmp_path, capsys):
    """It already printed where the cache is and then said nothing about it.

    This is the answer to "how far has the pull got" that costs nothing and is safe to run in a
    loop, which `slkit pull -n` is not.
    """
    from setlistkit.store.raw_cache import RawCache
    cache = RawCache(tmp_path / "state")
    cache.put("archive_org", "a", b"x" * 2048)
    cache.put("archive_org", "b", b"y" * 1024)
    assert main(["--config", _cfg(tmp_path), "store", "status"]) == EXIT_OK
    assert "archive_org: 2 entries, 3.0 KB" in capsys.readouterr().out


def test_store_status_before_init_flags_uninitialized(tmp_path, capsys):
    assert main(["--config", _cfg(tmp_path), "store", "status"]) == EXIT_OK
    assert "not initialized" in capsys.readouterr().out
    # test that a read-only status did not create the database file
    assert not (tmp_path / "state" / "setlistkit.sqlite").is_file()


def test_dump_after_init(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    main(["--config", cfg, "store", "init"])
    capsys.readouterr()
    assert main(["--config", cfg, "dump"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "setlistkit.sqlite — contents" in out
    assert "setlistkit_version" in out

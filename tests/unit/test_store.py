# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the Store: init, provenance, counts, and the plain-text dump."""

import pytest

from setlistkit.store.db import Store, _ident
from setlistkit.store.migrations import MIGRATIONS

# Read off the ledger rather than hard-coded, so adding a migration is one line in migrations.py
# instead of a hunt through the suite for every place a version number was written down.
EXPECTED_MIGRATIONS = [(m.version, m.name) for m in MIGRATIONS]
LATEST_SCHEMA = max(version for version, _ in EXPECTED_MIGRATIONS)


def test_init_applies_every_migration_in_order(tmp_path):
    data_root = tmp_path / "data"
    with Store(data_root) as store:
        assert store.init() == [version for version, _ in EXPECTED_MIGRATIONS]
        assert store.db_path.is_file()
        assert store.schema_version() == LATEST_SCHEMA
        assert "meta" in store.table_names()
        assert "schema_migrations" in store.table_names()


def test_init_is_idempotent(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        # test that a second init applies nothing new
        assert store.init() == []
        assert store.schema_version() == LATEST_SCHEMA


def test_init_stamps_provenance(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        meta = dict(store.conn.execute("SELECT key, value FROM meta").fetchall())
        assert meta["setlistkit_version"]
        assert meta["created_at"]


def test_created_at_is_preserved_across_reinit(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        first = store.conn.execute(
            "SELECT value FROM meta WHERE key='created_at'").fetchone()[0]
        store.init()
        second = store.conn.execute(
            "SELECT value FROM meta WHERE key='created_at'").fetchone()[0]
        assert first == second


def test_table_counts(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        counts = store.table_counts()
        assert counts["schema_migrations"] == LATEST_SCHEMA
        assert counts["meta"] >= 2


def test_dump_is_deterministic_and_drops_volatile_columns(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        first = store.dump()
        second = store.dump()
        assert first == second
        # applied_at is a volatile column and must not appear
        assert "applied_at" not in first
        # real content is present
        assert "setlistkit_version" in first
        assert "baseline" in first


def test_dump_of_empty_db_has_no_tables(tmp_path):
    # Opening without init() creates the file but runs no migrations.
    with Store(tmp_path) as store:
        assert store.table_names() == []
        assert store.dump().strip() == "# setlistkit.sqlite — contents"


def test_ident_rejects_non_identifiers():
    assert _ident("meta") == "meta"
    with pytest.raises(ValueError):
        _ident("meta; DROP TABLE meta")


def test_dump_and_counts_handle_reserved_word_identifiers(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        # A table and column named with SQL keywords must not break dump/table_counts:
        # identifiers are double-quoted, so this is valid SQL rather than a syntax error.
        store.conn.execute('CREATE TABLE "select"("order" TEXT, x TEXT)')
        store.conn.execute('INSERT INTO "select"("order", x) VALUES(?, ?)', ("a", "b"))
        assert store.table_counts()["select"] == 1
        out = store.dump()
        assert "## select (1 rows)" in out
        assert "a | b" in out

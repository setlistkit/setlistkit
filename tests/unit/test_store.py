# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the Store: init, provenance, counts, and the plain-text dump."""

import pytest

from setlistkit.store.db import Store, _ident


def test_init_creates_db_and_applies_baseline(tmp_path):
    data_root = tmp_path / "data"
    with Store(data_root) as store:
        assert store.init() == [1]
        assert store.db_path.is_file()
        assert store.schema_version() == 1
        assert "meta" in store.table_names()
        assert "schema_migrations" in store.table_names()


def test_init_is_idempotent(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        # test that a second init applies nothing new
        assert store.init() == []
        assert store.schema_version() == 1


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
        assert counts["schema_migrations"] == 1
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

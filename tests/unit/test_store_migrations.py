# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the schema migration runner."""

import sqlite3

import pytest

from setlistkit.store.migrations import (
    Migration,
    applied_versions,
    apply_pending,
    schema_version,
)


@pytest.fixture(name="conn")
def _conn_fixture():
    # Same posture the Store uses: autocommit off so we drive our own transactions.
    conn = sqlite3.connect(":memory:", isolation_level=None)
    try:
        yield conn
    finally:
        conn.close()


def _make(version, name, *creates, fail=False):
    def apply(conn):
        for stmt in creates:
            conn.execute(stmt)
        if fail:
            raise RuntimeError("boom")
    return Migration(version, name, apply)


def test_applies_all_pending_and_records_them(conn):
    migs = [_make(1, "one", "CREATE TABLE a(x)"), _make(2, "two", "CREATE TABLE b(y)")]
    assert apply_pending(conn, migs) == [1, 2]
    assert schema_version(conn) == 2
    recorded = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
    assert recorded == {1, 2}


def test_is_idempotent(conn):
    migs = [_make(1, "one", "CREATE TABLE a(x)")]
    assert apply_pending(conn, migs) == [1]
    # test that a second run with nothing new applies nothing
    assert apply_pending(conn, migs) == []
    assert schema_version(conn) == 1


def test_applies_in_version_order_regardless_of_list_order(conn):
    order = []
    migs = [
        Migration(2, "two", lambda c: order.append(2)),
        Migration(1, "one", lambda c: order.append(1)),
        Migration(3, "three", lambda c: order.append(3)),
    ]
    apply_pending(conn, migs)
    assert order == [1, 2, 3]


def test_failed_migration_rolls_back_and_stops(conn):
    migs = [
        _make(1, "one", "CREATE TABLE a(x)"),
        _make(2, "bad", "CREATE TABLE b(y)", fail=True),
    ]
    with pytest.raises(RuntimeError):
        apply_pending(conn, migs)
    # v1 committed; v2 rolled back whole, so table b is gone and v2 is not recorded.
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "a" in tables
    assert "b" not in tables
    assert schema_version(conn) == 1


def test_fresh_db_is_version_zero(conn):
    assert schema_version(conn) == 0


def test_reading_version_does_not_create_the_ledger(conn):
    # test that a status-style read leaves an unmigrated DB untouched
    assert applied_versions(conn) == set()
    assert schema_version(conn) == 0
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "schema_migrations" not in tables

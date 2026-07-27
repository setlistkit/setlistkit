# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the schema migration runner."""

import sqlite3

import pytest

from setlistkit.store.migrations import (
    MIGRATIONS,
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


def test_a_store_predating_the_transition_fold_in_gets_the_column(conn):
    """Migration 3 repairs a database that recorded migration 1 before `transition` was in it.

    Built the way the real one was: migration 1's own DDL, minus the column, with the version
    recorded. The runner then has nothing left to do at version 1, which is exactly why the
    column could never arrive on its own.
    """
    conn.execute("CREATE TABLE shows(date TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE show_entries("
                 "  date TEXT NOT NULL, section TEXT NOT NULL, set_no INTEGER,"
                 "  position INTEGER NOT NULL, song TEXT NOT NULL,"
                 "  segue INTEGER NOT NULL, non_song INTEGER NOT NULL)")
    # `recordings` is not migration 3's business and is declared anyway, because a store that
    # really stopped at version 2 has it -- migration 1 created it. Later migrations run over this
    # same fixture, and a stand-in that only builds the tables one migration cares about stops
    # being a store at version 2 and becomes a store no version describes.
    conn.execute("CREATE TABLE recordings("
                 "  identifier TEXT PRIMARY KEY, date TEXT NOT NULL, uploader TEXT NOT NULL,"
                 "  audio_format TEXT NOT NULL, n_tracks INTEGER NOT NULL)")
    conn.execute("CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT NOT NULL,"
                 " applied_at TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_migrations VALUES(1, 'initial', '2026-07-01')")
    conn.execute("INSERT INTO schema_migrations VALUES(2, 'picks', '2026-07-25')")

    assert apply_pending(conn, MIGRATIONS) == [3, 4]
    assert "transition" in {row[1] for row in conn.execute("PRAGMA table_info(show_entries)")}


def test_a_store_that_already_has_the_transition_column_is_left_alone(conn):
    """A fresh store gets it from migration 1, and `ADD COLUMN` on an existing name is an error."""
    apply_pending(conn, MIGRATIONS)
    before = [row[1] for row in conn.execute("PRAGMA table_info(show_entries)")]
    assert "transition" in before
    # Idempotent under a re-run of the whole list, which is what a second `store.init()` does.
    assert apply_pending(conn, MIGRATIONS) == []
    assert [row[1] for row in conn.execute("PRAGMA table_info(show_entries)")] == before


def test_a_store_predating_the_second_tape_supplier_gets_the_source_column(conn):
    """Migration 4 repairs a mirror written when archive.org was the only place tapes came from.

    Built like the real one: recordings as migration 1 declares it, with 1..3 recorded, so the
    runner has nothing to do below 4. That is the same trap migration 3 documents -- folding the
    column into migration 1 would leave every store on disk without it and nothing to say so.
    """
    conn.execute("CREATE TABLE recordings("
                 "  identifier TEXT PRIMARY KEY, date TEXT NOT NULL, uploader TEXT NOT NULL,"
                 "  audio_format TEXT NOT NULL, n_tracks INTEGER NOT NULL)")
    conn.execute("CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT NOT NULL,"
                 " applied_at TEXT NOT NULL)")
    for version, name in ((1, "initial"), (2, "picks"), (3, "transition_column")):
        conn.execute("INSERT INTO schema_migrations VALUES(?, ?, '2026-07-01')", (version, name))

    assert apply_pending(conn, MIGRATIONS) == [4]
    assert "source" in {row[1] for row in conn.execute("PRAGMA table_info(recordings)")}


def test_an_unmigrated_tape_reports_no_source_rather_than_archive_org(conn):
    """The backfill default is '' on purpose, and this is the assertion that keeps it that way.

    Defaulting to 'archive_org' would have been the convenient guess and it was right for every
    row that existed at the time. It would also make a row whose provenance was never recorded
    indistinguishable from one that was measured, which is the whole reason the column exists.
    """
    conn.execute("CREATE TABLE recordings("
                 "  identifier TEXT PRIMARY KEY, date TEXT NOT NULL, uploader TEXT NOT NULL,"
                 "  audio_format TEXT NOT NULL, n_tracks INTEGER NOT NULL)")
    conn.execute("CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT NOT NULL,"
                 " applied_at TEXT NOT NULL)")
    for version in (1, 2, 3):
        conn.execute("INSERT INTO schema_migrations VALUES(?, 'x', '2026-07-01')", (version,))
    conn.execute("INSERT INTO recordings VALUES('old-tape', '2019-01-01', 'taper', 'Flac', 12)")

    apply_pending(conn, MIGRATIONS)
    assert conn.execute("SELECT source FROM recordings").fetchone()[0] == ""


def test_the_source_column_is_idempotent_under_a_second_init(conn):
    """`ADD COLUMN` on a name that exists is an error, and `store.init()` runs the whole list."""
    apply_pending(conn, MIGRATIONS)
    before = [row[1] for row in conn.execute("PRAGMA table_info(recordings)")]
    assert "source" in before
    assert apply_pending(conn, MIGRATIONS) == []
    assert [row[1] for row in conn.execute("PRAGMA table_info(recordings)")] == before

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Schema migrations: a numbered, forward-only ledger the store walks on every open.

We version the schema so later phases can add tables without a migration framework bolted
on after the fact. Each migration is a function plus a version; the runner applies the ones
not yet recorded, each in its own transaction, so a half-applied migration never survives.

The connection MUST be in autocommit mode (``isolation_level=None``). Python's sqlite3 does
not wrap DDL in its implicit transactions, so relying on the default would leave a failed
``CREATE TABLE`` committed. We drive BEGIN/COMMIT/ROLLBACK ourselves instead.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

# The one bookkeeping table this module owns. Kept as a literal in the queries below so no
# SQL is built by string interpolation.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def transaction(conn: sqlite3.Connection):
    """One BEGIN/COMMIT, ROLLBACK on any exception. Requires ``isolation_level=None``."""
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


@dataclass(frozen=True)
class Migration:
    """One numbered schema change: a version, a name, and the function that applies it."""

    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


def _m0001_baseline(conn: sqlite3.Connection) -> None:
    # meta is a tiny key/value slate for provenance (version the DB was created under, etc.).
    # The interesting tables land in their own phases; this one just gives us somewhere to
    # write "who made this and when" from the very first open.
    conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")


# Forward-only. Never edit a shipped migration; add the next number instead.
MIGRATIONS: list[Migration] = [
    Migration(1, "baseline", _m0001_baseline),
]


def _ledger_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    return row is not None


def _ensure_ledger(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations("
        "  version INTEGER PRIMARY KEY,"
        "  name TEXT NOT NULL,"
        "  applied_at TEXT NOT NULL)"
    )


def applied_versions(conn: sqlite3.Connection) -> set[int]:
    """Versions already recorded. A pure read: an unmigrated DB reports nothing rather than
    getting a bookkeeping table written into it, so `store status` stays side-effect free."""
    if not _ledger_exists(conn):
        return set()
    return {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}


def apply_pending(conn: sqlite3.Connection,
                  migrations: list[Migration] | None = None) -> list[int]:
    """Apply every migration not yet recorded, in version order. Return the versions applied.

    Idempotent: a second call with nothing new does no work and touches no rows.
    """
    if migrations is None:
        migrations = MIGRATIONS
    _ensure_ledger(conn)
    applied = applied_versions(conn)
    done: list[int] = []
    for migration in sorted(migrations, key=lambda m: m.version):
        if migration.version in applied:
            continue
        with transaction(conn):
            migration.apply(conn)
            conn.execute(
                "INSERT INTO schema_migrations(version, name, applied_at) VALUES(?, ?, ?)",
                (migration.version, migration.name, _now()),
            )
        done.append(migration.version)
    return done


def schema_version(conn: sqlite3.Connection) -> int:
    """Highest applied version, or 0 for a fresh database."""
    applied = applied_versions(conn)
    return max(applied) if applied else 0

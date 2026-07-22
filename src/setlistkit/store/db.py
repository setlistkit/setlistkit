# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The store: one SQLite database for everything derived, plus the raw file cache.

Everything the pipeline computes — catalog, model outputs, the picks ledger — lands in one
database at ``<data_root>/setlistkit.sqlite``. Raw fetch results stay as files next to it
(see :mod:`setlistkit.store.raw_cache`). The store owns the connection, the PRAGMAs, and the
migration walk, and it can render itself to plain text so derived state stays reviewable.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from . import corpus, recordings
from .migrations import apply_pending, schema_version, transaction
from .raw_cache import RawCache

DB_FILENAME = "setlistkit.sqlite"

# Columns whose value is a timestamp or run marker: real content, but they change every run
# and would bury the signal in a diff of `slkit dump`. We drop them from the dump only.
VOLATILE_COLUMNS = frozenset({
    "applied_at", "fetched_at", "created_at", "updated_at", "run_ts", "scored_asof",
})

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Dates are stored as YYYY-MM-DD strings, which sort lexicographically, so a range is a plain
# string comparison. That only holds for the full shape: `--until 2023` would compare below every
# date IN 2023 and quietly exclude the year somebody asked for. Checked rather than trusted.
_DATE_SHAPE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# table -> the SQL expression giving a row's date. A table absent from here has no date axis and
# prints in full under any range, saying so in its header.
#
# recording_tracks is the case this map exists for. It carries no date of its own, every row
# belongs to a dated recording, and it is by a wide margin the largest table in the database --
# so an unranged dump of it is exactly what a range was reached for to avoid.
_DATE_EXPR = {
    "shows": '"date"',
    "show_entries": '"date"',
    "recordings": '"date"',
    "show_types": '"date"',
    "recording_tracks": ('(SELECT "date" FROM "recordings" '
                         'WHERE "recordings"."identifier" = "recording_tracks"."identifier")'),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ident(name: str) -> str:
    """Guard a schema identifier before it goes into a query string.

    Table and column names come from our own sqlite_master, never from a user, but building
    SQL by hand still deserves a fence: anything that isn't a plain identifier is a bug in
    the store, not input to trust.
    """
    if not _IDENT.match(name):
        raise ValueError(f"refusing to build SQL with identifier {name!r}")
    return name


# SQLite can't bind an identifier, so these build the query text by hand. Every name is a
# schema identifier from sqlite_master, run through _ident() first, and then double-quoted so
# a reserved word (a column literally named "order") is still valid SQL. The nosec is honest:
# no untrusted input, and no other way to name a table dynamically.
def _quote(name: str) -> str:
    return f'"{_ident(name)}"'


def _count_sql(table: str, where: str = "") -> str:
    return f"SELECT COUNT(*) FROM {_quote(table)}{where}"  # nosec B608


def _select_sql(table: str, columns: list[str], where: str = "") -> str:
    select = ", ".join(_quote(c) for c in columns)
    return f"SELECT {select} FROM {_quote(table)}{where} ORDER BY {select}"  # nosec B608


def _check_date(value: str | None, flag: str) -> str | None:
    """Reject a date that is not YYYY-MM-DD, rather than silently answering the wrong question."""
    if value is None:
        return None
    if not _DATE_SHAPE.match(value):
        raise ValueError(f"{flag} wants a YYYY-MM-DD date, not {value!r}")
    return value


def _where(table: str, since: str | None, until: str | None) -> tuple[str, tuple]:
    """The date-range clause for one table, as ``(sql, params)``. Empty when it does not apply.

    Inclusive at both ends. A half-open range is the classic off-by-one, and this is the view
    people reach for when they already suspect something is wrong -- an end date that means "up to
    but not including" is a second thing to be wrong about while debugging the first.
    """
    expr = _DATE_EXPR.get(table)
    if expr is None or (since is None and until is None):
        return "", ()
    tests, params = [], []
    if since is not None:
        tests.append(f"{expr} >= ?")
        params.append(since)
    if until is not None:
        tests.append(f"{expr} <= ?")
        params.append(until)
    return " WHERE " + " AND ".join(tests), tuple(params)


def _range_label(since: str | None, until: str | None) -> str:
    """How a range reads in a table header, in whichever of its three forms was asked for."""
    if since is not None and until is not None:
        return f"{since}..{until}"
    return f"from {since}" if since is not None else f"through {until}"


class Store:
    """Owns the database connection and the raw cache rooted at ``data_root``.

    Usable as a context manager so the connection closes cleanly::

        with Store(cfg.data_root) as store:
            store.init()
    """

    def __init__(self, data_root: str | Path, *, filename: str = DB_FILENAME) -> None:
        self.data_root = Path(data_root)
        self.db_path = self.data_root / filename
        self.raw = RawCache(self.data_root)
        self._conn: sqlite3.Connection | None = None

    # --- connection lifecycle ---

    @property
    def conn(self) -> sqlite3.Connection:
        """The open connection, opened lazily on first use."""
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _connect(self) -> sqlite3.Connection:
        self.data_root.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: we run our own transactions (see migrations.transaction), the
        # only way to keep DDL atomic across the Python versions we support.
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def close(self) -> None:
        """Close the connection if one is open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # --- schema ---

    def init(self) -> list[int]:
        """Create the data_root, open the DB, and apply pending migrations.

        Returns the versions applied this call (empty when already current). Also stamps
        provenance into meta on the very first init.
        """
        applied = apply_pending(self.conn)
        with transaction(self.conn):
            # created_at only on the first init; the version tracks whatever last touched it.
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('created_at', ?)", (_now(),))
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('setlistkit_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (__version__,))
        return applied

    def schema_version(self) -> int:
        """Highest applied migration version, or 0 for an uninitialized database."""
        return schema_version(self.conn)

    # --- the merged corpus ---
    #
    # Thin pass-throughs. The SQL lives in store/corpus.py so this class stays the handle on the
    # database rather than the place every table's queries accumulate.

    def replace_shows(self, shows) -> int:
        """Replace the whole corpus with ``shows``, in one transaction. Returns how many."""
        return corpus.replace_shows(self.conn, shows)

    def shows(self) -> list[dict]:
        """Every stored show, by date, sets and encore in the order they were played."""
        return corpus.shows(self.conn)

    def show_count(self) -> int:
        """How many shows are stored, without reading their setlists."""
        return corpus.show_count(self.conn)

    def song_count(self) -> int:
        """How many actual songs are stored, ignoring the tagged non-songs."""
        return corpus.song_count(self.conn)

    def show_sources(self) -> dict[str, str]:
        """date -> which source won it, without reading their setlists."""
        return corpus.show_sources(self.conn)

    # --- the recordings mirror ---

    def replace_recordings(self, records) -> tuple[int, int]:
        """Replace the mirror. Returns ``(recordings, tracks)`` written."""
        return recordings.replace_recordings(self.conn, records)

    def replace_show_types(self, types) -> int:
        """Replace every date's electric/acoustic/mixed/alterego tag. Returns how many."""
        return recordings.replace_show_types(self.conn, types)

    def recording_count(self) -> int:
        """How many tapes are mirrored, without reading their tracks."""
        return recordings.recording_count(self.conn)

    def track_count(self) -> int:
        """How many tracks are mirrored."""
        return recordings.track_count(self.conn)

    def show_type_counts(self) -> dict[str, int]:
        """kind -> how many dates carry it."""
        return recordings.show_type_counts(self.conn)

    def recordings(self) -> list[dict]:
        """Every mirrored tape with its tracks, by date then identifier, tracks in play order."""
        return recordings.recordings(self.conn)

    def show_types(self) -> dict[str, str]:
        """date -> kind."""
        return recordings.show_types(self.conn)

    def table_names(self) -> list[str]:
        """Every non-internal table, alphabetized."""
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        return [r[0] for r in rows]

    def table_counts(self) -> dict[str, int]:
        """Row count per table, for a quick sense of what's stored."""
        counts = {}
        for table in self.table_names():
            counts[table] = self.conn.execute(_count_sql(table)).fetchone()[0]
        return counts

    # --- plain-text view ---

    def dump(self, *, since: str | None = None, until: str | None = None) -> str:
        """Render derived state to deterministic plain text, for reviewable diffs.

        Volatile timestamp columns are dropped and rows are sorted, so two dumps of the same
        logical state are byte-identical. This is the ``slkit dump`` view: the thing that
        makes a bad write to an otherwise-opaque SQLite blob show up as a text diff.

        ``since`` and ``until`` narrow it to a date range, inclusive at both ends. Whole-database
        dumps stopped being readable somewhere around the seventy-six-thousandth track, and this
        is the view someone opens when they already suspect one night is wrong.

        A range never hides anything quietly. Every table's header says whether the range reached
        it and how many rows it holds in total, so a table with no date axis reads as "all of it,
        because there is no date to filter on" rather than as a table that happens to look small.
        """
        since = _check_date(since, "--since")
        until = _check_date(until, "--until")
        ranged = since is not None or until is not None
        lines = [f"# {DB_FILENAME} — contents"
                 + (f" ({_range_label(since, until)})" if ranged else ""), ""]
        for table in self.table_names():
            cols = [r[1] for r in self.conn.execute(f"PRAGMA table_info({_quote(table)})")]
            shown = [c for c in cols if c not in VOLATILE_COLUMNS]
            where, params = _where(table, since, until)
            count = self.conn.execute(_count_sql(table, where), params).fetchone()[0]
            lines.append(f"## {table} ({self._heading(table, count, ranged, since, until)})")
            if not shown:
                lines.append("")
                continue
            lines.append(" | ".join(shown))
            rows = self.conn.execute(_select_sql(table, shown, where), params).fetchall()
            lines.extend(" | ".join("" if v is None else str(v) for v in row) for row in rows)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _heading(self, table: str, count: int, ranged: bool,
                 since: str | None, until: str | None) -> str:
        """The parenthesised part of one table's dump header.

        Unchanged from an unranged dump when no range was asked for, so the existing view and its
        diffs are exactly what they were.
        """
        if not ranged:
            return f"{count} rows"
        if table not in _DATE_EXPR:
            return f"{count} rows, no date axis"
        total = self.conn.execute(_count_sql(table)).fetchone()[0]
        return f"{count} of {total} rows, {_range_label(since, until)}"

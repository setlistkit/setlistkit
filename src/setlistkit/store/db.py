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
from . import corpus, daterange, durations, recordings
from .migrations import apply_pending, schema_version, transaction
from .raw_cache import RawCache

DB_FILENAME = "setlistkit.sqlite"

# Columns whose value is a timestamp or run marker: real content, but they change every run
# and would bury the signal in a diff of `slkit dump`. We drop them from the dump only.
VOLATILE_COLUMNS = frozenset({
    "applied_at", "fetched_at", "created_at", "updated_at", "run_ts", "scored_asof",
})

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

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
    "performance_durations": '"date"',
    "duration_review": '"date"',
    "duration_abandoned": '"date"',
    "duration_edges": '"date"',
    "recording_listings": ('(SELECT "date" FROM "recordings" '
                           'WHERE "recordings"."identifier" = "recording_listings"."identifier")'),
    "recording_listing_entries": (
        '(SELECT "date" FROM "recordings" WHERE "recordings"."identifier" = '
        '"recording_listing_entries"."identifier")'),
    # song_length_stats is deliberately absent. Its `longest_date` is a date but not the row's
    # axis: a song's statistics are computed over every year it was played, so narrowing them to a
    # range would print a whole-corpus median under a header claiming it covers one month.
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


def _where(table: str, since: str | None, until: str | None) -> tuple[str, tuple]:
    """The date-range clause for one table, as ``(sql, params)``. Empty when it does not apply.

    The table's date axis is looked up here; the comparison itself is built by
    :mod:`setlistkit.store.daterange`, which is also what ``export`` narrows a bundle with. A table
    absent from the map has no date axis and is left whole.
    """
    expr = _DATE_EXPR.get(table)
    if expr is None:
        return "", ()
    return daterange.clause(expr, since, until)


class _Namespace:
    """One domain's queries, bound to a store's connection.

    The store used to carry every table's queries as flat methods on itself, and at three domains
    that was twenty-eight of them in one undifferentiated list -- ``recordings`` and ``shows`` and
    ``performances`` sitting side by side with nothing saying which layer each belonged to. The
    connection is still owned in exactly one place; what changed is that asking for a query now
    means naming the domain it belongs to.

    Subclasses declare their methods explicitly rather than forwarding by ``__getattr__``. A
    facade that answers to anything is a facade where a typo becomes a runtime AttributeError deep
    in a run instead of a name your editor could have completed.

    ``TABLES`` is what each domain owns, declared so a domain can be counted without anyone
    keeping a second list of which tables belong to which layer. That list has to exist somewhere
    for the report each command prints, and next to the queries is the only place it cannot drift
    away from them.
    """

    TABLES: tuple[str, ...] = ()

    def __init__(self, store: "Store") -> None:
        self._store = store

    @property
    def conn(self) -> sqlite3.Connection:
        """The store's connection.

        Read through the store on every access rather than captured at construction: the store
        opens lazily and reopens after a ``close()``, and a namespace holding the handle from
        before would be a namespace writing to a connection nobody else is using.
        """
        return self._store.conn

    def counts(self) -> dict[str, int]:
        """Row count for each table this domain owns, in declaration order.

        Declaration order rather than alphabetical, because the tables of a domain are written in
        dependency order and reading them that way is how a broken run reads: recordings before
        their tracks, performances before the statistics aggregated from them. A zero halfway down
        the list says which stage stopped.
        """
        return {table: self.conn.execute(_count_sql(table)).fetchone()[0]
                for table in self.TABLES}


class _Corpus(_Namespace):
    """The merged corpus: one show per date, and its entries in play order."""

    TABLES = ("shows", "show_entries")

    def replace_shows(self, shows) -> int:
        """Replace the whole corpus with ``shows``, in one transaction. Returns how many."""
        return corpus.replace_shows(self.conn, shows)

    def shows(self, *, since: str | None = None, until: str | None = None) -> list[dict]:
        """Every stored show, by date, sets and encore in the order they were played."""
        return corpus.shows(self.conn, since, until)

    def show_count(self) -> int:
        """How many shows are stored, without reading their setlists."""
        return corpus.show_count(self.conn)

    def song_count(self) -> int:
        """How many actual songs are stored, ignoring the tagged non-songs."""
        return corpus.song_count(self.conn)

    def show_sources(self) -> dict[str, str]:
        """date -> which source won it, without reading their setlists."""
        return corpus.show_sources(self.conn)


class _Tapes(_Namespace):
    """The recordings mirror: every tape of every night, its tracks, its listing, its date's kind.

    Every tape, not just the one that won the merge. A setlist is a single fact about a night, so
    the corpus keeps one; a duration is a measurement, and the same performance timed by four
    tapers is four measurements whose disagreement is the error bar.
    """

    TABLES = ("recordings", "recording_tracks", "show_types", "recording_listings",
              "recording_listing_entries")

    def replace_recordings(self, records) -> tuple[int, int]:
        """Replace the mirror. Returns ``(recordings, tracks)`` written."""
        return recordings.replace_recordings(self.conn, records)

    def replace_show_types(self, types) -> int:
        """Replace every date's electric/acoustic/mixed/alterego tag. Returns how many."""
        return recordings.replace_show_types(self.conn, types)

    def replace_listings(self, by_tape) -> tuple[int, int]:
        """Replace every tape's written tracklist. Returns ``(listings, entries)`` written."""
        return recordings.replace_listings(self.conn, by_tape)

    def recording_count(self, *, since: str | None = None,
                        until: str | None = None) -> int:
        """How many tapes are mirrored, without reading their tracks."""
        return recordings.recording_count(self.conn, since, until)

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

    def listings(self, *, matched_only: bool = True) -> dict[str, list[dict]]:
        """identifier -> its written tracklist, in play order. Matched listings only by default."""
        return recordings.listings(self.conn, matched_only=matched_only)

    def listing_readings(self) -> dict[str, int]:
        """reading -> how many tapes it explained, including the ones that did not line up."""
        return recordings.listing_readings(self.conn)

    def uploader_counts(self, *, since: str | None = None,
                        until: str | None = None) -> dict[str, int]:
        """who posted -> how many of their tapes we hold, most prolific first."""
        return recordings.uploader_counts(self.conn, since, until)


class _Durations(_Namespace):
    """What the durations chain concluded: performances, statistics, and what was skipped."""

    TABLES = ("performance_durations", "song_length_stats", "duration_review",
              "duration_abandoned", "duration_edges")

    def replace(self, performance_rows, stat_rows, **kwargs) -> dict[str, int]:
        """Replace every durations table in one transaction. Returns a count per table."""
        return durations.replace_durations(self.conn, performance_rows, stat_rows, **kwargs)

    def performances(self, *, since: str | None = None,
                     until: str | None = None) -> list[dict]:
        """Every stored performance, in play order."""
        return durations.performances(self.conn, since, until)

    def song_length_stats(self) -> list[dict]:
        """Every song's length statistics, longest first."""
        return durations.song_length_stats(self.conn)

    def review(self, *, since: str | None = None, until: str | None = None) -> list[dict]:
        """Tapes we hold and could not time."""
        return durations.duration_review(self.conn, since, until)

    def abandoned(self, *, since: str | None = None,
                  until: str | None = None) -> list[dict]:
        """Tapes with one track holding a whole set."""
        return durations.duration_abandoned(self.conn, since, until)

    def edges(self, *, since: str | None = None, until: str | None = None) -> list[dict]:
        """Every edge case recorded, with its detail as a mapping."""
        return durations.duration_edges(self.conn, since, until)

    def withheld_counts(self) -> dict[str, int]:
        """reason -> how many stored performances do not vote for their song's nominal length."""
        return durations.withheld_counts(self.conn)


class Store:
    """Owns the database connection and the raw cache rooted at ``data_root``.

    The tables are reached through one namespace per domain -- ``store.corpus``, ``store.tapes``,
    ``store.durations`` -- while the connection, the PRAGMAs, the migration walk and the plain-text
    dump stay here. Usable as a context manager so the connection closes cleanly::

        with Store(cfg.data_root) as store:
            store.init()
            store.corpus.replace_shows(shows)
    """

    def __init__(self, data_root: str | Path, *, filename: str = DB_FILENAME) -> None:
        self.data_root = Path(data_root)
        self.db_path = self.data_root / filename
        self.raw = RawCache(self.data_root)
        self._conn: sqlite3.Connection | None = None
        self.corpus = _Corpus(self)
        self.tapes = _Tapes(self)
        self.durations = _Durations(self)

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

    # --- the whole database, whatever is in it ---

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
        since = daterange.check_date(since, "--since")
        until = daterange.check_date(until, "--until")
        ranged = since is not None or until is not None
        lines = [f"# {DB_FILENAME} — contents"
                 + (f" ({daterange.label(since, until)})" if ranged else ""), ""]
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
        return f"{count} of {total} rows, {daterange.label(since, until)}"

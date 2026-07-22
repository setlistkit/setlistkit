# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""What the durations chain concluded, in SQLite: performances, statistics, and what was skipped.

Five tables written in ONE transaction, because they are five views of one run and any two of them
taken from different runs are a contradiction nothing would report. The statistics are an aggregate
of the performances; the review queue is the tapes those performances were not computed from; the
edges are the reasons. A half-applied write leaves a statistic whose supporting rows say something
else, which is worse than no statistic at all.

Replaced whole rather than patched, like the corpus and the mirror, and here that is not merely
convention: the reconciliation is a comparison BETWEEN tapes, so one tape arriving for one night
can change the answer for every performance of that night. There is no such thing as incrementally
updating a vote.

This module decides nothing and knows no catalog types -- it takes mappings keyed by column name,
exactly as :mod:`setlistkit.store.corpus` and :mod:`setlistkit.store.recordings` do, and the store
stays a layer that does not depend on the one above it. :func:`setlistkit.catalog.lengths.as_row`
produces those mappings, and lives there rather than here because flattening a performance is a
question about a performance -- in particular about ``withheld``, which is a property and which a
consumer flattening for itself would silently drop.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence

from . import daterange
from .migrations import transaction

# The columns of one performance row, in insert order, named once so the writer and the reader
# cannot drift into disagreeing about the order of nineteen of them. These are also exactly the
# keys catalog.lengths.as_row produces: the mapping between the two is the identity, so there is
# no third place a field can be misfiled.
_PERFORMANCE_COLUMNS = (
    "date", "set_label", "position", "song", "seconds",
    "n_tapes", "n_tapes_seen", "n_ballots", "spread_seconds", "spread_all_tapes",
    "suspect", "resolved_by", "segued", "show_type", "excluded", "withheld",
    "double_play_parts", "sandwich_total_seconds", "is_longest_part",
)

_STAT_COLUMNS = (
    "song", "n", "median_seconds", "mean_seconds", "min_seconds", "max_seconds",
    "p10_seconds", "p90_seconds", "stdev_seconds", "longest_date",
)

_REVIEW_COLUMNS = ("identifier", "date", "n_tracks", "n_setlist", "n_desc", "reason", "url")

_ABANDONED_COLUMNS = ("identifier", "date", "n_tracks", "longest_seconds")

# `id` is deliberately not here: it is the table's key, not a fact about the edge, and putting it
# in the export would publish an insertion counter as though it meant something.
_EDGE_COLUMNS = ("kind", "date", "identifier", "song", "detail_json")

# Columns SQLite has no type for and Python does: a bool stored as INTEGER comes back as 0 or 1,
# and `if row["suspect"]` would be true for both if we let the string "False" through on the way
# in. Coerced on write, coerced back on read, listed once.
_BOOLEAN_COLUMNS = frozenset({"suspect", "segued", "is_longest_part"})

_TABLES = ("performance_durations", "song_length_stats", "duration_review",
           "duration_abandoned", "duration_edges")


def _insert_sql(table: str, columns: Sequence[str]) -> str:
    """An INSERT for one of the column tuples above.

    Every name is a module constant, never input, so there is nothing to escape. What this buys is
    that nineteen placeholders are counted by the machine rather than typed out by hand beside
    nineteen column names -- a miscount in a row of interchangeable REALs publishes numbers that
    are wrong and look right.
    """
    placeholders = ", ".join("?" for _ in columns)
    return (f"INSERT INTO {table}({', '.join(columns)}) "     # nosec B608 - module constants only
            f"VALUES({placeholders})")


def _value(column: str, row: Mapping):
    """One column's value out of a row mapping, coerced for SQLite.

    Subscripted rather than ``.get``, so a mapping missing a column raises instead of storing
    NULL. That matters most for ``withheld``: a silent NULL there does not look like a bug, it
    looks like a performance that was counted, and the exclusion tally would quietly empty out
    while every other number in the run stayed plausible.
    """
    value = row[column]
    if column in _BOOLEAN_COLUMNS and value is not None:
        return int(bool(value))
    return value


def _rows(records: Iterable[Mapping], columns: Sequence[str]) -> list[tuple]:
    return [tuple(_value(column, record) for column in columns) for record in records]


def _edge_rows(edges: Iterable) -> list[tuple]:
    """Edges flattened, their detail dicts serialized.

    ``sort_keys`` so two runs over the same corpus write byte-identical JSON and a `slkit dump`
    diff stays about the findings rather than about dictionary ordering. ``default=str`` because a
    detail dict carries whatever the call that raised it thought was worth recording, and an edge
    that cannot be serialized would abort a run over something that is, by construction, a note.
    """
    return [(edge.kind, edge.date, edge.identifier, edge.song,
             json.dumps(edge.detail, sort_keys=True, default=str))
            for edge in edges]


def replace_durations(conn: sqlite3.Connection,
                      performance_rows: Iterable[Mapping],
                      stat_rows: Iterable[Mapping],
                      *,
                      review: Iterable[Mapping] = (),
                      abandoned: Iterable[Mapping] = (),
                      edges: Iterable = ()) -> dict[str, int]:
    """Replace everything the durations chain produces. Returns a count per table.

    Every count comes back, and every one is reported, because they fail independently and in ways
    that look like success. A run storing twenty thousand performances and no statistics has
    aggregated nothing. A run storing statistics over a tenth of the performances has published a
    page of numbers with n=1 on most of them. Neither is visible from the other's count.
    """
    written = {
        "performance_durations": _rows(performance_rows, _PERFORMANCE_COLUMNS),
        "song_length_stats": _rows(stat_rows, _STAT_COLUMNS),
        "duration_review": _rows(review, _REVIEW_COLUMNS),
        "duration_abandoned": _rows(abandoned, _ABANDONED_COLUMNS),
    }
    edge_rows = _edge_rows(edges)
    inserts = {"performance_durations": _PERFORMANCE_COLUMNS, "song_length_stats": _STAT_COLUMNS,
               "duration_review": _REVIEW_COLUMNS, "duration_abandoned": _ABANDONED_COLUMNS}
    with transaction(conn):
        for table in _TABLES:
            conn.execute(f"DELETE FROM {table}")               # nosec B608 - module constant
        for table, columns in inserts.items():
            conn.executemany(_insert_sql(table, columns), written[table])
        conn.executemany(
            "INSERT INTO duration_edges(kind, date, identifier, song, detail_json) "
            "VALUES(?, ?, ?, ?, ?)", edge_rows)
    counts = {table: len(rows) for table, rows in written.items()}
    counts["duration_edges"] = len(edge_rows)
    return counts


def _read(conn: sqlite3.Connection, table: str, columns: Sequence[str],
          order: str, *, since: str | None = None, until: str | None = None) -> list[dict]:
    """Rows back as mappings keyed by column name, booleans restored to bools.

    ``since``/``until`` narrow to an inclusive window on the table's own ``date`` column. Every
    table read through here has one; the clause is built by :mod:`setlistkit.store.daterange`, the
    same place ``slkit dump`` gets its range from, so the two commands cannot disagree about what
    "on or after" means.
    """
    where, params = daterange.clause('"date"', since, until)
    return [{column: (bool(row[column]) if column in _BOOLEAN_COLUMNS and row[column] is not None
                      else row[column])
             for column in columns}
            for row in conn.execute(
                f"SELECT {', '.join(columns)} FROM {table}{where} "  # nosec B608
                f"ORDER BY {order}", params)]


def performances(conn: sqlite3.Connection, since: str | None = None,
                 until: str | None = None) -> list[dict]:
    """Every stored performance, in play order, optionally narrowed to a date window.

    Ordered rather than left to rowid order, for the reason the mirror is: this is what the export
    reads, and a bundle whose row order depends on insertion order is one whose golden-file test
    passes or fails on something that is not the data.
    """
    return _read(conn, "performance_durations", _PERFORMANCE_COLUMNS,
                 "date, set_label, position",
                 since=since, until=until)


def song_length_stats(conn: sqlite3.Connection) -> list[dict]:
    """Every song's length statistics, longest first -- the order they are read in."""
    return _read(conn, "song_length_stats", _STAT_COLUMNS, "median_seconds DESC, song")


def duration_review(conn: sqlite3.Connection, since: str | None = None,
                    until: str | None = None) -> list[dict]:
    """Tapes we hold and could not time, worst mismatch first."""
    return _read(conn, "duration_review", _REVIEW_COLUMNS, "date, identifier",
                 since=since, until=until)


def duration_abandoned(conn: sqlite3.Connection, since: str | None = None,
                       until: str | None = None) -> list[dict]:
    """Tapes with one track holding a whole set. Nothing to do about them, but they are held."""
    return _read(conn, "duration_abandoned", _ABANDONED_COLUMNS, "date, identifier",
                 since=since, until=until)


def duration_edges(conn: sqlite3.Connection, since: str | None = None,
                   until: str | None = None) -> list[dict]:
    """Every edge case recorded, with its detail back as the mapping it went in as.

    Ordered by what the row SAYS rather than by ``id``, though the id is right there. Insertion
    order is the order the tapes happened to be read in, which is stable today only because
    :func:`setlistkit.store.recordings.recordings` sorts -- so ordering on it would make the
    export's row order depend on a decision made two modules away, and a golden file that fails
    when that decision changes is a golden file that fails for the wrong reason.
    """
    rows = _read(conn, "duration_edges", _EDGE_COLUMNS, "kind, date, identifier, song",
                 since=since, until=until)
    for row in rows:
        row["detail"] = json.loads(row.pop("detail_json"))
    return rows


def withheld_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """reason -> how many stored performances do not vote for their song's nominal length.

    Read back off the stored column rather than recomputed, so the tally and the statistics it
    explains can only ever come from the same run. A performance that DOES vote has NULL here and
    is not counted, which is why this sums to less than the table.
    """
    return {row["withheld"]: row["n"] for row in conn.execute(
        "SELECT withheld, COUNT(*) AS n FROM performance_durations "
        "WHERE withheld IS NOT NULL GROUP BY withheld ORDER BY withheld")}

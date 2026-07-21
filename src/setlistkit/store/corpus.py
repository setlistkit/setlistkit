# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The merged corpus in SQLite: one show per date, and its entries in the order they were played.

The catalog decides WHICH record wins a date (see :mod:`setlistkit.catalog.merge`); this module
only stores the answer and hands it back unchanged. It is deliberately dumb about setlists: it
does not count songs, does not ask whether an entry is music, and does not derive a year. Those
are the catalog's questions and they already have exactly one owner each.

Two things it does decide, because they are storage questions:

**A show is replaced whole, never patched.** The merge recomputes the entire corpus from every
source on every run, so a partial write would leave a date holding half of one run and half of
another. :func:`replace_shows` is one transaction: everything goes or nothing does.

**Order is a column, never an accident.** ``rowid`` order happens to match insertion order today
and is not promised to, and a setlist read back in the wrong order is wrong in a way that looks
right. So the set number, the section, and the position within the set are all stored, and every
read sorts on them explicitly.

Nothing derivable is stored. ``year`` is ``date[:4]`` and a song count is a tally of the entries,
and a column holding either can disagree with the rows it was computed from -- which is precisely
the failure ``merge._n_songs`` exists to prevent, where a record claiming a count nothing backs up
disqualifies every honest candidate for its date and enters the corpus empty.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping

from .migrations import transaction

# The sections of a night, as stored. Sets are numbered from 1 in the order they were played;
# the encore is its own section rather than a set with a magic number, so "the encore sorts
# last" is an explicit rule and not an arithmetic coincidence.
_SET, _ENCORE = 0, 1


def _entry_rows(date: str, record: Mapping) -> list[tuple]:
    """One show record flattened into ``show_entries`` rows.

    ``or ()`` rather than a default, matching ``merge._entries``: a key that is present and null
    is exactly what a truncated or hand-edited intermediate produces, and a default only covers
    a missing key.
    """
    rows: list[tuple] = []
    for set_no, one_set in enumerate(record.get("sets") or (), start=1):
        for position, entry in enumerate(one_set or ()):
            rows.append((date, _SET, set_no, position, str(entry.get("song") or ""),
                         int(bool(entry.get("segue"))), int(bool(entry.get("non_song")))))
    for position, entry in enumerate(record.get("encore") or ()):
        rows.append((date, _ENCORE, 1, position, str(entry.get("song") or ""),
                     int(bool(entry.get("segue"))), int(bool(entry.get("non_song")))))
    return rows


def replace_shows(conn: sqlite3.Connection, records: Iterable[Mapping]) -> int:
    """Replace the whole corpus with ``records``. Returns how many were written.

    One transaction, and the delete is part of it: a crash between the delete and the inserts
    would otherwise leave an empty corpus where a good one had been. Callers that want to refuse
    a suspiciously small replacement do that BEFORE calling here -- see the no-shrink guard in
    ``slkit ingest`` -- because by this point the decision to publish has been made.
    """
    rows = list(records)
    with transaction(conn):
        # show_entries cascades from shows, but delete it explicitly anyway: the cascade needs
        # PRAGMA foreign_keys to be on, and a silently orphaned entry table would come back as
        # songs attached to the next show that happens to reuse the date.
        conn.execute("DELETE FROM show_entries")
        conn.execute("DELETE FROM shows")
        conn.executemany(
            "INSERT INTO shows(date, source, identifier, reason) VALUES(?, ?, ?, ?)",
            [(str(show["date"]), str(show.get("source") or ""),
              str(show.get("identifier") or ""), show.get("reason")) for show in rows])
        entries = [row for show in rows for row in _entry_rows(str(show["date"]), show)]
        conn.executemany(
            "INSERT INTO show_entries(date, section, set_no, position, song, segue, non_song) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)", entries)
    return len(rows)


def shows(conn: sqlite3.Connection) -> list[dict]:
    """Every stored show, by date, with its sets and encore in the order they were played.

    Returns the record shape the merge produced, minus what was never stored: no ``year`` and no
    ``n_songs``. A caller that wants those asks the catalog for them, which is the layer that
    owns both answers. ``reason`` is present only where one was recorded, so a plain merged show
    and a manual override are still distinguishable after a round trip.
    """
    out: dict[str, dict] = {}
    for row in conn.execute("SELECT date, source, identifier, reason FROM shows ORDER BY date"):
        record = {"date": row["date"], "source": row["source"], "identifier": row["identifier"],
                  "sets": [], "encore": []}
        if row["reason"] is not None:
            record["reason"] = row["reason"]
        out[row["date"]] = record
    for row in conn.execute(
            "SELECT date, section, set_no, position, song, segue, non_song FROM show_entries "
            "ORDER BY date, section, set_no, position"):
        record = out.get(row["date"])
        if record is None:
            continue                       # orphaned by a hand-edit; the shows table is the truth
        entry = {"song": row["song"], "segue": bool(row["segue"]),
                 "non_song": bool(row["non_song"])}
        if row["section"] == _ENCORE:
            record["encore"].append(entry)
            continue
        # The set number is 1-based and the rows arrive in order, so a set is appended the first
        # time one of its entries is seen. Indexing by set_no instead would build a ragged list
        # if a set number were ever missing, and silently attach its songs to the wrong set.
        while len(record["sets"]) < row["set_no"]:
            record["sets"].append([])
        record["sets"][row["set_no"] - 1].append(entry)
    return list(out.values())


def show_count(conn: sqlite3.Connection) -> int:
    """How many shows are stored. The number the no-shrink guard measures against."""
    return conn.execute("SELECT COUNT(*) FROM shows").fetchone()[0]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The recordings mirror in SQLite: one row per tape, its tracks in play order, and each date's
kind.

The mirror answers a question the corpus cannot: the corpus keeps ONE show per date, the winner
of the merge, because a setlist is a single fact about a night. Durations are the opposite -- the
same performance timed by four different tapers is four measurements, and the disagreement between
them is the error bar. So every tape is stored, not just the one that won.

Like the corpus, it is replaced whole rather than patched. Ingest recomputes it from the entire
cache on every run, so a partial write would leave the database holding half of one run and half
of another, joined on nothing.

What this module does NOT do is decide anything. It does not choose a format, does not compute an
order, does not judge whether a tape is measurable. Those are :mod:`setlistkit.sources.archive_org`
and, from slice 3, the durations core; this stores the answers and hands them back unchanged. The
one thing it insists on is that an order arrives already computed, because ``idx`` is a column and
never an accident -- see the migration for why.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping

from .migrations import transaction


def _track_rows(identifier: str, record: Mapping) -> list[tuple]:
    """One recording's tracks flattened into ``recording_tracks`` rows.

    ``idx`` is taken from the projection rather than from this loop's counter. Re-numbering here
    would work perfectly right up until a caller handed over tracks in some other order, at which
    point the stored order and the computed order would part company with nothing to say so.
    """
    return [(identifier, int(track["idx"]), str(track.get("name") or ""),
             str(track.get("title") or ""), str(track.get("length_raw") or ""),
             track.get("seconds"))
            for track in record.get("duration_tracks") or ()]


def replace_recordings(conn: sqlite3.Connection, records: Iterable[Mapping]) -> tuple[int, int]:
    """Replace the whole mirror with ``records``. Returns ``(recordings, tracks)`` written.

    One transaction, and the delete is part of it, for the same reason
    :func:`setlistkit.store.corpus.replace_shows` does it that way: a crash between the delete and
    the inserts would otherwise leave an empty mirror where a good one had been.

    Both counts come back because they fail independently. A run that stores every tape and no
    tracks -- a format list that stopped matching what archive.org labels its derivatives, say --
    keeps the recording count identical and measures nothing at all.
    """
    rows = list(records)
    tracks = [row for record in rows
              for row in _track_rows(str(record["identifier"]), record)]
    with transaction(conn):
        # recording_tracks cascades from recordings, but delete it explicitly anyway: the cascade
        # needs PRAGMA foreign_keys to be on, and a silently orphaned track table would come back
        # as one tape's durations attached to the next tape that reused the identifier.
        conn.execute("DELETE FROM recording_tracks")
        conn.execute("DELETE FROM recordings")
        conn.executemany(
            "INSERT INTO recordings(identifier, date, uploader, audio_format, n_tracks) "
            "VALUES(?, ?, ?, ?, ?)",
            [(str(record["identifier"]), str(record["date"]), str(record.get("uploader") or ""),
              str(record.get("audio_format") or ""),
              len(record.get("duration_tracks") or ())) for record in rows])
        conn.executemany(
            "INSERT INTO recording_tracks(identifier, idx, name, title, length_raw, seconds) "
            "VALUES(?, ?, ?, ?, ?, ?)", tracks)
    return len(rows), len(tracks)


def replace_show_types(conn: sqlite3.Connection, types: Iterable[Mapping]) -> int:
    """Replace every date's show type. Returns how many rows were written."""
    rows = list(types)
    with transaction(conn):
        conn.execute("DELETE FROM show_types")
        conn.executemany(
            "INSERT INTO show_types(date, kind, evidence, identifier) VALUES(?, ?, ?, ?)",
            [(str(row["date"]), str(row["kind"]), row.get("evidence"), row.get("identifier"))
             for row in rows])
    return len(rows)


def replace_listings(conn: sqlite3.Connection, by_tape: Mapping[str, Mapping]) -> tuple[int, int]:
    """Replace every tape's written tracklist. Returns ``(listings, entries)`` written.

    Keyed by identifier, and written in the same run as the mirror it hangs off, for the reason
    :func:`replace_recordings` is written beside the corpus: the listing is read out of the same
    description the setlist is, and two projections of one payload made at two different moments
    are two projections of two different payloads.

    Unmatched listings are stored too, with ``matched`` false. See the migration for why that is
    a column and not an absence.
    """
    rows = sorted(by_tape.items())
    entries = [(identifier, int(entry["idx"]), str(entry["song"]), int(bool(entry["segue"])))
               for identifier, listing in rows
               for entry in listing.get("entries") or ()]
    with transaction(conn):
        conn.execute("DELETE FROM recording_listing_entries")
        conn.execute("DELETE FROM recording_listings")
        conn.executemany(
            "INSERT INTO recording_listings(identifier, reading, n_entries, matched) "
            "VALUES(?, ?, ?, ?)",
            [(identifier, str(listing["reading"]), len(listing.get("entries") or ()),
              int(bool(listing["matched"]))) for identifier, listing in rows])
        conn.executemany(
            "INSERT INTO recording_listing_entries(identifier, idx, song, segue) "
            "VALUES(?, ?, ?, ?)", entries)
    return len(rows), len(entries)


def listings(conn: sqlite3.Connection, *, matched_only: bool = True) -> dict[str, list[dict]]:
    """identifier -> its written tracklist, in play order.

    ``matched_only`` defaults to TRUE, so the caller that just wants to time songs gets only the
    listings that lined up with their tape's files -- which is the only thing anything may join
    on. Asking for the rest is possible and has to be spelled out, which is the point.
    """
    where = " WHERE matched = 1" if matched_only else ""
    wanted = {row["identifier"] for row in conn.execute(
        "SELECT identifier FROM recording_listings" + where)}      # nosec B608 - literal branch
    out: dict[str, list[dict]] = {identifier: [] for identifier in wanted}
    for row in conn.execute(
            "SELECT identifier, idx, song, segue FROM recording_listing_entries "
            "ORDER BY identifier, idx"):
        if row["identifier"] in out:
            out[row["identifier"]].append(
                {"idx": row["idx"], "song": row["song"], "segue": bool(row["segue"])})
    return out


def listing_readings(conn: sqlite3.Connection) -> dict[str, int]:
    """reading -> how many tapes it explained. What ingest reports, so a drift in the mix shows.

    UNMATCHED is one of the readings and is by far the most interesting one: it is the count of
    tapes whose taper wrote a listing we could not line up with their own files.
    """
    return {row["reading"]: row["n"] for row in conn.execute(
        "SELECT reading, COUNT(*) AS n FROM recording_listings GROUP BY reading ORDER BY reading")}


def recording_count(conn: sqlite3.Connection) -> int:
    """How many tapes are mirrored, without reading their tracks."""
    return conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]


def track_count(conn: sqlite3.Connection) -> int:
    """How many tracks are mirrored. The number that catches a mirror full of empty tapes."""
    return conn.execute("SELECT COUNT(*) FROM recording_tracks").fetchone()[0]


def show_type_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """kind -> how many dates carry it. What ingest reports, so a classifier change is visible.

    Ordered by kind so two runs over the same corpus print the same line, and a diff of two runs
    is about the numbers rather than about dictionary order.
    """
    return {row["kind"]: row["n"] for row in conn.execute(
        "SELECT kind, COUNT(*) AS n FROM show_types GROUP BY kind ORDER BY kind")}


def recordings(conn: sqlite3.Connection) -> list[dict]:
    """Every mirrored tape with its tracks, by date then identifier, tracks in play order.

    Sorted rather than left to rowid order: this is what the durations chain reads, and a
    reconciliation whose input order depends on insertion order is one whose output can change
    without its inputs changing.
    """
    out: dict[str, dict] = {}
    for row in conn.execute(
            "SELECT identifier, date, uploader, audio_format, n_tracks FROM recordings "
            "ORDER BY date, identifier"):
        out[row["identifier"]] = {"identifier": row["identifier"], "date": row["date"],
                                  "uploader": row["uploader"],
                                  "audio_format": row["audio_format"],
                                  "n_tracks": row["n_tracks"], "tracks": []}
    for row in conn.execute(
            "SELECT identifier, idx, name, title, length_raw, seconds FROM recording_tracks "
            "ORDER BY identifier, idx"):
        record = out.get(row["identifier"])
        if record is None:
            continue                       # orphaned by a hand-edit; recordings is the truth
        record["tracks"].append({"idx": row["idx"], "name": row["name"], "title": row["title"],
                                 "length_raw": row["length_raw"], "seconds": row["seconds"]})
    return list(out.values())


def show_types(conn: sqlite3.Connection) -> dict[str, str]:
    """date -> kind. The lookup the length statistics filter acoustic nights out with."""
    return {row["date"]: row["kind"] for row in conn.execute("SELECT date, kind FROM show_types")}

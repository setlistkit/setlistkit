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


def _m0002_corpus(conn: sqlite3.Connection) -> None:
    # The merged corpus: one show per date, and its entries in the order they were played.
    #
    # No `year` and no `n_songs`. Both are derivable (date[:4]; a tally of the entries), and a
    # stored copy of a derived value can disagree with what it was derived from. merge.py already
    # refuses to read a record's own n_songs for exactly that reason -- a record claiming a count
    # nothing backs up disqualifies every honest candidate for its date and enters the corpus
    # with no setlist in it -- so this schema does not offer somewhere to write one.
    #
    # `reason` is NULL for an ordinary merged show and carries text for a manual override, which
    # is the only kind that has to justify itself. See merge.overrides_from_mapping.
    conn.execute(
        "CREATE TABLE shows("
        "  date TEXT PRIMARY KEY,"
        "  source TEXT NOT NULL,"
        "  identifier TEXT NOT NULL,"
        "  reason TEXT)")
    # Order is stored, never inferred. rowid order matches insertion order today and is not
    # promised to; a setlist read back in the wrong order is wrong in a way that looks right.
    #
    # `section` separates the sets (0) from the encore (1) rather than giving the encore a magic
    # set number, so "the encore is last" is an explicit sort and not arithmetic that happens to
    # work. Every read orders by (section, set_no, position).
    conn.execute(
        "CREATE TABLE show_entries("
        "  date TEXT NOT NULL REFERENCES shows(date) ON DELETE CASCADE,"
        "  section INTEGER NOT NULL,"
        "  set_no INTEGER NOT NULL,"
        "  position INTEGER NOT NULL,"
        "  song TEXT NOT NULL,"
        "  segue INTEGER NOT NULL,"
        "  non_song INTEGER NOT NULL,"
        "  PRIMARY KEY (date, section, set_no, position))")
    # "how often has this song been played, and when" is the question every model layer asks, and
    # without this it is a full scan of every entry ever recorded.
    conn.execute("CREATE INDEX show_entries_song ON show_entries(song)")


def _m0003_recordings(conn: sqlite3.Connection) -> None:
    # The recordings mirror: one row per tape, and its tracks in the order they were played.
    #
    # A MIRROR, not a cache. Every column here is projected straight out of the raw payload at
    # ingest and rebuilt whole on the next one, which is what makes it safe to be a second copy
    # of something we already hold. What it buys is that no later stage has to open the raw
    # cache: that cache is gitignored, so a consumer that reaches into it works on the machine
    # it was written on and comes back empty on the server. `uploader` is exactly the field that
    # happened to in the previous implementation -- 425 tapes, silently uncredited.
    conn.execute(
        "CREATE TABLE recordings("
        "  identifier TEXT PRIMARY KEY,"
        "  date TEXT NOT NULL,"
        "  uploader TEXT NOT NULL,"
        "  audio_format TEXT NOT NULL,"
        "  n_tracks INTEGER NOT NULL)")
    # Several tapes per night is the normal case and the whole point -- independent timings of
    # one performance are what the durations chain votes with -- so every read of a night's
    # tapes is a lookup by date, not by identifier.
    conn.execute("CREATE INDEX recordings_date ON recordings(date)")
    # `idx` is the computed play order, stored rather than inferred, for the same reason
    # show_entries stores position: an order recomputed on read is an order that can silently
    # differ from the one the join was built against. See sources.archive_org._ordered for what
    # computing it costs and what believing archive.org's own `track` field cost before that.
    #
    # `length_raw` sits beside `seconds` because the raw layer stays raw. A duration that could
    # not be parsed is NULL seconds with the source string intact, so a parser bug is diagnosable
    # from the database rather than needing a re-pull of four thousand items to reproduce.
    conn.execute(
        "CREATE TABLE recording_tracks("
        "  identifier TEXT NOT NULL REFERENCES recordings(identifier) ON DELETE CASCADE,"
        "  idx INTEGER NOT NULL,"
        "  name TEXT NOT NULL,"
        "  title TEXT NOT NULL,"
        "  length_raw TEXT NOT NULL,"
        "  seconds REAL,"
        "  PRIMARY KEY (identifier, idx))")
    # electric / acoustic / mixed / alterego, per date. A tag and never a deletion -- see
    # catalog/showtypes.py for why, at length. Stored so that every consumer can decide for
    # itself: length statistics exclude the acoustic nights, and nothing else has to change.
    #
    # `identifier` is the tape the verdict was read off. The design listed three columns; this is
    # the fourth, because `evidence` without it is a claim with no citation -- "notes describe an
    # acoustic set" names no notes, and re-finding them means re-scanning every tape of that date.
    conn.execute(
        "CREATE TABLE show_types("
        "  date TEXT PRIMARY KEY,"
        "  kind TEXT NOT NULL,"
        "  evidence TEXT,"
        "  identifier TEXT)")


def _m0004_listings(conn: sqlite3.Connection) -> None:
    # The taper's own written tracklist, read at INGEST and stored, because the description it is
    # read out of exists nowhere else. Descriptions live only in the raw cache; the cache is
    # gitignored; so a derive that read tracklists for itself would work on the machine that
    # pulled and come back with nothing on the server. That is the same failure `uploader` had --
    # see the recordings mirror above -- and the fix is the same one: project it once, at the only
    # moment the raw payload is open, and let every later stage read a table.
    #
    # Shaped exactly like recordings/recording_tracks (a header row, then the entries) rather than
    # repeating the reading on every entry. `reading` is which of the seven readings in
    # catalog/tracklists.py produced this, kept so a wrong listing can be argued with instead of
    # guessed at.
    #
    # `matched` is stored, and stored as a column rather than implied by presence, because it is
    # the only thing a consumer may join on. A listing that did not line up with the tape's files
    # is kept -- the best failed attempt is what makes a mismatch diagnosable, and it is what the
    # review queue counts -- but anything using it to time a song has to ignore this column on
    # purpose. Dropping the unmatched rows instead would have made "no listing" and "a listing we
    # could not trust" the same absence, and only one of those is worth a human's attention.
    conn.execute(
        "CREATE TABLE recording_listings("
        "  identifier TEXT PRIMARY KEY REFERENCES recordings(identifier) ON DELETE CASCADE,"
        "  reading TEXT NOT NULL,"
        "  n_entries INTEGER NOT NULL,"
        "  matched INTEGER NOT NULL)")
    conn.execute(
        "CREATE TABLE recording_listing_entries("
        "  identifier TEXT NOT NULL REFERENCES recording_listings(identifier) ON DELETE CASCADE,"
        "  idx INTEGER NOT NULL,"
        "  song TEXT NOT NULL,"
        "  segue INTEGER NOT NULL,"
        "  PRIMARY KEY (identifier, idx))")


def _m0005_durations(conn: sqlite3.Connection) -> None:
    # One row per performance: one song, one night, one place in the setlist, and how long it took
    # once every taper who recorded it had voted. The primary key is the SLOT and not the song,
    # because a song played twice in a night is two performances and keying on the song would
    # silently keep one of them.
    #
    # `set_label` and not `set`, which is a reserved word. The export bundle spells it `set` and
    # is the only place the two names meet.
    #
    # The consensus columns are stored beside the answer rather than left derivable, because they
    # are not derivable: n_tapes_seen counts tapes this row DISCARDED, and after the discard there
    # is nothing left in the database that remembers they existed. Publishing `seconds` without
    # them turns a resolved three-way dispute into an uncorroborated measurement that reads
    # exactly like a well-attested one.
    #
    # `withheld` is the one derived value here that is stored anyway, against the rule the corpus
    # tables follow. The rule is about a stored copy disagreeing with what it was derived from,
    # and the risk runs the other way for this column: the derivation is a priority ordering over
    # four other columns, so re-deriving it in SQL would be a SECOND implementation of it, and two
    # implementations of "why was this held back" is exactly the drift Performance.withheld exists
    # to prevent. One implementation, in Python, written down here.
    #
    # No `double_play` boolean beside `double_play_parts`. It would be true precisely when the
    # parts column is not NULL, and the corpus tables already refuse to keep a column that is a
    # second copy of one beside it.
    conn.execute(
        "CREATE TABLE performance_durations("
        "  date TEXT NOT NULL,"
        "  set_label TEXT NOT NULL,"
        "  position INTEGER NOT NULL,"
        "  song TEXT NOT NULL,"
        "  seconds REAL NOT NULL,"
        "  n_tapes INTEGER NOT NULL,"
        "  n_tapes_seen INTEGER NOT NULL,"
        "  n_ballots INTEGER NOT NULL,"
        "  spread_seconds REAL NOT NULL,"
        "  spread_all_tapes REAL NOT NULL,"
        "  suspect INTEGER NOT NULL,"
        "  resolved_by TEXT,"
        "  segued INTEGER NOT NULL,"
        "  show_type TEXT NOT NULL,"
        "  excluded TEXT,"
        "  withheld TEXT,"
        "  double_play_parts INTEGER,"
        "  sandwich_total_seconds REAL,"
        "  is_longest_part INTEGER,"
        "  PRIMARY KEY (date, set_label, position))")
    # "how long does this song usually run, and when did they stretch it" is every question asked
    # of this table, and without the index it is a full scan of twenty thousand performances.
    conn.execute("CREATE INDEX performance_durations_song ON performance_durations(song)")
    # The published statistic. Derived from the table above by song_stats() and stored because the
    # export reads it directly -- but note what is NOT here: no exclusion counts. Those are read
    # back off `withheld` above, so the tally of what was left out and the statistic it was left
    # out of can never be computed from different runs.
    conn.execute(
        "CREATE TABLE song_length_stats("
        "  song TEXT PRIMARY KEY,"
        "  n INTEGER NOT NULL,"
        "  median_seconds REAL NOT NULL,"
        "  mean_seconds REAL NOT NULL,"
        "  min_seconds REAL NOT NULL,"
        "  max_seconds REAL NOT NULL,"
        "  p10_seconds REAL NOT NULL,"
        "  p90_seconds REAL NOT NULL,"
        "  stdev_seconds REAL NOT NULL,"
        "  longest_date TEXT NOT NULL)")
    # Tapes we hold and could not time, kept as two separate lists because they want two different
    # things from a person. A REVIEW is a tape whose tracks would not line up with its night: the
    # counts that failed to reconcile are stored beside it, so the next reader starts from "17
    # files, 15 songs, 15 in the description" rather than from re-running the join by hand. `url`
    # is stored rather than built at read time so the row is self-contained in a dump.
    conn.execute(
        "CREATE TABLE duration_review("
        "  identifier TEXT PRIMARY KEY,"
        "  date TEXT NOT NULL,"
        "  n_tracks INTEGER NOT NULL,"
        "  n_setlist INTEGER NOT NULL,"
        "  n_desc INTEGER NOT NULL,"
        "  reason TEXT NOT NULL,"
        "  url TEXT NOT NULL)")
    # ABANDONED is the other kind, and it is not a review queue: a tape whose longest track runs
    # past an hour is a bounced set, one file holding a whole set, and no amount of looking at it
    # will make it time a song. Listed anyway so the tape count and the timed-tape count add up.
    conn.execute(
        "CREATE TABLE duration_abandoned("
        "  identifier TEXT PRIMARY KEY,"
        "  date TEXT NOT NULL,"
        "  n_tracks INTEGER NOT NULL,"
        "  longest_seconds REAL NOT NULL)")
    # Every fuzzy call the chain made, kept whole. An edge is never a failure -- it is a decision
    # with its reasons -- and `detail_json` holds those reasons verbatim rather than flattened
    # into columns, because the kinds carry different facts and a shared column set would be
    # mostly NULL and still wrong for the next kind added.
    #
    # A synthetic id because there is no natural key: one tape can raise several edges of one kind
    # about one song, and each of them is a separate observation about a separate track.
    conn.execute(
        "CREATE TABLE duration_edges("
        "  id INTEGER PRIMARY KEY,"
        "  kind TEXT NOT NULL,"
        "  date TEXT NOT NULL,"
        "  identifier TEXT NOT NULL,"
        "  song TEXT NOT NULL,"
        "  detail_json TEXT NOT NULL)")
    conn.execute("CREATE INDEX duration_edges_kind ON duration_edges(kind)")


# Forward-only. Never edit a shipped migration; add the next number instead.
MIGRATIONS: list[Migration] = [
    Migration(1, "baseline", _m0001_baseline),
    Migration(2, "corpus", _m0002_corpus),
    Migration(3, "recordings", _m0003_recordings),
    Migration(4, "listings", _m0004_listings),
    Migration(5, "durations", _m0005_durations),
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

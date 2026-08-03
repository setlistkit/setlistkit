# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""What was predicted, when, and how it turned out. The one table set that cannot be rebuilt.

Every other table in this database is derived and disposable: ``replace_shows`` replaces the whole
corpus on every ingest, the durations chain recomputes from the tapes, and losing any of it costs
a re-run. A prediction is different in kind. It is a claim made at a moment, and once the corpus
has moved past that moment it cannot be produced again -- re-running the model against today's
data does not recover what it would have said in March, it produces a new and better-informed
answer wearing March's date.

Two protections follow from that, and both are here rather than in the caller:

* **The no-shrink rule.** A run that leaves a table with less than half of what it started with is
  assumed broken rather than authoritative. Ported from the version this comes from, which arrived
  at it the hard way, and it is the same guard ``slkit build`` applies to the corpus.
* **Nothing else writes these tables.** ``replace_shows`` does not touch them, and a test asserts
  it: an ingest that quietly took the ledger with it would be unrecoverable and silent.

TRANSACTIONS ARE THE CALLER'S TO OPEN. The connection runs in autocommit mode (see
:mod:`setlistkit.store.db`), so a bare :func:`record` commits itself, which is what a one-shot
write wants. A replay wraps its whole walk in one :func:`setlistkit.store.migrations.transaction`
instead, because the no-shrink guard has to run BEFORE the commit -- a guard with nothing left to
refuse is not a guard.

:func:`score_outcome` is a pure function of ``(ranked, actual)`` and touches no database, so the
metric arithmetic can be pinned without one. That is how it arrived and it is worth keeping: the
three numbers below are what the whole facility is judged on, and they should be checkable by
hand.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from ..diagnostics import ERROR, Diagnostic, DiagnosticError
from ..model.rollout import SongChance

# The three tables this module owns. Named here so the no-shrink guard, `slkit show picks` and the
# ingest-does-not-touch-them test all count the same set rather than three hand-typed lists.
LEDGER_TABLES = ("pick_iterations", "picks", "pick_outcomes")

# Below this fraction of what a run started with, a table is assumed to have lost rows by accident.
# The same number and the same reasoning as the corpus guard in `slkit build`.
NO_SHRINK_FRAC = 0.5


@dataclass(frozen=True)
class Outcome:
    """How one night's prediction actually did.

    ``recall_at_n`` is what it got right, out of what was played. ``expected_hits`` is what it
    SAID it would get right -- the sum of the confidences it published. ``calibration_ratio`` is
    the first over the second, and it is the number worth watching: a model can be right rarely
    and honest about it, which is useful, or right rarely while claiming otherwise, which is
    broken in a way no hit rate reveals.

    ``n_actual`` is how many songs were played, carried because every other number here is a share
    or a sum over it and a reader cannot check any of them without it.
    """

    recall_at_n: float
    expected_hits: float
    actual_hits: int
    calibration_ratio: float
    n_actual: int


def score_outcome(ranked: Sequence[SongChance], actual: Iterable[str]) -> Outcome:
    """Grade a ranked list against what was played. Pure: no database, no clock.

    Both guards below return 0.0 rather than raising, and they mean different things. An empty
    ``actual`` is a night with no songs, which the gradeability floor should already have
    excluded. A zero ``expected`` is a prediction that claimed nothing -- every confidence at
    zero -- and dividing by it would be reporting infinite calibration for a model that made no
    claim at all.
    """
    played = set(actual)
    hits = len({chance.song for chance in ranked} & played)
    expected = sum(chance.confidence for chance in ranked)
    return Outcome(
        recall_at_n=hits / len(played) if played else 0.0,
        expected_hits=expected,
        actual_hits=hits,
        calibration_ratio=hits / expected if expected > 0 else 0.0,
        n_actual=len(played),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iteration_id(conn: sqlite3.Connection, cutoff: str) -> int | None:
    """The stored iteration for ``cutoff``, or None if none was ever recorded."""
    row = conn.execute("SELECT id FROM pick_iterations WHERE cutoff_date = ?",
                       (cutoff,)).fetchone()
    return row[0] if row else None


def record(conn: sqlite3.Connection, cutoff: str,
           nights: Mapping[str, Sequence[SongChance]],
           config: Mapping[str, object]) -> int:
    """Store one iteration's picks. Returns how many pick rows were written.

    ``nights`` maps a night to its ranked list, rather than taking a single night, even though the
    replay writes exactly one night per iteration. The table shape needs it -- ``picks`` is keyed
    ``(iteration_id, night_date, rank)`` -- and a multi-night run is the whole point of the
    forward-sim work downstream, so a single-night signature would only have to be widened later.
    The replay passes a one-entry mapping and that is not a special case.

    Re-recording a cutoff REPLACES its picks rather than adding to them: a second run at the same
    cutoff is a correction, and leaving the old ranks behind would produce a list with two songs
    at rank 3 and no way to tell which run each came from.
    """
    conn.execute(
        "INSERT INTO pick_iterations(cutoff_date, run_ts, config_json) VALUES(?, ?, ?) "
        "ON CONFLICT(cutoff_date) DO UPDATE SET run_ts = excluded.run_ts, "
        "config_json = excluded.config_json",
        (cutoff, _now(), json.dumps(config, sort_keys=True, default=str)))
    ident = iteration_id(conn, cutoff)
    conn.execute("DELETE FROM picks WHERE iteration_id = ?", (ident,))
    rows = [(ident, night, chance.rank, chance.song, chance.confidence)
            for night, ranked in nights.items() for chance in ranked]
    conn.executemany(
        "INSERT INTO picks(iteration_id, night_date, rank, song, confidence) "
        "VALUES(?, ?, ?, ?, ?)", rows)
    return len(rows)


def honest_iteration(conn: sqlite3.Connection, night: str) -> tuple[int, str] | None:
    """The iteration that is allowed to be graded against ``night``: the latest cutoff before it.

    Strictly before, and in SQL, so there is nowhere for an off-by-one to hide. The version this
    comes from reconstructed the same relation in Python after the fact
    (``_last_honest_cutoff``) because its iterations were computed out of order; a replay that
    advances one show at a time makes it a lookup.
    """
    row = conn.execute(
        "SELECT id, cutoff_date FROM pick_iterations WHERE cutoff_date < ? "
        "ORDER BY cutoff_date DESC LIMIT 1", (night,)).fetchone()
    return (row[0], row[1]) if row else None


def picks_for(conn: sqlite3.Connection, ident: int, night: str) -> list[SongChance]:
    """One iteration's ranked list for one night, in rank order."""
    return [SongChance(rank=row[0], song=row[1], confidence=row[2])
            for row in conn.execute(
                "SELECT rank, song, confidence FROM picks "
                "WHERE iteration_id = ? AND night_date = ? ORDER BY rank", (ident, night))]


def grade(conn: sqlite3.Connection, night: str, actual: Iterable[str]) -> Outcome | None:
    """Score the stored prediction for ``night`` against what was played, and record the result.

    Returns None when there is nothing to grade -- no iteration predates the night, or the one
    that does never covered it. That is an ordinary state at the start of a replay, not an error.

    The picks are read BACK OUT of the table rather than taken from the caller, which costs a
    query and buys a round trip: what is graded is what was stored, so a write that dropped or
    reordered rows shows up as a bad grade instead of passing unnoticed.
    """
    honest = honest_iteration(conn, night)
    if honest is None:
        return None
    ident, cutoff = honest
    ranked = picks_for(conn, ident, night)
    if not ranked:
        return None
    played = sorted(set(actual))
    outcome = score_outcome(ranked, played)
    conn.execute(
        "INSERT INTO pick_outcomes(night_date, actual_json, recall_at_n, expected_hits, "
        "actual_hits, calibration_ratio, scored_asof) VALUES(?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(night_date) DO UPDATE SET actual_json = excluded.actual_json, "
        "recall_at_n = excluded.recall_at_n, expected_hits = excluded.expected_hits, "
        "actual_hits = excluded.actual_hits, "
        "calibration_ratio = excluded.calibration_ratio, scored_asof = excluded.scored_asof",
        (night, json.dumps(played), outcome.recall_at_n, outcome.expected_hits,
         outcome.actual_hits, outcome.calibration_ratio, cutoff))
    return outcome


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row count for each ledger table. What the no-shrink guard measures, before and after."""
    return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # nosec B608
            for table in LEDGER_TABLES}


def shrinkage(before: Mapping[str, int], after: Mapping[str, int]) -> list[str]:
    """Tables that lost more than half their rows this run, described one per line.

    A list rather than a boolean because which table shrank is the diagnosis. Iterations shrinking
    means the walk covered fewer cutoffs; picks shrinking with iterations steady means ``top_n``
    moved or a night came back empty. Reporting "the ledger shrank" would tell a reader there is a
    problem and nothing about where.
    """
    return [f"{table}: {before[table]} -> {after[table]} rows" for table in LEDGER_TABLES
            if before.get(table, 0) > 0 and after[table] < NO_SHRINK_FRAC * before[table]]


def guard_no_shrink(before: Mapping[str, int], after: Mapping[str, int], *,
                    force: bool) -> None:
    """Refuse a run that lost more than half of any ledger table.

    Here rather than in :mod:`setlistkit.picks.replay`, where it started, because a second walk
    now needs it. The rule, the fraction it uses and the tables it counts all live in this module,
    so a guard that reads across to another one for its enforcement is the one piece of the rule
    sitting away from the rest -- and the alternative, a second walk importing a private name,
    means changing the rule in one place and breaking a caller that was never looked at.
    """
    complaints = shrinkage(before, after)
    if not complaints or force:
        return
    raise DiagnosticError(Diagnostic(
        severity=ERROR,
        summary="refusing to write: this run shrank the picks ledger",
        detail="\n".join(f"  {line}" for line in complaints) + "\n\n"
               f"A run leaving a table under {NO_SHRINK_FRAC:.0%} of what it started with\n"
               "is assumed broken rather than authoritative. The ledger is the one thing in this\n"
               "database that cannot be rebuilt: a prediction is a claim made at a moment, and\n"
               "re-running the model now produces a better-informed answer, not the same one.\n\n"
               "A LOWER top_n does this legitimately, and is the usual cause: re-recording every\n"
               "cutoff with a shorter list replaces long rows with short ones. A narrower window\n"
               "does NOT -- a walk never deletes an iteration it did not re-record, so covering\n"
               "fewer nights leaves the earlier ones in place and cannot shrink anything.\n\n"
               "Nothing was written; --force publishes a shrink you meant, and --dry-run reports\n"
               "the walk without touching the ledger at all.",
    ))

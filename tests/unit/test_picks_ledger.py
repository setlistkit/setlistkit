# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the picks ledger: the metric arithmetic, and what SQLite hands back.

``score_outcome`` is pure, so the three numbers the whole facility is judged on can be pinned by
hand -- which is the point of it having no database and no clock. Those tests are first, and every
expected value below is worked out in the assertion rather than copied from a run.

The rest are about storage, and one of them is the reason ``grade`` reads picks back out of the
table instead of grading the list it was handed: a write that dropped or reordered rows would
otherwise never show up.
"""

import json
import sqlite3

import pytest

from setlistkit.model.rollout import SongChance
from setlistkit.picks.ledger import (
    LEDGER_TABLES,
    NO_SHRINK_FRAC,
    grade,
    honest_iteration,
    iteration_id,
    picks_for,
    record,
    score_outcome,
    shrinkage,
    table_counts,
)
from setlistkit.store.migrations import MIGRATIONS, apply_pending

CONFIG = {"score": {"halflife": 200}, "predict": {"top_n": 3}}


@pytest.fixture(name="conn")
def _conn_fixture():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    apply_pending(conn, MIGRATIONS)
    try:
        yield conn
    finally:
        conn.close()


def _ranked(*pairs):
    return [SongChance(rank=index, song=song, confidence=conf)
            for index, (song, conf) in enumerate(pairs, start=1)]


def test_a_perfect_and_perfectly_honest_prediction_scores_one_across_the_board():
    outcome = score_outcome(_ranked(("A", 1.0), ("B", 1.0)), ["A", "B"])
    assert outcome.recall_at_n == 1.0
    assert outcome.actual_hits == 2
    assert outcome.expected_hits == 2.0
    assert outcome.calibration_ratio == 1.0


def test_recall_is_over_what_was_played_not_over_what_was_listed():
    """Thirty listed songs and a fifteen-song night: the denominator is fifteen.

    Over the list instead, a longer ``top_n`` would lower recall for free while predicting strictly
    more, which is backwards.
    """
    outcome = score_outcome(_ranked(*[(f"S{i}", 0.1) for i in range(30)]),
                            [f"S{i}" for i in range(5)] + [f"X{i}" for i in range(10)])
    assert outcome.n_actual == 15
    assert outcome.actual_hits == 5
    assert outcome.recall_at_n == pytest.approx(5 / 15)


def test_expected_hits_is_the_sum_of_what_the_model_claimed():
    outcome = score_outcome(_ranked(("A", 0.9), ("B", 0.5), ("C", 0.1)), ["A"])
    assert outcome.expected_hits == pytest.approx(1.5)
    assert outcome.calibration_ratio == pytest.approx(1 / 1.5)


def test_a_model_that_claimed_more_than_it_got_calibrates_below_one():
    assert score_outcome(_ranked(("A", 1.0), ("B", 1.0)), ["A"]).calibration_ratio == 0.5


def test_a_model_that_claimed_less_than_it_got_calibrates_above_one():
    assert score_outcome(_ranked(("A", 0.25), ("B", 0.25)), ["A", "B"]).calibration_ratio == 4.0


def test_a_prediction_that_claimed_nothing_does_not_divide_by_zero():
    """Every confidence at zero. Infinite calibration for a model that made no claim is worse
    than useless: it is the best-looking number in the report."""
    outcome = score_outcome(_ranked(("A", 0.0), ("B", 0.0)), ["A", "B"])
    assert outcome.expected_hits == 0.0
    assert outcome.calibration_ratio == 0.0
    assert outcome.actual_hits == 2


def test_a_night_with_no_songs_scores_zero_rather_than_raising():
    outcome = score_outcome(_ranked(("A", 0.5)), [])
    assert outcome.recall_at_n == 0.0 and outcome.n_actual == 0


def test_an_empty_prediction_scores_zero_rather_than_raising():
    outcome = score_outcome([], ["A", "B"])
    assert outcome.actual_hits == 0 and outcome.recall_at_n == 0.0
    assert outcome.calibration_ratio == 0.0


def test_a_song_played_twice_in_a_night_is_one_hit_and_one_song():
    """Setlists repeat songs -- a reprise, a tease reprised late. A hit is a hit once."""
    outcome = score_outcome(_ranked(("A", 1.0)), ["A", "B", "A"])
    assert outcome.actual_hits == 1 and outcome.n_actual == 2


def test_picks_come_back_out_in_rank_order(conn):
    assert record(conn, "2026-02-01", {"2026-02-02": _ranked(("A", 0.9), ("B", 0.4))},
                  CONFIG) == 2
    ident = iteration_id(conn, "2026-02-01")
    assert [(row.rank, row.song) for row in picks_for(conn, ident, "2026-02-02")] \
        == [(1, "A"), (2, "B")]


def test_one_iteration_can_hold_several_nights(conn):
    """The forward-sim work downstream writes many nights per cutoff; the replay writes one."""
    written = record(conn, "2026-02-01",
                     {"2026-02-02": _ranked(("A", 0.9)),
                      "2026-02-03": _ranked(("B", 0.8), ("C", 0.7))}, CONFIG)
    assert written == 3
    ident = iteration_id(conn, "2026-02-01")
    assert len(picks_for(conn, ident, "2026-02-03")) == 2


def test_re_recording_a_cutoff_replaces_its_picks_rather_than_doubling_them(conn):
    """Otherwise the table holds two songs at rank 3 and no way to tell the runs apart."""
    record(conn, "2026-02-01", {"2026-02-02": _ranked(("A", 0.9), ("B", 0.4))}, CONFIG)
    record(conn, "2026-02-01", {"2026-02-02": _ranked(("C", 0.7))}, CONFIG)
    ident = iteration_id(conn, "2026-02-01")
    assert [row.song for row in picks_for(conn, ident, "2026-02-02")] == ["C"]
    assert conn.execute("SELECT COUNT(*) FROM pick_iterations").fetchone()[0] == 1


def test_the_stored_config_is_the_config_the_run_was_made_under(conn):
    record(conn, "2026-02-01", {"2026-02-02": _ranked(("A", 0.9))}, CONFIG)
    stored = conn.execute("SELECT config_json FROM pick_iterations").fetchone()[0]
    assert json.loads(stored) == CONFIG


def test_the_honest_iteration_for_a_night_is_the_latest_cutoff_strictly_before_it(conn):
    for cutoff in ("2026-01-10", "2026-01-20", "2026-02-05"):
        record(conn, cutoff, {"2026-02-01": _ranked(("A", 0.5))}, CONFIG)
    assert honest_iteration(conn, "2026-02-01")[1] == "2026-01-20"


def test_an_iteration_cut_off_on_the_night_itself_is_not_honest_about_it(conn):
    """Strictly before. A cutoff of D saw everything up to D, which includes D."""
    record(conn, "2026-02-01", {"2026-02-01": _ranked(("A", 0.5))}, CONFIG)
    assert honest_iteration(conn, "2026-02-01") is None


def test_grading_a_night_nothing_predates_returns_nothing(conn):
    assert grade(conn, "2026-02-01", ["A"]) is None


def test_grading_a_night_the_honest_iteration_never_covered_returns_nothing(conn):
    """An iteration exists and predates the night, but it holds picks for a different one."""
    record(conn, "2026-01-10", {"2026-01-11": _ranked(("A", 0.5))}, CONFIG)
    assert grade(conn, "2026-02-01", ["A"]) is None


def test_a_grade_is_computed_from_what_the_table_holds_not_from_what_was_passed_in(conn):
    """The round trip. Delete a row behind ``grade``'s back and the grade has to notice.

    This is the whole reason ``grade`` costs an extra query instead of scoring the in-memory list:
    a write that dropped rows would otherwise be graded as though it had not.
    """
    record(conn, "2026-01-10", {"2026-02-01": _ranked(("A", 1.0), ("B", 1.0))}, CONFIG)
    conn.execute("DELETE FROM picks WHERE song = 'B'")
    outcome = grade(conn, "2026-02-01", ["A", "B"])
    assert outcome.actual_hits == 1
    assert outcome.expected_hits == 1.0


def test_a_grade_lands_in_the_outcomes_table_with_what_was_played(conn):
    record(conn, "2026-01-10", {"2026-02-01": _ranked(("A", 0.9), ("B", 0.4))}, CONFIG)
    outcome = grade(conn, "2026-02-01", ["B", "A", "Z"])
    row = conn.execute("SELECT actual_json, actual_hits, recall_at_n, scored_asof "
                       "FROM pick_outcomes WHERE night_date = '2026-02-01'").fetchone()
    assert json.loads(row[0]) == ["A", "B", "Z"]
    assert row[1] == outcome.actual_hits == 2
    assert row[2] == pytest.approx(2 / 3)
    # The cutoff it was honest as of, so a stored grade says what it was allowed to know.
    assert row[3] == "2026-01-10"


def test_re_grading_a_night_replaces_its_outcome(conn):
    record(conn, "2026-01-10", {"2026-02-01": _ranked(("A", 0.9))}, CONFIG)
    grade(conn, "2026-02-01", ["A"])
    grade(conn, "2026-02-01", ["Z"])
    rows = conn.execute("SELECT actual_hits FROM pick_outcomes").fetchall()
    assert rows == [(0,)]


def test_deleting_an_iteration_takes_its_picks_with_it(conn):
    """ON DELETE CASCADE, which needs foreign keys ON -- so this also asserts the Store's pragma."""
    conn.execute("PRAGMA foreign_keys = ON")
    record(conn, "2026-01-10", {"2026-02-01": _ranked(("A", 0.9), ("B", 0.4))}, CONFIG)
    conn.execute("DELETE FROM pick_iterations WHERE cutoff_date = '2026-01-10'")
    assert conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0] == 0


def test_table_counts_reports_every_ledger_table(conn):
    assert set(table_counts(conn)) == set(LEDGER_TABLES)
    record(conn, "2026-01-10", {"2026-02-01": _ranked(("A", 0.9), ("B", 0.4))}, CONFIG)
    counts = table_counts(conn)
    assert counts["pick_iterations"] == 1 and counts["picks"] == 2


def test_a_table_that_lost_more_than_half_its_rows_is_named(conn):
    complaints = shrinkage({"pick_iterations": 100, "picks": 100, "pick_outcomes": 100},
                           {"pick_iterations": 100, "picks": 10, "pick_outcomes": 100})
    assert len(complaints) == 1 and complaints[0].startswith("picks: 100 -> 10")


def test_shrinking_to_exactly_the_threshold_is_not_a_shrink(conn):
    """Strictly below, so the boundary is not a coin flip on a rounding error."""
    half = int(100 * NO_SHRINK_FRAC)
    assert not shrinkage({table: 100 for table in LEDGER_TABLES},
                         {table: half for table in LEDGER_TABLES})


def test_a_first_run_filling_an_empty_ledger_is_not_a_shrink(conn):
    assert not shrinkage({table: 0 for table in LEDGER_TABLES},
                         {table: 0 for table in LEDGER_TABLES})

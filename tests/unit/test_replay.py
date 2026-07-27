# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for walking the corpus one night at a time and grading each before revealing it.

Three of these are protections rather than behavior, and they exist because the ledger is the one
thing in this database that cannot be rebuilt. A prediction is a claim made at a moment; re-running
the model after the corpus has moved does not recover it, it produces a better-informed answer
wearing the old date. So: an ingest must not touch these tables, a shrinking run must be refused,
and a refused run must leave nothing behind.

The fourth is ``test_a_night_is_predicted_from_the_show_before_it``, which is the honesty of the
walk itself. Everything the replay reports is a number about that cutoff being right.
"""

import sqlite3
from inspect import signature

import pytest

from setlistkit.diagnostics import DiagnosticError
from setlistkit.model.backtest import backtest
from setlistkit.model.scores import ScoreConfig
from setlistkit.model.setlists import PredictConfig
from setlistkit.picks import ledger
from setlistkit.picks.replay import MIN_GRADEABLE_SONGS, ReplaySpec, replay
from setlistkit.store.corpus import replace_shows
from setlistkit.store.migrations import MIGRATIONS, apply_pending

LOOSE = ScoreConfig(min_show_songs=1, cooldown=())
QUICK = PredictConfig(rollouts=25, top_n=10)
SPEC = ReplaySpec(score=LOOSE, predict=QUICK)

REGULARS = ("Aurora", "Rebubula", "Meat", "Zelda", "Buster", "Wormhole", "Gumbo", "Cavern",
            "Timmy", "Moth")


@pytest.fixture(name="conn")
def _conn_fixture():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    apply_pending(conn, MIGRATIONS)
    try:
        yield conn
    finally:
        conn.close()


def _entry(song):
    return {"song": song, "segue": False, "non_song": False}


def _show(date, *songs):
    return {"date": date, "source": "description", "identifier": f"t{date}",
            "sets": [[_entry(s) for s in songs]], "encore": []}


def _nights(days=range(1, 13), songs=REGULARS):
    return [_show(f"2026-01-{day:02d}", *songs) for day in days]


def test_every_night_but_the_first_is_graded(conn):
    report = replay(conn, _nights(), SPEC)
    assert len(report.nights) == 11
    assert report.skipped == 1


def test_the_corpus_first_night_is_skipped_because_nothing_precedes_it(conn):
    """Reported rather than swallowed: "11 nights" against "12 in the window" is what says so."""
    report = replay(conn, _nights(), SPEC)
    assert "2026-01-01" not in {row.night for row in report.nights}
    assert report.skipped == 1


def test_a_night_is_predicted_from_the_show_before_it(conn):
    report = replay(conn, _nights(), SPEC)
    assert [(row.night, row.cutoff) for row in report.nights][:3] == [
        ("2026-01-02", "2026-01-01"),
        ("2026-01-03", "2026-01-02"),
        ("2026-01-04", "2026-01-03"),
    ]


def test_a_windowed_night_is_still_predicted_from_the_show_before_it_even_outside_the_window(conn):
    """The window bounds what is GRADED, not what is learned from. The show before the window's
    first night is outside it and is still the honest place to predict from."""
    report = replay(conn, _nights(), ReplaySpec(score=LOOSE, predict=QUICK, since="2026-01-06"))
    assert report.nights[0].night == "2026-01-06"
    assert report.nights[0].cutoff == "2026-01-05"
    assert report.skipped == 0


def test_the_window_is_inclusive_at_both_ends(conn):
    report = replay(conn, _nights(),
                    ReplaySpec(score=LOOSE, predict=QUICK, since="2026-01-04",
                               until="2026-01-07"))
    assert [row.night for row in report.nights] == ["2026-01-04", "2026-01-05", "2026-01-06",
                                                    "2026-01-07"]


def test_a_night_too_short_to_grade_is_not_graded(conn):
    """A night recorded as two songs grades the taper's notes, not the band."""
    shows = _nights(range(1, 6)) + [_show("2026-01-06", "Aurora", "Meat")] + _nights(range(7, 10))
    report = replay(conn, shows, SPEC)
    assert "2026-01-06" not in {row.night for row in report.nights}


def test_the_gradeability_floor_matches_the_backtest():
    """Two reports that disagree about which nights count disagree about the same corpus.

    Read off the signature rather than restated as a literal, so moving one moves this.
    """
    assert MIN_GRADEABLE_SONGS == signature(backtest).parameters["min_songs"].default


def test_a_perfectly_predictable_band_is_recalled_completely(conn):
    """Ten songs every night, ten recorded. There is nothing else it could have named."""
    report = replay(conn, _nights(), SPEC)
    assert all(row.outcome.recall_at_n == 1.0 for row in report.nights)
    assert report.mean_recall == 1.0


def test_a_band_that_repeats_itself_is_predicted_confidently_and_calibrates_near_one(conn):
    report = replay(conn, _nights(), SPEC)
    assert report.calibration == pytest.approx(1.0, abs=0.05)


def test_the_aggregate_pools_hits_over_expected_rather_than_averaging_ratios(conn):
    """A night that expected 0.4 and got two scores 5.0 and would drag a mean around alone."""
    report = replay(conn, _nights(), SPEC)
    assert report.total_hits == sum(row.outcome.actual_hits for row in report.nights)
    assert report.calibration == pytest.approx(report.total_hits / report.total_expected)


def test_an_empty_window_reports_nothing_rather_than_dividing_by_zero(conn):
    report = replay(conn, _nights(), ReplaySpec(score=LOOSE, predict=QUICK, since="2027-01-01"))
    assert report.nights == ()
    assert report.mean_recall == 0.0 and report.calibration == 0.0


def test_a_walk_writes_one_iteration_per_night_and_one_outcome(conn):
    replay(conn, _nights(), SPEC)
    counts = ledger.table_counts(conn)
    assert counts["pick_iterations"] == 11
    assert counts["pick_outcomes"] == 11
    assert counts["picks"] == 11 * 10


def test_a_dry_run_writes_nothing_at_all(conn):
    report = replay(conn, _nights(), SPEC, dry_run=True)
    assert report.nights
    assert ledger.table_counts(conn) == {table: 0 for table in ledger.LEDGER_TABLES}


def test_a_dry_run_reports_the_same_numbers_as_the_run_that_writes(conn):
    """The metrics come from the same arithmetic; writing only adds a round trip through SQLite."""
    dry = replay(conn, _nights(), SPEC, dry_run=True)
    wet = replay(conn, _nights(), SPEC)
    assert [row.outcome for row in dry.nights] == [row.outcome for row in wet.nights]


def test_the_walk_is_reproducible(conn):
    first = replay(conn, _nights(), SPEC, dry_run=True)
    second = replay(conn, _nights(), SPEC, dry_run=True)
    assert first == second


def test_progress_is_reported_as_each_night_lands(conn):
    """Silence for the couple of minutes a real walk takes is indistinguishable from a hang."""
    seen = []
    report = replay(conn, _nights(), SPEC, progress=seen.append)
    assert seen == list(report.nights)


def test_a_narrower_window_cannot_shrink_the_ledger(conn):
    """Stated as a test because the guard's advice text asserts it.

    A replay never deletes an iteration it did not re-record, so covering fewer nights leaves the
    earlier ones in place. Which means the only ordinary way to trip the guard is a lower
    ``top_n``, and the message says so rather than sending a reader to widen a window that was
    never the cause.
    """
    replay(conn, _nights(), SPEC)
    before = ledger.table_counts(conn)
    replay(conn, _nights(), ReplaySpec(score=LOOSE, predict=QUICK, since="2026-01-11"))
    assert ledger.table_counts(conn) == before


def test_a_run_that_shrinks_the_ledger_is_refused(conn):
    """A lower ``top_n`` replaces every long list with a short one: 110 pick rows become 33."""
    replay(conn, _nights(), SPEC)
    with pytest.raises(DiagnosticError) as caught:
        replay(conn, _nights(), ReplaySpec(score=LOOSE, predict=PredictConfig(rollouts=25,
                                                                              top_n=3)))
    assert "shrank the picks ledger" in caught.value.diagnostic.summary
    assert "picks: 110 -> 33 rows" in caught.value.diagnostic.detail


def test_a_refused_run_leaves_the_previous_ledger_exactly_as_it_was(conn):
    """The guard runs inside the transaction, so tripping it rolls the whole walk back.

    A guard that fires after the commit is a report, not a guard.
    """
    replay(conn, _nights(), SPEC)
    before = ledger.table_counts(conn)
    with pytest.raises(DiagnosticError):
        replay(conn, _nights(), ReplaySpec(score=LOOSE, predict=PredictConfig(rollouts=25,
                                                                              top_n=3)))
    assert ledger.table_counts(conn) == before


def test_a_shrink_can_be_published_on_purpose(conn):
    replay(conn, _nights(), SPEC)
    replay(conn, _nights(), ReplaySpec(score=LOOSE, predict=PredictConfig(rollouts=25, top_n=3)),
           force=True)
    assert ledger.table_counts(conn)["picks"] == 33


def test_a_first_run_against_an_empty_ledger_is_not_a_shrink(conn):
    assert replay(conn, _nights(), SPEC).nights


def test_the_stored_config_records_what_the_run_was_made_under(conn):
    replay(conn, _nights(), SPEC)
    stored = conn.execute("SELECT config_json FROM pick_iterations LIMIT 1").fetchone()[0]
    assert '"rollouts": 25' in stored
    assert '"min_songs": 8' in stored


def test_an_ingest_does_not_touch_the_picks_tables(conn):
    """The protection that matters most, because losing this is silent and unrecoverable.

    ``replace_shows`` replaces the WHOLE corpus on every ingest -- that is what makes every other
    table in this database disposable. If it ever grew a `DELETE FROM picks`, nothing else in the
    suite would notice.
    """
    replace_shows(conn, _nights())
    replay(conn, _nights(), SPEC)
    before = ledger.table_counts(conn)
    replace_shows(conn, _nights(range(1, 20)))
    assert ledger.table_counts(conn) == before


def test_a_night_the_ledger_already_holds_is_re_predicted_rather_than_doubled(conn):
    """Running the same window twice is a correction, not a second opinion."""
    replay(conn, _nights(), SPEC)
    replay(conn, _nights(), SPEC)
    counts = ledger.table_counts(conn)
    assert counts["pick_iterations"] == 11 and counts["picks"] == 110

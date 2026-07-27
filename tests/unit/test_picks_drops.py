# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for predicting one night from several successive cutoffs.

The walk adds one new way to leak. A plain replay predicts from the show immediately before the
night, and the boundary inside ``rollout_night`` is the only thing enforcing it. A drop also
PRE-FILTERS the corpus, which means there are now two places a cutoff can be applied and two
chances to apply it to the wrong date. So the first tests here are about what each drop was
allowed to see, and one of them checks the pre-filter is doing work rather than being decorative.

The rest cover the writing shape, which is where the schema and the walk disagree: the walk is
ordered by night and the ledger is written by cutoff, and one iteration has to end up holding
every night that cutoff looked ahead to.
"""

import sqlite3

import pytest

from setlistkit.model.scores import ScoreConfig
from setlistkit.model.setlists import PredictConfig
from setlistkit.picks import ledger
from setlistkit.picks.drops import DEFAULT_DEPTH, drops
from setlistkit.picks.replay import ReplaySpec, replay
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


def test_no_drop_is_ever_dated_on_or_after_the_night_it_predicts(conn):
    """The honesty of the whole walk in one assertion.

    Every other number here is a claim about a prediction made before the night; a single drop
    dated on its own night would make all of them wrong, and quietly.
    """
    report = drops(conn, _nights(), SPEC, depth=4, dry_run=True)
    assert report.nights
    for night in report.nights:
        for drop in night.drops:
            assert drop.cutoff < night.night


def test_the_last_drop_is_the_cutoff_a_plain_replay_would_have_used(conn):
    """Otherwise the graded number here and the one on the Prediction page are not comparable.

    Run both walks over the same corpus and check they agree about where each night was predicted
    from. This is the join between the two facilities, and it is the kind of thing that drifts
    silently the first time either walk changes how it picks a cutoff.
    """
    walked = drops(conn, _nights(), SPEC, depth=3, dry_run=True)
    plain = replay(conn, _nights(), SPEC, dry_run=True)
    by_night = {row.night: row.cutoff for row in plain.nights}
    for night in walked.nights:
        assert night.drops[-1].cutoff == by_night[night.night]


def test_an_earlier_drop_cannot_see_a_show_that_a_later_one_can(conn):
    """The pre-filter is doing work, and this is what proves it.

    A song played only on 2026-01-05 must be absent from the shortlists of nights predicted from
    cutoffs before it and present afterwards. Without the corpus filter every drop would train on
    the whole history and the drops would be identical -- which reads as a model that settled
    early rather than as a broken walk.
    """
    corpus = _nights(days=range(1, 13))
    corpus[4] = _show("2026-01-05", *REGULARS, "Meteor")
    # top_n has to leave room for an eleventh song. At the shared top_n=10 the ten nightly
    # regulars fill the list exactly, Meteor ranks eleventh on its one play, and the test would
    # report "the filter works" while measuring only that the list is too short to show it.
    wide = ReplaySpec(score=LOOSE, predict=PredictConfig(rollouts=25, top_n=15))
    report = drops(conn, corpus, wide, depth=8, dry_run=True)
    late = next(row for row in report.nights if row.night == "2026-01-12")
    before = [drop for drop in late.drops if drop.cutoff < "2026-01-05"]
    after = [drop for drop in late.drops if drop.cutoff >= "2026-01-05"]
    assert before and after
    assert not any("Meteor" in drop.songs for drop in before)
    assert any("Meteor" in drop.songs for drop in after)


def test_one_iteration_is_written_per_cutoff_holding_every_night_it_looked_ahead_to(conn):
    """``ledger.record`` replaces a cutoff's picks wholesale.

    So a walk that wrote once per night would leave each iteration holding only the last night
    that touched it, discarding the rest with no error. This asserts the grouping that prevents
    it: a middle cutoff must carry several nights.
    """
    drops(conn, _nights(), SPEC, depth=4)
    rows = dict(conn.execute(
        "SELECT cutoff_date, COUNT(DISTINCT night_date) FROM pick_iterations "
        "JOIN picks ON picks.iteration_id = pick_iterations.id GROUP BY cutoff_date"))
    assert max(rows.values()) > 1


def test_a_night_is_graded_against_the_last_drop_and_not_the_best_one(conn):
    """Grading every drop would invite quoting the best, which is picking after seeing the answer.

    ``ledger.grade`` reads the latest iteration strictly before the night, so the graded outcome
    must match scoring the final drop by hand.
    """
    report = drops(conn, _nights(), SPEC, depth=4)
    night = report.nights[-1]
    expected = ledger.score_outcome(night.drops[-1].chances, night.played)
    assert night.outcome.actual_hits == expected.actual_hits
    assert night.outcome.recall_at_n == pytest.approx(expected.recall_at_n)


def test_a_dry_run_writes_nothing_and_still_reports_the_same_grades(conn):
    """The page can be built without touching a ledger a real run wrote."""
    dry = drops(conn, _nights(), SPEC, depth=4, dry_run=True)
    assert ledger.table_counts(conn) == {"pick_iterations": 0, "picks": 0, "pick_outcomes": 0}
    wet = drops(conn, _nights(), SPEC, depth=4)
    assert [row.outcome.actual_hits for row in dry.nights] == \
           [row.outcome.actual_hits for row in wet.nights]


def test_nights_with_less_history_behind_them_than_depth_are_counted_not_padded(conn):
    """A three-drop night beside twelve-drop ones has a much shorter story.

    Reported rather than hidden, for the same reason the replay reports ``skipped``: a chart that
    does not say so reads as a model that settled early on those nights.
    """
    report = drops(conn, _nights(days=range(1, 8)), SPEC, depth=5, dry_run=True)
    assert report.short > 0
    assert report.nights[0].night == "2026-01-02"
    assert len(report.nights[0].drops) == 1


def test_the_corpus_first_night_is_skipped_because_no_cutoff_precedes_it(conn):
    report = drops(conn, _nights(), SPEC, depth=4, dry_run=True)
    assert "2026-01-01" not in {row.night for row in report.nights}
    assert report.skipped == 1


def test_drift_and_step_are_indexed_with_the_drops_and_both_open_at_zero(conn):
    """An off-by-one between the three series is the failure this catches."""
    report = drops(conn, _nights(), SPEC, depth=4, dry_run=True)
    night = report.nights[-1]
    assert len(night.drift) == len(night.step) == len(night.rbo_drift) == len(night.drops)
    assert night.drift[0] == night.step[0] == night.rbo_drift[0] == 0.0


def test_a_corpus_that_never_changes_produces_drops_that_never_move(conn):
    """The control case, and the one that would catch a walk ignoring its cutoff entirely.

    Every night here is the same ten songs, so there is nothing for a later cutoff to learn. Every
    drop must be identical and every churn number zero.

    ``settled_early`` counts every night EXCEPT the one with a single drop, and that exclusion is
    correct rather than an off-by-one: a night predicted once has no earlier list to have already
    agreed with, so it cannot have settled early on any evidence. Asserting the count equals the
    night total would have written that confusion into the suite.
    """
    report = drops(conn, _nights(), SPEC, depth=4, dry_run=True)
    assert report.mean_step == 0.0
    assert all(value == 0.0 for night in report.nights for value in night.drift)
    multi_drop = [night for night in report.nights if len(night.drops) > 1]
    assert report.settled_early == len(multi_drop) < len(report.nights)


def test_the_default_depth_is_a_default_and_not_a_limit(conn):
    """Passing a depth must override it, or the knob that decides the build cost does nothing."""
    assert DEFAULT_DEPTH != 3
    report = drops(conn, _nights(), SPEC, depth=3, dry_run=True)
    assert max(len(night.drops) for night in report.nights) == 3

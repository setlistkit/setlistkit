# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for walking the scorer forward honestly.

A backtest exists to be believed, so the tests that matter are the ones that catch it lying.
Two do: one proves a show cannot inform its own prediction, and one proves the held-out half of
a tuning split never touches the search. Both failures produce a better-looking number and no
error, which is the only kind of bug a backtest can have that matters.
"""

import pytest

from setlistkit.model.backtest import backtest, holdout, naive_baseline, recency_baseline
from setlistkit.model.scores import ScoreConfig

LOOSE = ScoreConfig(min_show_songs=1, cooldown=())
REGULARS = ("Meat", "Rebubula", "Aurora", "Zelda", "Buster", "Timmy")


def _show(date, *songs):
    return {"date": date, "source": "description", "identifier": "t1",
            "sets": [[{"song": s, "segue": False, "non_song": False} for s in songs]],
            "encore": []}


def _steady(n=40, songs=REGULARS, start=1):
    """A band that plays the same set every night. Perfectly predictable by construction."""
    return [_show(f"2026-{1 + (start + i) // 28:02d}-{1 + (start + i) % 28:02d}", *songs)
            for i in range(n)]


def test_a_perfectly_predictable_band_is_predicted_perfectly():
    result = backtest(_steady(), LOOSE, count=5, min_songs=6)
    assert result.hit_rate == pytest.approx(1.0)
    assert result.n_shows == 5


def test_a_band_that_shares_nothing_between_nights_is_not_predicted():
    shows = [_show(f"2026-01-{day:02d}", *[f"S{day}x{i}" for i in range(6)])
             for day in range(1, 21)]
    result = backtest(shows, LOOSE, count=5, min_songs=6)
    assert result.hit_rate == pytest.approx(0.0)


def test_a_show_cannot_inform_its_own_prediction():
    """The no-leak test, and the reason to trust anything else this module reports.

    One earlier night, then one night of songs played nowhere else. An honest scorer has only
    the earlier night to go on and gets none of them; one that reads the night it is predicting
    finds those songs at full recency weight, ranks them top, and reports 1.0.

    The corpus is deliberately this small. With twenty nights of a regular set behind it, the
    leaked show is outweighed twelve to one, the ranking does not move, and the test passes
    whether the boundary is enforced or not -- which is exactly what the first draft did.
    """
    shows = [_show("2026-01-01", *REGULARS),
             _show("2026-06-01", *[f"Never{i}" for i in range(6)])]
    result = backtest(shows, LOOSE, count=1, min_songs=6)
    assert result.through == "2026-06-01"
    assert result.hit_rate == pytest.approx(0.0)


def test_shows_after_the_window_do_not_reach_back_into_it():
    """The other half of the boundary: later shows must not inform earlier predictions."""
    shows = _steady(20) + [_show("2026-06-01", *[f"Never{i}" for i in range(6)])]
    full = backtest(shows, LOOSE, count=3, min_songs=6)
    truncated = backtest([s for s in shows if s["date"] <= full.through], LOOSE,
                         count=3, min_songs=6)
    assert truncated.hit_rate == pytest.approx(full.hit_rate)
    assert truncated.through == full.through


def test_the_scorer_gets_exactly_as_many_guesses_as_songs_were_played():
    """Top-m, where m is what the band actually played. Handing it more guesses buys recall for
    free and makes every config look better than it is.

    The night under test is the six LOWEST-ranked songs in the corpus, so a scorer held to six
    guesses gets none of them and one allowed twelve gets all of them. A corpus where the test
    night shares nothing at all cannot catch this, because widening the guess list still finds
    nothing.
    """
    low = [f"Rare{i}" for i in range(6)]
    shows = _steady(20) + [_show("2025-01-01", *low), _show("2026-06-01", *low)]
    result = backtest(shows, LOOSE, count=1, min_songs=6)
    assert result.through == "2026-06-01"
    assert result.hit_rate == pytest.approx(0.0)


def test_the_evaluation_window_is_the_most_recent_shows():
    result = backtest(_steady(40), LOOSE, count=3, min_songs=6)
    assert result.n_shows == 3
    assert result.through == max(s["date"] for s in _steady(40))


def test_a_show_too_short_to_grade_is_not_evaluated():
    """Grading a two-song night says more about the taper's notes than about the model."""
    shows = _steady(20) + [_show("2026-06-01", "Meat", "Rebubula")]
    result = backtest(shows, LOOSE, count=3, min_songs=6)
    assert "2026-06-01" not in result.through


def test_the_standard_error_reports_the_spread_not_zero():
    shows = _steady(20, songs=REGULARS) + \
            [_show(f"2026-03-{day:02d}", *[f"N{day}x{i}" for i in range(6)])
             for day in range(1, 6)]
    result = backtest(shows, LOOSE, count=8, min_songs=6)
    assert 0.0 < result.hit_rate < 1.0
    assert result.std_error > 0.0


def test_one_evaluated_show_has_no_standard_error_rather_than_a_wrong_one():
    result = backtest(_steady(20), LOOSE, count=1, min_songs=6)
    assert result.std_error == 0.0


def test_an_empty_corpus_backtests_to_nothing_rather_than_crashing():
    result = backtest([], LOOSE, count=5, min_songs=6)
    assert result.n_shows == 0 and result.hit_rate == 0.0


def test_the_baselines_run_and_a_predictable_band_saturates_them():
    shows = _steady(20)
    tests = shows[-3:]
    assert naive_baseline(shows, tests) == pytest.approx(1.0)
    assert recency_baseline(shows, tests, LOOSE) == pytest.approx(1.0)


def test_holdout_tunes_on_the_older_half_and_scores_the_newer_half():
    grid = {"gap_weight": [0.0, 0.35], "recency_halflife_days": [200.0, 400.0]}
    result = holdout(_steady(40), LOOSE, grid, window=8, min_songs=6)
    assert result.train_shows == 4 and result.held_shows == 4
    assert set(result.best) == {"gap_weight", "recency_halflife_days"}


def test_the_held_out_half_never_reaches_the_search():
    """A grid searched on the shows it is later scored on reports its own best guess back.

    Replacing the newer half with songs the band never played must not move ``in_sample``,
    because the search never saw those shows. It must move ``out_of_sample``, because that is
    the half being graded. Asserting only that the chosen config is unchanged would pass on a
    predictable corpus where every config ties, so both halves are asserted here.
    """
    grid = {"gap_weight": [0.0, 0.35], "recency_halflife_days": [200.0, 400.0]}
    base = _steady(40)
    scrambled = base[:36] + [_show(f"2026-06-{day:02d}", *[f"X{day}y{i}" for i in range(6)])
                             for day in range(1, 5)]
    honest = holdout(base, LOOSE, grid, window=8, min_songs=6)
    swapped = holdout(scrambled, LOOSE, grid, window=8, min_songs=6)
    assert swapped.in_sample == pytest.approx(honest.in_sample)
    assert swapped.out_of_sample < honest.out_of_sample


def test_an_empty_grid_still_reports_the_incumbent_config():
    result = holdout(_steady(40), LOOSE, {}, window=8, min_songs=6)
    assert result.best == {}
    assert result.out_of_sample == pytest.approx(1.0)

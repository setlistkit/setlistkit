# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for measuring a model against chance and against the dumb answer.

Two kinds of thing are checked here and they need different treatment.

The CLOSED FORMS -- the hypergeometric mean and variance, the odds of a perfect night, the Murphy
identity -- are arithmetic, so they are checked against values worked out by hand on cases small
enough to verify without the code. A test that recomputes the formula it is testing proves only
that the author typed it twice.

The DIRECTION CLAIMS are the ones the module docstring makes in prose: that a wider repertoire
flatters the model against chance while raising popularity's bar, that a forecast which always
states the base rate has no resolution, that a song outside the repertoire is not charged to
chance. Those are the claims a reader will rely on and cannot check, so each has a test.
"""

import math

from setlistkit.model.rollout import SongChance
from setlistkit.picks.skill import (Brier, Rung, brier_decomposition, chance_overlap, measure,
                                    perfect_night_odds, popularity_picks, repertoire,
                                    scored_pairs)

REGULARS = ("Aurora", "Rebubula", "Meat", "Zelda", "Buster", "Wormhole", "Gumbo", "Cavern")


def _entry(song):
    return {"song": song, "segue": False, "non_song": False}


def _show(date, *songs):
    return {"date": date, "source": "description", "identifier": f"t{date}",
            "sets": [[_entry(s) for s in songs]], "encore": []}


def _nights(*pairs):
    """(date, songs) pairs in the shape :func:`gradeable` returns."""
    return [(date, set(songs)) for date, songs in pairs]


def _chances(*songs):
    return [SongChance(rank=index + 1, song=song, confidence=0.5)
            for index, song in enumerate(songs)]


# --------------------------------------------------------------------------- the closed forms

def test_the_hypergeometric_mean_and_variance_match_the_hand_computed_case():
    """N=10, K=4, n=5. Mean = nK/N = 2.0; variance = n(K/N)(1-K/N)(N-n)/(N-1) = 5(.4)(.6)(5/9)."""
    mean, variance = chance_overlap(pool_size=10, drawable=4, picks=5)
    assert mean == 2.0
    assert math.isclose(variance, 5 * 0.4 * 0.6 * (5 / 9))


def test_a_pool_no_bigger_than_the_shortlist_yields_nothing_rather_than_dividing_by_zero():
    """Drawing 30 from 30 is not a prediction, and N-1 in the variance would be a trap at N=1."""
    assert chance_overlap(pool_size=30, drawable=10, picks=30) == (0.0, 0.0)
    assert chance_overlap(pool_size=1, drawable=1, picks=30) == (0.0, 0.0)


def test_a_night_with_nothing_drawable_expects_nothing():
    assert chance_overlap(pool_size=100, drawable=0, picks=30) == (0.0, 0.0)


def test_the_odds_of_a_perfect_night_match_the_hand_computed_case():
    """C(3,2)/C(5,2) = 3/10. Small enough to check without the formula."""
    assert math.isclose(perfect_night_odds(pool_size=5, drawable=2, picks=3), 0.3)


def test_a_perfect_night_is_certain_when_the_draw_covers_the_pool():
    assert math.isclose(perfect_night_odds(pool_size=4, drawable=4, picks=4), 1.0)


def test_more_songs_than_picks_can_never_all_be_drawn():
    assert perfect_night_odds(pool_size=100, drawable=31, picks=30) == 0.0


def test_the_odds_survive_a_pool_far_too_big_for_factorials():
    """The real case: the answer is around 1e-38 and every intermediate overflows a float."""
    odds = perfect_night_odds(pool_size=282, drawable=29, picks=30)
    assert 0.0 < odds < 1e-30, odds


# --------------------------------------------------------------------------- the repertoire

def test_the_repertoire_forgets_songs_older_than_the_span():
    nights = _nights(("2020-01-01", ["Shelved"]), ("2025-06-01", ["Current"]))
    assert repertoire(nights, "2026-01-01", years=2) == {"Current"}
    assert repertoire(nights, "2026-01-01", years=10) == {"Shelved", "Current"}


def test_the_repertoire_cannot_see_the_night_it_is_being_built_for():
    """The same leak boundary the rest of the walk obeys, restated where it would be easy to lose."""
    nights = _nights(("2026-01-01", ["Tonight"]))
    assert repertoire(nights, "2026-01-01", years=5) == set()


def test_a_song_outside_the_repertoire_is_not_charged_to_chance():
    """A debut cannot be drawn by any strategy, so counting it as a miss would weaken the null --
    the one direction that flatters the model."""
    with_debut = chance_overlap(pool_size=100, drawable=10, picks=30)
    # The same night, if the debut were wrongly counted as drawable.
    if_charged = chance_overlap(pool_size=100, drawable=11, picks=30)
    assert with_debut[0] < if_charged[0]


# --------------------------------------------------------------------------- popularity

def test_popularity_counts_nights_containing_and_not_total_performances():
    """Matches naive_baseline. A song played twice in one night is one night, not two."""
    nights = _nights(("2025-01-01", ["Twice", "Twice", "Once"]), ("2025-02-01", ["Once"]))
    assert popularity_picks(nights, "2026-01-01", picks=1, years=5) == {"Once"}


def test_popularity_breaks_ties_on_the_name_so_the_baseline_is_reproducible():
    nights = _nights(("2025-01-01", ["Beta", "Alpha"]))
    assert popularity_picks(nights, "2026-01-01", picks=1, years=5) == {"Alpha"}


def test_popularity_reads_only_what_came_before():
    nights = _nights(("2025-01-01", ["Old"]), ("2026-06-01", ["New"]))
    assert popularity_picks(nights, "2026-01-01", picks=5, years=5) == {"Old"}


# --------------------------------------------------------------------------- Brier and Murphy

def test_murphys_identity_closes_exactly_on_the_binned_forecast():
    """The decomposition is exact for the BINNED forecast, and only for that one."""
    pairs = [(0.9, 1), (0.8, 1), (0.7, 0), (0.3, 0), (0.2, 1), (0.05, 0), (0.0, 0), (0.0, 1)]
    scored = brier_decomposition(pairs)
    assert math.isclose(scored.binned,
                        scored.reliability - scored.resolution + scored.uncertainty,
                        abs_tol=1e-12)


def test_the_raw_score_closes_only_once_the_within_bin_spread_is_added_back():
    """The gap the three terms cannot see, kept as a reported quantity rather than a tolerance.

    Two forecasts of 0.9 and 0.8 land in the same tenth and are decomposed as though both were
    0.85. That is a real approximation, it is what `within_bin` measures, and hiding it inside an
    `abs_tol` would turn "the bins are too coarse for this forecast" into "the test still passes".
    """
    pairs = [(0.9, 1), (0.8, 1), (0.7, 0), (0.3, 0), (0.2, 1), (0.05, 0), (0.0, 0), (0.0, 1)]
    scored = brier_decomposition(pairs)
    assert math.isclose(scored.score, scored.binned + scored.within_bin, abs_tol=1e-12)
    assert scored.within_bin != 0.0, "these forecasts do vary inside their bins"


def test_a_forecast_that_never_varies_inside_a_bin_has_no_residual():
    """The case where binning throws nothing away, so the identity closes on the raw score too."""
    scored = brier_decomposition([(0.25, 1), (0.25, 0), (0.85, 1), (0.85, 1)])
    assert math.isclose(scored.within_bin, 0.0, abs_tol=1e-12)
    assert math.isclose(scored.score, scored.binned, abs_tol=1e-12)


def test_a_perfectly_sharp_and_correct_forecast_scores_zero():
    assert brier_decomposition([(1.0, 1), (0.0, 0)]).score == 0.0


def test_a_forecast_that_always_states_the_base_rate_has_no_resolution():
    """The failure only resolution catches: perfectly calibrated, entirely useless.

    Four events, half of which happen, every one forecast at 0.5. Reliability is zero because 0.5
    is exactly right on average, and resolution is zero because the forecast never distinguished
    one event from another. A calibration figure alone would call this a good model.
    """
    scored = brier_decomposition([(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)])
    assert math.isclose(scored.resolution, 0.0, abs_tol=1e-12)
    assert math.isclose(scored.reliability, 0.0, abs_tol=1e-12)
    assert math.isclose(scored.score, 0.25)


def test_an_overconfident_forecast_pays_for_it_in_reliability():
    confident = brier_decomposition([(1.0, 1), (1.0, 0), (1.0, 1), (1.0, 0)])
    assert confident.reliability > 0.0


def test_an_empty_forecast_is_zeroed_rather_than_dividing_by_nothing():
    assert brier_decomposition([]).n == 0


def test_the_brier_skill_score_is_positive_only_against_a_worse_forecast():
    better = Brier(score=0.10, reliability=0.0, resolution=0.0, uncertainty=0.0, within_bin=0.0, n=1)
    worse = Brier(score=0.20, reliability=0.0, resolution=0.0, uncertainty=0.0, within_bin=0.0, n=1)
    assert math.isclose(better.skill_against(worse), 0.5)
    assert worse.skill_against(better) < 0.0


# --------------------------------------------------------------------------- the ladder

def test_the_skill_score_is_negative_when_the_model_loses_to_popularity():
    """A signed number on purpose: losing to the baseline is a real outcome and has to be sayable."""
    losing = Rung(years=2, played=100, chance=_flat_chance(), popularity_hits=50, model_hits=40)
    assert losing.skill < 0.0


def test_the_skill_score_is_the_share_of_the_baselines_misses_recovered():
    """Popularity missed 50; the model recovered 10 of them."""
    rung = Rung(years=2, played=100, chance=_flat_chance(), popularity_hits=50, model_hits=60)
    assert math.isclose(rung.skill, 0.2)


def test_a_baseline_that_called_everything_leaves_no_headroom_to_divide_by():
    rung = Rung(years=2, played=100, chance=_flat_chance(), popularity_hits=100, model_hits=100)
    assert rung.skill == 0.0


def _flat_chance():
    from setlistkit.picks.skill import Chance
    return Chance(expected=10.0, sd=2.0, pool_median=100)


def test_sigma_is_zero_rather_than_infinite_when_chance_cannot_vary():
    from setlistkit.picks.skill import Chance
    assert Chance(expected=0.0, sd=0.0, pool_median=0).sigma(50) == 0.0


def test_a_wider_repertoire_flatters_the_model_against_chance():
    """The direction claim the docstring makes, and the reason a span is never chosen alone.

    More songs in the pool means a blind draw is less likely to land on tonight's, so the same
    model looks better the wider the repertoire is drawn.
    """
    corpus = ([_show(f"2019-01-{day:02d}", "Shelved") for day in range(1, 4)]
              + [_show(f"2026-01-{day:02d}", *REGULARS[:4]) for day in range(1, 10)])
    graded = [("2026-02-01", _chances(*REGULARS[:2]), set(REGULARS[:4]))]
    narrow, wide = measure(corpus, graded, picks=3, spans=(2, 10))
    assert wide.lift_over_chance > narrow.lift_over_chance


def test_measure_grades_the_nights_it_was_handed_and_never_the_whole_corpus():
    """The walk is passed in so the page and the skill table cannot report on different nights."""
    corpus = [_show(f"2026-01-{day:02d}", *REGULARS[:4]) for day in range(1, 12)]
    graded = [("2026-01-11", _chances(*REGULARS[:2]), set(REGULARS[:4]))]
    rung, = measure(corpus, graded, picks=3, spans=(5,))
    assert rung.played == 4
    assert rung.model_hits == 2


def test_a_song_the_model_never_named_still_counts_against_it():
    """Scoring only the songs a model chose to talk about is marking your own homework."""
    corpus = [_show(f"2026-01-{day:02d}", *REGULARS[:5]) for day in range(1, 10)]
    graded = [("2026-02-01", _chances("Aurora"), {"Aurora", "Meat"})]
    scored = scored_pairs(corpus, graded, years=5)
    assert (0.0, 1) in scored.strict, "the played song the model never named has to be scored"


def test_both_forecasts_answer_for_exactly_the_same_songs():
    """Two Brier scores are only comparable over one set of events."""
    corpus = [_show(f"2026-01-{day:02d}", *REGULARS[:5]) for day in range(1, 10)]
    graded = [("2026-02-01", _chances("Aurora"), {"Aurora", "Meat"})]
    scored = scored_pairs(corpus, graded, years=5)
    assert len(scored.strict) == len(scored.climate) == len(scored.backed_off)
    assert ([outcome for _f, outcome in scored.strict]
            == [outcome for _f, outcome in scored.climate])


def test_backing_off_replaces_a_hard_zero_with_the_songs_long_run_rate():
    """The distinction the whole three-way split exists for.

    Strict scores an unlisted song at 0.0 -- "this will not happen" -- while backed-off scores it
    at whatever it usually does. Same ranking, different claim about everything the ranking left
    out, and the two can land on opposite sides of the baseline.
    """
    corpus = [_show(f"2026-01-{day:02d}", "Aurora", "Meat") for day in range(1, 10)]
    graded = [("2026-02-01", _chances("Aurora"), {"Aurora"})]
    scored = scored_pairs(corpus, graded, years=5)
    assert (0.0, 0) in scored.strict, "Meat is unlisted, so strict calls it impossible"
    assert (1.0, 0) in scored.backed_off, "Meat played every night, so backing off says it will"
    assert (1.0, 0) in scored.climate


def test_a_listed_song_is_untouched_by_backing_off():
    corpus = [_show(f"2026-01-{day:02d}", "Aurora") for day in range(1, 10)]
    graded = [("2026-02-01", _chances("Aurora"), {"Aurora"})]
    scored = scored_pairs(corpus, graded, years=5)
    assert scored.strict == scored.backed_off


def test_a_song_that_played_from_outside_the_repertoire_is_still_scored():
    corpus = [_show(f"2026-01-{day:02d}", "Aurora") for day in range(1, 10)]
    graded = [("2026-02-01", _chances("Aurora"), {"Aurora", "Debut"})]
    scored = scored_pairs(corpus, graded, years=5)
    assert len(scored.strict) == 2, "the debut belongs in the universe even though nothing saw it"
    assert (0.0, 1) in scored.climate

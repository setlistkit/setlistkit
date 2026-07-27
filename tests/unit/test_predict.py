# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for turning a ranking into stated probabilities, and both halves into a night.

THE TEST THAT MATTERS IS ``test_the_future_cannot_change_the_prediction``. A prediction made with
the future present and one made with it deleted must be the same object, field for field. Every
other guarantee in this facility is downstream of that one: a replay grades a prediction against a
night, and if any trace of the night reached the predictor the grade is a number about nothing.

It is asserted as whole-object equality rather than as a spot check on the top few songs, because
the ways a leak shows up are not all in the ranking. A leaked bigram moves the ORDER and not the
chances. A leaked encore moves the encore alone. ``Prediction`` is all tuples so that ``==`` sees
every one of them.
"""

import pytest

from setlistkit.model.predict import predict_night
from setlistkit.model.rollout import rollout_night
from setlistkit.model.scores import ScoreConfig
from setlistkit.model.setlists import PredictConfig

# min_show_songs=1 so short synthetic nights are not thrown out before anything can be measured;
# no cooldown so a song's chance is a function of the corpus and not of what it just played.
LOOSE = ScoreConfig(min_show_songs=1, cooldown=())
QUICK = PredictConfig(rollouts=60, set_len=5, top_n=12)


def _entry(song, segue=False):
    return {"song": song, "segue": segue, "non_song": False}


def _show(date, *songs, encore=()):
    return {"date": date, "source": "description", "identifier": f"t{date}",
            "sets": [[_entry(s.rstrip(">"), s.endswith(">")) for s in songs]],
            "encore": [_entry(s) for s in encore]}


REGULARS = ("Aurora", "Rebubula", "Meat", "Zelda", "Buster", "Wormhole", "Gumbo", "Cavern")


def _history(n=24, start=1):
    """A rotation with a stable habit in it: Aurora opens and runs into Rebubula."""
    return [_show(f"2026-{1 + (start + i) // 28:02d}-{1 + (start + i) % 28:02d}",
                  "Aurora>", *REGULARS[1:], encore=("Timmy",))
            for i in range(n)]


CUTOFF = "2026-03-01"


def _after_the_cutoff():
    """Nights past the boundary, built to be as loud a leak as a corpus can be.

    Songs that appear nowhere else, at full recency weight, adjacent to each other and to the
    regulars. If any of it can reach the predictor, it lands at the top of the ranking.
    """
    return [_show("2026-03-01", "Leak One>", "Leak Two", "Leak Three", encore=("Leak Four",)),
            _show("2026-03-02", "Leak Two>", "Aurora", "Leak Five", encore=("Leak Four",)),
            _show("2026-04-11", "Leak Six", "Leak Seven", *REGULARS)]


def test_the_future_cannot_change_the_prediction():
    """Same corpus with and without everything from the as-of date on. Same prediction."""
    past = _history()
    blind = predict_night(past, LOOSE, QUICK, asof=CUTOFF)
    with_future = predict_night(past + _after_the_cutoff(), LOOSE, QUICK, asof=CUTOFF)
    assert with_future == blind


def test_no_song_from_on_or_after_the_as_of_date_appears_anywhere_in_the_output():
    """Stated separately from the equality test, because it names the failure in the message.

    Whole-object equality proves the two runs agree; it does not say WHAT leaked when they do not.
    This one fails with the leaked song in the assertion.
    """
    prediction = predict_night(_history() + _after_the_cutoff(), LOOSE, QUICK, asof=CUTOFF)
    named = ({chance.song for chance in prediction.chances}
             | {entry.song for entry in prediction.setlist}
             | {entry.song for entry in prediction.encore})
    assert not {song for song in named if song.startswith("Leak")}


def test_the_prediction_states_the_boundary_it_was_made_under():
    assert predict_night(_history(), LOOSE, QUICK, asof=CUTOFF).asof == CUTOFF


def test_the_same_seed_predicts_the_same_night_twice():
    shows = _history()
    assert (predict_night(shows, LOOSE, QUICK, asof=CUTOFF)
            == predict_night(shows, LOOSE, QUICK, asof=CUTOFF))


def test_a_different_seed_is_allowed_to_disagree():
    """The reproducibility above is the seed's doing, not an artifact of a tiny corpus.

    Without this, a bug that ignored the seed entirely would pass the determinism test.
    """
    shows = [_show(f"2026-01-{day:02d}", *[f"S{(day + i) % 40}" for i in range(12)])
             for day in range(1, 29)]
    left = predict_night(shows, LOOSE, PredictConfig(rollouts=40, set_len=5, top_n=30, seed=1),
                         asof=CUTOFF)
    right = predict_night(shows, LOOSE, PredictConfig(rollouts=40, set_len=5, top_n=30, seed=2),
                          asof=CUTOFF)
    assert left.chances != right.chances


def test_the_setlist_is_drawn_from_the_top_of_the_ranking():
    prediction = predict_night(_history(), LOOSE, QUICK, asof=CUTOFF)
    top = [chance.song for chance in prediction.chances[:prediction.set_len]]
    assert sorted(entry.song for entry in prediction.setlist) == sorted(top)


def test_the_encore_is_never_a_song_the_set_already_played():
    prediction = predict_night(_history(), LOOSE, QUICK, asof=CUTOFF)
    played = {entry.song for entry in prediction.setlist}
    assert not played & {entry.song for entry in prediction.encore}


def test_the_caller_can_override_the_set_length_for_one_night():
    """How the replay tells the predictor how long the night it is grading actually was."""
    prediction = predict_night(_history(), LOOSE, QUICK, asof=CUTOFF, set_len=3)
    assert prediction.set_len == 3
    assert len(prediction.setlist) == 3


def test_confidences_are_shares_of_the_draws_and_come_back_ranked():
    ranked = rollout_night(_history(), LOOSE, QUICK, asof=CUTOFF, set_len=5)
    assert [chance.rank for chance in ranked] == list(range(1, len(ranked) + 1))
    assert all(0.0 < chance.confidence <= 1.0 for chance in ranked)
    confidences = [chance.confidence for chance in ranked]
    assert confidences == sorted(confidences, reverse=True)


def test_a_song_played_every_single_night_is_predicted_with_near_certainty():
    """Nine songs the band always plays, and a draw of eight: the habit has to show.

    Nine and not eight -- the encore song is played every night too, and a draw of eight out of a
    vocabulary of nine leaves whichever one it is scored like the rest.
    """
    ranked = rollout_night(_history(), LOOSE, QUICK, asof=CUTOFF, set_len=8)
    assert set(REGULARS) <= {chance.song for chance in ranked[:9]}
    assert all(chance.confidence > 0.8 for chance in ranked[:8])


def test_the_recorded_list_reaches_top_n_even_on_a_night_shorter_than_the_pool_multiple():
    """The pool is ``max(set_len * 4, top_n)`` wide, and the max is not decoration.

    At ``set_len=3`` a pool of ``set_len * 4`` holds twelve songs, so a top-30 request could only
    ever record twelve -- understating ``expected_hits``, which is the denominator of calibration,
    and flattering the ratio for free. Short nights are ordinary.

    100 rollouts and not the 40 the other tests use, because the list is also bounded by what got
    DRAWN: a song in the pool that no draw happened to take has no confidence to record and does
    not appear. At 40 draws of three, one song reliably misses and the count is 29, which would
    make this assert the noise floor rather than the pool width.
    """
    shows = [_show(f"2026-01-{day:02d}", *[f"S{(day * 3 + i) % 50}" for i in range(14)])
             for day in range(1, 29)]
    ranked = rollout_night(shows, LOOSE, PredictConfig(rollouts=100, top_n=30), asof=CUTOFF,
                           set_len=3)
    assert len(ranked) == 30


def test_the_recorded_list_stops_at_what_the_corpus_can_name():
    """A vocabulary of eight cannot fill a top-30, and padding it would be inventing songs."""
    ranked = rollout_night(_history(), LOOSE, PredictConfig(rollouts=40, top_n=30), asof=CUTOFF,
                           set_len=5)
    assert {chance.song for chance in ranked} <= set(REGULARS) | {"Timmy"}


def test_an_empty_corpus_predicts_nothing_rather_than_raising():
    """The CLI refuses this case with a diagnostic; the model layer just has nothing to say."""
    prediction = predict_night([], LOOSE, QUICK, asof=CUTOFF)
    assert prediction.chances == () and prediction.setlist == () and prediction.encore == ()


def test_a_corpus_entirely_after_the_boundary_is_the_same_as_no_corpus():
    assert (predict_night(_after_the_cutoff(), LOOSE, QUICK, asof=CUTOFF)
            == predict_night([], LOOSE, QUICK, asof=CUTOFF))


@pytest.mark.parametrize("bad", ["", "2026-13-01", "tomorrow"])
def test_an_unusable_as_of_date_is_refused(bad):
    with pytest.raises(ValueError):
        predict_night(_history(), LOOSE, QUICK, asof=bad)

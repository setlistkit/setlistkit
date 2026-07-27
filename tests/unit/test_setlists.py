# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for laying a chosen bag of songs out as a night.

Two of these matter more than the rest. One proves a bigram cannot be learned from the night it is
used to predict, which is the same obligation ``song_scores`` carries and the reason the as-of
filter sits inside ``transitions`` rather than in whoever calls it. The other proves the ordering
is stable across processes -- a greedy walk over a pool of ten ties constantly, and a dict's
iteration order is hash-seed randomized, so an ordering that only sorts on the rate is an ordering
that changes between runs of the same replay.

The rest pin what the walk is FOR. Nothing here is graded, so a test asserting a particular setlist
is prettier than another would be asserting a preference; these assert structure instead.
"""

import random

import pytest

from setlistkit.model.scores import ScoreConfig, song_scores
from setlistkit.model.setlists import (
    PredictConfig,
    draw_set,
    order_set,
    pick_encore,
    transitions,
)

CONFIG = PredictConfig()


def _entry(song, segue=False):
    return {"song": song, "segue": segue, "non_song": False}


def _show(date, *songs, encore=()):
    """One night. A song given as ``"X>"`` runs into the next one."""
    return {"date": date, "source": "description", "identifier": f"t{date}",
            "sets": [[_entry(s.rstrip(">"), s.endswith(">")) for s in songs]],
            "encore": [_entry(s) for s in encore]}


# A band with an unmistakable habit: it opens with Aurora, Aurora always runs into Rebubula, and
# it encores with Timmy. Everything a construction test needs to have an expected answer at all.
HABIT = [_show(f"2026-01-{day:02d}", "Aurora>", "Rebubula", "Meat", "Zelda", "Buster",
               encore=("Timmy",))
         for day in range(1, 13)]


def test_the_boundary_is_strictly_before_the_as_of_date():
    """A show ON the as-of date is not read, which is the whole point of the parameter.

    ``corpus.shows(until=D)`` is INCLUSIVE, so a caller pre-filtering its own corpus hands the
    predicted night straight into the graph that predicts it. This is the test that the filter
    lives here instead.
    """
    corpus = HABIT + [_show("2026-02-01", "Newcomer>", "Stranger")]
    blind = transitions(corpus, asof="2026-02-01")
    assert "Newcomer" not in blind.features
    seeing = transitions(corpus, asof="2026-02-02")
    assert "Newcomer" in seeing.features


def test_no_bigram_comes_from_a_night_on_or_after_the_boundary():
    """The pair, not just the songs. A leak can arrive as two known songs newly adjacent."""
    corpus = HABIT + [_show("2026-02-01", "Meat>", "Aurora")]
    blind = transitions(corpus, asof="2026-02-01")
    assert "Aurora" not in blind.partners("Meat")
    assert "Aurora" in transitions(corpus, asof="2026-02-02").partners("Meat")


def test_an_unparseable_as_of_date_is_refused_rather_than_treated_as_open():
    """A boundary that silently means "no boundary" is the failure this facility cannot have."""
    with pytest.raises(ValueError, match="not a date"):
        transitions(HABIT, asof="last tuesday")


def test_a_song_the_corpus_has_never_seen_has_no_rates_and_no_partners():
    trans = transitions(HABIT, asof="2026-06-01")
    assert trans.opener_rate("Nothing") == 0.0
    assert trans.encore_rate("Nothing") == 0.0
    assert trans.segue_rate("Nothing") == 0.0
    assert not trans.partners("Nothing")


def test_the_walk_opens_with_the_opener_and_follows_the_learned_pair():
    trans = transitions(HABIT, asof="2026-06-01")
    order = order_set(["Zelda", "Rebubula", "Aurora", "Meat"], trans, CONFIG)
    assert [entry.song for entry in order][:2] == ["Aurora", "Rebubula"]


def test_a_song_that_always_ran_into_the_next_is_marked_as_segueing():
    trans = transitions(HABIT, asof="2026-06-01")
    order = order_set(["Aurora", "Rebubula", "Meat"], trans, CONFIG)
    marks = {entry.song: entry.segue for entry in order}
    assert marks["Aurora"] is True
    assert marks["Rebubula"] is False


def test_the_last_song_never_segues_however_often_it_ran_into_things():
    """There is nothing after it. A trailing `>` is a setlist that was cut off, not a night."""
    trans = transitions(HABIT, asof="2026-06-01")
    order = order_set(["Aurora"], trans, CONFIG)
    assert order == [type(order[0])(song="Aurora", segue=False)]


def test_the_walk_covers_the_pool_exactly_once():
    trans = transitions(HABIT, asof="2026-06-01")
    pool = ["Zelda", "Rebubula", "Aurora", "Meat", "Buster"]
    order = [entry.song for entry in order_set(pool, trans, CONFIG)]
    assert sorted(order) == sorted(pool)


def test_an_empty_pool_orders_to_an_empty_night():
    assert order_set([], transitions(HABIT, asof="2026-06-01"), CONFIG) == []


def test_the_same_pool_orders_the_same_way_however_it_arrived():
    """Ties break on the song name, so iteration order cannot reach the output.

    Shuffling the input is the cheap half of this. The expensive half -- that a hash-seed change
    between processes does not move it either -- is what the tie-break exists for and what this
    approximates: every pair below ties on every rate the walk consults.
    """
    trans = transitions([_show("2026-01-01", "P", "Q", "R", "S")] * 6, asof="2026-06-01")
    pool = ["P", "Q", "R", "S"]
    once = [entry.song for entry in order_set(pool, trans, CONFIG)]
    for _ in range(8):
        random.shuffle(pool)
        assert [entry.song for entry in order_set(pool, trans, CONFIG)] == once


def test_the_encore_comes_from_songs_the_corpus_has_actually_encored():
    trans = transitions(HABIT, asof="2026-06-01")
    scores = song_scores(HABIT, ScoreConfig(), asof="2026-06-01")
    assert [entry.song for entry in pick_encore(scores, trans, CONFIG, set())] == ["Timmy"]


def test_the_encore_will_not_reuse_a_song_the_set_already_played():
    trans = transitions(HABIT, asof="2026-06-01")
    scores = song_scores(HABIT, ScoreConfig(), asof="2026-06-01")
    assert pick_encore(scores, trans, CONFIG, {"Timmy"}) == []


def test_picking_an_encore_marks_it_used_so_a_second_night_cannot_repeat_it():
    """``used`` is mutated on purpose: it is how a multi-night run gets no-repeat for free."""
    corpus = HABIT + [_show("2026-01-20", "Aurora", "Meat", encore=("Cavern",))]
    trans = transitions(corpus, asof="2026-06-01")
    scores = song_scores(corpus, ScoreConfig(), asof="2026-06-01")
    used: set[str] = set()
    first = pick_encore(scores, trans, CONFIG, used)
    second = pick_encore(scores, trans, CONFIG, used)
    assert first and second
    assert first[0].song != second[0].song
    assert used == {first[0].song, second[0].song}


def test_asking_for_no_encore_gets_none_and_marks_nothing_used():
    trans = transitions(HABIT, asof="2026-06-01")
    scores = song_scores(HABIT, ScoreConfig(), asof="2026-06-01")
    used: set[str] = set()
    assert pick_encore(scores, trans, PredictConfig(encore_len=0), used) == []
    assert not used


def test_a_draw_takes_distinct_songs_and_stops_at_the_length_asked_for():
    pool = [f"S{i}" for i in range(20)]
    drawn = draw_set(pool, {song: 1.0 for song in pool}, 8, random.Random(1))
    assert len(drawn) == len(set(drawn)) == 8


def test_a_draw_cannot_take_more_songs_than_the_pool_holds():
    """A short pool shortens the night rather than raising -- and never repeats to fill it."""
    drawn = draw_set(["A", "B"], {"A": 1.0, "B": 1.0}, 10, random.Random(1))
    assert sorted(drawn) == ["A", "B"]


def test_a_song_weighted_zero_is_never_drawn():
    """What a cooldown of 0.0 means, and it has to survive normalization to get here."""
    pool = ["Cold", "Warm"]
    for seed in range(20):
        assert draw_set(pool, {"Cold": 0.0, "Warm": 1.0}, 2, random.Random(seed)) == ["Warm"]


def test_a_heavier_song_is_drawn_more_often_than_a_lighter_one():
    rng = random.Random(7)
    firsts = [draw_set(["Heavy", "Light"], {"Heavy": 9.0, "Light": 1.0}, 1, rng)[0]
              for _ in range(400)]
    assert firsts.count("Heavy") > firsts.count("Light") * 4


def test_the_same_seed_draws_the_same_night_twice():
    pool = [f"S{i}" for i in range(30)]
    weights = {song: 1.0 + index for index, song in enumerate(pool)}
    assert (draw_set(pool, weights, 10, random.Random(99))
            == draw_set(pool, weights, 10, random.Random(99)))

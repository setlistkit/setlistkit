# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for scoring every song as of a date.

The one that matters most is the no-lookahead test. A scorer that can see the show it is
predicting will look excellent and be worthless, and the failure is invisible in the output --
it just quietly reports a number nobody should believe. Everything else here isolates one
signal at a time, because the score is a product of four of them and a test that moves two at
once proves nothing about either.
"""

import datetime as dt

import pytest

from setlistkit.model.scores import OVERDUE_FALLBACK_RATIO, ScoreConfig, overdue_ratio, rotation
from setlistkit.model.scores import song_scores


def _show(date, *songs, non_songs=()):
    entries = [{"song": s, "segue": False, "non_song": False} for s in songs]
    entries += [{"song": s, "segue": False, "non_song": True} for s in non_songs]
    return {"date": date, "source": "description", "identifier": "t1",
            "sets": [entries], "encore": []}


def _filler(date, tag="F"):
    """A show that meets the song floor without colliding with anything under test."""
    return _show(date, *[f"{tag}{i}" for i in range(6)])


def _by_song(rows):
    return {row.song: row for row in rows}


# The floor exists to reject truncated parses, not short shows, so the fixtures above clear it
# deliberately rather than by accident.
LOOSE = ScoreConfig(min_show_songs=1)


def test_a_song_nobody_has_played_is_not_a_candidate():
    rows = _by_song(song_scores([_show("2026-01-01", "Meat")], LOOSE, asof="2026-06-01"))
    assert set(rows) == {"Meat"}


def test_only_shows_before_the_as_of_date_are_training_data():
    """The whole point of an as-of date. A scorer that sees the night it is predicting
    reports a number that looks excellent and means nothing."""
    shows = [_show("2026-01-01", "Meat"),
             _show("2026-06-01", "Tomorrow"),        # the as-of date itself, excluded
             _show("2026-07-01", "Later")]           # after, excluded
    rows = _by_song(song_scores(shows, LOOSE, asof="2026-06-01"))
    assert set(rows) == {"Meat"}


def test_a_recent_play_outscores_an_old_one():
    shows = [_show("2020-01-01", "Ancient"), _show("2026-05-01", "Recent")]
    rows = _by_song(song_scores(shows, LOOSE, asof="2026-06-01"))
    assert rows["Recent"].base > rows["Ancient"].base


def test_a_truncated_parse_is_not_training_data():
    """Two songs off a taper's stray note is not a record of a night, and the default floor
    rejects it. A storm-shortened five-song show is a complete record and is kept."""
    shows = [_show("2026-01-01", "A", "B"), _filler("2026-02-01")]
    rows = _by_song(song_scores(shows, ScoreConfig(), asof="2026-06-01"))
    assert "A" not in rows and "F0" in rows


def test_tagged_non_songs_never_become_candidates():
    """Counted, "break" would outrank every song the band has ever written."""
    shows = [_show("2026-01-01", "Meat", non_songs=["break", "Drums"])]
    rows = _by_song(song_scores(shows, LOOSE, asof="2026-06-01"))
    assert set(rows) == {"Meat"}


def test_a_song_played_last_show_is_suppressed_by_the_cooldown():
    """The band essentially never repeats across consecutive shows, yet a just-played song
    otherwise ranks UP: its gap is zero, so the due boost is neutral and its base is at peak
    recency. Without this the scorer's top pick is regularly last night's closer."""
    shows = [_filler("2026-05-01"), _filler("2026-05-02", tag="G"),
             _show("2026-05-03", "JustPlayed", "A", "B", "C", "D", "E")]
    rows = _by_song(song_scores(shows, ScoreConfig(), asof="2026-06-01"))
    assert rows["JustPlayed"].cooldown == pytest.approx(0.12)
    assert rows["JustPlayed"].score < rows["JustPlayed"].base


def test_a_song_left_alone_long_enough_is_off_cooldown():
    shows = [_show("2026-05-01", "Old", "A", "B", "C", "D", "E")] + \
            [_filler(f"2026-05-{day:02d}", tag=f"G{day}") for day in range(2, 8)]
    rows = _by_song(song_scores(shows, ScoreConfig(), asof="2026-06-01"))
    assert rows["Old"].cooldown == 1.0


def test_an_overdue_song_gets_a_due_boost():
    """A song played every other show that has now missed several is worth more than its raw
    rate says. That is the rotation signal, and it is the reason gap_ratio exists."""
    shows = []
    for day in range(1, 9):                      # Regular plays on alternating early shows
        songs = ["Regular"] if day <= 4 and day % 2 else []
        shows.append(_show(f"2026-05-{day:02d}", *songs, *[f"F{i}" for i in range(6)]))
    rows = _by_song(song_scores(shows, ScoreConfig(), asof="2026-06-01"))
    assert rows["Regular"].gap_ratio > 1.0
    assert rows["Regular"].score > rows["Regular"].base * rows["Regular"].cooldown


def test_the_due_boost_is_capped():
    """Uncapped, a song played once in 1994 and never again becomes the top pick forever."""
    tight = ScoreConfig(gap_cap=3.0, gap_weight=0.35, min_show_songs=1, cooldown=())
    shows = [_show("1994-01-01", "Once")] + \
            [_filler(f"2026-05-{day:02d}", tag=f"G{day}") for day in range(1, 9)]
    rows = _by_song(song_scores(shows, tight, asof="2026-06-01"))
    # due saturates at 1 + gap_weight * (gap_cap - 1); it must not run away with the gap
    assert rows["Once"].score <= rows["Once"].base * (1 + 0.35 * (3.0 - 1)) + 1e-9


def test_lookback_years_hard_excludes_older_history():
    shows = [_show("2000-01-01", "Ancient"), _show("2026-05-01", "Recent")]
    rows = _by_song(song_scores(shows, ScoreConfig(min_show_songs=1, lookback_years=5),
                                asof="2026-06-01"))
    assert set(rows) == {"Recent"}


def test_era_weighting_discounts_shows_before_the_threshold():
    """Separate knob from recency and it has to be shown to be one, or it is just a second
    half-life wearing a different name."""
    shows = [_show("2026-01-01", "Meat")]
    full = ScoreConfig(min_show_songs=1, era_full_from_year=2026)
    halved = ScoreConfig(min_show_songs=1, era_full_from_year=2027, era_weight_before=0.5)
    hot = _by_song(song_scores(shows, full, asof="2026-06-01"))["Meat"]
    cold = _by_song(song_scores(shows, halved, asof="2026-06-01"))["Meat"]
    assert cold.base == pytest.approx(hot.base)      # sole show: weight cancels in the ratio
    assert cold.weighted_plays < hot.weighted_plays  # but the evidence really is worth less


def test_rows_come_back_best_first_with_ties_broken_on_name():
    shows = [_show("2026-05-01", "Zelda", "Aurora", "A", "B", "C", "D")]
    rows = song_scores(shows, ScoreConfig(), asof="2026-06-01")
    assert [r.song for r in rows] == sorted(
        [r.song for r in rows], key=lambda s: (-_by_song(rows)[s].score, s))
    assert rows[0].score >= rows[-1].score


def test_an_empty_corpus_scores_nothing_rather_than_dividing_by_zero():
    assert song_scores([], ScoreConfig(), asof="2026-06-01") == []


def test_a_show_with_an_unreadable_date_is_skipped_not_crashed_on():
    """`2008-31-08` is in the real corpus. A scorer that raises on it cannot run at all."""
    shows = [_show("2008-31-08", "Bad"), _show("2026-05-01", "Good", "A", "B", "C", "D", "E")]
    rows = _by_song(song_scores(shows, ScoreConfig(), asof="2026-06-01"))
    assert "Good" in rows and "Bad" not in rows


def test_the_config_is_frozen():
    with pytest.raises(AttributeError):
        ScoreConfig().gap_weight = 99


def test_overdue_ratio_is_the_single_shared_definition():
    """The formula published in songbook.py and pinned again in
    test_rotation_parity.py against the JS side -- this is the ordinary-case check that it
    computes what it says: shows since last played, over the typical gap between plays."""
    assert overdue_ratio(gap=6, mean_gap=3.0) == 2.0


def test_overdue_ratio_falls_back_when_the_mean_gap_is_zero():
    """The fallback that used to differ silently between this scorer and the Scorecard's
    ancestor (0.0 there, 1.0 here, per the design doc's own measurement) -- settled in slice
    2a and pinned here as the one module constant both this function and
    test_rotation_parity.py read, rather than two independently typed literals."""
    assert overdue_ratio(gap=5, mean_gap=0.0) == OVERDUE_FALLBACK_RATIO


def test_rotation_is_public_and_still_correct():
    """_rotation became rotation() in slice 4, as groundwork for the Scorecard port -- see
    the function's own docstring for why the Songbook itself never calls it. This test only
    re-checks that the rename did not change the arithmetic."""
    past = [(dt.date(2026, 1, 1), ["A"]), (dt.date(2026, 1, 2), ["B"]),
            (dt.date(2026, 1, 3), ["A"])]
    since, typical = rotation(past)
    assert since["A"] == 0 and typical["A"] == 2.0

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for the per-song structural features.

Where a song sits in a night is a property of the song, not of the night. These lock the six
rates, the slot average and the segue partners against hand-built setlists small enough to
check by eye.
"""

from dataclasses import FrozenInstanceError

import pytest

from setlistkit.catalog.features import Rates, SongFeature, song_features


def _entry(song, segue=False, non_song=False):
    return {"song": song, "segue": segue, "non_song": non_song}


def _show(date, sets, encore=()):
    return {"date": date, "source": "description", "identifier": f"t{date}",
            "sets": [[_entry(*s) if isinstance(s, tuple) else _entry(s) for s in one]
                     for one in sets],
            "encore": [_entry(*e) if isinstance(e, tuple) else _entry(e) for e in encore]}


def _by_song(features):
    return {f.song: f for f in features}


def test_opener_and_set_closer_rates():
    shows = [_show("2025-01-01", [["Aurora", "Wormhole", "Bearsong"]]),
             _show("2025-01-02", [["Aurora", "Bearsong", "Wormhole"]])]
    feat = _by_song(song_features(shows))
    assert feat["Aurora"].rates.opener == 1.0          # opened both sets
    assert feat["Aurora"].rates.set_closer == 0.0
    assert feat["Bearsong"].rates.set_closer == 0.5    # closed one of its two plays


def test_a_tagged_non_song_is_never_a_feature():
    """The corpus stores 'break' and 'Drums' as tagged non-songs. They are not songs.

    moe-pack tags fourteen of these. Counted, 'break' would be the strongest set-closer in the
    band's history on 755 plays, and every real closer's rate would be diluted by it.
    """
    shows = [_show("2025-01-01", [["Aurora", ("break", False, True)]])]
    feat = _by_song(song_features(shows))
    assert "break" not in feat
    assert feat["Aurora"].rates.set_closer == 1.0   # the non-song did not steal the close


def test_encore_rate_and_show_closer():
    shows = [_show("2025-01-01", [["Aurora", "Bearsong"]], encore=["Plane Crash"])]
    feat = _by_song(song_features(shows))
    assert feat["Plane Crash"].rates.encore == 1.0
    # An encore closes the night, so nothing in the final SET is the show closer.
    assert feat["Bearsong"].rates.show_closer == 0.0


def test_show_closer_when_there_is_no_encore():
    shows = [_show("2025-01-01", [["Aurora"], ["Bearsong", "Wormhole"]])]
    feat = _by_song(song_features(shows))
    assert feat["Wormhole"].rates.show_closer == 1.0
    assert feat["Aurora"].rates.show_closer == 0.0    # closed set 1, not the show


def test_segue_rates_and_partners():
    shows = [_show("2025-01-01", [[("Aurora", True), "Wormhole"]]),
             _show("2025-01-02", [[("Aurora", True), "Wormhole"]]),
             _show("2025-01-03", [[("Aurora", True), "Bearsong"]])]
    feat = _by_song(song_features(shows))
    assert feat["Aurora"].rates.segue_out == 1.0
    assert feat["Wormhole"].rates.segue_in == 1.0
    assert feat["Aurora"].top_partners[0] == ("Wormhole", 2)


def test_mean_slot_places_a_song_in_the_set():
    shows = [_show("2025-01-01", [["Aurora", "Wormhole", "Bearsong"]])]
    feat = _by_song(song_features(shows))
    assert feat["Aurora"].mean_slot == 0.0     # first of three
    assert feat["Wormhole"].mean_slot == 0.5   # middle
    assert feat["Bearsong"].mean_slot == 1.0   # last


def test_a_one_song_set_lands_in_the_middle_rather_than_dividing_by_zero():
    """A set of one has no first-to-last axis. 0.5 is the honest answer, not 0.0."""
    shows = [_show("2025-01-01", [["Aurora"]])]
    assert _by_song(song_features(shows))["Aurora"].mean_slot == 0.5


def test_first_and_last_seen_span_the_plays():
    shows = [_show("1998-06-01", [["Aurora"]]), _show("2026-07-18", [["Aurora"]])]
    feat = _by_song(song_features(shows))
    assert feat["Aurora"].first_seen == "1998-06-01"
    assert feat["Aurora"].last_seen == "2026-07-18"
    assert feat["Aurora"].n_plays == 2


def test_ordering_is_stable_for_equal_play_counts():
    shows = [_show("2025-01-01", [["Wormhole"], ["Aurora"]])]
    assert [f.song for f in song_features(shows)] == ["Aurora", "Wormhole"]


def test_returns_song_feature_instances():
    shows = [_show("2025-01-01", [["Aurora"]])]
    assert isinstance(song_features(shows)[0], SongFeature)


def test_the_rates_travel_together_and_stay_immutable():
    """The six share a denominator, so they are one object, and it is a frozen one.

    A feature is a value: 1,972 of them come back from one call and a consumer will sort,
    compare and de-duplicate them. A mutable member would defeat that quietly.
    """
    feature = song_features([_show("2025-01-01", [["Aurora"]])])[0]
    assert isinstance(feature.rates, Rates)
    assert hash(feature) is not None            # every member is immutable, so this works
    with pytest.raises(FrozenInstanceError):
        feature.rates.opener = 0.5

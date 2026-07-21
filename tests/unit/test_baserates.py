# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for per-year play counts, the prior a rotation model starts from.

These lock the counting semantics against the retired artifact this replaces, because a prior
that counts differently is a different prior and nothing downstream would say so. A song played
twice in a night counts twice; a show with no songs still counts in the denominator; a tagged
non-song counts never.
"""

import pytest

from setlistkit.catalog.baserates import YearRate, base_rates


def _show(date, sets, encore=()):
    """A stored show, in the shape :func:`setlistkit.store.corpus.shows` returns."""
    def entries(names):
        return [{"song": n, "segue": False, "non_song": False} if isinstance(n, str) else n
                for n in names]
    return {"date": date, "source": "description", "identifier": "t1",
            "sets": [entries(one) for one in sets], "encore": entries(encore)}


def _non_song(name):
    return {"song": name, "segue": False, "non_song": True}


def _by_year(rows):
    return {row.year: row for row in rows}


def test_one_show_becomes_one_year_carrying_its_songs():
    rows = _by_year(base_rates([_show("2025-06-01", [["Rebubula", "Meat"]])]))
    assert rows["2025"].shows == 1
    assert dict(rows["2025"].songs) == {"Rebubula": 1, "Meat": 1}


def test_shows_in_the_same_year_accumulate():
    rows = _by_year(base_rates([_show("2025-06-01", [["Rebubula"]]),
                                _show("2025-06-02", [["Rebubula", "Meat"]])]))
    assert rows["2025"].shows == 2
    assert dict(rows["2025"].songs) == {"Rebubula": 2, "Meat": 1}


def test_years_are_kept_apart_and_come_back_in_order():
    rows = base_rates([_show("2026-01-01", [["Meat"]]),
                       _show("1997-01-01", [["Rebubula"]]),
                       _show("2011-01-01", [["Meat"]])])
    assert [row.year for row in rows] == ["1997", "2011", "2026"]


def test_a_song_played_twice_in_one_night_counts_twice():
    """The artifact this replaces counted plays, not shows-containing.

    A rate of plays-per-show is what the consumer divides by its show denominator, and it can
    exceed 1.0 for a song the band played in both sets. Counting the night once instead would
    silently cap every such song at the show count and nothing downstream would notice.
    """
    rows = _by_year(base_rates([_show("2025-06-01", [["Rebubula"], ["Rebubula"]])]))
    assert dict(rows["2025"].songs) == {"Rebubula": 2}
    assert rows["2025"].shows == 1


def test_the_encore_counts_as_plays():
    rows = _by_year(base_rates([_show("2025-06-01", [["Meat"]], encore=["Rebubula"])]))
    assert dict(rows["2025"].songs) == {"Meat": 1, "Rebubula": 1}


def test_tagged_non_songs_never_reach_the_counts():
    """Counted as songs, "break" would be the most-played item in the band's history.

    The corpus deliberately stores set breaks, drum segments and banter as real records of the
    night. Every derivation has to filter them and this is the one that proves this one does.
    """
    show = _show("2025-06-01", [["Rebubula", _non_song("break"), _non_song("Drums")]],
                 encore=[_non_song("banter"), "Meat"])
    rows = _by_year(base_rates([show]))
    assert dict(rows["2025"].songs) == {"Rebubula": 1, "Meat": 1}


def test_a_show_with_no_songs_still_counts_in_the_denominator():
    """The denominator is nights the band played, not nights we could read a setlist off.

    Dropping an empty show would quietly inflate every rate in that year, because the
    numerator loses nothing and the denominator loses one.
    """
    rows = _by_year(base_rates([_show("2025-06-01", [["Meat"]]),
                                _show("2025-06-02", [[_non_song("break")]])]))
    assert rows["2025"].shows == 2
    assert dict(rows["2025"].songs) == {"Meat": 1}


def test_distinct_songs_is_derived_and_cannot_disagree_with_the_counts():
    rows = _by_year(base_rates([_show("2025-06-01", [["Rebubula", "Meat", "Rebubula"]])]))
    assert rows["2025"].distinct_songs == 2
    assert rows["2025"].distinct_songs == len(rows["2025"].songs)


def test_songs_come_back_most_played_first_with_ties_broken_on_name():
    """Two runs over one corpus must agree, and so must anything computed downstream."""
    rows = _by_year(base_rates([_show("2025-06-01", [["Zelda", "Meat", "Meat", "Aurora"]])]))
    assert rows["2025"].songs == (("Meat", 2), ("Aurora", 1), ("Zelda", 1))


def test_a_show_without_a_readable_year_is_skipped():
    rows = base_rates([_show("", [["Meat"]]), _show("nope", [["Meat"]]),
                       _show("2025-06-01", [["Meat"]])])
    assert [row.year for row in rows] == ["2025"]


def test_an_impossible_date_still_yields_its_year():
    """`2008-31-08` is in the real corpus -- a transposed month that the parser let through.

    Base rates read a year, not a calendar date, so the transposition does not move the show
    between years and this derivation is not the place to fix it. Locking that in so the
    eventual date fix does not silently change these numbers as a side effect.
    """
    rows = _by_year(base_rates([_show("2008-31-08", [["Meat"]])]))
    assert rows["2008"].shows == 1


def test_a_year_rate_is_frozen_and_hashable():
    """Immutable all the way down, so a prior cannot be edited after the fact by a consumer."""
    row = base_rates([_show("2025-06-01", [["Meat"]])])[0]
    assert hash(row)
    with pytest.raises(AttributeError):
        row.shows = 99


def test_no_shows_is_no_years_rather_than_an_error():
    assert base_rates([]) == []


def test_the_row_is_the_shape_the_retired_artifact_published():
    """{shows, songs, distinct_songs} per year, so a consumer written against the old JSON
    reads this without a translation layer that could drift."""
    row = base_rates([_show("2025-06-01", [["Meat"]])])[0]
    assert isinstance(row, YearRate)
    assert (row.year, row.shows, dict(row.songs), row.distinct_songs) == ("2025", 1,
                                                                          {"Meat": 1}, 1)

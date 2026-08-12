# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for predicting nights that have not happened.

Two things here can go wrong quietly, and both get more attention than the happy path.

The first is the PARSE. A phish.net /upcoming listing has no separator between records -- it is
four lines, then four more -- so reading it is an assumption about line counts wearing the clothes
of a parse. A file that gains or loses one line in the middle would not fail; it would slide every
later record up by one and hand back venues as artists and cities as venues, with dates attached to
the wrong shows. So the tests below care most about what happens when the file drifts.

The second is the ASYMMETRY with a scored night. A forecast has no outcome and no known set length,
and both absences have to survive into the result rather than being filled in with something
plausible.
"""

import json
from pathlib import Path

from setlistkit.model.scores import ScoreConfig
from setlistkit.model.setlists import PredictConfig
from setlistkit.picks.replay import ReplaySpec
from setlistkit.picks.upcoming import (TourDate, after, forecast, parse_json_tour,
                                       parse_tour, read_schedule, schedule_path)

LOOSE = ScoreConfig(min_show_songs=1, cooldown=())
QUICK = PredictConfig(rollouts=25, top_n=10, set_len=7)
SPEC = ReplaySpec(score=LOOSE, predict=QUICK)

REGULARS = ("Aurora", "Rebubula", "Meat", "Zelda", "Buster", "Wormhole", "Gumbo", "Cavern",
            "Timmy", "Moth")

LISTING = """# source: https://phish.net/upcoming

Phish
Saturday, 07/25/2026
Madison Square Garden
New York, NY
Bob Wagner & Friends
Tuesday, 09/01/2026
Champlain Valley Exposition
Essex Junction, VT
Phish
Friday, 09/04/2026
Dick's Sporting Goods Park
Commerce City, CO
"""


def _entry(song):
    return {"song": song, "segue": False, "non_song": False}


def _show(date, *songs):
    return {"date": date, "source": "description", "identifier": f"t{date}",
            "sets": [[_entry(s) for s in songs]], "encore": []}


def _corpus(days=range(1, 13), songs=REGULARS):
    return [_show(f"2026-01-{day:02d}", *songs) for day in days]


def test_a_listing_is_read_four_lines_at_a_time():
    schedule = parse_tour(LISTING)
    assert [row.date for row in schedule.dates] == ["2026-07-25", "2026-09-01", "2026-09-04"]
    assert schedule.dates[0].venue == "Madison Square Garden"
    assert schedule.dates[0].city == "New York, NY"
    assert schedule.problems == ()


def test_comments_and_blank_lines_do_not_shift_the_records():
    """The header comment and the blank line after it are why this is not a plain chunk-by-four."""
    assert parse_tour(LISTING).dates[0].artist == "Phish"


def test_a_file_that_drifts_out_of_step_complains_instead_of_inventing_a_schedule():
    """The failure the format invites, and the reason the date slot is checked.

    One extra line early on pushes every later record up by one. Nothing raises, nothing looks
    wrong, and the result would be a board full of shows at venues named after cities.
    """
    drifted = LISTING.replace("Phish\nSaturday, 07/25/2026",
                              "Phish\nSPECIAL GUEST TBA\nSaturday, 07/25/2026")
    schedule = parse_tour(drifted)
    assert schedule.problems, "a listing that no longer parses as dates must say so"
    assert not any(row.date == "2026-09-04" for row in schedule.dates)


def test_a_trailing_partial_record_is_reported_rather_than_dropped():
    schedule = parse_tour(LISTING + "Phish\nSunday, 09/06/2026\n")
    assert any("not a whole record" in problem for problem in schedule.problems)


def test_a_side_project_is_not_the_band():
    """A tour listing carries other acts, and predicting a Phish setlist for one of them would be
    a confident answer to a question nobody asked."""
    kept = parse_tour(LISTING).by_artist("Phish")
    assert [row.date for row in kept] == ["2026-07-25", "2026-09-04"]


def test_the_artist_match_ignores_case_but_not_identity():
    schedule = parse_tour(LISTING)
    assert len(schedule.by_artist("phish")) == 2
    assert schedule.by_artist("Mike Gordon") == ()


def test_shows_the_corpus_already_holds_are_not_upcoming():
    dates = parse_tour(LISTING).by_artist("Phish")
    assert [row.date for row in after(dates, "2026-08-01")] == ["2026-09-04"]


def test_the_cut_is_the_corpus_and_not_the_calendar():
    """A build is a pure function of the store. Cutting on today's date would render a different
    board from the same store depending on when the command ran."""
    dates = (TourDate("2020-01-01", "Phish", "v", "c"),)
    assert after(dates, "2019-12-31") == dates       # long past, still ahead of THIS corpus


def test_a_forecast_has_no_outcome_and_nothing_played():
    nights = forecast(_corpus(), SPEC, parse_tour(LISTING).by_artist("Phish"))
    assert len(nights) == 2
    assert all(night.outcome is None for night in nights)
    assert all(night.played == () for night in nights)


def test_a_forecast_draws_the_configured_set_length_because_nobody_knows_the_real_one():
    """The asymmetry that makes a forecast card harder than any scored card beside it.

    A scored night hands the model the number of songs actually played. Here that number does not
    exist yet, so the configured default stands in -- and the page has to say so, because the two
    kinds of card are not comparable on recall when one of them was told the answer's size.
    """
    night, = forecast(_corpus(), SPEC, (TourDate("2026-09-04", "Phish", "v", "c"),))
    assert night.drops, "a forecast with a corpus behind it must produce drops"


def test_every_upcoming_night_shares_its_cutoffs_because_no_show_happens_between_them():
    """Surprising and correct: cutoffs are played shows, and there are none in the future."""
    first, second = forecast(_corpus(), SPEC, parse_tour(LISTING).by_artist("Phish"))
    assert [drop.cutoff for drop in first.drops] == [drop.cutoff for drop in second.drops]


def test_no_cutoff_is_ever_dated_on_or_after_the_night_it_predicts():
    """The same honesty assertion the graded walk makes, restated for a night with no answer."""
    for night in forecast(_corpus(), SPEC, parse_tour(LISTING).by_artist("Phish")):
        assert all(drop.cutoff < night.night for drop in night.drops)


def test_a_corpus_with_nothing_gradeable_forecasts_nothing_rather_than_guessing():
    assert forecast([], SPEC, parse_tour(LISTING).by_artist("Phish")) == ()


# The JSON form, written by scripts/tour_dates.py. The failure this guards is the opposite of the
# four-line file's: nothing here can slide out of step, so what matters is that an entry missing a
# field is REPORTED rather than filled in with a plausible blank.

JSON_LISTING = json.dumps({
    "dates": [
        {"artist": "moe.", "date": "2026-09-04", "venue": "Ravinia", "city": "Highland Park, IL",
         "id": "a", "review": "listing runs through 2026-09-06"},
        {"artist": "moe.", "date": "2026-09-05", "venue": "The Wilbur", "city": "Boston, MA",
         "id": "b"},
    ],
    "seen": ["a", "b"],
})


def test_the_json_form_reads_the_four_fields_and_ignores_the_rest():
    """`id` and `review` belong to the review loop, not to a prediction."""
    schedule = parse_json_tour(JSON_LISTING)
    assert schedule.problems == ()
    assert schedule.dates == (
        TourDate("2026-09-04", "moe.", "Ravinia", "Highland Park, IL"),
        TourDate("2026-09-05", "moe.", "The Wilbur", "Boston, MA"),
    )


def test_an_entry_missing_a_field_is_reported_and_the_rest_still_read():
    """Keep and flag: one bad entry must not take the whole schedule down with it."""
    schedule = parse_json_tour(json.dumps({"dates": [
        {"artist": "moe.", "date": "2026-09-04", "venue": "", "city": "Boston, MA"},
        {"artist": "moe.", "date": "2026-09-05", "venue": "The Wilbur", "city": "Boston, MA"},
    ]}))
    assert [row.date for row in schedule.dates] == ["2026-09-05"]
    assert "venue" in schedule.problems[0]


def test_a_date_that_is_not_iso_is_refused_rather_than_sorted_as_text():
    """`after` compares dates as strings, so a non-ISO date would cut against the wrong boundary."""
    schedule = parse_json_tour(json.dumps({"dates": [
        {"artist": "moe.", "date": "09/04/2026", "venue": "v", "city": "c"},
    ]}))
    assert schedule.dates == ()
    assert "ISO" in schedule.problems[0]


def test_a_file_with_no_dates_list_says_so_instead_of_reading_as_empty():
    """An empty schedule and an unreadable one look identical downstream. They are not."""
    assert parse_json_tour(json.dumps({"seen": []})).problems == ("no 'dates' list in this file",)


def test_the_schedule_lives_at_a_reserved_name_in_the_pack():
    """Not configurable: the writer and every reader derive one path from the pack."""
    assert schedule_path("/packs/moe") == Path("/packs/moe/tour.json")


def test_read_schedule_picks_its_parser_by_suffix(tmp_path):
    """Both formats stay readable: one band's schedule is still gathered by hand."""
    as_json = tmp_path / "tour.json"
    as_json.write_text(JSON_LISTING, encoding="utf-8")
    dates, problems = read_schedule(str(as_json), "moe.", "2026-09-04")
    assert problems == ()
    assert [row.date for row in dates] == ["2026-09-05"]

    pasted = tmp_path / "phish-next-tour.txt"
    pasted.write_text(LISTING, encoding="utf-8")
    dates, _ = read_schedule(str(pasted), "Phish", "1990-01-01")
    assert dates, "a pasted phish.net listing must still read"

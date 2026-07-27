# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""phish.in projected as mirror recordings, so the durations chain can read it.

The interesting cases are all about what a MIRROR is for, which is not what a merge record is for.
A merge record is the night as we believe it happened; a mirror row is what the recording actually
contains, including the parts nobody performed to a room. The two disagree on purpose, and these
tests pin the places they do.
"""

import pytest

from setlistkit.catalog.parse_phish_in import parse_phish_in, phish_in_tapes
from setlistkit.catalog.normalizer import Normalizer

NZ = Normalizer(["Tweezer", "Reba", "Sand", "Ghost"])


def _track(title, position, duration_ms, **extra):
    return {"title": title, "position": position, "duration": duration_ms,
            "set_name": "Set 1", "exclude_from_stats": False, **extra}


def _show(date, *tracks, **extra):
    return {"date": date, "id": "1234", "audio_status": "complete",
            "tracks": list(tracks), **extra}


def test_a_show_becomes_one_recording_with_its_tracks_in_position_order():
    """Order comes from ``position``, never from the order the array happened to arrive in."""
    tapes = phish_in_tapes([_show("2026-01-05", _track("Reba", 2, 720_000),
                                  _track("Tweezer", 1, 1_170_000))])
    assert len(tapes) == 1
    tape = tapes[0]
    assert tape["date"] == "2026-01-05"
    assert [t["title"] for t in tape["duration_tracks"]] == ["Tweezer", "Reba"]
    assert [t["idx"] for t in tape["duration_tracks"]] == [0, 1]
    assert [t["seconds"] for t in tape["duration_tracks"]] == [1170.0, 720.0]


def test_the_tape_and_the_setlist_of_one_night_name_the_same_identifier():
    """Otherwise a night's tape and its merge record are two rows nothing joins.

    Both projections read the same payload, so this is cheap to hold and expensive to lose: the
    identifier is how a stored recording is traced back to the show it came from.
    """
    show = _show("2026-01-05", _track("Tweezer", 1, 1_170_000), _track("Reba", 2, 720_000),
                 _track("Sand", 3, 700_000))
    assert phish_in_tapes([show])[0]["identifier"] == \
        parse_phish_in([show], normalizer=NZ)[0]["identifier"]


def test_a_zero_length_track_is_stored_as_no_measurement_and_not_as_zero_seconds():
    """A gap inside a real tape: one track phish.in has no length for, the rest timed.

    Stored as NULL with the source string intact, which is the mirror's rule for a length it
    could not read. Zero would be a claim that a song took no time, and it would drag every
    statistic that song appears in toward the floor.

    The show carries a second, timed track deliberately. A show where EVERY duration is zero is
    a different fact -- a night with no audio at all -- and it is not mirrored; see below. Testing
    the two through one fixture is what let them be one behavior for as long as they were.
    """
    tape = phish_in_tapes([_show("2026-01-05", _track("Ghost", 1, 0),
                                 _track("Reba", 2, 720_000))])[0]
    track = tape["duration_tracks"][0]
    assert track["seconds"] is None
    assert track["length_raw"] == "0"


def test_a_track_with_no_duration_field_at_all_is_also_left_unmeasured():
    """A missing key and a zero are the same absence of a measurement, and must not diverge."""
    show = _show("2026-01-05", {"title": "Ghost", "position": 1, "set_name": "Set 1"},
                 _track("Reba", 2, 720_000))
    track = phish_in_tapes([show])[0]["duration_tracks"][0]
    assert track["seconds"] is None
    assert track["length_raw"] == ""


def test_a_night_phish_in_lists_but_holds_no_audio_for_is_not_mirrored_as_a_tape():
    """The catalog covers shows nobody has audio for, and it hands them over looking identical.

    Every track present, every ``duration`` zero, ``audio_status: missing``. Mirrored as-is that
    is a row claiming we hold a tape of a night nobody taped -- indistinguishable in the table
    from a real one, and inflating every count of "recordings held" that reads it. Measured on the
    live 2020-2026 cache when this was found: 14 of 249.

    The setlist for those nights is untouched, which is the whole reason for consulting a catalog
    that lists them.
    """
    dark = _show("2026-01-05", _track("Ghost", 1, 0), _track("Reba", 2, 0),
                 audio_status="missing")
    assert phish_in_tapes([dark]) == []
    assert [rec["date"] for rec in parse_phish_in([dark], normalizer=NZ)] == ["2026-01-05"]


def test_the_rule_is_the_durations_and_not_the_label_the_source_put_on_them():
    """``audio_status`` explains the 14; the durations are what decides.

    They agree exactly on today's cache. If they ever stop agreeing, a mirror row is worth having
    when it carries a measurement, whatever the payload calls itself -- because a measurement is
    the only thing a mirror row is for.
    """
    mislabeled = _show("2026-01-05", _track("Tweezer", 1, 1_170_000), audio_status="missing")
    assert len(phish_in_tapes([mislabeled])) == 1


def test_soundcheck_tracks_are_kept_in_the_tape_though_the_setlist_flags_them_non_song():
    """The two projections disagree here on purpose, and this is the assertion that says so.

    A mirror records what the recording contains. Dropping the soundcheck would make the stored
    track count disagree with the audio it claims to mirror; keeping it costs nothing, because it
    matches nothing in the setlist and is reported as unclaimed.
    """
    show = _show("2026-01-05", _track("Tweezer", 1, 1_170_000),
                 _track("Sand", 2, 700_000, set_name="Soundcheck", exclude_from_stats=True))
    tape = phish_in_tapes([show])[0]
    assert len(tape["duration_tracks"]) == 2
    record = parse_phish_in([show], normalizer=NZ)[0]
    assert record["n_songs"] == 1


def test_the_synthesized_filename_never_replaces_the_stated_title():
    """``name`` is invented so the recognition has something filename-shaped to read.

    ``title`` holds the source's own string unaltered beside it. If these ever became one column
    a synthesized name would be indistinguishable from a stated one.
    """
    track = phish_in_tapes([_show("2026-01-05", _track("Tweezer", 1, 1_170_000))])[0][
        "duration_tracks"][0]
    assert track["title"] == "Tweezer"
    assert track["name"] == "01 Tweezer.mp3"


def test_a_show_with_no_tracks_or_no_date_produces_no_recording():
    """A recording with nothing in it is not a recording, and neither is one nothing can date."""
    assert phish_in_tapes([_show("2026-01-05"), _show("", _track("Reba", 1, 720_000)),
                           {}, None]) == []


def test_min_year_floors_the_projection_and_the_result_is_sorted():
    tapes = phish_in_tapes([_show("2026-01-05", _track("Reba", 1, 720_000)),
                            _show("2019-01-05", _track("Reba", 1, 720_000))], min_year=2020)
    assert [tape["date"] for tape in tapes] == ["2026-01-05"]


def test_every_transfer_is_credited_to_the_archive_rather_than_to_a_person():
    """There is no taper to name, and inventing one would be crediting nobody as somebody."""
    assert phish_in_tapes([_show("2026-01-05", _track("Reba", 1, 720_000))])[0]["uploader"] \
        == "phish.in"


def test_the_projection_is_defensive_about_junk_rather_than_raising():
    assert phish_in_tapes([{"date": "2026-01-05", "tracks": [None, "nonsense"]}]) == []


@pytest.mark.parametrize("milliseconds,expected", [(1_170_000, 1170.0), (1, 0.001), (0, None)])
def test_durations_are_milliseconds_and_convert_by_exactly_one_thousand(milliseconds, expected):
    """The unit is the one thing here nothing else can catch.

    A factor of a thousand in either direction produces a page of plausible-looking numbers --
    Tweezer at 19 seconds or at 19,000 -- and every downstream consistency check still passes.
    """
    track = phish_in_tapes([_show("2026-01-05", _track("Tweezer", 1, milliseconds),
                                  _track("Reba", 2, 720_000))])[0]["duration_tracks"][0]
    assert track["seconds"] == expected

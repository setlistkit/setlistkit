# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The measuring projection of an item's files, and its promise not to disturb the other one.

Three of these are regression tests for bugs that have already happened once, in the previous
implementation, and cost real corpus damage before anyone noticed:

* the format pick, which returned durations rounded to the nearest second while the validation
  it fed was measured at five;
* the track order, which interleaved the sets of 65 of 436 tapes;
* the container drop, which put a whole set in slot 1 and shifted a night by one, reporting a
  six-minute song at 62 minutes.

The fourth guarantee is the one this slice is built on: :func:`_tracks` cannot move. Every
setlist in the corpus is decided against it, so a projection added beside it that changed it by
a single element would rewrite shows this slice has no business touching.
"""

from setlistkit.sources.archive_org import (_AUDIO_FORMATS, _DURATION_FORMATS, _tracks,
                                            duration_tracks)


def test_the_two_format_lists_cover_exactly_the_same_formats():
    """A format known to one projection and not the other is silently unmeasurable.

    _tracks would happily read a tracklist out of it while duration_tracks passed it over, so
    every tape carrying only that format would store a setlist and no durations, and the only
    symptom would be a track count that came out slightly low.
    """
    assert set(_DURATION_FORMATS) == _AUDIO_FORMATS
    assert len(_DURATION_FORMATS) == len(set(_DURATION_FORMATS))     # no duplicate ranking


def test_the_format_pick_prefers_precision_over_density():
    """Finding A. FLAC states "575.47"; VBR MP3 states "09:35" and throws away the rest.

    The MP3 derivative has more files here, so _tracks takes it -- which is right for a parser
    mining titles. Rounding thirty songs a night to the nearest second is error at the scale of
    the signal the durations chain measures, so this one goes the other way.
    """
    files = [
        {"format": "VBR MP3", "name": "01.mp3", "track": "1", "length": "09:35"},
        {"format": "VBR MP3", "name": "02.mp3", "track": "2", "length": "04:12"},
        {"format": "Flac", "name": "d1.flac", "track": "1", "length": "575.47"},
    ]
    audio_format, tracks = duration_tracks(files)
    assert audio_format == "Flac"
    assert [t["seconds"] for t in tracks] == [575.47]
    # ...and the parser still reads what it read before, off the very same payload.
    assert [t["name"] for t in _tracks(files)] == ["01.mp3", "02.mp3"]


def test_a_format_that_states_no_length_at_all_is_passed_over():
    """Otherwise it wins the preference order and hands back a tracklist that measures nothing."""
    files = [
        {"format": "Flac", "name": "d1t01.flac", "track": "1"},
        {"format": "VBR MP3", "name": "01.mp3", "track": "1", "length": "09:35"},
    ]
    audio_format, tracks = duration_tracks(files)
    assert audio_format == "VBR MP3"
    assert [t["seconds"] for t in tracks] == [575.0]


def test_one_unreadable_length_inside_the_winning_format_is_kept_as_none():
    """Kept, not dropped: dropping renumbers the tape and hides the parser bug that caused it.

    The source string rides along so the bug is diagnosable from the database rather than
    needing a re-pull of four thousand items to reproduce.
    """
    files = [
        {"format": "Flac", "name": "d1t01.flac", "track": "1", "length": "575.47"},
        {"format": "Flac", "name": "d1t02.flac", "track": "2", "length": "unknown"},
        {"format": "Flac", "name": "d1t03.flac", "track": "3", "length": "120.0"},
    ]
    _, tracks = duration_tracks(files)
    assert [t["seconds"] for t in tracks] == [575.47, None, 120.0]
    assert [t["idx"] for t in tracks] == [0, 1, 2]            # no gap where the unreadable one is
    assert tracks[1]["length_raw"] == "unknown"


def test_an_item_with_no_audio_measures_nothing_and_says_so():
    """A photo set, a stub, an artwork-only upload. Not an error, just not a recording."""
    assert duration_tracks([{"format": "JPEG", "name": "cover.jpg"}]) == ("", [])
    assert duration_tracks(None) == ("", [])
    assert duration_tracks([]) == ("", [])


# --- Finding B: play order -------------------------------------------------------------------

def test_a_track_field_that_restarts_per_set_does_not_interleave_the_tape():
    """Finding B, as a regression test. This corrupted 65 of 436 tapes before it was caught.

    archive.org's `track` restarts at 1 for each set, so both s1t01 and s2t01 report 1. Sorting
    by it gives set 1 track 1, set 2 track 1, set 1 track 2 -- a shuffled deck, against which
    every downstream mapping lines up the taper's tracklist and gets a different song.
    """
    files = [
        {"format": "Flac", "name": "moe2024-01-01.s2t01.flac", "track": "1", "length": "100.0"},
        {"format": "Flac", "name": "moe2024-01-01.s1t01.flac", "track": "1", "length": "200.0"},
        {"format": "Flac", "name": "moe2024-01-01.s2t02.flac", "track": "2", "length": "300.0"},
        {"format": "Flac", "name": "moe2024-01-01.s1t02.flac", "track": "2", "length": "400.0"},
    ]
    _, tracks = duration_tracks(files)
    assert [t["name"].split(".")[1] for t in tracks] == ["s1t01", "s1t02", "s2t01", "s2t02"]
    assert [t["idx"] for t in tracks] == [0, 1, 2, 3]


def test_a_track_field_unique_across_the_item_is_believed():
    """Where it IS unique it is the taper's own numbering and beats guessing from a filename."""
    files = [
        {"format": "Flac", "name": "zzz-last.flac", "track": "3", "length": "300.0"},
        {"format": "Flac", "name": "aaa-first.flac", "track": "1", "length": "100.0"},
        {"format": "Flac", "name": "mmm-middle.flac", "track": "2", "length": "200.0"},
    ]
    _, tracks = duration_tracks(files)
    assert [t["seconds"] for t in tracks] == [100.0, 200.0, 300.0]


def test_filenames_sort_the_way_a_human_reads_them():
    """t2 before t10. Plain string order puts "10" before "2" and reorders every long set."""
    files = [{"format": "Flac", "name": f"moe.t{n}.flac", "track": "1", "length": f"{n}.0"}
             for n in (10, 2, 1)]
    _, tracks = duration_tracks(files)
    assert [t["seconds"] for t in tracks] == [1.0, 2.0, 10.0]


def test_mixed_filename_shapes_sort_instead_of_raising():
    """A number must never have to compare against a string.

    re.split on a digit group keeps text and digits aligned for names of the same shape, and one
    tape uploaded under a different convention is enough to break that alignment -- which is a
    TypeError partway through an ingest of four thousand items, not a mis-sort.
    """
    files = [
        {"format": "Flac", "name": "01-opener.flac", "track": "1", "length": "100.0"},
        {"format": "Flac", "name": "encore.flac", "track": "1", "length": "200.0"},
    ]
    _, tracks = duration_tracks(files)
    assert len(tracks) == 2


# --- containers ------------------------------------------------------------------------------

def _split_set(prefix, lengths, container=None):
    files = [{"format": "Flac", "name": f"{prefix}.t{n:02d}.flac", "track": "1",
              "length": str(length)} for n, length in enumerate(lengths, start=1)]
    if container is not None:
        files.insert(0, {"format": "Flac", "name": f"{prefix}.flac", "track": "1",
                         "length": str(container)})
    return files


def test_a_whole_set_uploaded_beside_its_own_tracks_is_not_a_track():
    """It natural-sorts into slot 1 and shifts the whole night by one.

    The previous implementation reported a six-minute song at 62 minutes, the next at 102 and
    the next at 78 -- numbers no listener would believe, and ones no aggregate flagged.
    """
    files = _split_set("moe2023-06-27.s01", [375.0, 277.0, 500.0], container=1152.0)
    _, tracks = duration_tracks(files)
    assert [t["seconds"] for t in tracks] == [375.0, 277.0, 500.0]
    assert [t["idx"] for t in tracks] == [0, 1, 2]     # renumbered AFTER the bag came out


def test_one_nested_file_of_the_same_length_is_not_enough_to_call_the_parent_a_bag():
    """Two children is the floor, and this is the case it buys.

    A single nested file matching its parent's duration is a duplicate upload, a rename, or one
    song a taper happened to file under a longer name -- and "these two are the same length" is
    not evidence that one CONTAINS the other. Requiring two makes the test arithmetic again:
    a sum of parts, not a coincidence of one.
    """
    files = _split_set("moe2023-06-27.s01", [652.0])
    files.insert(0, {"format": "Flac", "name": "moe2023-06-27.s01.flac", "track": "1",
                     "length": "652.0"})
    _, tracks = duration_tracks(files)
    assert len(tracks) == 2


def test_a_file_that_merely_shares_a_prefix_is_not_a_container():
    """The test is arithmetic. A whole-set file that does not sum to its parts is a real track."""
    files = _split_set("moe2023-06-27.s01", [375.0, 277.0, 500.0], container=99.0)
    _, tracks = duration_tracks(files)
    assert len(tracks) == 4


def test_an_unreadable_length_never_makes_a_bag_look_right():
    """An unknown does not sum. Guessing it to be zero makes every bag look slightly too long."""
    files = _split_set("moe2023-06-27.s01", [375.0, 277.0], container=652.0)
    files[1]["length"] = "unknown"                    # one child now unmeasured
    _, tracks = duration_tracks(files)
    assert len(tracks) == 3                           # the container is kept, not guessed away


# --- the promise to the parser ---------------------------------------------------------------

def test_the_parser_projection_is_untouched_by_all_of_it():
    """Every setlist in the corpus is decided against _tracks. Slice 1 must not move one.

    Density, source order, and the raw length string, over a payload that exercises every rule
    the other projection applies: a denser lossy format, a per-disc track field, and a container.
    """
    files = [
        {"format": "Flac", "name": "moe.s01.flac", "track": "1", "title": "Set One",
         "length": "652.0"},
        {"format": "Flac", "name": "moe.s01.t02.flac", "track": "2", "title": "ATL",
         "length": "277.0"},
        {"format": "Flac", "name": "moe.s01.t01.flac", "track": "1", "title": "Rebubula",
         "length": "375.0"},
        {"format": "VBR MP3", "name": "02.mp3", "track": "2", "title": "ATL", "length": "04:37"},
        {"format": "VBR MP3", "name": "01.mp3", "track": "1", "title": "Rebubula",
         "length": "06:15"},
        {"format": "VBR MP3", "name": "03.mp3", "track": "3", "title": "Wormhole",
         "length": "08:20"},
    ]
    assert _tracks(files) == [
        {"track": "2", "title": "ATL", "name": "02.mp3", "length": "04:37"},
        {"track": "1", "title": "Rebubula", "name": "01.mp3", "length": "06:15"},
        {"track": "3", "title": "Wormhole", "name": "03.mp3", "length": "08:20"},
    ]
    # ...while the same payload measures off FLAC, in play order, with the bag removed.
    audio_format, tracks = duration_tracks(files)
    assert audio_format == "Flac"
    assert [t["title"] for t in tracks] == ["Rebubula", "ATL"]

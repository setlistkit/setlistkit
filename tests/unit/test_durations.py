"""Reading one tape against one night.

Nearly every case here is a bug that actually happened: a song title sitting inside another's,
a naming convention mistaken for a claim, a description offset by a block of gear lines. They are
written as the shapes that broke, so a regression names the original failure rather than a line
number.
"""
import re

import pytest

from setlistkit.catalog import durations as D
from setlistkit.catalog.normalizer import Normalizer


class _Pack(Normalizer):
    """A normalizer with just enough pack policy to exercise the matching rules."""

    def __init__(self, vocabulary=None, aliases=None, non_songs=()):
        super().__init__(vocabulary or [])
        self._aliases = aliases or {}
        self._non_songs = [re.compile(p) for p in non_songs]

    def aliases(self):
        return dict(self._aliases)

    def non_song_patterns(self):
        return list(self._non_songs)


def _show(sets, encore=()):
    return {"sets": [[dict(e) for e in one] for one in sets],
            "encore": [dict(e) for e in encore]}


def _entry(song, segue=False, non_song=False):
    return {"song": song, "segue": segue, "non_song": non_song}


def _tracks(*pairs):
    """(filename, seconds) pairs into the track shape the mirror stores."""
    return [{"idx": i, "name": name, "title": None, "length_raw": None, "seconds": secs}
            for i, (name, secs) in enumerate(pairs)]


def _night(show, normalizer=None):
    return D.Night.of(show, normalizer or _Pack())


# ---- the shapes of the data ------------------------------------------------------------------

def test_basename_drops_the_mic_rig_directory_and_the_extension():
    assert D.basename("moe. 2023-01-19 Neumann AK40/01 Stranger Than Fiction.flac") == \
        "01 Stranger Than Fiction"
    assert D.basename("t04.shn") == "t04"


def test_flatten_setlist_numbers_sets_from_one_and_labels_the_encore():
    show = _show([[_entry("Buster"), _entry("Moth")], [_entry("Meat")]], [_entry("Gone")])
    assert D.flatten_setlist(show) == [
        ("1", 1, "Buster"), ("1", 2, "Moth"), ("2", 1, "Meat"), ("E", 1, "Gone")]


@pytest.mark.parametrize("position,expected", [(1, True), (2, True), (3, False)])
def test_a_slot_touches_a_segue_from_either_side(position, expected):
    """Either the song segues out, or the one before segued into it. Both mean the boundary was
    drawn by ear, and two tapers will draw it differently."""
    show = _show([[_entry("Buster", segue=True), _entry("Moth"), _entry("Meat")]])
    assert D.touches_segue(show, "1", position) is expected


def test_touches_segue_is_false_past_the_end_of_a_set():
    show = _show([[_entry("Buster")]])
    assert D.touches_segue(show, "1", 9) is False
    assert D.touches_segue(show, "E", 1) is False


# ---- recognizing a song in a filename --------------------------------------------------------

def test_a_song_is_recognised_through_punctuation_spacing_and_case():
    night = _night(_show([[_entry("Stranger Than Fiction")]]))
    for written in ("01 Stranger Than Fiction", "01-stranger_than_fiction",
                    "01 STRANGER THAN FICTION", "01.StrangerThanFiction"):
        assert [h[0] for h in D.claims_in(written, night, _Pack())] == [0], written


def test_a_bare_track_index_names_nothing():
    night = _night(_show([[_entry("Buster")]]))
    assert D.claims_in("moe2026-02-05t04", night, _Pack()) == []
    assert D.claims_in("", night, _Pack()) == []


def test_a_short_title_must_line_up_with_a_whole_word():
    """"bud" sits inside "buddy". Squashing reopens substring collisions, so short names are
    looked up among word runs rather than anywhere in the text."""
    night = _night(_show([[_entry("Bud")]]))
    assert [h[0] for h in D.claims_in("03 Bud", night, _Pack())] == [0]
    assert D.claims_in("03 Buddy Holly", night, _Pack()) == []


def test_a_short_match_is_flagged_so_nobody_has_to_trust_it_quietly():
    night = _night(_show([[_entry("It"), _entry("Moth"), _entry("Meat")]]))
    tape = D.Tape("t", "2024-01-01", tuple(_tracks(("01 It.flac", 300.0),
                                                   ("02 Moth.flac", 400.0),
                                                   ("03 Meat.flac", 500.0))))
    reading = D.match_tape(tape, night, _Pack(), D.FILENAME)
    kinds = [e.kind for e in reading.edges]
    assert "short_name_match" in kinds


def test_a_title_sitting_inside_a_longer_one_does_not_also_claim_it():
    """A band with a song called "Time" makes "time" a whole word in "Deep This Time". A track
    named "Deep This Time" claimed both, the cursor jumped past Time's slot near the end of the
    night, and the ten songs in between became unreachable."""
    show = _show([[_entry("Deep This Time"), _entry("Moth"), _entry("Time")]])
    night = _night(show)
    hits = D.claims_in("05 Deep This Time", night, _Pack())
    assert [h[0] for h in hits] == [0]


def test_a_real_segue_pair_in_one_file_still_reports_both_songs():
    """The overlap rule must not silence a real pair: "Time > Breathe Reprise" names two
    distinct songs and neither title sits inside the other."""
    show = _show([[_entry("Time"), _entry("Breathe Reprise")]])
    night = _night(show)
    hits = D.claims_in("07 Time > Breathe Reprise", night, _Pack())
    assert sorted(h[0] for h in hits) == [0, 1]


def test_an_alias_recognises_the_song_it_points_at():
    pack = _Pack(vocabulary=["Rebubula"], aliases={"rebubula i": "Rebubula"})
    night = D.Night.of(_show([[_entry("Rebubula")]]), pack)
    assert [h[0] for h in D.claims_in("04 Rebubula I", night, pack)] == [0]


# ---- reading a whole tape --------------------------------------------------------------------

def test_the_cursor_only_ever_moves_forward_through_the_night():
    """A filename mentioning a song from earlier in the night must not drag the alignment back."""
    show = _show([[_entry("Buster"), _entry("Moth"), _entry("Meat")]])
    tape = D.Tape("t", "2024-01-01", tuple(_tracks(
        ("01 Buster.flac", 300.0), ("02 Moth.flac", 400.0),
        ("03 Buster tease.flac", 60.0), ("04 Meat.flac", 500.0))))
    reading = D.match_tape(tape, _night(show), _Pack(), D.FILENAME)
    assert [(r.song, r.seconds) for r in reading.rows] == [
        ("Buster", 300.0), ("Moth", 400.0), ("Meat", 500.0)]


def test_a_song_played_twice_keeps_the_first_free_slot_not_both():
    """claims() reports every slot whose song is in the filename. Treating two Moths as a segue
    pair advances the cursor past the later one and skips the whole setlist in between."""
    show = _show([[_entry("Moth"), _entry("Buster"), _entry("Meat"), _entry("Moth")]])
    tape = D.Tape("t", "2024-01-01", tuple(_tracks(
        ("01 Moth.flac", 300.0), ("02 Buster.flac", 200.0),
        ("03 Meat.flac", 400.0), ("04 Moth.flac", 500.0))))
    reading = D.match_tape(tape, _night(show), _Pack(), D.FILENAME)
    assert [(r.song, r.position) for r in reading.rows] == [
        ("Moth", 1), ("Buster", 2), ("Meat", 3), ("Moth", 4)]


def test_a_song_naming_every_file_is_a_convention_not_a_claim():
    """One night went dark for this. Its setlist carried an entry naming the show's format, and
    every file was named after that format, so every track claimed that one slot."""
    show = _show([[_entry("Acoustic Night"), _entry("Buster"), _entry("Moth"),
                   _entry("Meat"), _entry("Gone")]])
    night = _night(show)
    tape = D.Tape("t", "2024-01-01", tuple(_tracks(
        ("acousticnight2024s1t01 Buster.flac", 300.0),
        ("acousticnight2024s1t02 Moth.flac", 400.0),
        ("acousticnight2024s1t03 Meat.flac", 500.0),
        ("acousticnight2024s1t04 Gone.flac", 200.0),
        ("acousticnight2024s1t05 Buster.flac", 100.0))))
    assert 0 in D.tape_named_slots(tape, night, _Pack())
    reading = D.match_tape(tape, night, _Pack(), D.FILENAME)
    assert [r.song for r in reading.rows] == ["Buster", "Moth", "Meat", "Gone"]


def test_the_convention_rule_needs_five_tracks_before_it_fires():
    """On a two- or three-track fragment "the same song names every file" is a plausible accident
    -- a song, its reprise, its jam -- not evidence of a convention."""
    show = _show([[_entry("Meat")]])
    tape = D.Tape("t", "2024-01-01", tuple(_tracks(
        ("01 Meat.flac", 300.0), ("02 Meat reprise.flac", 100.0))))
    assert D.tape_named_slots(tape, _night(show), _Pack()) == set()


def test_a_segue_pair_in_one_file_records_the_other_song_and_an_edge():
    show = _show([[_entry("Bring You Down", segue=True), _entry("Brent Black")]])
    tape = D.Tape("t", "2024-01-01", tuple(_tracks(
        ("01 Bring You Down > Brent Black.flac", 900.0),)))
    reading = D.match_tape(tape, _night(show), _Pack(), D.FILENAME)
    assert reading.rows[0].combined_with == ("Brent Black",)
    assert [e.kind for e in reading.edges] == ["one_file_many_songs"]


# ---- which reading to believe ----------------------------------------------------------------

def _two_source_tape(desc_names, file_names, seconds=300.0):
    tracks = []
    for i, (desc, fname) in enumerate(zip(desc_names, file_names)):
        tracks.append({"idx": i, "name": f"{i + 1:02d} {fname}.flac", "title": None,
                       "length_raw": None, "seconds": seconds, "desc_song": desc})
    return D.Tape("t", "2024-01-01", tuple(tracks))


def test_when_both_readings_agree_the_verdict_says_so():
    """"agreed" is counted separately, or "filename won" swallows every tape where there was
    nothing to win and the tally stops meaning anything."""
    show = _show([[_entry("Buster"), _entry("Moth"), _entry("Meat")]])
    tape = _two_source_tape(["Buster", "Moth", "Meat"], ["Buster", "Moth", "Meat"])
    assert D.best_reading(tape, _night(show), _Pack()).verdict == D.AGREED


def test_a_source_that_names_nothing_defers_instead_of_silencing_the_other():
    """A taper who writes "t01" in their notes and puts the song on the file is still read, and
    so is the reverse. Because the fallback is per TRACK, a tape whose filenames are bare indexes
    reads identically whichever source leads -- which is why it comes back "agreed" rather than
    as a win for the description."""
    show = _show([[_entry("Buster"), _entry("Moth"), _entry("Meat"), _entry("Gone")]])
    tape = _two_source_tape(["Buster", "Moth", "Meat", "Gone"], ["t01", "t02", "t03", "t04"])
    reading = D.best_reading(tape, _night(show), _Pack())
    assert [r.song for r in reading.rows] == ["Buster", "Moth", "Meat", "Gone"]
    assert reading.verdict == D.AGREED


def test_a_description_that_explains_far_more_of_the_night_is_taken():
    """One tape's track 07 is called "Plane Crash" on disk while the notes call it Recreational
    Chemistry. The notes are right -- the real Plane Crash is track 14 -- and trusting the
    filename skips six songs and rejects the tape.

    The filenames here are wrong rather than absent, which is what makes the two readings
    really differ: track one names the LAST song of the night, so the forward-only cursor
    jumps to the end and everything after it is unreachable.
    """
    show = _show([[_entry("Buster"), _entry("Moth"), _entry("Meat"), _entry("Gone")]])
    tape = _two_source_tape(["Buster", "Moth", "Meat", "Gone"],
                            ["Gone", "Moth", "Meat", "Gone"])
    reading = D.best_reading(tape, _night(show), _Pack())
    assert reading.verdict == D.DESCRIPTION
    assert [r.song for r in reading.rows] == ["Buster", "Moth", "Meat", "Gone"]


def test_a_description_winning_by_a_single_row_does_not_get_to_win():
    """A near-tie is the SIGNATURE of an off-by-one description, not weak evidence for it. One
    tape's description opened with a date header, so every song sat one track late, and the
    shifted reading beat the correct one 13 to 12 -- booking a song at the length of another."""
    show = _show([[_entry("Buster"), _entry("Moth"), _entry("Meat")]])
    # The description is offset by one: it names Moth where Buster plays, and so on. It matches
    # one more slot than the filenames, which are right but leave one track unrecognized.
    tape = _two_source_tape(["Moth", "Meat", "Buster"], ["Buster", "Moth", "zzz"])
    reading = D.best_reading(tape, _night(show), _Pack())
    assert reading.verdict == D.FILENAME
    assert [r.song for r in reading.rows] == ["Buster", "Moth"]


# ---- putting a misnumbered file back -----------------------------------------------------

def test_a_misnumbered_file_is_put_back_where_the_listing_says_it_belongs():
    """One tape has TWO files numbered 02 and no 12, so "02 She.flac" natural-sorts into third
    place -- but She was played tenth. Zipping the listing onto that order glues the third song's
    name to the tenth song's audio."""
    tape = D.Tape("t", "2024-01-01", tuple(_tracks(
        ("01 Understand.flac", 100.0), ("02 It.flac", 200.0),
        ("02 She.flac", 300.0), ("03 Threw It All Away.flac", 400.0))))
    listing = [{"song": "Understand"}, {"song": "It"},
               {"song": "Threw It All Away"}, {"song": "She"}]
    moved = D.reorder_to_listing(tape, listing, _Pack())
    assert [t["seconds"] for t in moved.tracks] == [100.0, 200.0, 400.0, 300.0]


def test_a_listing_that_does_not_pair_completely_leaves_the_order_alone():
    """A partial pairing would mean guessing about the rest, and this chain does not guess."""
    tape = D.Tape("t", "2024-01-01", tuple(_tracks(
        ("01 Understand.flac", 100.0), ("02 It.flac", 200.0))))
    listing = [{"song": "Understand"}, {"song": "Something Nobody Named"}]
    assert D.reorder_to_listing(tape, listing, _Pack()).tracks == tape.tracks


def test_a_listing_of_the_wrong_length_is_never_applied():
    tape = D.Tape("t", "2024-01-01", tuple(_tracks(("01 Understand.flac", 100.0),)))
    assert D.reorder_to_listing(tape, [{"song": "A"}, {"song": "B"}], _Pack()).tracks == \
        tape.tracks


# ---- re-seating an offset description ---------------------------------------------------

@pytest.mark.parametrize("shift,expected", [
    (1, ["", "Buster", "Moth"]),
    (-1, ["Moth", "Meat", ""]),
    (0, ["Buster", "Moth", "Meat"]),
])
def test_shifting_a_listing_keeps_it_the_same_length_as_the_tape(shift, expected):
    """A dropped row is blanked rather than deleted: the equal-count check is what licenses using
    the listing at all. One track unidentified beats nineteen confidently mislabelled."""
    assert D.shift_listing(["Buster", "Moth", "Meat"], shift) == expected


def test_alignment_error_is_scored_on_log_ratio_not_raw_seconds():
    """Raw seconds would let the jam vehicles decide every alignment: a 2x error on a 4-minute
    song has to weigh the same as a 2x error on a 20-minute one."""
    pack = _Pack()
    medians = {"short": 100.0, "long": 1000.0}
    short_off, n1 = D.alignment_error(["Short"], [200.0], 0, medians, pack)
    long_off, n2 = D.alignment_error(["Long"], [2000.0], 0, medians, pack)
    assert n1 == n2 == 1
    assert short_off == pytest.approx(long_off)


def test_alignment_error_reports_no_evidence_when_nothing_carries_a_prior():
    error, scored = D.alignment_error(["Unknown"], [300.0], 0, {}, _Pack())
    assert scored == 0 and error == 99.0


def test_a_listing_is_only_re_seated_when_the_shift_is_far_better():
    """At least MIN_OFFSET_EVIDENCE songs must carry a prior and the shift must beat the status
    quo by the whole margin. Measured, the real cases score 3-5x better."""
    songs = ["Buster", "Moth", "Meat", "Gone", "Time", "Bud"]
    show = _show([[_entry(s) for s in songs]])
    # The tape's files are bare indexes, so only the description can read it -- which is exactly
    # when the offset test has to do the work alone.
    tracks = _tracks(*[(f"t{i:02d}.flac", 300.0) for i in range(1, 8)])
    record = {"identifier": "t", "date": "2024-01-01", "tracks": tracks}
    # The listing names a lineage line first, so every song sits one track late.
    listing = [{"song": "Channel Mix Lineage"}] + [{"song": s} for s in songs]
    priors = {"buster": 300.0, "moth": 300.0, "meat": 300.0,
              "gone": 300.0, "time": 300.0, "bud": 300.0}

    shifted = D.shift_listing([e["song"] for e in listing], -1)
    assert shifted[0] == "Buster"
    # With every song at its usual length the shifted reading is perfect and the unshifted one is
    # not, because its first row names nothing we have a prior for.
    before, _ = D.alignment_error([e["song"] for e in listing],
                                  [t["seconds"] for t in tracks], 0, priors, _Pack())
    after, scored = D.alignment_error(shifted, [t["seconds"] for t in tracks], -1, priors, _Pack())
    assert scored >= D.MIN_OFFSET_EVIDENCE
    assert after <= before
    assert len(show["sets"][0]) == len(songs)
    assert record["identifier"] == "t"


_OFFSET_SONGS = [("Buster", 300.0), ("Moth", 600.0), ("Meat", 900.0),
                 ("Gone", 1200.0), ("Time", 1500.0), ("Bud", 1800.0)]


def _well_named_tape(identifier):
    """A tape whose filenames explain the night on their own -- so it may supply a prior."""
    return {"identifier": identifier, "date": "2024-01-01",
            "tracks": _tracks(*[(f"{i:02d} {song}.flac", secs)
                                for i, (song, secs) in enumerate(_OFFSET_SONGS, start=1)])}


def _offset_corpus():
    show = _show([[_entry(song) for song, _ in _OFFSET_SONGS]])
    # Three tapes name their own songs, at consistent lengths, and become the priors.
    recordings = [_well_named_tape(f"good{n}") for n in range(3)]
    # One tape has bare indexes, so only its description can read it -- and that description
    # opens with a lineage line, so every song sits one track late.
    bare = {"identifier": "bare", "date": "2024-01-01",
            "tracks": _tracks(*[(f"t{i:02d}.flac", secs)
                                for i, (_, secs) in enumerate(_OFFSET_SONGS, start=1)],
                              ("t07.flac", 120.0))}
    recordings.append(bare)
    listing = [{"song": "Channel Mix Lineage: MixPre-10 II"}] + \
              [{"song": song} for song, _ in _OFFSET_SONGS]
    return recordings, {"2024-01-01": show}, {"bare": listing}


def test_priors_come_only_from_tapes_whose_filenames_explain_themselves():
    """The offset test is scored against these, so they must not come from tapes the offset test
    could have corrected. The two sets cannot overlap: a tape can only supply a prior if its
    FILENAMES read the night, and can only be corrected if they read nothing."""
    recordings, shows, _ = _offset_corpus()
    pack = _Pack()
    medians = D.provisional_medians(recordings, shows, pack)
    assert medians == {"buster": 300.0, "moth": 600.0, "meat": 900.0,
                       "gone": 1200.0, "time": 1500.0, "bud": 1800.0}


def test_a_description_opening_with_a_lineage_line_is_re_seated():
    """best_reading cannot catch this: with bare filenames there is nothing to compete, so the
    offset listing wins unopposed and the night is published confidently wrong. Only the
    DURATIONS can see it -- a four-minute song is not 25 minutes."""
    recordings, shows, tracklists = _offset_corpus()
    fixed, notes = D.correct_tracklist_offsets(recordings, shows, tracklists, _Pack())
    assert [n["identifier"] for n in notes] == ["bare"]
    assert notes[0]["shift"] == -1
    assert notes[0]["dropped_row"].startswith("Channel Mix Lineage")
    assert notes[0]["error_after"] < notes[0]["error_before"]
    # Every song now sits on the track that actually holds it, and the row that fell off the end
    # is blanked rather than deleted, so the listing still matches the file count.
    assert [row["song"] for row in fixed["bare"]] == \
        ["Buster", "Moth", "Meat", "Gone", "Time", "Bud", ""]
    assert len(fixed["bare"]) == len(recordings[-1]["tracks"])


def test_a_listing_that_is_already_right_is_left_alone():
    """The threshold sits in open space, not on top of the data: a correct listing must not be
    shifted by a marginally better-scoring alternative."""
    recordings, shows, _ = _offset_corpus()
    correct = [{"song": song} for song, _ in _OFFSET_SONGS]
    good = dict(recordings[0])
    tracklists = {good["identifier"]: correct}
    _, notes = D.correct_tracklist_offsets(recordings, shows, tracklists, _Pack())
    assert notes == []


def test_a_tape_that_bounced_a_whole_set_into_one_file_gives_itself_away():
    """There is no honest way to split a 79-minute file between the songs inside it. Any rule
    dividing it invents the boundary, which is the guessing this chain exists to refuse."""
    tape = D.Tape("t", "2024-01-01", tuple(_tracks(
        ("01 Opium into Timmy Tucker.flac", 4735.0), ("02 Meat.flac", 400.0))))
    assert tape.longest_seconds > D.GIVE_UP_TRACK_SECONDS


def test_a_night_counts_only_the_music_when_judging_whether_a_tape_was_read():
    """A taper who named all the songs but not the MC's introduction has named the whole setlist
    as far as this chain is concerned."""
    pack = _Pack(non_songs=[r"^intro$"])
    night = D.Night.of(_show([[_entry("Intro"), _entry("Buster"), _entry("Moth")]]), pack)
    assert len(night) == 3
    assert night.real_songs(pack) == 2


def test_override_labels_are_read_the_way_a_tapers_are():
    """An author writes what they read off the page. The canonical spelling, the encore marker and
    a trailing '>' are the machine's problem -- an override that had to be written in canonical
    form is one nobody could write without first querying the vocabulary."""
    pack = _Pack()
    listing = D.listing_from_labels(["captain america >", "  waiting for the punchline  ",
                                     "e. rebubula"], pack)
    assert [row["song"] for row in listing] == ["captain america", "waiting for the punchline",
                                                "e. rebubula"]
    assert [row["segue"] for row in listing] == [True, False, False]
    assert [row["idx"] for row in listing] == [0, 1, 2]

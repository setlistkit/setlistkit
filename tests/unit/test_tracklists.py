# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Reading a taper's own tracklist out of a description.

The safety net under all of this is the file count, and these lock it there. A listing that does
not line up with the tape's files is refused, because using it would mean guessing where the
discrepancy is -- and a misplaced guess writes a wrong duration that looks exactly like a right
one. Several of the tests below exist because a previous implementation guessed.
"""

import re

from setlistkit.catalog.normalizer import Normalizer
from setlistkit.catalog.tracklists import (BARE, BARE_SPLIT, MIN_PLAYS, NUMBERED_TRUNCATED,
                                           UNMATCHED, read_tracklist, song_vocabulary)

SONGS = ["Faker", "Rebubula", "Wormhole", "Aurora", "Timmy Tucker", "Bearsong", "Crushing",
         "Silver Sun", "Blue Jeans Pizza", "Moth", "Buster", "Plane Crash", "Wurm"]


class _Pack(Normalizer):
    """A normalizer with a vocabulary and the one classifier an interstitial needs."""

    def non_song_patterns(self):
        return [re.compile(r"^tuning$"), re.compile(r"^setbreak$"), re.compile(r"^crowd")]


def _norm():
    return _Pack(SONGS)


def _read(description, n_tracks, *, shows=(), gear=()):
    normalizer = _norm()
    return read_tracklist(description, n_tracks,
                          vocab=song_vocabulary(normalizer, shows),
                          normalizer=normalizer, gear_patterns=gear)


# --- the numbered reading --------------------------------------------------------------------

def test_a_numbered_listing_is_read_in_the_indexes_the_taper_wrote():
    desc = ("01. Aurora\n02. Rebubula\n03. Wormhole\n04. Bearsong\n05. Crushing\n")
    listing = _read(desc, 5)
    assert listing.matched
    assert [e.song for e in listing.entries] == ["Aurora", "Rebubula", "Wormhole", "Bearsong",
                                                 "Crushing"]
    assert [e.index for e in listing.entries] == [1, 2, 3, 4, 5]


def test_every_numbering_style_one_taper_or_another_uses():
    """Disc prefixes, encore prefixes, and a separator that is punctuation, a space, or nothing.

    "10.Wurm" is the one with no space at all, and that single missing character once hid an
    entire fully-labeled show.
    """
    desc = ("d1t01. Aurora\n2) Rebubula\n3 - Wormhole\n4 Bearsong\n10.Wurm\n")
    listing = _read(desc, 5)
    assert listing.matched
    assert [e.song for e in listing.entries] == ["Aurora", "Rebubula", "Wormhole", "Bearsong",
                                                 "Wurm"]


def test_an_encore_index_sorts_after_the_main_set_rather_than_among_it():
    """"E01" restarts at 1. Read as a bare index it would land at the front of the night."""
    desc = ("01. Aurora\n02. Rebubula\n03. Wormhole\n04. Bearsong\nE01 Plane Crash\n")
    listing = _read(desc, 5)
    assert listing.matched
    assert [e.song for e in listing.entries][-1] == "Plane Crash"


# --- the bare reading ------------------------------------------------------------------------

def test_a_bare_listing_is_found_by_recognising_the_songs():
    """No numbers, no pattern -- just words on lines. Recognition is the only way in."""
    desc = ("SET ONE\nAurora\nRebubula\nWormhole\nBearsong\nCrushing\n")
    listing = _read(desc, 5)
    assert listing.matched and listing.reading == BARE
    assert [e.song for e in listing.entries] == ["Aurora", "Rebubula", "Wormhole", "Bearsong",
                                                 "Crushing"]


def test_a_line_the_pack_calls_a_non_song_is_still_a_track():
    """Tuning occupies a file. Dropping it breaks the count that is the only safety net."""
    desc = ("SET ONE\nTuning\nAurora\nRebubula\nWormhole\nBearsong\n")
    listing = _read(desc, 5)
    assert listing.matched
    assert [e.song for e in listing.entries][0] == "Tuning"


def test_a_taper_note_about_the_show_is_not_a_track():
    """"* = with Suke Cerulo on guitar" is a line ABOUT the show, not a file on the tape.

    The normalizer knows that shape, but it knows it as a non-song -- which is also what it
    calls tuning. Asking WHICH pack pattern fired separates them: a guest note trips a shape
    rule that belongs to the normalizer and fires no pack pattern at all.
    """
    desc = ("Aurora\nRebubula\nWormhole\nBearsong\nCrushing\n* = with Suke Cerulo on guitar\n")
    listing = _read(desc, 5)
    assert listing.matched
    assert all("Suke" not in e.song for e in listing.entries)


def test_the_taper_describing_their_rig_is_not_a_track():
    desc = ("Recorded with Schoeps mk4 > sound devices\nsource: AUD\n"
            "Aurora\nRebubula\nWormhole\nBearsong\nCrushing\n")
    listing = _read(desc, 5)
    assert listing.matched
    assert [e.song for e in listing.entries] == ["Aurora", "Rebubula", "Wormhole", "Bearsong",
                                                 "Crushing"]


def test_a_pack_gear_word_is_dropped_too():
    """The lineage filter is the parser's, with the pack's shorthand folded in -- one answer."""
    desc = ("zoomf8\nAurora\nRebubula\nWormhole\nBearsong\nCrushing\n")
    listing = _read(desc, 5, gear=(r"zoomf\d",))
    assert listing.matched
    assert all("zoom" not in e.song.lower() for e in listing.entries)


# --- what decides between the readings -------------------------------------------------------

def test_the_file_count_picks_the_reading_and_nothing_else_does():
    """A numbering restart is really ambiguous: footnotes, or set two renumbered?

    Both readings are produced and the tape decides. Here the tape has 5 files, so the truncated
    reading is the right one -- the trailing "1." and "2." were footnotes.
    """
    desc = ("01. Aurora\n02. Rebubula\n03. Wormhole\n04. Bearsong\n05. Crushing\n"
            "1. taped by someone\n2. thanks for listening\n")
    listing = _read(desc, 5)
    assert listing.matched and listing.reading == NUMBERED_TRUNCATED
    assert [e.song for e in listing.entries] == ["Aurora", "Rebubula", "Wormhole", "Bearsong",
                                                 "Crushing"]


def test_the_same_restart_is_kept_whole_when_the_tape_says_it_was_set_two():
    """Identical shape, opposite answer, and only the file count can tell them apart.

    Cutting this one throws away half the show, which an earlier implementation did silently to
    every taper who numbers per set.
    """
    desc = ("01. Aurora\n02. Rebubula\n03. Wormhole\n"
            "01. Bearsong\n02. Crushing\n03. Moth\n")
    listing = _read(desc, 6)
    assert listing.matched
    assert [e.song for e in listing.entries] == ["Aurora", "Rebubula", "Wormhole", "Bearsong",
                                                 "Crushing", "Moth"]


def test_a_taper_who_wrote_a_setlist_has_their_segue_runs_split_to_match_the_files():
    """Some tapers group by segue on one line while the TAPE splits them into separate files."""
    desc = ("SET ONE\nCrushing > Silver Sun > Blue Jeans Pizza\nMoth > Buster\n"
            "Aurora\nRebubula\nWormhole\n")
    listing = _read(desc, 8)
    assert listing.matched and listing.reading == BARE_SPLIT
    assert [e.song for e in listing.entries] == ["Crushing", "Silver Sun", "Blue Jeans Pizza",
                                                 "Moth", "Buster", "Aurora", "Rebubula",
                                                 "Wormhole"]
    # ...and every piece but the last of a run segued, by construction: that is what put them
    # on one line in the first place.
    assert [e.segue for e in listing.entries][:3] == [True, True, False]


def test_the_same_run_is_left_whole_when_the_taper_kept_it_in_one_file():
    """We never have to know which kind of taper we are dealing with. The tape says."""
    desc = ("SET ONE\nCrushing > Silver Sun > Blue Jeans Pizza\nMoth > Buster\n"
            "Aurora\nRebubula\nWormhole\n")
    listing = _read(desc, 5)
    assert listing.matched
    assert [e.song for e in listing.entries][0] == "Crushing > Silver Sun > Blue Jeans Pizza"


def test_a_listing_that_lines_up_with_nothing_says_so_instead_of_being_believed():
    """The best attempt still comes back, because a mismatch has to be diagnosable. It just
    comes back SAYING it did not line up, so using it has to be a decision."""
    desc = ("01. Aurora\n02. Rebubula\n03. Wormhole\n04. Bearsong\n05. Crushing\n")
    listing = _read(desc, 11)
    assert not listing.matched and listing.reading == UNMATCHED
    assert len(listing.entries) == 5


def test_prose_that_merely_mentions_a_song_is_not_a_tracklist():
    """Under the floor there is no listing here, only a sentence with a song name in it."""
    listing = _read("Great show! Aurora was the highlight.\n", 1)
    assert not listing.matched and listing.entries == ()


# --- cleaning, which has to agree with the setlist parser ------------------------------------

def test_the_checksum_table_is_cut_off_the_end():
    desc = ("01. Aurora\n02. Rebubula\n03. Wormhole\n04. Bearsong\n05. Crushing\n"
            "shntool output\nlength expanded size cdr fmt ratio filename\n"
            "6:08.007 211971914 B cxx -- ---xx flac 0.5850 08 No Rain.flac\n")
    listing = _read(desc, 5)
    assert listing.matched
    assert all("flac" not in e.song for e in listing.entries)


def test_every_tag_is_a_line_break():
    """Where this parts company with the setlist parser, which turns most tags into a space.

    A tracklist is one entry per line and tapers mark those lines with whatever markup came to
    hand. A tag that becomes a space fuses two tracks and the tape comes up a file short.
    """
    desc = "<li>Aurora</li><li>Rebubula</li><li>Wormhole</li><li>Bearsong</li><li>Crushing</li>"
    listing = _read(desc, 5)
    assert listing.matched
    assert [e.song for e in listing.entries] == ["Aurora", "Rebubula", "Wormhole", "Bearsong",
                                                 "Crushing"]


def test_a_double_encoded_segue_is_resolved_before_it_is_read():
    """archive.org descriptions are routinely double-encoded; one pass leaves &gt; in a title."""
    desc = ("01. Aurora &amp;gt;\n02. Rebubula\n03. Wormhole\n04. Bearsong\n05. Crushing\n")
    listing = _read(desc, 5)
    assert listing.matched
    assert listing.entries[0].song == "Aurora" and listing.entries[0].segue is True


def test_a_trailing_taper_note_is_not_part_of_the_song_name():
    desc = ("01. Aurora (NH)\n02. Rebubula\n03. Wormhole\n04. Bearsong\n05. Crushing\n")
    listing = _read(desc, 5)
    assert listing.entries[0].song == "Aurora"


def test_a_note_in_the_middle_of_a_segue_run_comes_off_each_piece():
    """It is only TRAILING once the line has been split, so the strip has to run again there.

    Left alone it survives into a song called "Crushing (NH)", which joins to nothing.
    """
    desc = ("SET ONE\nCrushing (NH) > Silver Sun\nAurora\nRebubula\nWormhole\nBearsong\n")
    listing = _read(desc, 6)
    assert listing.matched
    assert [e.song for e in listing.entries][:2] == ["Crushing", "Silver Sun"]


def test_a_footnote_mark_is_stripped_including_the_ones_that_were_splitting_songs():
    """'$' was missing from the parser's footnote set, so "Buster $" was a song of its own."""
    desc = ("01. Buster $\n02. Moth*\n03. Aurora #\n04. Bearsong^\n05. Crushing\n")
    listing = _read(desc, 5)
    assert [e.song for e in listing.entries] == ["Buster", "Moth", "Aurora", "Bearsong",
                                                 "Crushing"]


def test_a_segue_arrow_is_never_welded_to_the_title():
    desc = ("01. Aurora >\n02. Rebubula ->\n03. Wormhole\n04. Bearsong\n05. Crushing\n")
    listing = _read(desc, 5)
    assert [e.song for e in listing.entries][:2] == ["Aurora", "Rebubula"]
    assert [e.segue for e in listing.entries][:3] == [True, True, False]


# --- the vocabulary --------------------------------------------------------------------------

def test_recognition_goes_through_the_packs_own_normalization():
    """The taper writes "The Faker"; the pack's canonical name is "Faker". One song.

    Comparing raw squashed strings misses this, and missing it costs the whole tape -- the
    listing comes up one entry short and lines up with nothing.
    """
    desc = ("SET ONE\nThe Faker\nAurora\nRebubula\nWormhole\nBearsong\n")
    listing = _read(desc, 5)
    assert listing.matched
    assert [e.song for e in listing.entries][0] == "The Faker"


def test_a_song_the_corpus_has_seen_often_enough_can_vouch_for_a_line():
    """The pack cannot list a cover the band played twice in 1998. Our own setlists can."""
    shows = [{"sets": [[{"song": "Cracker Cover"}]], "encore": []}] * MIN_PLAYS
    desc = ("SET ONE\nCracker Cover\nAurora\nRebubula\nWormhole\nBearsong\n")
    assert _read(desc, 5, shows=shows).matched


def test_a_title_the_corpus_saw_once_does_not_get_to_vouch_for_anything():
    """The corpus is parser OUTPUT, not a curated list, and its singletons are its accidents.

    Feeding all of them back makes the vocabulary a laundering channel: junk certifies junk. The
    cost is a once-played song going unrecognized, which leaves the tape unmatched and
    unused -- the safe direction, unlike a junk-inflated listing that MATCHES.
    """
    shows = [{"sets": [[{"song": "Setlist"}]], "encore": []}]
    desc = ("SET ONE\nSetlist\nAurora\nRebubula\nWormhole\nBearsong\n")
    assert not _read(desc, 5, shows=shows).matched


def test_a_tagged_non_song_never_enters_the_vocabulary_as_repertoire():
    shows = [{"sets": [[{"song": "Tuning", "non_song": True}]], "encore": []}] * 5
    vocab = song_vocabulary(_norm(), shows)
    assert "tuning" not in vocab.normalized


def test_the_vocabulary_is_sorted_so_a_fuzzy_match_cannot_move_between_runs():
    """difflib returns matches in the order it finds them, so a set makes the fuzzy branch
    depend on iteration order. A tape would read one way today and another tomorrow."""
    vocab = song_vocabulary(_norm())
    assert list(vocab.squashed) == sorted(vocab.squashed)


def test_a_typo_still_finds_the_song_it_meant():
    """Tapers type in the dark. The candidate pool is one band's repertoire, not English."""
    desc = ("SET ONE\nRebubulaa\nAurora\nWormhole\nBearsong\nCrushing\n")
    assert _read(desc, 5).matched

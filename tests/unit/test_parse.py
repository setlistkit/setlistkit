# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for the archive description parser.

Two things are being locked here. The first is the mechanism the old parse_archive.py learned
the hard way: the checksum table that became an encore, the credit roll that became twenty
songs, the venue that became a song, the gear preamble that had to be skipped without eating
the setlist behind it.

The second is the seam. There is now exactly one answer to "is this a song" --
``Normalizer.is_non_song`` -- and this parser TAGS what it finds instead of dropping it. The
tests below assert both halves of that: the tag is set, and the entry is still there.

No pack is loaded. A synthetic normalizer supplies a small vocabulary and a couple of non-song
patterns, which is enough to exercise every path without any band's data.
"""

import re

from setlistkit.catalog import (ArchivePolicy, Normalizer, count_songs, parse_archive_item,
                                parse_archive_items, title_band_filter)
from setlistkit.catalog.parse import clean_html

_VOCAB = ["Rebubula", "Meat", "Plane Crash", "Ophelia", "The Faker", "Recreational Chemistry",
          "Hi & Lo", "Timmy Tucker"]


# A synthetic pack: enough policy to exercise the hooks, none of a real band's data.
class _StubNormalizer(Normalizer):
    def __init__(self, extra_vocab=(), protected=()):
        super().__init__([*_VOCAB, *extra_vocab])
        self._protected = set(protected)

    def non_song_patterns(self):
        return [re.compile(r"^(?:intro|outro|tuning|crowd|banter|applause|drums|basssolo)+$"),
                re.compile(r"^setbreak$")]

    def protected_titles(self):
        return self._protected


def _item(**overrides):
    """An archive item with a normal, well-behaved description unless told otherwise."""
    item = {
        "identifier": "band2026-01-31.akg",
        "title": "band. Live at The Fillmore on 2026-01-31",
        "date": "2026-01-31",
        "venue": "The Fillmore",
        "coverage": "Denver, CO",
        "description": ("band. 2026-01-31 The Fillmore\n"
                        "Denver, CO\n"
                        "\n"
                        "Source: AKG mics > MixPre, transferred with XAct\n"
                        "\n"
                        "Set 1:\n"
                        "01. Rebubula\n"
                        "02. Tuning\n"
                        "03. Meat > Plane Crash\n"
                        "04. Ophelia\n"
                        "\n"
                        "Encore:\n"
                        "05. The Faker\n"),
    }
    item.update(overrides)
    return item


def _parse(item=None, policy=None):
    return parse_archive_item(item or _item(), normalizer=_StubNormalizer(), policy=policy)


def _titles(record):
    return [entry["song"] for one_set in record["sets"] for entry in one_set]


# --- the seam: one answer, and non-songs are kept ------------------------------------------

def test_non_song_is_tagged_and_kept_not_dropped():
    """the headline fix. The tape says they tuned, so the corpus says they tuned."""
    record = _parse()
    tuning = [entry for one_set in record["sets"] for entry in one_set
              if entry["song"] == "Tuning"]
    assert len(tuning) == 1
    assert tuning[0]["non_song"] is True
    assert _titles(record) == ["Rebubula", "Tuning", "Meat", "Plane Crash", "Ophelia"]


def test_real_songs_are_not_tagged():
    record = _parse()
    songs = [entry for one_set in record["sets"] for entry in one_set
             if entry["song"] != "Tuning"]
    assert all(entry["non_song"] is False for entry in songs)
    assert record["encore"][0] == {"song": "The Faker", "segue": False, "non_song": False}


def test_n_songs_counts_music_only():
    """Tuning is recorded but it is not repertoire, and completeness is measured in songs."""
    record = _parse()
    assert record["n_songs"] == 5           # four in the set, one encore, Tuning excluded
    assert count_songs(record["sets"], record["encore"]) == 5


def test_slash_joined_banter_gets_the_same_answer_as_a_bare_one():
    """the combo case that needed its own function in the old parser.

    ``is_non_song`` squashes before matching, so "Intro/Crowd/Banter" and "Intro" reach the
    same pattern. That is the whole point of routing through one place.
    """
    record = _parse(_item(description="Set 1:\n01. Intro/Crowd/Banter\n02. Meat\n03. Ophelia\n"))
    entries = record["sets"][0]
    assert entries[0]["song"] == "Intro/Crowd/Banter"
    assert entries[0]["non_song"] is True
    assert record["n_songs"] == 2


def test_standalone_guest_credit_is_tagged_rather_than_dropped():
    """"with Jake" is a note about who sat in. The old parser deleted it in two different
    places; now the normalizer says what it is and the parser writes that down."""
    record = _parse(_item(description="Set 1:\n01. Meat\n02. with Jake\n03. Ophelia\n"))
    assert [entry["song"] for entry in record["sets"][0]] == ["Meat", "with Jake", "Ophelia"]
    assert record["sets"][0][1]["non_song"] is True
    assert record["n_songs"] == 2


def test_slashed_guest_credit_is_tagged_once_the_numbering_comes_off():
    """"02. w/ Andy Frasco" only reaches the seam because the track number is stripped first.

    Splitting on "w/" ahead of the numbering truncated this entry to "02." and dropped it,
    which is a third accidental answer to a question that is supposed to have one. With the
    order fixed the normalizer gets asked, and it says what the entry actually is.
    """
    record = _parse(_item(description="Set 1:\n01. Meat\n02. w/ Andy Frasco\n03. Ophelia\n"))
    entries = record["sets"][0]
    assert [entry["song"] for entry in entries] == ["Meat", "w/ Andy Frasco", "Ophelia"]
    assert entries[1]["non_song"] is True
    assert record["n_songs"] == 2


def test_a_guest_note_appended_to_a_real_song_is_still_just_stripped():
    """the other half of the ordering: "Meat w/ Jake" is a Meat, annotated."""
    record = _parse(_item(description="Set 1:\n01. Meat w/ Jake\n02. Ophelia\n03. Rebubula\n"))
    assert _titles(record) == ["Meat", "Ophelia", "Rebubula"]


def test_ampersand_guest_note_gets_the_same_answer_as_the_slashed_one():
    """two spellings of one annotation used to get opposite treatment: tagged, and deleted."""
    record = _parse(_item(description="Set 1:\n01. Meat\n02. & Andy Frasco\n03. Ophelia\n"))
    entries = record["sets"][0]
    assert [entry["song"] for entry in entries] == ["Meat", "& Andy Frasco", "Ophelia"]
    assert entries[1]["non_song"] is True


def test_a_bare_instrument_is_tagged_rather_than_deleted():
    """a drum solo happened. The old parser answered this with its own word list and threw the
    entry away, so the corpus said nothing happened between song one and song three."""
    record = _parse(_item(description="Set 1:\n01. Meat\n02. Drums\n03. Ophelia\n"))
    entries = record["sets"][0]
    assert [entry["song"] for entry in entries] == ["Meat", "Drums", "Ophelia"]
    assert entries[1]["non_song"] is True
    assert record["n_songs"] == 2


def test_a_band_member_credit_line_is_not_a_setlist_entry_at_all():
    """the other side of that line: a name, a separator and instruments is not a performance."""
    record = _parse(_item(description=("Set 1:\n01. Meat\n"
                                       "Al Schnier - guitar, vocals\n"
                                       "02. Ophelia\n03. Rebubula\n")))
    assert _titles(record) == ["Meat", "Ophelia", "Rebubula"]


# --- segues ---------------------------------------------------------------------------------

def test_internal_segue_splits_one_token_into_two_songs():
    record = _parse()
    entries = {entry["song"]: entry for entry in record["sets"][0]}
    assert entries["Meat"]["segue"] is True
    assert entries["Plane Crash"]["segue"] is False


# --- structure: sets, encores, and where the setlist starts ---------------------------------

def test_encore_is_separated_from_the_sets():
    record = _parse()
    assert len(record["sets"]) == 1
    assert [entry["song"] for entry in record["encore"]] == ["The Faker"]


def test_gear_preamble_sits_above_the_setlist_and_is_skipped():
    """the region starts at the set header, so the lineage never gets parsed at all."""
    assert _titles(_parse()) == ["Rebubula", "Tuning", "Meat", "Plane Crash", "Ophelia"]


def test_gear_words_written_as_a_track_are_not_songs():
    """and when a taper does put the lineage in the numbered list, the words catch it."""
    record = _parse(_item(description=("Set 1:\n01. Rebubula\n02. AKG mics > MixPre\n"
                                       "03. Meat\n04. Ophelia\n")))
    assert _titles(record) == ["Rebubula", "Meat", "Ophelia"]


def test_a_song_in_the_vocabulary_survives_a_gear_word_collision():
    """"Wave" is a tape format AND a perfectly ordinary song title. Gear is the widest filter
    here and it cannot tell them apart, so anything the pack claims is never deleted on shape."""
    record = parse_archive_item(_item(description="Set 1:\n01. Meat\n02. Wave\n03. Ophelia\n"),
                                normalizer=_StubNormalizer(extra_vocab=("Wave",)))
    assert _titles(record) == ["Meat", "Wave", "Ophelia"]


def test_a_protected_title_survives_a_gear_word_collision():
    """the same guard, reached through protected_titles rather than the vocabulary."""
    record = parse_archive_item(_item(description="Set 1:\n01. Meat\n02. Wave\n03. Ophelia\n"),
                                normalizer=_StubNormalizer(protected=("Wave",)))
    assert _titles(record) == ["Meat", "Wave", "Ophelia"]


def test_first_numbered_track_marks_where_the_setlist_starts():
    """without it the region is the whole text and the taper's header comes along as a song."""
    record = _parse(_item(description=("Ophelia Hall\n"
                                       "01. Rebubula\n02. Meat\n03. Plane Crash\n")))
    assert _titles(record) == ["Rebubula", "Meat", "Plane Crash"]


def test_track_durations_are_stripped_rather_than_read():
    record = _parse(_item(description=("Set 1:\n01. Rebubula (7:43)\n02. Meat (12:01)\n"
                                       "03. Ophelia\n")))
    assert _titles(record) == ["Rebubula", "Meat", "Ophelia"]


def test_numbered_tracks_with_no_set_header_still_parse():
    """a taper who numbers the songs and writes only "Enc:" used to lose the entire main set."""
    record = _parse(_item(description=("Source: AKG mics, transferred\n"
                                       "01. Rebubula\n"
                                       "02. Meat\n"
                                       "03. Ophelia\n"
                                       "Enc: The Faker\n")))
    assert [entry["song"] for entry in record["sets"][0]] == ["Rebubula", "Meat", "Ophelia"]
    assert [entry["song"] for entry in record["encore"]] == ["The Faker"]


def test_bare_segue_run_with_no_markers_parses_whole_text():
    record = _parse(_item(description="Rebubula > Meat > Ophelia"))
    assert [entry["song"] for entry in record["sets"][0]] == ["Rebubula", "Meat", "Ophelia"]


# --- the tails: machine output and credit rolls ---------------------------------------------

def test_checksum_table_does_not_become_an_encore():
    """35 shows once had an encore made of an shntool report."""
    record = _parse(_item(description=("Set 1:\n01. Rebubula\n02. Meat\n03. Ophelia\n"
                                       "SHNTOOL OUTPUT\n"
                                       "length   expanded size   cdr  fmt  ratio  filename\n"
                                       "6:08.007   211971914 B   cxx   --   ---xx   flac\n")))
    assert record["encore"] == []
    assert _titles(record) == ["Rebubula", "Meat", "Ophelia"]


def test_credit_roll_named_by_the_band_is_cut():
    policy = ArchivePolicy(band_name="band.")
    record = _parse(_item(description=("Set 1:\n01. Rebubula\n02. Meat\n03. Ophelia\n"
                                       "About band\n"
                                       "Lauded by American critics for a brace of songs\n")),
                    policy=policy)
    assert _titles(record) == ["Rebubula", "Meat", "Ophelia"]


def test_credit_roll_is_cut_for_a_band_name_ending_in_punctuation():
    """a \\b cannot fire at the end of a line after "!", so the whole bio came in as songs."""
    record = _parse(_item(description=("Set 1:\n01. Rebubula\n02. Meat\n03. Ophelia\n"
                                       "About !!!\n"
                                       "Brace Of Songs\n")),
                    policy=ArchivePolicy(band_name="!!!"))
    assert _titles(record) == ["Rebubula", "Meat", "Ophelia"]


def test_a_lineage_line_naming_ffp_does_not_cut_the_show():
    """CHECKSUM_TAIL is line-anchored for this reason: a loose \\bffp\\b once ate 23 setlists."""
    record = _parse(_item(description=("Source: XAct(FLAC 8,ffp,tagging)\n"
                                       "Set 1:\n01. Rebubula\n02. Meat\n03. Ophelia\n")))
    assert _titles(record) == ["Rebubula", "Meat", "Ophelia"]


def test_total_time_credit_block_is_cut_without_a_band_name():
    record = _parse(_item(description=("Set 1:\n01. Rebubula\n02. Meat\n03. Ophelia\n"
                                       "Total Time: [02:40:38]\n"
                                       "Steve Young: FOH\n"
                                       "Poster Artist\n")))
    assert _titles(record) == ["Rebubula", "Meat", "Ophelia"]


# --- venues are not songs -------------------------------------------------------------------

def test_venue_named_by_the_item_is_never_a_song():
    """the item tells us what the room is called, so we do not have to out-guess the header.

    The venue is written INSIDE the numbered list here, which is the case the header regex
    cannot reach and the reason _place_terms exists at all.
    """
    record = _parse(_item(venue="Northlands", coverage="Swanzey NH",
                          description=("Set 1:\n01. Rebubula\n02. Swanzey NH\n"
                                       "03. Northlands\n04. Meat\n05. Ophelia\n")))
    assert _titles(record) == ["Rebubula", "Meat", "Ophelia"]


def test_words_that_belong_to_buildings_are_not_songs():
    record = _parse(_item(description=("Set 1:\n01. Rebubula\n02. Peach Music Festival\n"
                                       "03. Meat\n04. Ophelia\n")))
    assert _titles(record) == ["Rebubula", "Meat", "Ophelia"]


def test_a_known_song_survives_a_venue_shaped_name():
    """the vocabulary guard: anything the band actually plays is never dropped."""
    record = parse_archive_item(
        _item(description="Set 1:\n01. Rebubula\n02. Main Stage\n03. Meat\n"),
        normalizer=_StubNormalizer(extra_vocab=("Main Stage",)))
    assert "Main Stage" in _titles(record)


# --- the policy: what the pack supplies -----------------------------------------------------

def test_junk_patterns_from_the_policy_filter_cover_artists():
    desc = "Set 1:\n01. Rebubula\n02. Umphrey's McGee\n03. Meat\n04. Ophelia\n"
    kept = _parse(_item(description=desc))
    assert "Umphrey's McGee" in _titles(kept)              # nothing generic catches it
    filtered = _parse(_item(description=desc), policy=ArchivePolicy(junk_patterns=("umphrey",)))
    assert _titles(filtered) == ["Rebubula", "Meat", "Ophelia"]


def test_a_pack_fragment_that_starts_with_punctuation_still_matches():
    """a \\b would demand a word character inside the match, so this would match nothing."""
    record = _parse(_item(description=("Set 1:\n01. Rebubula\n02. Some Tune (cover)\n"
                                       "03. Meat\n04. Ophelia\n")),
                    policy=ArchivePolicy(junk_patterns=(r"\(cover\)",)))
    assert _titles(record) == ["Rebubula", "Meat", "Ophelia"]


def test_a_support_act_annotation_is_not_a_song():
    """locks `opened$`, an anchored fragment sitting inside a built alternation."""
    record = _parse(_item(description=("Set 1:\n01. Rebubula\n02. Marco Benevento opened\n"
                                       "03. Meat\n04. Ophelia\n")))
    assert _titles(record) == ["Rebubula", "Meat", "Ophelia"]


def test_drop_dates_removes_the_show_entirely():
    policy = ArchivePolicy(drop_dates=frozenset({"2026-01-31"}))
    assert _parse(policy=policy) is None


def test_date_override_moves_a_misfiled_show():
    policy = ArchivePolicy(date_overrides={"band2026-01-31.akg": "2025-01-31"})
    record = _parse(policy=policy)
    assert record["date"] == "2025-01-31"
    assert record["year"] == "2025"


def test_item_with_an_unreadable_date_is_skipped():
    assert _parse(_item(date="sometime in 2026")) is None


def test_band_filter_rejects_a_side_project():
    policy = ArchivePolicy(band_filter=title_band_filter("band."))
    assert _parse(_item(title="bob. Live at Ophelia's on 2026-01-31"), policy=policy) is None
    assert _parse(policy=policy) is not None


def test_band_filter_reads_a_multi_word_band_name():
    """the capture used to be one alphabetic token, which is the shape of "moe." and of almost
    nothing else -- so for every other band the filter quietly became "accept everything"."""
    policy = ArchivePolicy(band_filter=title_band_filter("Goose"))
    leak = _item(title="Umphrey's McGee Live at The Fillmore on 2026-01-31")
    assert _parse(leak, policy=policy) is None
    assert _parse(_item(title="Goose Live at The Fillmore"), policy=policy) is not None
    assert _parse(_item(title="Umphrey's McGee Live at The Fillmore"),
                  policy=ArchivePolicy(band_filter=title_band_filter("Umphrey's McGee")))


def test_band_filter_ignores_a_date_written_before_the_marker():
    policy = ArchivePolicy(band_filter=title_band_filter("band."))
    assert _parse(_item(title="band. 2026-01-31 Live at The Fillmore"), policy=policy) is not None


def test_band_filter_keeps_a_title_it_cannot_read():
    """being unable to read a title is not evidence that a show is fake."""
    policy = ArchivePolicy(band_filter=title_band_filter("band."))
    assert _parse(_item(title="2026-01-31 soundboard"), policy=policy) is not None


# --- the tracklist fallback -----------------------------------------------------------------

def test_tracks_are_used_when_the_description_is_prose():
    record = _parse(_item(description="A lovely evening was had by all.",
                          tracks=[{"title": "01 Rebubula"}, {"title": "02 Meat"},
                                  {"title": "03 Plane Crash"}, {"title": "04 Ophelia"}]))
    assert record["source"] == "tracks"
    assert _titles(record) == ["Rebubula", "Meat", "Plane Crash", "Ophelia"]


def test_tracks_dedupe_repeated_formats_and_discs():
    record = _parse(_item(description="",
                          tracks=[{"title": "d1t01 Rebubula"}, {"title": "d1t02 Meat"},
                                  {"title": "d2t01 Rebubula"}, {"title": "d2t02 Ophelia"},
                                  {"title": ""}]))
    assert _titles(record) == ["Rebubula", "Meat", "Ophelia"]


def test_a_good_description_beats_the_tracklist():
    record = _parse(_item(tracks=[{"title": "01 Timmy Tucker"}]))
    assert record["source"] == "description"


# --- unknown titles --------------------------------------------------------------------------

def test_short_unknown_title_is_kept_as_a_new_song():
    record = _parse(_item(description="Set 1:\n01. Rebubula\n02. Brand New Tune\n03. Meat\n"))
    assert "Brand New Tune" in _titles(record)


def test_long_or_numeric_unknown_titles_are_dropped():
    record = _parse(_item(description=("Set 1:\n01. Rebubula\n"
                                       "02. Six Whole Words Of Unlikely Prose Here\n"
                                       "03. Track 7 unknown\n04. Meat\n")))
    assert _titles(record) == ["Rebubula", "Meat"]


# --- many items -------------------------------------------------------------------------------

def test_every_tape_of_a_date_comes_back():
    """Four tapes of one night are four records. Picking between them is the merge's job.

    This layer used to keep only the richest, which was a second answer to a question pick_show
    already owns -- and the tapes it discarded never reached override_disagreements, which is
    the one thing that can tell you an override has gone stale.
    """
    thin = _item(identifier="band2026-01-31.aud",
                 description="Set 1:\n01. Rebubula\n02. Meat\n03. Ophelia\n")
    records = parse_archive_items([thin, _item()], normalizer=_StubNormalizer()).shows
    assert [record["identifier"] for record in records] == ["band2026-01-31.akg",
                                                            "band2026-01-31.aud"]
    assert [record["n_songs"] for record in records] == [5, 3]


def test_order_does_not_depend_on_the_order_the_source_returned_items():
    first = _item(identifier="band2026-01-31.aaa")
    second = _item(identifier="band2026-01-31.zzz")
    forward = parse_archive_items([first, second], normalizer=_StubNormalizer())
    backward = parse_archive_items([second, first], normalizer=_StubNormalizer())
    assert forward.shows == backward.shows
    assert forward.skipped == backward.skipped


def test_refusals_come_back_in_a_reproducible_order():
    band = title_band_filter("band.")
    items = [{"identifier": ident, "date": "2026-01-31",
              "title": "Other Band Live at The Fillmore on 2026-01-31"}
             for ident in ("zzz", "aaa", "mmm")]
    result = parse_archive_items(items, normalizer=_StubNormalizer(),
                                 policy=ArchivePolicy(band_filter=band))
    assert [skip.identifier for skip in result.skipped] == ["aaa", "mmm", "zzz"]


def test_records_come_back_sorted_by_date():
    """identifiers deliberately ascend while the dates do not, so insertion order is wrong."""
    items = [_item(identifier="a", date="2026-03-01"),
             _item(identifier="b", date="2026-01-31"),
             _item(identifier="c", date="2026-02-14")]
    records = parse_archive_items(items, normalizer=_StubNormalizer()).shows
    assert [record["date"] for record in records] == ["2026-01-31", "2026-02-14", "2026-03-01"]


# --- clean_html --------------------------------------------------------------------------------

def test_clean_html_unescapes_twice_and_keeps_line_breaks():
    assert clean_html("Hi &amp;amp; Lo<br/>Meat") == "Hi & Lo\nMeat"


def test_clean_html_joins_a_list_and_survives_none():
    assert clean_html(["one", "two"]) == "one two"
    assert clean_html(None) == ""


def test_gear_patterns_from_the_policy_filter_a_local_dialect():
    """setlistkit ships the gear words every taper writes; the scene's own shorthand is the
    pack's. `kcy`, `nbob`, `pfa`, `ela` and `cf` came over with the port carrying no recorded
    explanation, and a three-letter token nobody can read is the shape that deleted ATL."""
    desc = "Set 1:\n01. Rebubula\n02. AKG CK61 > KCY\n03. Meat\n04. Ophelia\n"
    kept = _parse(_item(description=desc))
    assert "KCY" in _titles(kept)                          # nothing generic catches it now
    filtered = _parse(_item(description=desc), policy=ArchivePolicy(gear_patterns=("kcy",)))
    assert _titles(filtered) == ["Rebubula", "Meat", "Ophelia"]


def test_a_song_survives_a_gear_word_supplied_by_the_pack():
    """the pack's own gear words are guarded exactly like the built-in ones."""
    record = parse_archive_item(
        _item(description="Set 1:\n01. Meat\n02. Reaper\n03. Ophelia\n"),
        normalizer=_StubNormalizer(extra_vocab=("Reaper",)),
        policy=ArchivePolicy(gear_patterns=("reaper",)))
    assert _titles(record) == ["Meat", "Reaper", "Ophelia"]


def test_a_greeting_is_tagged_rather_than_dropped():
    """"Greeting By Al" is Al saying hello: it happened on the stage and it is not music, so
    it is a TAG. It used to sit in the generic junk filter, which drops -- and a DROP rule is
    only allowed to answer "is this an entry", never "is this music"."""
    class _Greets(_StubNormalizer):
        def non_song_patterns(self):
            return [*super().non_song_patterns(), re.compile("greeting")]

    record = parse_archive_item(
        _item(description="Set 1:\n01. Greeting By Al\n02. Meat\n03. Ophelia\n"),
        normalizer=_Greets())
    assert _titles(record) == ["Greeting By Al", "Meat", "Ophelia"]
    assert record["sets"][0][0]["non_song"] is True
    assert record["n_songs"] == 2          # tagged, so it is recorded but never counted


def test_no_drop_rule_can_delete_a_song_the_pack_claims():
    """The invariant every DROP rule in this module defers to, asserted rule by rule.

    Each of these once deleted a real title. The shape gate and the venue check already asked
    the pack first; the annotation filter, the credit-line filter and the crew-role filter did
    not, and ``protected.json`` could not save a title from them because it is consulted before
    they run. A pack had no way to defend "Notes", "Lighting" or "Total Time".

    The guard is narrow on purpose: it fires only when the whole token normalizes onto a known
    song, which is why the same filters still drop the credits and cover artists below.
    """
    desc = "Set 1:\n01. {}\n02. Meat\n"

    for title, filter_named in [("Plane Crash", "the shape gate, via a gear word"),
                                ("Notes", "the annotation filter"),
                                ("Lighting", "the crew-role filter")]:
        record = parse_archive_item(
            _item(description=desc.format(title)),
            normalizer=_StubNormalizer(extra_vocab=(title,)),
            policy=ArchivePolicy(gear_patterns=("crash",), junk_patterns=("crash",)))
        assert _titles(record) == [title, "Meat"], f"{title!r} deleted by {filter_named}"

    # ...and protected.json reaches every one of them too, not just the two it used to
    record = parse_archive_item(_item(description=desc.format("Notes")),
                                normalizer=_StubNormalizer(protected=("Notes",)))
    assert _titles(record) == ["Notes", "Meat"]


def test_the_claim_guard_is_narrow_enough_to_leave_the_filters_working():
    """the other half: a token that is not a title is still dropped by all of them."""
    desc = "Set 1:\n01. {}\n02. Meat\n03. Ophelia\n"
    for token, policy in [("Notes", None),
                          ("Al Schnier - guitar, vocals", None),
                          ("Umphrey's McGee", ArchivePolicy(junk_patterns=("umphrey",))),
                          ("KCY", ArchivePolicy(gear_patterns=("kcy",)))]:
        record = _parse(_item(description=desc.format(token)), policy=policy)
        assert _titles(record) == ["Meat", "Ophelia"], f"{token!r} survived"


def test_an_alias_key_is_as_protected_as_a_canonical_title():
    """an alias is the pack saying "this spelling IS that song", and every DROP rule runs
    before canonicalize -- so the guard has to resolve aliases itself or the rules never see
    the claim at all."""
    class _Aliased(_StubNormalizer):
        def aliases(self):
            return {"rebubula ii": "Rebubula"}

    record = parse_archive_item(
        _item(description="Set 1:\n01. Rebubula II\n02. Meat\n"),
        normalizer=_Aliased(),
        policy=ArchivePolicy(junk_patterns=(r"rebubula\s+ii",)))
    assert _titles(record) == ["Rebubula", "Meat"]


def test_an_override_date_that_is_not_a_bare_date_is_refused():
    """`stated` is sliced to ten characters; an override is not, so it was the way a date with
    a newline in it got into the corpus and joined to nothing for the rest of time.

    Anchored with \\Z rather than `$`, which in Python also matches before a trailing newline.
    """
    for bad in ("2025-06-14\n", "2025-06-14T00:00:00", "2025-06-14 (set two)"):
        record = parse_archive_item(
            _item(description="Set 1:\n01. Meat\n02. Ophelia\n"),
            normalizer=_StubNormalizer(),
            policy=ArchivePolicy(date_overrides={"band2026-01-31.akg": bad}))
        assert record is None, f"{bad!r} was accepted as a date"

    good = _parse(_item(), policy=ArchivePolicy(
        date_overrides={"band2026-01-31.akg": "2025-06-14"}))
    assert good["date"] == "2025-06-14" and good["year"] == "2025"

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for cross-source precedence and manual overrides.

The headline is the three-source case in merge.py's docstring: the reason the pick is global
rather than a chain of pairwise merges. Folding the sources together two at a time gets a
different, wrong answer, and the test below is what says so.

The rest lock the rules that were learned by getting them wrong: a stub can never win, a refused
date is refused in every source at once, and an override wins without arguing about song counts.
"""

import re

from pathlib import Path

import pytest

from setlistkit.catalog.jsonpos import JSONSource

from setlistkit.catalog import (COMPLETE_FRAC, MergePolicy, Normalizer, apply_overrides,
                                count_songs, merge_shows, override_disagreements,
                                overrides_from_mapping, pick_show)
from setlistkit.diagnostics import DiagnosticError

# "Intro" is in the vocabulary on purpose: a setlist service lists it as though it were a song,
# which is what separates the vocabulary filter from the non-song filter in _real_songs.
_VOCAB = ["Meat", "Rebubula", "Ophelia", "Plane Crash", "The Faker", "Recreational Chemistry",
          "Timmy Tucker", "Hi & Lo", "Intro"]


class _StubNormalizer(Normalizer):
    def __init__(self):
        super().__init__(_VOCAB)

    def aliases(self):
        return {"rec chem": "Recreational Chemistry"}

    def non_song_patterns(self):
        return [re.compile(r"^(?:intro|outro|tuning|banter)+$")]


def _entries(names):
    return [{"song": name, "segue": False, "non_song": False} for name in names]


def _record(date, source, songs, *, identifier=None, encore=()):
    sets = [_entries(songs)] if songs else []
    enc = _entries(encore)
    return {"date": date, "year": date[:4], "sets": sets, "encore": enc,
            "n_songs": count_songs(sets, enc), "source": source,
            "identifier": identifier or f"{source}-{date}"}


def _named(count, prefix="Song"):
    return [f"{prefix} {index}" for index in range(count)]


# --- the pick ---------------------------------------------------------------------------------

def test_the_global_pick_beats_folding_sources_together_two_at_a_time():
    """the reason this module exists.

    A 12-song tape, a 15-song listing and a 20-song graphic, ranked 3 / 2 / 1. The tape is under
    75% of 20 so it is partial and drops out; among what is left the best rank wins, which is the
    15-song listing. Merging pairwise picks the 20-song graphic instead, because the tape loses to
    the graphic first and the graphic then outnumbers the listing.
    """
    tape = _record("2026-01-31", "description", _named(12))
    listing = _record("2026-01-31", "setlistfm", _named(15))
    graphic = _record("2026-01-31", "instagram", _named(20))
    assert pick_show([tape, listing, graphic])["source"] == "setlistfm"


def test_the_best_source_wins_when_it_is_complete_enough():
    tape = _record("2026-01-31", "description", _named(16))
    graphic = _record("2026-01-31", "instagram", _named(20))
    assert pick_show([tape, graphic])["source"] == "description"


def test_a_thin_parse_from_a_good_source_yields_to_a_fuller_one():
    tape = _record("2026-01-31", "description", _named(4))
    graphic = _record("2026-01-31", "instagram", _named(20))
    assert pick_show([tape, graphic])["source"] == "instagram"


def test_a_lone_candidate_clears_its_own_bar():
    only = _record("2026-01-31", "instagram", _named(3))
    assert pick_show([only])["source"] == "instagram"


def test_ties_break_on_song_count_then_identifier():
    """asserted from both argument orders, or the identifier rule could be deleted and the
    first-listed candidate would keep winning by position."""
    first = _record("2026-01-31", "setlistfm", _named(10), identifier="zzz")
    second = _record("2026-01-31", "setlistfm", _named(10), identifier="aaa")
    assert pick_show([first, second])["identifier"] == "zzz"
    assert pick_show([second, first])["identifier"] == "zzz"
    richer = _record("2026-01-31", "setlistfm", _named(11), identifier="aaa")
    assert pick_show([first, richer])["identifier"] == "aaa"


def test_picking_from_nothing_says_so():
    with pytest.raises(ValueError, match="at least one candidate"):
        pick_show([])


def test_a_completeness_bar_outside_zero_to_one_is_refused():
    """above 1.0 nothing clears the bar and the rule inverts into "thinnest top rank wins"."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        MergePolicy(complete_frac=1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        MergePolicy(complete_frac=-0.1)


def test_an_unranked_source_never_outranks_a_configured_one():
    known = _record("2026-01-31", "setlistfm", _named(10))
    stranger = _record("2026-01-31", "carrier-pigeon", _named(10))
    assert pick_show([known, stranger])["source"] == "setlistfm"


def test_an_unranked_source_still_takes_a_date_nobody_else_has():
    stranger = _record("2026-01-31", "carrier-pigeon", _named(10))
    assert merge_shows([stranger]).shows[0]["source"] == "carrier-pigeon"


def test_the_completeness_bar_is_tunable():
    tape = _record("2026-01-31", "description", _named(12))
    graphic = _record("2026-01-31", "instagram", _named(20))
    assert pick_show([tape, graphic])["source"] == "instagram"
    generous = MergePolicy(complete_frac=0.5)
    assert pick_show([tape, graphic], generous)["source"] == "description"


# --- what never reaches the pick ---------------------------------------------------------------

def test_an_empty_stub_can_never_win():
    """what makes "the best source is final" safe: a stub is not evidence of an empty show."""
    stub = _record("2026-01-31", "description", [])
    graphic = _record("2026-01-31", "instagram", _named(9))
    result = merge_shows([stub, graphic])
    assert len(result.shows) == 1
    assert result.shows[0]["source"] == "instagram"


def test_a_stub_does_not_take_a_date_of_its_own():
    result = merge_shows([_record("2026-01-31", "description", [])])
    assert result.shows == []


def test_a_refused_date_is_refused_in_every_source():
    """refusing it in one parser was not enough. The other sources carry the same night, so the
    merge quietly picked one of those copies up instead and the show came straight back."""
    policy = MergePolicy(drop_dates=frozenset({"2025-10-31"}))
    records = [_record("2025-10-31", "description", _named(9)),
               _record("2025-10-31", "setlistfm", _named(11)),
               _record("2026-01-31", "description", _named(9))]
    result = merge_shows(records, policy=policy)
    assert [show["date"] for show in result.shows] == ["2026-01-31"]


def test_a_record_with_no_date_is_skipped():
    assert merge_shows([{"source": "description", "n_songs": 5}]).shows == []


def test_a_stated_song_count_cannot_talk_a_record_past_the_stub_filter():
    """an empty record claiming 99 songs used to clear the stub filter, set the bar every honest
    candidate was measured against, disqualify all of them, and win the date with no setlist."""
    liar = {"date": "2026-01-31", "source": "description", "sets": [], "encore": [],
            "n_songs": 99, "identifier": "liar"}
    honest = _record("2026-01-31", "instagram", _named(9))
    result = merge_shows([liar, honest])
    assert [show["identifier"] for show in result.shows] == [honest["identifier"]]


def test_a_null_setlist_is_not_a_crash():
    """a key present and null is what a truncated intermediate produces, and a default misses it."""
    assert merge_shows([{"date": "2026-01-31", "source": "description",
                         "sets": None, "encore": None}]).shows == []


def test_a_refused_date_stays_refused_even_with_an_override():
    """the empty-override diagnostic tells people to use drop_dates to delete a date, so it has
    to actually delete the date."""
    overrides = overrides_from_mapping(
        {"overrides": {"2025-10-31": {"reason": "confirmed", "sets": [["Meat"]]}}},
        _StubNormalizer())
    result = merge_shows([_record("2025-10-31", "description", _named(9))],
                         overrides=overrides,
                         policy=MergePolicy(drop_dates=frozenset({"2025-10-31"})))
    assert result.shows == []
    assert result.applied == []


# --- the merged corpus ---------------------------------------------------------------------------

def test_shows_come_back_one_per_date_sorted():
    records = [_record("2026-03-01", "description", _named(9)),
               _record("2026-01-31", "description", _named(9)),
               _record("2026-01-31", "setlistfm", _named(9)),
               _record("2026-02-14", "instagram", _named(9))]
    result = merge_shows(records)
    assert [show["date"] for show in result.shows] == ["2026-01-31", "2026-02-14", "2026-03-01"]


def test_candidates_are_kept_for_reporting():
    records = [_record("2026-01-31", "description", _named(9)),
               _record("2026-01-31", "setlistfm", _named(9))]
    result = merge_shows(records)
    assert len(result.candidates["2026-01-31"]) == 2


def test_the_merge_does_not_mutate_the_records_it_was_given():
    """including the nested setlists, which a shallow copy would still share."""
    record = _record("2026-01-31", "description", _named(9))
    result = merge_shows([record])
    result.shows[0]["source"] = "tampered"
    result.shows[0]["sets"][0].append({"song": "Meat", "segue": False, "non_song": False})
    assert record["source"] == "description"
    assert len(record["sets"][0]) == 9


def test_song_count_is_computed_when_a_record_did_not_bring_one():
    bare = {"date": "2026-01-31", "source": "instagram",
            "sets": [_entries(["Meat", "Ophelia"])], "encore": []}
    assert merge_shows([bare]).shows[0]["source"] == "instagram"


def test_a_tagged_non_song_does_not_pad_a_record_past_the_bar():
    """the padding failure that made overrides necessary, now that non-songs are tagged."""
    padded = _record("2026-01-31", "description", _named(4))
    padded["sets"][0].extend({"song": "Tuning", "segue": False, "non_song": True}
                             for _ in range(10))
    honest = _record("2026-01-31", "setlistfm", _named(8))
    assert pick_show([padded, honest])["source"] == "setlistfm"


# --- overrides ------------------------------------------------------------------------------------

def _override_doc(**entry):
    base = {"reason": "listened to the tape twice", "sets": [["Meat", "Ophelia >", "Rebubula"]]}
    base.update(entry)
    return {"overrides": {"2026-01-31": base}}


def test_an_override_wins_without_arguing_about_song_counts():
    """an honest short setlist beats a junk-padded long one, which ranking alone cannot do."""
    overrides = overrides_from_mapping(_override_doc(), _StubNormalizer())
    padded = _record("2026-01-31", "description", _named(14))
    result = merge_shows([padded], overrides=overrides)
    assert len(result.shows) == 1
    assert result.shows[0]["source"] == "override"
    assert result.shows[0]["n_songs"] == 3
    assert result.applied == ["2026-01-31"]


def test_an_override_for_an_untaped_date_grows_the_corpus():
    """folded in before anything counts the result, so it does not look like a shrink."""
    overrides = overrides_from_mapping(_override_doc(), _StubNormalizer())
    result = merge_shows([_record("2026-03-01", "description", _named(9))], overrides=overrides)
    assert [show["date"] for show in result.shows] == ["2026-01-31", "2026-03-01"]


def test_apply_overrides_never_reads_a_song_count():
    """A two-song override beats a fourteen-song tape, because it was written by someone who
    listened rather than by something that argued its way past a threshold."""
    shows = [_record("2026-01-31", "description", _named(14))]
    merged, applied = apply_overrides(shows, {"2026-01-31": {
        "date": "2026-01-31", "source": "override", "identifier": "override-2026-01-31",
        "sets": [_named(2)], "encore": []}})
    assert merged[0]["source"] == "override"
    assert applied == ["2026-01-31"]


def test_a_date_correction_cannot_be_applied_as_a_setlist_override():
    """The two kinds of override live in one policy object and are entirely different things.

    corpus.json's date_overrides move a show to another night; overrides.json says what was
    played. Fed in here as the second, a date correction becomes a "show" made of a date and a
    paragraph of prose -- and since an override always wins, it replaces the genuine record for
    that night. This shipped once, from a single rebound local name.
    """
    shows = [_record("2026-01-31", "description", _named(14))]
    date_correction = {"date": "2026-01-31", "why": "the description says January 2026"}
    with pytest.raises(ValueError, match="is not a show record"):
        apply_overrides(shows, {"2026-01-31": date_correction})


def test_an_override_whose_date_disagrees_with_its_key_is_refused():
    """The key is what gets REPLACED; record["date"] is what gets STORED.

    Let them disagree and one override deletes a real night and invents another, while `applied`
    reports a date that ends up in neither. Same failure as the one above, one field further in.
    """
    shows = [_record("2026-01-31", "description", _named(14))]
    misfiled = {"2026-01-31": {"date": "2026-02-02", "sets": [_named(3)], "encore": [],
                               "source": "override", "identifier": "override-x"}}
    with pytest.raises(ValueError, match="would delete '2026-01-31' and create '2026-02-02'"):
        apply_overrides(shows, misfiled)


def test_override_songs_run_through_the_normalizer():
    """aliases resolve and a trailing '>' sets the segue, exactly as for every other source."""
    doc = _override_doc(sets=[["Rec Chem", "Ophelia >", "Meat"]])
    record = overrides_from_mapping(doc, _StubNormalizer())["2026-01-31"]
    songs = record["sets"][0]
    assert songs[0]["song"] == "Recreational Chemistry"
    assert songs[1]["segue"] is True
    assert songs[2]["segue"] is False


def test_override_non_songs_are_tagged_like_everywhere_else():
    doc = _override_doc(sets=[["Meat", "Tuning", "Ophelia"]])
    record = overrides_from_mapping(doc, _StubNormalizer())["2026-01-31"]
    assert record["sets"][0][1]["non_song"] is True
    assert record["n_songs"] == 2


def test_an_override_carries_its_reason_and_its_own_identifier():
    record = overrides_from_mapping(_override_doc(), _StubNormalizer())["2026-01-31"]
    assert record["reason"] == "listened to the tape twice"
    assert record["identifier"] == "override-2026-01-31"
    assert record["year"] == "2026"


def test_an_encore_may_be_given():
    doc = _override_doc(encore=["The Faker"])
    record = overrides_from_mapping(doc, _StubNormalizer())["2026-01-31"]
    assert [entry["song"] for entry in record["encore"]] == ["The Faker"]
    assert record["n_songs"] == 4


# --- overrides that must be refused ----------------------------------------------------------------

@pytest.mark.parametrize("doc, expected", [
    ({}, "top-level"),
    ({"overrides": []}, "top-level"),
    ({"overrides": {"2026-01-31": "Meat"}}, "must be an object"),
])
def test_a_malformed_document_is_refused(doc, expected):
    with pytest.raises(DiagnosticError) as caught:
        overrides_from_mapping(doc, _StubNormalizer())
    assert expected in caught.value.diagnostic.summary


def test_an_override_without_a_reason_is_refused():
    """nothing goes in on a hunch."""
    with pytest.raises(DiagnosticError) as caught:
        overrides_from_mapping(_override_doc(reason="  "), _StubNormalizer())
    assert "reason" in caught.value.diagnostic.summary


def test_an_override_with_no_songs_is_refused():
    """an empty one would silently hand the date back to the sources it was written to correct."""
    with pytest.raises(DiagnosticError) as caught:
        overrides_from_mapping(_override_doc(sets=[]), _StubNormalizer())
    assert "no songs" in caught.value.diagnostic.summary


def test_an_override_of_nothing_but_non_songs_is_refused():
    with pytest.raises(DiagnosticError) as caught:
        overrides_from_mapping(_override_doc(sets=[["Tuning", "Banter"]]), _StubNormalizer())
    assert "no songs" in caught.value.diagnostic.summary


@pytest.mark.parametrize("sets, expected", [
    ("Meat", "must be a list of lists"),
    ([["Meat", 7]], "must be strings"),
    ([["   "]], "empty song name"),
    (["Meat"], "must be a list"),          # a bare string set would iterate into four letters
])
def test_a_malformed_setlist_is_refused(sets, expected):
    with pytest.raises(DiagnosticError) as caught:
        overrides_from_mapping(_override_doc(sets=sets), _StubNormalizer())
    assert expected in caught.value.diagnostic.summary


def test_a_malformed_encore_is_refused():
    with pytest.raises(DiagnosticError) as caught:
        overrides_from_mapping(_override_doc(encore="The Faker"), _StubNormalizer())
    assert "must be a list" in caught.value.diagnostic.summary


@pytest.mark.parametrize("reason", [None, False, 0, [], {}])
def test_a_reason_that_is_not_a_string_is_refused(reason):
    """str(None) is "None", so stringifying before the test let every falsy non-string pass."""
    with pytest.raises(DiagnosticError) as caught:
        overrides_from_mapping(_override_doc(reason=reason), _StubNormalizer())
    assert "reason" in caught.value.diagnostic.summary


@pytest.mark.parametrize("date", ["", "   ", 20260131])
def test_an_unusable_date_key_is_refused(date):
    doc = {"overrides": {date: {"reason": "confirmed", "sets": [["Meat"]]}}}
    with pytest.raises(DiagnosticError) as caught:
        overrides_from_mapping(doc, _StubNormalizer())
    assert "not a usable date" in caught.value.diagnostic.summary


def test_the_diagnostic_names_the_file_it_came_from():
    with pytest.raises(DiagnosticError) as caught:
        overrides_from_mapping(_override_doc(reason=""), _StubNormalizer(),
                               src=JSONSource(file=Path("overrides.json")))
    assert caught.value.diagnostic.path == "overrides.json"
    assert caught.value.diagnostic.detail


# --- disagreements ------------------------------------------------------------------------------

def test_a_source_holding_a_real_song_the_override_lacks_is_reported():
    """an override always wins, so nothing else will ever tell us it went stale."""
    overrides = overrides_from_mapping(_override_doc(), _StubNormalizer())
    tape = _record("2026-01-31", "description", ["Meat", "Ophelia", "Rebubula", "Timmy Tucker"])
    result = merge_shows([tape], overrides=overrides)
    reports = override_disagreements(result.candidates, overrides, _StubNormalizer())
    assert len(reports) == 1
    assert reports[0]["missing"] == ["Timmy Tucker"]
    assert reports[0]["source"] == "description"
    assert reports[0]["n_override"] == 3


def test_junk_a_source_carries_is_not_reported_as_a_disagreement():
    """the record an override replaces usually has junk in it -- that is generally why. If that
    counted, the date would print a review line on every run forever."""
    overrides = overrides_from_mapping(_override_doc(), _StubNormalizer())
    tape = _record("2026-01-31", "description",
                   ["Meat", "Ophelia", "Rebubula", "Technical Difficulties", "SD Card"])
    result = merge_shows([tape], overrides=overrides)
    assert override_disagreements(result.candidates, overrides, _StubNormalizer()) == []


def test_a_song_only_in_an_encore_is_still_a_disagreement():
    """the encore is part of the record, and _entries has to reach it."""
    overrides = overrides_from_mapping(_override_doc(), _StubNormalizer())
    tape = _record("2026-01-31", "description", ["Meat", "Ophelia", "Rebubula"],
                   encore=["Timmy Tucker"])
    result = merge_shows([tape], overrides=overrides)
    reports = override_disagreements(result.candidates, overrides, _StubNormalizer())
    assert reports[0]["missing"] == ["Timmy Tucker"]


def test_a_non_song_that_IS_in_the_vocabulary_is_not_a_disagreement():
    """the case the vocabulary filter cannot catch: a service lists "Intro" as a song, and it is
    in the dictionary, so only is_non_song can tell it is not music."""
    overrides = overrides_from_mapping(_override_doc(), _StubNormalizer())
    tape = _record("2026-01-31", "description", ["Meat", "Ophelia", "Rebubula", "Intro"])
    result = merge_shows([tape], overrides=overrides)
    assert override_disagreements(result.candidates, overrides, _StubNormalizer()) == []


def test_an_agreeing_source_produces_no_report():
    overrides = overrides_from_mapping(_override_doc(), _StubNormalizer())
    tape = _record("2026-01-31", "description", ["Meat", "Ophelia"])
    result = merge_shows([tape], overrides=overrides)
    assert override_disagreements(result.candidates, overrides, _StubNormalizer()) == []


def test_the_default_completeness_bar_is_three_quarters():
    assert COMPLETE_FRAC == 0.75

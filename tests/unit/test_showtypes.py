# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for classifying a date as electric, acoustic, mixed or alter-ego.

The detection is deliberately narrow and these lock it there. An acoustic night is a different
band playing shorter songs, and pooling its lengths with the electric ones is what put a
5-minute Lazarus in the length table. Matching the bare word "acoustic" is what swept seven
electric nights in with it.
"""

import re

from setlistkit.catalog.showtypes import (ACOUSTIC, ALTEREGO, ELECTRIC, MIXED,
                                          show_types)

# The acoustic billing is the PACK's now, not this module's. moe.'s is used throughout
# because these tests were written against a real band's brand name and what they check is
# the mechanism around it -- that it beats a gear list, loses to an alter-ego billing, and
# survives HTML. `_types` supplies it so each test reads as it did before.
MOESTLY = (re.compile(r"moe\.?stly", re.I),)


def _types(items, **kwargs):
    kwargs.setdefault("acoustic", MOESTLY)
    return show_types(items, **kwargs)


def _item(identifier, date, description="", title="", venue=""):
    return {"identifier": identifier, "meta_date": date, "description": description,
            "title": title, "venue": venue, "list_title": ""}


def _by_date(rows):
    return {r.date: r for r in rows}


def test_a_date_with_no_evidence_is_electric():
    rows = _by_date(_types([_item("t1", "2025-01-01", description="Set One\n1. Aurora")]))
    assert rows["2025-01-01"].kind == ELECTRIC
    assert rows["2025-01-01"].evidence is None


def test_the_brand_name_marks_an_acoustic_show():
    rows = _by_date(_types([_item("t1", "2025-01-01", title="moe.stly Acoustic")]))
    assert rows["2025-01-01"].kind == ACOUSTIC


def test_the_brand_name_is_matched_without_the_dot_too():
    rows = _by_date(_types([_item("t1", "2025-01-01", title="Moestly Acoustic Evening")]))
    assert rows["2025-01-01"].kind == ACOUSTIC


def test_gear_lists_do_not_make_a_show_acoustic():
    """The bug this module exists to avoid.

    A taper listing "acoustic guitar" among the instruments is describing a microphone setup,
    not the kind of night it was. Matching the bare word took New Year's Eve and six other
    full electric shows with it.
    """
    gear = "Jim Loughlin - percussion, MalletKat, flute, acoustic guitar"
    rows = _by_date(_types([_item("t1", "2025-12-31", description=gear)]))
    assert rows["2025-12-31"].kind == ELECTRIC


def test_an_acoustic_set_inside_an_electric_show_is_mixed():
    notes = "Set1: Al, Chuck & Rob - acoustic"
    rows = _by_date(_types([_item("t1", "2023-12-31", description=notes)]))
    assert rows["2023-12-31"].kind == MIXED


def test_the_band_playing_as_somebody_else_is_an_alter_ego_night():
    rows = _by_date(_types([_item("t1", "2025-11-22",
                                  description="performing as Monkeys On Ecstasy")]))
    assert rows["2025-11-22"].kind == ALTEREGO


def test_the_strongest_evidence_on_a_date_wins():
    """A date can be taped four times over and the tapes need not agree."""
    items = [_item("t1", "2025-01-01", description="Set One\n1. Aurora"),
             _item("t2", "2025-01-01", title="moe.stly Acoustic")]
    assert _by_date(_types(items))["2025-01-01"].kind == ACOUSTIC


def test_alter_ego_outranks_acoustic():
    items = [_item("t1", "2025-01-01", title="moe.stly Acoustic"),
             _item("t2", "2025-01-01", description="performing as Monkeys On Ecstasy")]
    assert _by_date(_types(items))["2025-01-01"].kind == ALTEREGO


def test_the_brand_inside_markup_is_not_evidence_about_the_night():
    """Descriptions carry HTML, and a link's href is not a claim about the show.

    A taper linking to the acoustic tour's page from an ordinary electric night writes the
    brand into an attribute, never into the prose. Reading the raw blob would take the night
    for acoustic on the strength of a URL. Stripping the tags first is what stops that, so
    this fails if the stripping goes away.
    """
    link = '<a href="https://archive.org/details/moe.stlyAcoustic2023">Full Electric Show</a>'
    rows = _by_date(_types([_item("t1", "2025-01-01", description=link)]))
    assert rows["2025-01-01"].kind == ELECTRIC


def test_markup_around_the_evidence_does_not_hide_it():
    """The other direction: tags wrapping the brand must not stop it being read."""
    rows = _by_date(_types([_item("t1", "2025-01-01",
                                  description="<b>moe.stly</b> Acoustic")]))
    assert rows["2025-01-01"].kind == ACOUSTIC


def test_an_undated_item_contributes_nothing():
    assert _types([_item("t1", "", title="moe.stly Acoustic")]) == []


def test_rows_come_back_sorted_by_date():
    items = [_item("t2", "2025-06-01"), _item("t1", "2024-01-01")]
    assert [r.date for r in _types(items)] == ["2024-01-01", "2025-06-01"]


def test_the_evidence_and_identifier_are_recorded():
    """The evidence names the RULE that fired, not a fixed sentence.

    With the billing in the pack there can be several patterns, and "it looked acoustic" tells a
    reader who doubts a night nothing they can act on. The pattern plus the tape is a citation.
    """
    rows = _by_date(_types([_item("tape-a", "2025-01-01", title="moe.stly Acoustic")]))
    assert rows["2025-01-01"].identifier == "tape-a"
    assert rows["2025-01-01"].evidence == r"tape metadata matches /moe\.?stly/"


def test_a_pack_that_declares_no_acoustic_billing_never_tags_one():
    """The fix for a brand name that was hardcoded into a band-agnostic layer.

    "moe.stly" is a fact about moe., exactly like a drop date. Left in code, this module could
    never tag an acoustic night for anybody else's pack -- it would report every one of them
    electric while running clean. Refusing to guess is the same answer the band filter gives
    when it is handed no band name.
    """
    items = [_item("t1", "2025-01-01", title="moe.stly Acoustic")]
    assert _by_date(show_types(items))["2025-01-01"].kind == ELECTRIC
    assert _by_date(show_types(items, acoustic=MOESTLY))["2025-01-01"].kind == ACOUSTIC


def test_any_of_several_billings_can_mark_an_acoustic_night():
    """A pack may spell it more than one way; the first that fires supplies the evidence."""
    acoustic = (re.compile(r"moe\.?stly", re.I), re.compile(r"al\s+and\s+rob\s+duo", re.I))
    rows = _by_date(show_types([_item("t1", "2025-01-01", title="Al and Rob Duo")],
                               acoustic=acoustic))
    assert rows["2025-01-01"].kind == ACOUSTIC
    assert "al" in rows["2025-01-01"].evidence


def test_a_corrected_date_tags_the_night_the_show_was_actually_played():
    """An uploader can type any date they like, so a pack carries corrections.

    A tag computed off the stated date lands on the wrong night for exactly the tapes whose
    metadata was already known to be wrong -- and it lands there as a real row, which reads as
    a night that was tagged rather than a night that was missed.
    """
    items = [{"identifier": "a", "meta_date": "2024-06-14",
              "description": "a moe.stly duo set"}]
    tagged, = _types(items, dates={"a": "2025-06-14"})
    assert tagged.date == "2025-06-14" and tagged.kind == ACOUSTIC


def test_an_item_with_no_correction_still_uses_its_own_date():
    """The fallback is what keeps a standalone call over raw items honest."""
    items = [{"identifier": "a", "meta_date": "2024-06-14", "description": ""},
             {"identifier": "b", "meta_date": "2024-06-15", "description": ""}]
    tagged = _types(items, dates={"a": "2025-06-14"})
    assert [t.date for t in tagged] == ["2024-06-15", "2025-06-14"]

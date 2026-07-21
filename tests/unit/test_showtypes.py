# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for classifying a date as electric, acoustic, mixed or alter-ego.

The detection is deliberately narrow and these lock it there. An acoustic night is a different
band playing shorter songs, and pooling its lengths with the electric ones is what put a
5-minute Lazarus in the length table. Matching the bare word "acoustic" is what swept seven
electric nights in with it.
"""

from setlistkit.catalog.showtypes import ACOUSTIC, ALTEREGO, ELECTRIC, MIXED, show_types


def _item(identifier, date, description="", title="", venue=""):
    return {"identifier": identifier, "meta_date": date, "description": description,
            "title": title, "venue": venue, "list_title": ""}


def _by_date(rows):
    return {r.date: r for r in rows}


def test_a_date_with_no_evidence_is_electric():
    rows = _by_date(show_types([_item("t1", "2025-01-01", description="Set One\n1. Aurora")]))
    assert rows["2025-01-01"].kind == ELECTRIC
    assert rows["2025-01-01"].evidence is None


def test_the_brand_name_marks_an_acoustic_show():
    rows = _by_date(show_types([_item("t1", "2025-01-01", title="moe.stly Acoustic")]))
    assert rows["2025-01-01"].kind == ACOUSTIC


def test_the_brand_name_is_matched_without_the_dot_too():
    rows = _by_date(show_types([_item("t1", "2025-01-01", title="Moestly Acoustic Evening")]))
    assert rows["2025-01-01"].kind == ACOUSTIC


def test_gear_lists_do_not_make_a_show_acoustic():
    """The bug this module exists to avoid.

    A taper listing "acoustic guitar" among the instruments is describing a microphone setup,
    not the kind of night it was. Matching the bare word took New Year's Eve and six other
    full electric shows with it.
    """
    gear = "Jim Loughlin - percussion, MalletKat, flute, acoustic guitar"
    rows = _by_date(show_types([_item("t1", "2025-12-31", description=gear)]))
    assert rows["2025-12-31"].kind == ELECTRIC


def test_an_acoustic_set_inside_an_electric_show_is_mixed():
    notes = "Set1: Al, Chuck & Rob - acoustic"
    rows = _by_date(show_types([_item("t1", "2023-12-31", description=notes)]))
    assert rows["2023-12-31"].kind == MIXED


def test_the_band_playing_as_somebody_else_is_an_alter_ego_night():
    rows = _by_date(show_types([_item("t1", "2025-11-22",
                                      description="performing as Monkeys On Ecstasy")]))
    assert rows["2025-11-22"].kind == ALTEREGO


def test_the_strongest_evidence_on_a_date_wins():
    """A date can be taped four times over and the tapes need not agree."""
    items = [_item("t1", "2025-01-01", description="Set One\n1. Aurora"),
             _item("t2", "2025-01-01", title="moe.stly Acoustic")]
    assert _by_date(show_types(items))["2025-01-01"].kind == ACOUSTIC


def test_alter_ego_outranks_acoustic():
    items = [_item("t1", "2025-01-01", title="moe.stly Acoustic"),
             _item("t2", "2025-01-01", description="performing as Monkeys On Ecstasy")]
    assert _by_date(show_types(items))["2025-01-01"].kind == ALTEREGO


def test_html_in_a_description_does_not_hide_the_evidence():
    rows = _by_date(show_types([_item("t1", "2025-01-01",
                                      description="<b>moe.stly</b> Acoustic")]))
    assert rows["2025-01-01"].kind == ACOUSTIC


def test_an_undated_item_contributes_nothing():
    assert show_types([_item("t1", "", title="moe.stly Acoustic")]) == []


def test_rows_come_back_sorted_by_date():
    items = [_item("t2", "2025-06-01"), _item("t1", "2024-01-01")]
    assert [r.date for r in show_types(items)] == ["2024-01-01", "2025-06-01"]


def test_the_evidence_and_identifier_are_recorded():
    rows = _by_date(show_types([_item("tape-a", "2025-01-01", title="moe.stly Acoustic")]))
    assert rows["2025-01-01"].identifier == "tape-a"
    assert "moe.stly" in rows["2025-01-01"].evidence

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `titles_from_cache` -- the in-process stand-in for the dirty-stripper jq filter.

It hands `suggest` the same bag of raw `files[].title` strings the jq did, but sourced from the
cached item records `cached_items` produces. The properties that matter: it reads the `tracks`
projection (not the raw `files`), dedupes across every item, and never emits an empty title.
"""

from setlistkit.catalog.suggest import titles_from_cache


def _item(*titles):
    """A cached item record shaped like the one archive_org builds, carrying just its tracks."""
    return {"tracks": [{"track": str(i + 1), "title": t} for i, t in enumerate(titles)]}


def test_titles_are_deduped_across_items_and_sorted():
    items = [_item("Ophelia", "1348"), _item("1348", "Bridgeless")]
    assert titles_from_cache(items) == ["1348", "Bridgeless", "Ophelia"]


def test_empty_and_whitespace_titles_are_dropped():
    items = [_item("Ophelia", "", "   "), {"tracks": [{"track": "1"}]}]  # missing title key
    assert titles_from_cache(items) == ["Ophelia"]


def test_item_with_no_tracks_contributes_nothing():
    items = [{"tracks": None}, {}, _item("Ophelia")]     # None, absent, real
    assert titles_from_cache(items) == ["Ophelia"]


def test_empty_cache_is_an_empty_list():
    assert titles_from_cache([]) == []

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the raw file cache."""

from datetime import datetime, timedelta, timezone

import pytest

from setlistkit.store.raw_cache import RawCache


def test_put_get_roundtrip(tmp_path):
    cache = RawCache(tmp_path)
    cache.put("archive_org", "moe2026-07-04", b"raw bytes")
    assert cache.get("archive_org", "moe2026-07-04") == b"raw bytes"
    assert cache.get_text("archive_org", "moe2026-07-04") == "raw bytes"


def test_missing_entry_returns_none(tmp_path):
    cache = RawCache(tmp_path)
    assert cache.get("archive_org", "nope") is None
    assert cache.meta("archive_org", "nope") is None
    assert cache.age("archive_org", "nope") is None
    assert cache.has("archive_org", "nope") is False


def test_meta_records_provenance(tmp_path):
    cache = RawCache(tmp_path)
    cache.put("setlistfm", "abc", b"12345",
              url="https://api.setlist.fm/x", content_type="application/json",
              etag="W/\"1\"", last_modified="Mon, 01 Jan 2026 00:00:00 GMT")
    meta = cache.meta("setlistfm", "abc")
    assert meta["key"] == "abc"
    assert meta["url"] == "https://api.setlist.fm/x"
    assert meta["etag"] == "W/\"1\""
    assert meta["bytes"] == 5
    assert meta["fetched_at"]


def test_age_uses_fetched_at(tmp_path):
    cache = RawCache(tmp_path)
    fetched = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    cache.put("archive_org", "k", b"x", fetched_at=fetched)
    now = fetched + timedelta(hours=6)
    assert cache.age("archive_org", "k", now=now) == timedelta(hours=6)


def test_namespaces_do_not_collide(tmp_path):
    cache = RawCache(tmp_path)
    cache.put("archive_org", "same", b"A")
    cache.put("setlistfm", "same", b"B")
    assert cache.get("archive_org", "same") == b"A"
    assert cache.get("setlistfm", "same") == b"B"


def test_delete(tmp_path):
    cache = RawCache(tmp_path)
    cache.put("archive_org", "k", b"x")
    assert cache.delete("archive_org", "k") is True
    assert cache.has("archive_org", "k") is False
    assert cache.meta("archive_org", "k") is None
    # test that deleting an absent entry reports nothing was there
    assert cache.delete("archive_org", "k") is False


def test_long_key_is_hashed_but_roundtrips(tmp_path):
    cache = RawCache(tmp_path)
    key = "https://archive.org/details/" + "x" * 400
    cache.put("archive_org", key, b"payload")
    # test that an overlong key still stores, retrieves, and keeps the original in meta
    assert cache.get("archive_org", key) == b"payload"
    assert cache.meta("archive_org", key)["key"] == key


def test_slash_in_key_does_not_create_subdirs(tmp_path):
    cache = RawCache(tmp_path)
    cache.put("archive_org", "a/b/c", b"x")
    assert cache.get("archive_org", "a/b/c") == b"x"
    # the slash is quoted, so the payload is one flat file under blob/, not nested dirs
    entries = list((tmp_path / "raw" / "archive_org" / "blob").iterdir())
    assert entries == [p for p in entries if p.is_file()]
    assert len(entries) == 1


def test_key_ending_in_meta_json_does_not_corrupt_a_sidecar(tmp_path):
    cache = RawCache(tmp_path)
    cache.put("archive_org", "foo", b"A", url="u")
    # test that a key literally ending in .meta.json can't land on foo's sidecar path
    cache.put("archive_org", "foo.meta.json", b"B")
    assert cache.meta("archive_org", "foo")["url"] == "u"
    assert cache.get("archive_org", "foo") == b"A"
    assert cache.get("archive_org", "foo.meta.json") == b"B"


def test_naive_fetched_at_is_treated_as_utc(tmp_path):
    cache = RawCache(tmp_path)
    naive = datetime(2026, 7, 1, 12, 0, 0)  # no tzinfo, as some HTTP Date headers parse
    cache.put("archive_org", "k", b"x", fetched_at=naive)
    # test that age() does not blow up subtracting an aware now from a naive fetched_at
    assert cache.age("archive_org", "k", now=datetime(2026, 7, 1, 18, 0, 0)) == timedelta(hours=6)


def test_bad_namespace_rejected(tmp_path):
    cache = RawCache(tmp_path)
    with pytest.raises(ValueError):
        cache.put("bad/ns", "k", b"x")

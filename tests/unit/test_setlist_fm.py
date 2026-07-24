# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the setlist.fm source: the polite pager (no network) and the catalog parser.

The two halves are tested on the two sides of the layer line they live on. The client
(`sources/setlist_fm.py`) is exercised against a fake transport -- pagination, the newest-first
year floor that stops it early, the x-api-key header, the cache round-trip -- and never touches a
song name. The parser (`catalog/parse.py:parse_setlistfm`) turns raw pages into merge records, and
that is where the date conversion, the encore pooling, and the tape-is-not-a-play rule are checked.
"""

import json
import os

import pytest

from setlistkit.catalog.normalizer import Normalizer
from setlistkit.catalog.parse import count_songs
from setlistkit.catalog.parse_setlistfm import parse_setlistfm
from setlistkit.sources.client import Response
from setlistkit.sources.setlist_fm import NAMESPACE, SetlistFmClient
from setlistkit.config import load_config
from setlistkit.store.raw_cache import RawCache

NZ = Normalizer(vocabulary=[])


# -- parser: raw pages -> merge records ----------------------------------------------------

def _song(name, **flags):
    return {"name": name, **flags}


def _setlist(sid, event_date, *sets):
    """A raw setlist object; each `set` is (songs_list, encore_number_or_None)."""
    return {"id": sid, "eventDate": event_date,
            "sets": {"set": [{"song": songs, **({"encore": n} if n else {})}
                             for songs, n in sets]}}


def _page(*setlists):
    return {"setlist": list(setlists)}


def test_event_date_becomes_iso():
    [rec] = parse_setlistfm([_page(_setlist("a", "23-08-2024", ([_song("Mantis")], None)))],
                            normalizer=NZ)
    assert rec["date"] == "2024-08-23" and rec["year"] == "2024"
    assert rec["source"] == "setlist_fm" and rec["identifier"] == "setlist_fm-a"


def test_encore_sets_pool_into_encore():
    [rec] = parse_setlistfm(
        [_page(_setlist("a", "01-02-2024",
                        ([_song("Bridgeless"), _song("Ophelia")], None),
                        ([_song("Hangover")], 1)))],
        normalizer=NZ)
    assert [e["song"] for e in rec["sets"][0]] == ["Bridgeless", "Ophelia"]
    assert [e["song"] for e in rec["encore"]] == ["Hangover"]


def test_tape_song_is_kept_but_flagged_non_song():
    """A song played over the PA is recorded but does not count as a performance."""
    [rec] = parse_setlistfm(
        [_page(_setlist("a", "01-02-2024",
                        ([_song("Intro Music", tape=True), _song("Mantis")], None)))],
        normalizer=NZ)
    tape_entry = rec["sets"][0][0]
    assert tape_entry["song"] == "Intro Music" and tape_entry["non_song"] is True
    assert count_songs(rec["sets"], rec["encore"]) == 1        # only Mantis counts


def test_min_year_floors_the_parse():
    pages = [_page(_setlist("new", "01-02-2024", ([_song("Mantis")], None)),
                   _setlist("old", "01-02-2019", ([_song("Mantis")], None)))]
    dates = {r["date"] for r in parse_setlistfm(pages, normalizer=NZ, min_year=2020)}
    assert dates == {"2024-02-01"}


def test_no_date_and_songless_setlists_are_skipped():
    pages = [_page(_setlist("nodate", "", ([_song("Mantis")], None)),
                   _setlist("empty", "01-02-2024"),
                   _setlist("good", "02-02-2024", ([_song("Mantis")], None)))]
    recs = parse_setlistfm(pages, normalizer=NZ)
    assert [r["identifier"] for r in recs] == ["setlist_fm-good"]


# -- client: pagination + year floor + auth, all against a fake transport ------------------

def _keyfile(tmp_path):
    path = tmp_path / "sfm.key"
    path.write_text("SECRET-KEY", encoding="utf-8")
    os.chmod(path, 0o600)                          # read_api_key refuses a world-readable key
    return path


def _config(tmp_path):
    body = ('data_root = "state"\n'
            'user_agent = "famoe.ly nightly (you@example.com)"\n'
            'mbid = "the-mbid"\n'
            f'[sources.setlist_fm]\napi_key_file = "{_keyfile(tmp_path)}"\n')
    path = tmp_path / "slkit.toml"
    path.write_text(body, encoding="utf-8")
    return load_config(str(path))


class FakeApi:
    """A setlist.fm stand-in: maps page number -> setlists, records every request's headers."""

    def __init__(self, pages, total=None, per_page=20):
        self._pages = pages                        # {page_number: [setlist, ...]}
        self._total = total if total is not None else sum(len(v) for v in pages.values())
        self._per = per_page
        self.headers_seen = []
        self.pages_requested = []

    def __call__(self, url, headers, *, timeout=60):
        self.headers_seen.append(headers)
        page = int(url.split("p=")[-1])
        self.pages_requested.append(page)
        setlists = self._pages.get(page)
        if setlists is None:
            return Response(404, {}, b"")
        body = json.dumps({"total": self._total, "itemsPerPage": self._per,
                           "page": page, "setlist": setlists}).encode("utf-8")
        return Response(200, {}, body)


def _client(tmp_path, api):
    return SetlistFmClient(_config(tmp_path), RawCache(tmp_path / "state"),
                           transport=api, sleeper=lambda _s: None)


def test_pull_pages_until_the_total_is_reached(tmp_path):
    api = FakeApi({1: [_setlist(f"a{i}", "01-02-2024", ([_song("X")], None)) for i in range(20)],
                   2: [_setlist("b", "01-02-2024", ([_song("X")], None))]}, total=21)
    result = _client(tmp_path, api).pull("the-mbid")
    assert result.listed == 21 and result.fetched == 21


def test_pull_sends_the_api_key_on_every_request(tmp_path):
    api = FakeApi({1: [_setlist("a", "01-02-2024", ([_song("X")], None))]}, total=1)
    _client(tmp_path, api).pull("the-mbid")
    assert api.headers_seen and all(h.get("x-api-key") == "SECRET-KEY" for h in api.headers_seen)


def test_pull_stops_at_the_year_floor(tmp_path):
    # Page 1 is all 2024; page 2 dips to 2019, so page 3 must never be requested.
    api = FakeApi({1: [_setlist(f"n{i}", "01-02-2024", ([_song("X")], None)) for i in range(20)],
                   2: [_setlist("boundary", "31-12-2019", ([_song("X")], None))],
                   3: [_setlist("older", "01-02-2018", ([_song("X")], None))]}, total=60)
    client = _client(tmp_path, api)
    result = client.pull("the-mbid", min_year=2020)
    assert 3 not in api.pages_requested                              # stopped at the boundary page
    assert result.fetched == 21                                      # 20 (page 1) + 1 (page 2)
    assert len(client.cached_pages("the-mbid")) == 2                 # only pages 1 and 2 cached


def test_dry_run_fetches_only_the_first_page(tmp_path):
    api = FakeApi({1: [_setlist("a", "01-02-2024", ([_song("X")], None))],
                   2: [_setlist("b", "01-02-2024", ([_song("X")], None))]}, total=2)
    result = _client(tmp_path, api).pull("the-mbid", dry_run=True)
    assert api.pages_requested == [1]
    assert result.listed == 2 and result.fetched == 0


def test_unknown_artist_returns_an_empty_result(tmp_path):
    result = _client(tmp_path, FakeApi({}, total=0)).pull("nobody")
    assert result.listed == 0 and result.fetched == 0


def test_cached_pages_round_trips(tmp_path):
    api = FakeApi({1: [_setlist("a", "01-02-2024", ([_song("Mantis")], None))]}, total=1)
    client = _client(tmp_path, api)
    client.pull("the-mbid")
    pages = client.cached_pages("the-mbid")
    assert len(pages) == 1
    [rec] = parse_setlistfm(pages, normalizer=NZ)
    assert rec["sets"][0][0]["song"] == "Mantis"


def test_world_readable_key_is_refused(tmp_path):
    path = tmp_path / "sfm.key"
    path.write_text("SECRET", encoding="utf-8")
    os.chmod(path, 0o644)                          # world-readable
    body = ('data_root = "state"\n'
            'user_agent = "famoe.ly nightly (you@example.com)"\n'
            f'[sources.setlist_fm]\napi_key_file = "{path}"\n')
    cfg_path = tmp_path / "slkit.toml"
    cfg_path.write_text(body, encoding="utf-8")
    from setlistkit.diagnostics import DiagnosticError
    with pytest.raises(DiagnosticError):
        SetlistFmClient(load_config(str(cfg_path)), RawCache(tmp_path / "state"))

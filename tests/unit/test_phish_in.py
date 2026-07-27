# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the phish.in source: the two-phase index-then-shows pull, against a fake transport.

phish.in is the first source since archive.org to need two phases, and for the same reason: the
index does not carry the payload. These lock that the listing is walked, that each show is fetched
once, that a re-run costs the host nothing, and that a show the index names but the item endpoint
does not have is recorded as missing rather than sinking the run -- which is a systematic case
here, not a fluke, because the catalog includes shows nobody ever taped.
"""

import inspect
import json
import urllib.parse

import pytest

from setlistkit.catalog.normalizer import Normalizer
from setlistkit.catalog.parse_phish_in import parse_phish_in
from setlistkit.config import load_config
from setlistkit.sources.base import SourceClient
from setlistkit.sources.client import Response
from setlistkit.sources.phish_in import (NAMESPACE, PhishInClient, show_dates_in, shows_in,
                                         tracks_in)
from setlistkit.store.raw_cache import RawCache


NZ = Normalizer(vocabulary=[])


# -- parser: show payloads -> merge show-records ---------------------------------------------

def test_set_names_map_to_sets_and_the_encore():
    show = _show("2024-07-01",
                 _track("Llama", 1, 1, set_name="Set 1"),
                 _track("Maze", 2, 1, set_name="Set 2"),
                 _track("Tube", 3, 1, set_name="Encore"))
    [rec] = parse_phish_in([show], normalizer=NZ)
    assert [[e["song"] for e in s] for s in rec["sets"]] == [["Llama"], ["Maze"]]
    assert [e["song"] for e in rec["encore"]] == ["Tube"]


def test_soundcheck_and_pre_show_are_kept_but_never_counted_as_plays():
    """Measured on the live cache: all 46 Soundcheck and 14 Pre-Show tracks carry
    exclude_from_stats, and no set or encore track does. Folding them into the sets would inflate
    a song's totals with material nobody performed to a room."""
    show = _show("2024-07-01",
                 _track("Sample in a Jar", 1, 1, set_name="Soundcheck", exclude_from_stats=True),
                 _track("Llama", 2, 1, set_name="Set 1"))
    [rec] = parse_phish_in([show], normalizer=NZ)
    assert rec["n_songs"] == 1                              # only the real play counts
    flat = [e for s in rec["sets"] for e in s]
    assert {"Sample in a Jar", "Llama"} == {e["song"] for e in flat}   # both are still present
    assert [e["song"] for e in flat if e["non_song"]] == ["Sample in a Jar"]


def test_a_soundcheck_never_takes_the_opening_set_slot():
    """A set's INDEX carries meaning downstream -- index 0 is the show's opening set -- so a
    soundcheck sorting first would rewrite what "opener" means for every night that had one."""
    show = _show("2024-07-01",
                 _track("Checked", 1, 1, set_name="Soundcheck", exclude_from_stats=True),
                 _track("Real Opener", 2, 1, set_name="Set 1"))
    [rec] = parse_phish_in([show], normalizer=NZ)
    assert rec["sets"][0][0]["song"] == "Real Opener"
    assert rec["sets"][-1][0]["song"] == "Checked"          # kept, at the end


def test_exclude_from_stats_is_what_flags_a_non_song_not_the_set_name():
    """The upstream's own judgement, kept current by someone watching, rather than a blocklist
    here that goes stale the first time they add a category."""
    show = _show("2024-07-01", _track("Weird Thing", 1, 1, set_name="Set 1",
                                      exclude_from_stats=True),
                 _track("Llama", 2, 1, set_name="Set 1"))
    [rec] = parse_phish_in([show], normalizer=NZ)
    assert [(e["song"], e["non_song"]) for e in rec["sets"][0]] == [("Weird Thing", True),
                                                                    ("Llama", False)]


def test_segue_is_always_false_because_the_source_has_no_segue_field():
    """Not empty -- absent. Claiming one either way would be inventing data, and it is why
    phish.net outranks this source on a date they both carry."""
    show = _show("2024-07-01", _track("Mike's Song", 1, 1), _track("Weekapaug Groove", 2, 1))
    [rec] = parse_phish_in([show], normalizer=NZ)
    assert all(e["segue"] is False for e in rec["sets"][0])


def test_tracks_are_ordered_by_position_not_array_order():
    show = _show("2024-07-01", _track("Third", 3, 1), _track("First", 1, 1), _track("Second", 2, 1))
    [rec] = parse_phish_in([show], normalizer=NZ)
    assert [e["song"] for e in rec["sets"][0]] == ["First", "Second", "Third"]


def test_a_show_with_no_audio_still_parses():
    """phish.in carries setlists for nights nobody taped, which is exactly why it is worth
    consulting for them."""
    show = _show("2024-02-22", _track("Llama", 1, None), audio_status="missing")
    [rec] = parse_phish_in([show], normalizer=NZ)
    assert rec["n_songs"] == 1 and rec["date"] == "2024-02-22"


def test_min_year_floors_the_parse_and_records_are_sorted():
    shows = [_show("2024-07-02", _track("B", 1, 1)), _show("2019-07-01", _track("Old", 1, 1)),
             _show("2024-07-01", _track("A", 1, 1))]
    recs = parse_phish_in(shows, normalizer=NZ, min_year=2020)
    assert [r["date"] for r in recs] == ["2024-07-01", "2024-07-02"]
    assert all(r["source"] == "phish_in" for r in recs)


def test_junk_shows_parse_to_nothing_rather_than_raising():
    assert parse_phish_in([None, {}, {"date": "2024-07-01"}, {"date": "", "tracks": []}],
                          normalizer=NZ) == []


def _config(tmp_path):
    body = ('data_root = "state"\n'
            'user_agent = "setlistkit-research (you@example.com)"\n'
            '[sources.phish_in]\n')
    path = tmp_path / "phish-slkit.toml"
    path.write_text(body, encoding="utf-8")
    return load_config(str(path))


def _track(title, position, duration_ms, **extra):
    return {"title": title, "position": position, "duration": duration_ms,
            "set_name": "Set 1", "exclude_from_stats": False, **extra}


def _show(date, *tracks, **extra):
    return {"date": date, "venue_name": "Madison Square Garden", "tour_name": "2026 Summer Tour",
            "audio_status": "complete", "tracks": list(tracks), **extra}


class FakeApi:
    """A phish.in stand-in: an index paged at `per_page`, plus a per-date show endpoint."""

    def __init__(self, shows, per_page=100, absent=()):
        self._shows = {s["date"]: s for s in shows}
        self._dates = sorted(self._shows)
        self._per = per_page
        self._absent = set(absent)                 # dates the index names but /shows/<date> 404s
        self.urls_seen = []

    def __call__(self, url, headers, *, timeout=60):
        self.urls_seen.append(url)
        if "/shows?" in url:
            # Parse the query properly. Splitting on "page=" finds "per_page=" first, which is
            # exactly the kind of thing a fake gets wrong quietly and then blames the code for.
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            page = int(query["page"][0])
            start = (page - 1) * self._per
            window = self._dates[start:start + self._per]
            total_pages = max(1, -(-len(self._dates) // self._per))
            body = {"shows": [{"date": d} for d in window], "total_pages": total_pages,
                    "current_page": page, "total_entries": len(self._dates)}
            return Response(200, {}, json.dumps(body).encode("utf-8"))
        date = url.rstrip("/").split("/")[-1]
        if date in self._absent or date not in self._shows:
            return Response(404, {}, b"")
        return Response(200, {}, json.dumps(self._shows[date]).encode("utf-8"))


def _client(tmp_path, api):
    return PhishInClient(_config(tmp_path), RawCache(tmp_path / "state"),
                         transport=api, sleeper=lambda _s: None)


def _keyword_only_shape(func):
    """The pull contract minus `self` and the identity slot, whose NAME may differ per source."""
    params = list(inspect.signature(func).parameters.values())
    return [(p.name, p.kind, p.default) for p in params[2:]]


# -- projections ----------------------------------------------------------------------------

def test_projections_are_defensive_about_junk():
    assert shows_in(None) == [] and shows_in({"shows": "nope"}) == []
    assert show_dates_in({"shows": [{"date": ""}, {"date": "2024-07-01"}]}) == ["2024-07-01"]
    assert tracks_in(None) == [] and tracks_in({"tracks": [None, 3]}) == []


def test_track_durations_are_milliseconds():
    """The reason this source exists: an integer from a database, not a string off a tracklist."""
    payload = _show("2023-12-31", _track("Everything's Right", 1, 668400))
    [track] = tracks_in(payload)
    assert track["duration"] == 668400            # 11:08
    assert track["duration"] // 1000 == 668


# -- the two-phase pull ---------------------------------------------------------------------

def test_pull_lists_then_fetches_each_show(tmp_path, monkeypatch):
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2024)
    api = FakeApi([_show("2024-07-01", _track("Llama", 1, 300000)),
                   _show("2024-07-02", _track("Maze", 1, 400000))])
    result = _client(tmp_path, api).pull(min_year=2024)
    assert result.listed == 2 and result.fetched == 2 and result.cached == 0
    assert sum("/shows?" in u for u in api.urls_seen) == 1        # one index page
    assert sum("/shows/2024-" in u for u in api.urls_seen) == 2   # one fetch per show


def test_the_index_is_paged(tmp_path, monkeypatch):
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2024)
    shows = [_show(f"2024-07-{d:02d}", _track("Llama", 1, 300000)) for d in range(1, 8)]
    api = FakeApi(shows, per_page=3)
    result = _client(tmp_path, api).pull(min_year=2024)
    assert result.listed == 7
    assert sum("/shows?" in u for u in api.urls_seen) == 3        # 3 + 3 + 1
    assert result.fetched == 7


def test_a_second_pull_refetches_the_index_but_no_shows(tmp_path, monkeypatch):
    """The index is the phase that discovers what exists, so it is always re-asked -- a stale
    index is how a show that landed yesterday goes unnoticed forever. The shows behind it are
    immutable-ish and come from cache, which is where the savings actually are."""
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2024)
    api = FakeApi([_show("2024-07-01", _track("Llama", 1, 300000))])
    client = _client(tmp_path, api)
    client.pull(min_year=2024)
    before = sum("/shows/2024-" in u for u in api.urls_seen)
    second = client.pull(min_year=2024)
    after = sum("/shows/2024-" in u for u in api.urls_seen)
    assert second.fetched == 0 and second.cached == 1
    assert after == before                          # no show was re-fetched


def test_a_show_the_index_names_but_the_item_endpoint_lacks_is_missing_not_fatal(tmp_path,
                                                                                 monkeypatch):
    """Systematic, not a fluke: the catalog carries shows with no circulating recording, and they
    cluster where nobody smuggles a rig in -- the Mexico destination runs, most of them."""
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2024)
    api = FakeApi([_show("2024-07-01", _track("Llama", 1, 300000)), _show("2024-02-22")],
                  absent={"2024-02-22"})
    result = _client(tmp_path, api).pull(min_year=2024)
    assert result.missing == ("2024-02-22",)
    assert result.fetched == 1 and result.listed == 2


def test_every_listed_show_lands_in_exactly_one_counter(tmp_path, monkeypatch):
    """The rule PullResult's docstring states: a counter an item can fall out of is how a lost
    show goes unnoticed."""
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2024)
    api = FakeApi([_show("2024-07-01", _track("Llama", 1, 1)), _show("2024-07-02"),
                   _show("2024-07-03", _track("Maze", 1, 1))], absent={"2024-07-02"})
    result = _client(tmp_path, api).pull(min_year=2024)
    assert result.fetched + result.cached + len(result.missing) == result.listed


def test_dry_run_stops_after_the_index(tmp_path, monkeypatch):
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2024)
    api = FakeApi([_show("2024-07-01", _track("Llama", 1, 1)),
                   _show("2024-07-02", _track("Maze", 1, 1))])
    result = _client(tmp_path, api).pull(min_year=2024, dry_run=True)
    assert not any("/shows/2024-" in u for u in api.urls_seen)     # no item was fetched
    assert result.listed == 2 and result.planned == 2 and result.fetched == 0
    assert result.probe_requests == 1              # the one index page it had to read


def test_a_dry_run_over_a_cached_index_reports_spending_nothing(tmp_path, monkeypatch):
    """`probe_requests` counts what left the box, not what was asked for -- so a second dry run
    against a warm index honestly reports zero rather than re-claiming the first one's cost."""
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2024)
    api = FakeApi([_show("2024-07-01", _track("Llama", 1, 1))])
    client = _client(tmp_path, api)
    client.pull(min_year=2024, dry_run=True)
    assert client.pull(min_year=2024, dry_run=True).probe_requests == 0


def test_dry_run_discounts_shows_already_held(tmp_path, monkeypatch):
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2024)
    api = FakeApi([_show("2024-07-01", _track("Llama", 1, 1)),
                   _show("2024-07-02", _track("Maze", 1, 1))])
    client = _client(tmp_path, api)
    client.pull(min_year=2024)
    assert client.pull(min_year=2024, dry_run=True).planned == 0


def test_the_year_range_is_built_from_min_year_to_now(tmp_path, monkeypatch):
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2026)
    api = FakeApi([_show("2024-07-01", _track("Llama", 1, 1))])
    _client(tmp_path, api).pull(min_year=2020)
    assert any("year_range=2020-2026" in u for u in api.urls_seen)


def test_cached_shows_round_trips_with_its_tracks(tmp_path, monkeypatch):
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2024)
    api = FakeApi([_show("2024-07-01", _track("Llama", 1, 300000), _track("Maze", 2, 400000))])
    client = _client(tmp_path, api)
    client.pull(min_year=2024)
    [show] = client.cached_shows(2024)
    assert [t["title"] for t in tracks_in(show)] == ["Llama", "Maze"]
    assert [t["duration"] for t in tracks_in(show)] == [300000, 400000]


def test_cached_reads_touch_no_network(tmp_path, monkeypatch):
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2024)
    api = FakeApi([_show("2024-07-01", _track("Llama", 1, 1))])
    client = _client(tmp_path, api)
    client.pull(min_year=2024)
    before = len(api.urls_seen)
    assert client.cached_show_dates(2024) == ["2024-07-01"]
    assert len(client.cached_shows(2024)) == 1
    assert len(api.urls_seen) == before


def test_progress_is_reported_per_show(tmp_path, monkeypatch):
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2024)
    api = FakeApi([_show(f"2024-07-{d:02d}", _track("Llama", 1, 1)) for d in (1, 2, 3)])
    seen = []
    _client(tmp_path, api).pull(min_year=2024, progress=lambda i, n: seen.append((i, n)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_the_client_satisfies_the_source_contract_by_shape():
    """Locked now, before an importer registers it -- the pull surface is structural, so drift
    here is what would stop a registry entry from being a drop-in."""
    assert issubclass(PhishInClient, SourceClient)
    assert _keyword_only_shape(PhishInClient.pull) == _keyword_only_shape(SourceClient.pull)
    assert SourceClient not in PhishInClient.__mro__


def test_no_key_is_required(tmp_path, monkeypatch):
    """The API is anonymous. A config with no api_key_file must construct and pull fine -- this is
    the only source here that needs no credential, and it should stay that way."""
    monkeypatch.setattr("setlistkit.sources.phish_in._current_year", lambda: 2024)
    api = FakeApi([_show("2024-07-01", _track("Llama", 1, 1))])
    assert _client(tmp_path, api).pull(min_year=2024).fetched == 1
    assert not any("apikey" in u for u in api.urls_seen)

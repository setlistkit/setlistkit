# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the phish.net source: year-at-a-time fetching, envelope errors, and key hygiene.

Everything here runs against a fake transport -- no socket, no key, no requests spent on a host
whose docs warn that apps demanding data too frequently get disabled.

Three of these tests exist because of traps specific to this API rather than to this codebase:
the credential travels in the query string (so it must not reach the cache sidecar), logical
failures arrive as HTTP 200 (so the polite client cannot see them), and a year payload mixes
Phish with Trey solo and with shows that were called off (so a naive count reports a corpus that
never happened).
"""

import inspect
import json
import os

import pytest

from setlistkit.catalog.normalizer import Normalizer
from setlistkit.catalog.parse_phish_net import NON_CONCERT, parse_phish_net
from setlistkit.config import load_config
from setlistkit.diagnostics import DiagnosticError
from setlistkit.sources.base import SourceClient
from setlistkit.sources.client import Response
from setlistkit.sources.phish_net import (NAMESPACE, PhishNetAPIError, PhishNetClient,
                                          show_dates_in)
from setlistkit.store.raw_cache import RawCache

KEY = "SECRET-PNET-KEY"
NZ = Normalizer(vocabulary=[])


# -- parser: flat rows -> merge show-records -------------------------------------------------

def _srow(showdate, song, set_label="1", position=1, transition=1, **extra):
    """One setlist row with the fields the parser reads."""
    return {"showdate": showdate, "song": song, "set": set_label, "position": position,
            "transition": transition, "artist_slug": "phish", "exclude": "0",
            "showid": 1234, **extra}


def _year(*rows):
    return {"error": 0, "error_message": "", "data": list(rows)}


def test_position_is_show_relative_and_is_reranked_per_set():
    """THE trap. `position` is a continuous 1-based counter across the whole show and never
    resets, so set 2 starts at 10, not at 1. Indexing a set by it puts every set's songs at the
    wrong offset and `features.py` reads openers as mid-set songs."""
    payload = _year(
        _srow("2018-10-31", "Set One Opener", "1", 1),
        _srow("2018-10-31", "Set One Closer", "1", 9),
        _srow("2018-10-31", "Set Two Opener", "2", 10),
        _srow("2018-10-31", "Set Two Closer", "2", 19),
        _srow("2018-10-31", "Encore Song", "e", 27),
    )
    [rec] = parse_phish_net([payload], normalizer=NZ)
    assert [e["song"] for e in rec["sets"][0]] == ["Set One Opener", "Set One Closer"]
    assert [e["song"] for e in rec["sets"][1]] == ["Set Two Opener", "Set Two Closer"]
    assert [e["song"] for e in rec["encore"]] == ["Encore Song"]
    # Each set starts with ITS opener, which is the whole point of the re-rank.
    assert rec["sets"][1][0]["song"] == "Set Two Opener"


def test_rows_arriving_out_of_order_are_sorted_by_position():
    payload = _year(_srow("2024-07-01", "Third", "1", 3), _srow("2024-07-01", "First", "1", 1),
                    _srow("2024-07-01", "Second", "1", 2))
    [rec] = parse_phish_net([payload], normalizer=NZ)
    assert [e["song"] for e in rec["sets"][0]] == ["First", "Second", "Third"]


def test_numbered_sets_stay_in_order_and_every_encore_pools():
    payload = _year(_srow("2024-07-01", "S2", "2", 10), _srow("2024-07-01", "S1", "1", 1),
                    _srow("2024-07-01", "E1", "e", 20), _srow("2024-07-01", "E2", "e2", 21))
    [rec] = parse_phish_net([payload], normalizer=NZ)
    assert [[e["song"] for e in s] for s in rec["sets"]] == [["S1"], ["S2"]]
    assert [e["song"] for e in rec["encore"]] == ["E1", "E2"]      # e and e2 pool together


def test_an_unrecognized_set_label_is_treated_as_an_encore():
    """The vocabulary is open and phish.net's own renderer falls through to "Encore" for anything
    it does not know, so an unfamiliar label must not become a phantom set or vanish."""
    payload = _year(_srow("2024-07-01", "Regular", "1", 1),
                    _srow("2024-07-01", "Mystery", "z9", 2))
    [rec] = parse_phish_net([payload], normalizer=NZ)
    assert [e["song"] for e in rec["sets"][0]] == ["Regular"]
    assert [e["song"] for e in rec["encore"]] == ["Mystery"]


@pytest.mark.parametrize("code,expected", [(1, False), (2, True), (3, True),
                                           (4, False), (5, False), (6, False), (7, False)])
def test_segue_comes_from_the_transition_code(code, expected):
    """2 renders " > " and 3 renders " -> "; 4/5/6 are structural boundaries that render empty,
    and 7 is undocumented but observed only at the end of an encore, also rendering empty."""
    [rec] = parse_phish_net([_year(_srow("2024-07-01", "Tube", "1", 1, transition=code))],
                            normalizer=NZ)
    assert rec["sets"][0][0]["segue"] is expected


@pytest.mark.parametrize("code", [1, 2, 3, 4, 5, 6, 7])
def test_the_raw_transition_code_is_carried_through_unflattened(code):
    """The boolean cannot hold `->`, so the code rides along beside it.

    Measured on 2020-2026: `>` and `->` are near-disjoint relations -- 1,398 and 304 distinct
    song pairs with 36 in common -- and folding them together buries the 325 jam transitions in
    1,600 routine ones. 4/5/6 encode set structure and 7 is undocumented; all are kept, because
    a code we cannot explain yet is still a fact the source stated.
    """
    [rec] = parse_phish_net([_year(_srow("2024-07-01", "Tube", "1", 1, transition=code))],
                            normalizer=NZ)
    assert rec["sets"][0][0]["transition"] == code


def test_an_entry_with_no_transition_omits_the_key_rather_than_defaulting():
    """Absent must not become 1. "Not told" and "the band stopped" are different claims, and a
    placeholder here would be a value phish.net never sends arriving as though it had."""
    [rec] = parse_phish_net([_year(_srow("2024-07-01", "Tube", "1", 1, transition=None))],
                            normalizer=NZ)
    assert "transition" not in rec["sets"][0][0]


def test_only_the_named_artists_rows_are_parsed():
    """Across 2020-2026 the non-Phish rows outnumber the Phish ones, so this is not a nicety."""
    payload = _year(_srow("2024-07-01", "Llama"),
                    _srow("2024-07-02", "Jibboo", artist_slug="trey-anastasio"))
    recs = parse_phish_net([payload], normalizer=NZ)
    assert [r["date"] for r in recs] == ["2024-07-01"]


def test_a_non_concert_appearance_is_kept_and_tagged_not_dropped():
    """A Tiny Desk is real songs really played. Dropping it loses repertoire; counting it as a
    show corrupts every shows-since-last-play gap that steps over it. So: keep, and tag."""
    payload = _year(_srow("2024-07-17", "Sigma Oasis", exclude="1"),
                    _srow("2024-07-01", "Llama"))
    by_date = {r["date"]: r for r in parse_phish_net([payload], normalizer=NZ)}
    assert by_date["2024-07-17"]["reason"] == NON_CONCERT
    assert by_date["2024-07-17"]["n_songs"] == 1             # the play is still counted...
    assert "reason" not in by_date["2024-07-01"]            # ...and an ordinary show is untagged


def test_field_types_are_coerced_not_branched_on():
    """Older captures return every value as a string, newer ones return ints, and a single row can
    mix the two. A parser that branched on type would silently mis-order one vintage."""
    as_strings = _year({"showdate": "2024-07-01", "song": "Second", "set": "1", "position": "2",
                        "transition": "2", "artist_slug": "phish", "exclude": "0", "showid": "99"},
                       {"showdate": "2024-07-01", "song": "First", "set": "1", "position": "1",
                        "transition": "1", "artist_slug": "phish", "exclude": "0", "showid": "99"})
    [rec] = parse_phish_net([as_strings], normalizer=NZ)
    assert [e["song"] for e in rec["sets"][0]] == ["First", "Second"]
    assert rec["sets"][0][0]["segue"] is False and rec["sets"][0][1]["segue"] is True
    assert rec["identifier"] == "phish_net-99"


def test_min_year_floors_the_phish_net_parse():
    payload = _year(_srow("2024-07-01", "New"), _srow("2019-07-01", "Old"))
    dates = {r["date"] for r in parse_phish_net([payload], normalizer=NZ, min_year=2020)}
    assert dates == {"2024-07-01"}


def test_rows_without_a_date_or_a_song_contribute_nothing():
    payload = _year(_srow("", "Orphan"), _srow("2024-07-01", ""), _srow("2024-07-02", "Real"))
    assert [r["date"] for r in parse_phish_net([payload], normalizer=NZ)] == ["2024-07-02"]


def test_records_are_sorted_and_carry_the_source_tag():
    payload = _year(_srow("2024-07-02", "B"), _srow("2024-07-01", "A"))
    recs = parse_phish_net([payload], normalizer=NZ)
    assert [r["date"] for r in recs] == ["2024-07-01", "2024-07-02"]
    assert all(r["source"] == "phish_net" for r in recs)


def test_a_junk_payload_parses_to_nothing_rather_than_raising():
    assert parse_phish_net([None, {"data": "nope"}, {"data": [1, None]}], normalizer=NZ) == []


def _row(showdate, song, *, artist_slug="phish", exclude="0", **extra):
    """One setlist row, in the shape the v5 `setlists` method returns them."""
    return {"showdate": showdate, "song": song, "artist_slug": artist_slug,
            "exclude": exclude, "set": "1", "position": 1, **extra}


def _envelope(*rows, error=0, message=""):
    return {"error": error, "error_message": message, "data": list(rows)}


def _keyfile(tmp_path, contents=KEY, name="pnet.key"):
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")
    os.chmod(path, 0o600)                          # read_api_key refuses a world-readable key
    return path


def _config(tmp_path, *, key_path=None, field=None):
    field_line = f'api_key_field = "{field}"\n' if field else ""
    body = ('data_root = "state"\n'
            'user_agent = "setlistkit-research (you@example.com)"\n'
            '[sources.phish_net]\n'
            f'api_key_file = "{key_path or _keyfile(tmp_path)}"\n'
            f'{field_line}')
    path = tmp_path / "phish-slkit.toml"
    path.write_text(body, encoding="utf-8")
    return load_config(str(path))


class FakeApi:
    """A phish.net stand-in: maps year -> envelope, and records every URL it was asked for."""

    def __init__(self, years, shows_envelope=None):
        self._years = years                        # {year: envelope_dict}
        self._shows = shows_envelope if shows_envelope is not None else _envelope()
        self.urls_seen = []

    def __call__(self, url, headers, *, timeout=60):
        self.urls_seen.append(url)
        if "/shows/artist/" in url:
            return Response(200, {}, json.dumps(self._shows).encode("utf-8"))
        year = int(url.split("/showyear/")[-1].split(".json")[0])
        envelope = self._years.get(year)
        if envelope is None:
            return Response(200, {}, json.dumps(_envelope()).encode("utf-8"))
        return Response(200, {}, json.dumps(envelope).encode("utf-8"))


def _client(tmp_path, api, **cfg):
    return PhishNetClient(_config(tmp_path, **cfg), RawCache(tmp_path / "state"),
                          transport=api, sleeper=lambda _s: None)


def _keyword_only_shape(func):
    """The pull contract minus `self` and the identity slot, whose NAME may differ per source."""
    params = list(inspect.signature(func).parameters.values())
    return [(p.name, p.kind, p.default) for p in params[2:]]


# -- the projection: what counts as a show -------------------------------------------------

def test_only_the_named_artists_rows_are_counted():
    """The showyear endpoint returns Trey solo and guest appearances beside Phish."""
    payload = _envelope(_row("2024-07-01", "Llama"),
                        _row("2024-07-02", "Gotta Jibboo", artist_slug="trey-anastasio"))
    assert show_dates_in(payload, "phish") == {"2024-07-01"}


def test_appearances_that_were_not_concerts_are_not_counted_as_shows():
    """`exclude` does not mark a show that was called off -- one nobody played has no song rows to
    return. Measured against the live 2020-2026 corpus it marks 8 dates that really happened but
    were not concerts: a Tonight Show spot (1 song), an NPR Tiny Desk (5), and the short poolside
    sets at Moon Palace. Right to drop when COUNTING shows; wrong to drop when parsing plays."""
    payload = _envelope(_row("2020-02-21", "Wolfman's Brother"),
                        _row("2024-07-17", "Sigma Oasis", exclude="1"))    # NPR Tiny Desk
    assert show_dates_in(payload, "phish") == {"2020-02-21"}


def test_a_show_is_counted_once_however_many_songs_it_has():
    payload = _envelope(_row("2024-07-01", "Llama"), _row("2024-07-01", "Maze"),
                        _row("2024-07-01", "Tube"))
    assert show_dates_in(payload, "phish") == {"2024-07-01"}


def test_a_junk_payload_projects_to_nothing_rather_than_raising():
    assert show_dates_in(None, "phish") == set()
    assert show_dates_in({"data": "not-a-list"}, "phish") == set()
    assert show_dates_in({"data": [None, 7, "x"]}, "phish") == set()


# -- key hygiene: the reason `extra_query` exists -------------------------------------------

def test_the_key_is_sent_as_a_query_parameter(tmp_path):
    api = FakeApi({2024: _envelope(_row("2024-07-01", "Llama"))})
    _client(tmp_path, api).pull("phish", min_year=2024)
    setlist_urls = [u for u in api.urls_seen if "/showyear/" in u]
    assert setlist_urls and all(f"apikey={KEY}" in u for u in setlist_urls)


def test_the_key_never_reaches_the_cache_sidecar(tmp_path):
    """The whole point of applying auth at the transport call. A cache sidecar is exactly the kind
    of file that gets pasted into a bug report, and keyfile.py refuses a world-readable key
    precisely so the secret does not sit somewhere casual -- writing it here would undo that."""
    api = FakeApi({2024: _envelope(_row("2024-07-01", "Llama"))})
    _client(tmp_path, api).pull("phish", min_year=2024)
    meta_files = list((tmp_path / "state" / "raw" / NAMESPACE / "meta").glob("*.json"))
    assert meta_files, "expected the pull to have written cache sidecars"
    for path in meta_files:
        assert KEY not in path.read_text(encoding="utf-8")


def test_the_key_never_reaches_an_exception_message(tmp_path):
    """The other half of the same rule: a traceback is a place secrets leak into logs."""
    api = FakeApi({2024: _envelope(error=2, message="invalid key")})
    with pytest.raises(PhishNetAPIError) as caught:
        _client(tmp_path, api).pull("phish", min_year=2024)
    assert KEY not in str(caught.value)


def test_a_json_credential_blob_is_read_by_field(tmp_path):
    """phish.net's key page issues {"apikey": ..., "publickey": ...}; both values are real, so the
    file cannot simply be read whole."""
    blob = json.dumps({"version": 1, "apikey": KEY, "publickey": "PUBLIC-NOT-THE-SECRET"})
    api = FakeApi({2024: _envelope(_row("2024-07-01", "Llama"))})
    client = _client(tmp_path, api, key_path=_keyfile(tmp_path, blob, "keys.json"),
                     field="apikey")
    client.pull("phish", min_year=2024)
    setlist_urls = [u for u in api.urls_seen if "/showyear/" in u]
    assert all(f"apikey={KEY}" in u for u in setlist_urls)
    assert not any("PUBLIC-NOT-THE-SECRET" in u for u in api.urls_seen)


def test_a_credential_blob_missing_the_named_field_is_a_diagnostic(tmp_path):
    blob = json.dumps({"version": 1, "publickey": "PUBLIC"})
    with pytest.raises(DiagnosticError) as caught:
        _client(tmp_path, FakeApi({}), key_path=_keyfile(tmp_path, blob, "keys.json"),
                field="apikey")
    assert "apikey" in str(caught.value.diagnostic.summary)


def test_a_world_readable_key_is_refused_before_any_request(tmp_path):
    path = tmp_path / "pnet.key"
    path.write_text(KEY, encoding="utf-8")
    os.chmod(path, 0o644)
    api = FakeApi({})
    with pytest.raises(DiagnosticError):
        _client(tmp_path, api, key_path=path)
    assert api.urls_seen == []                     # nothing left the box


# -- envelope errors: HTTP 200 that is not a success ----------------------------------------

def test_an_envelope_error_raises_even_though_the_status_was_200(tmp_path):
    api = FakeApi({2024: _envelope(error=4, message="rate limited")})
    with pytest.raises(PhishNetAPIError) as caught:
        _client(tmp_path, api).pull("phish", min_year=2024)
    assert caught.value.code == 4
    assert "rate limited" in str(caught.value)


def test_an_envelope_error_is_evicted_from_the_cache(tmp_path):
    """The failure this guards is silent: PoliteClient has already written the error body to disk
    by the time the envelope is readable, and a cached error parses back as a year holding no
    shows -- indistinguishable from a year the band did not tour."""
    api = FakeApi({2024: _envelope(error=4, message="rate limited")})
    client = _client(tmp_path, api)
    with pytest.raises(PhishNetAPIError):
        client.pull("phish", min_year=2024)
    assert client.cached_years(2024) == []
    blobs = tmp_path / "state" / "raw" / NAMESPACE / "blob"
    assert not list(blobs.glob("*")) or client.cached_show_dates("phish", 2024) == set()


def test_no_data_matched_is_an_empty_year_not_a_failure(tmp_path):
    """Error 11 means the query matched nothing, which for a year endpoint is a year off the road."""
    api = FakeApi({2024: _envelope(error=11, message="no data")})
    result = _client(tmp_path, api).pull("phish", min_year=2024)
    assert result.fetched == 0 and result.listed == 0


# -- the pull loop ---------------------------------------------------------------------------

def test_pull_asks_for_one_request_per_year(tmp_path, monkeypatch):
    monkeypatch.setattr("setlistkit.sources.phish_net._current_year", lambda: 2026)
    api = FakeApi({y: _envelope(_row(f"{y}-07-01", "Llama")) for y in (2024, 2025, 2026)})
    result = _client(tmp_path, api).pull("phish", min_year=2024)
    year_urls = [u for u in api.urls_seen if "/showyear/" in u]
    assert len(year_urls) == 3                     # 2024, 2025, 2026 -- no pagination anywhere
    assert result.fetched == 3                     # one show apiece
    assert result.planned_requests == 4            # 3 years + the show-metadata call


def test_pull_also_caches_the_show_metadata_that_carries_updated_at(tmp_path, monkeypatch):
    """The setlist rows have no updated_at, and phish.net has no `since` parameter, so this call
    is the only basis a later incremental sync has for noticing revised history."""
    monkeypatch.setattr("setlistkit.sources.phish_net._current_year", lambda: 2024)
    api = FakeApi({2024: _envelope(_row("2024-07-01", "Llama"))})
    _client(tmp_path, api).pull("phish", min_year=2024)
    assert any("/shows/artist/phish.json" in u for u in api.urls_seen)


def test_a_second_pull_reports_cached_and_spends_no_requests(tmp_path, monkeypatch):
    monkeypatch.setattr("setlistkit.sources.phish_net._current_year", lambda: 2024)
    api = FakeApi({2024: _envelope(_row("2024-07-01", "Llama"), _row("2024-07-02", "Maze"))})
    client = _client(tmp_path, api)
    first = client.pull("phish", min_year=2024)
    before = len(api.urls_seen)
    second = client.pull("phish", min_year=2024)
    assert first.fetched == 2 and first.cached == 0
    assert second.fetched == 0 and second.cached == 2
    assert len(api.urls_seen) == before            # the cache answered; nothing left the box


def test_a_show_spanning_two_cached_years_is_counted_once(tmp_path, monkeypatch):
    """listed is a DISTINCT show count, so the fetched/cached split must not double-count."""
    monkeypatch.setattr("setlistkit.sources.phish_net._current_year", lambda: 2025)
    api = FakeApi({2024: _envelope(_row("2024-12-31", "Llama")),
                   2025: _envelope(_row("2024-12-31", "Llama"), _row("2025-01-01", "Maze"))})
    result = _client(tmp_path, api).pull("phish", min_year=2024)
    assert result.listed == 2
    assert result.fetched + result.cached == result.listed


def test_dry_run_spends_nothing_at_all(tmp_path, monkeypatch):
    """Unlike a paged source there is no first page to fetch to size the job -- the years come
    from config -- so the honest dry run costs zero requests and reports the request count."""
    monkeypatch.setattr("setlistkit.sources.phish_net._current_year", lambda: 2026)
    api = FakeApi({y: _envelope(_row(f"{y}-07-01", "Llama")) for y in (2024, 2025, 2026)})
    result = _client(tmp_path, api).pull("phish", min_year=2024, dry_run=True)
    assert api.urls_seen == []
    assert result.planned_requests == 4            # 3 unfetched years + show metadata
    assert result.fetched == 0
    # Reported, not inferred: the CLI used to deduce "the first page was fetched" from the mere
    # presence of planned_requests, and would say so about a run that made no request at all.
    assert result.probe_requests == 0


def test_cached_years_round_trips_and_skips_a_gap(tmp_path, monkeypatch):
    """Years are independent requests, not a contiguous walk, so a missing 2025 means 2025 was
    never fetched -- not that 2026 is untrustworthy. Stopping at the gap would discard real data."""
    monkeypatch.setattr("setlistkit.sources.phish_net._current_year", lambda: 2026)
    api = FakeApi({2024: _envelope(_row("2024-07-01", "Llama")),
                   2026: _envelope(_row("2026-07-01", "Maze"))})
    client = _client(tmp_path, api)
    client.pull("phish", min_year=2024)
    client._cache.delete(NAMESPACE, "setlists/showyear/2025")
    assert client.cached_show_dates("phish", 2024) == {"2024-07-01", "2026-07-01"}


def test_the_phish_net_client_satisfies_the_source_contract_by_shape():
    """Locked now, before an importer registers it. The pull surface is structural -- no base
    class enforces it -- so the moment this client drifts from the keyword block every other
    source presents is the moment a registry entry would stop being a drop-in."""
    want = _keyword_only_shape(SourceClient.pull)
    assert issubclass(PhishNetClient, SourceClient)
    assert _keyword_only_shape(PhishNetClient.pull) == want
    assert SourceClient not in PhishNetClient.__mro__      # satisfied structurally, not inherited


def test_progress_is_reported_per_year(tmp_path, monkeypatch):
    monkeypatch.setattr("setlistkit.sources.phish_net._current_year", lambda: 2026)
    api = FakeApi({y: _envelope(_row(f"{y}-07-01", "Llama")) for y in (2024, 2025, 2026)})
    seen = []
    _client(tmp_path, api).pull("phish", min_year=2024, progress=lambda i, n: seen.append((i, n)))
    assert seen == [(1, 3), (2, 3), (3, 3)]

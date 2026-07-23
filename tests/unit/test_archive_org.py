# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the archive.org client (built on the polite client, driven by a fake transport)."""

import json

import pytest

from setlistkit.config import Config
from setlistkit.sources.archive_org import ArchiveOrgClient, _tracks, track_seconds
from setlistkit.sources.client import PoliteClient, Response
from setlistkit.store.raw_cache import RawCache

_UA = "famoe.ly nightly (you@example.com)"


def _config(tmp_path):
    return Config(data_root=tmp_path, user_agent=_UA, source_path=tmp_path / "slkit.toml", raw={})


class FakeTransport:
    """Serves queued responses in order and records the URLs it was asked for."""

    def __init__(self, *responses):
        self._queue = list(responses)
        self.urls = []
        self.calls = []

    def __call__(self, url, headers):
        self.urls.append(url)
        self.calls.append((url, dict(headers)))
        return self._queue.pop(0)


def _archive(tmp_path, transport):
    client = PoliteClient(_config(tmp_path), RawCache(tmp_path), namespace="archive_org",
                          transport=transport, sleeper=lambda _s: None)
    return ArchiveOrgClient(_config(tmp_path), RawCache(tmp_path), client=client)


def _json_response(payload):
    return Response(200, {"Content-Type": "application/json"}, json.dumps(payload).encode("utf-8"))


def test_metadata_fetches_and_parses(tmp_path):
    blob = {"metadata": {"title": "moe. 2026-07-04", "description": "Set 1: ..."}}
    archive = _archive(tmp_path, FakeTransport(_json_response(blob)))
    assert archive.metadata("moe2026-07-04") == blob


def test_metadata_targets_the_right_url(tmp_path):
    transport = FakeTransport(_json_response({"metadata": {}}))
    archive = _archive(tmp_path, transport)
    archive.metadata("moe2026-07-04")
    assert transport.urls == ["https://archive.org/metadata/moe2026-07-04"]


def test_metadata_missing_item_is_none(tmp_path):
    archive = _archive(tmp_path, FakeTransport(Response(404, {}, b"")))
    assert archive.metadata("nope") is None


def test_list_items_pages_until_numfound_is_exhausted(tmp_path):
    page1 = {"response": {"numFound": 3, "docs": [{"identifier": "a"}, {"identifier": "b"}]}}
    page2 = {"response": {"numFound": 3, "docs": [{"identifier": "c"}]}}
    transport = FakeTransport(_json_response(page1), _json_response(page2))
    archive = _archive(tmp_path, transport)
    docs = archive.list_items("moe")
    assert [d["identifier"] for d in docs] == ["a", "b", "c"]
    assert len(transport.urls) == 2                  # stopped once numFound was reached


def test_list_items_stops_on_an_empty_page(tmp_path):
    page1 = {"response": {"numFound": 999, "docs": [{"identifier": "a"}]}}
    page2 = {"response": {"numFound": 999, "docs": []}}
    transport = FakeTransport(_json_response(page1), _json_response(page2))
    archive = _archive(tmp_path, transport)
    docs = archive.list_items("moe")
    # an empty page ends the walk even when numFound claims there is more
    assert [d["identifier"] for d in docs] == ["a"]
    assert len(transport.urls) == 2


def test_list_items_encodes_the_collection_query(tmp_path):
    page = {"response": {"numFound": 0, "docs": []}}
    transport = FakeTransport(_json_response(page))
    archive = _archive(tmp_path, transport)
    archive.list_items("moe", min_year=2020)
    url = transport.urls[0]
    assert "advancedsearch.php" in url
    assert "collection%3Amoe" in url                 # the ':' is percent-encoded
    assert "year%3A%5B2020%20TO%209999%5D" in url    # spaces -> %20, brackets -> %5B/%5D
    assert "fl[]=identifier" in url
    # Two sort keys. addeddate alone is not unique, and a non-unique sort makes the page
    # boundaries move between requests -- see the duplicate-boundary test below.
    assert "sort[]=addeddate+asc" in url and "sort[]=identifier+asc" in url


# --- pull ------------------------------------------------------------------------------------

def _listing(*identifiers, **extra):
    docs = [{"identifier": ident, "date": "2026-07-04", "title": f"listing {ident}", **extra}
            for ident in identifiers]
    return {"response": {"numFound": len(docs), "docs": docs}}


def _meta(**metadata):
    return {"metadata": metadata, "files": []}


def test_pull_fetches_metadata_for_every_listed_item(tmp_path):
    transport = FakeTransport(_json_response(_listing("a", "b")),
                              _json_response(_meta(title="A")), _json_response(_meta(title="B")))
    result = _archive(tmp_path, transport).pull("moe")
    assert (result.listed, result.fetched, result.cached) == (2, 2, 0)
    assert transport.urls[1:] == ["https://archive.org/metadata/a",
                                  "https://archive.org/metadata/b"]


def test_pull_skips_an_item_already_cached(tmp_path):
    first = FakeTransport(_json_response(_listing("a", "b")),
                          _json_response(_meta(title="A")), _json_response(_meta(title="B")))
    _archive(tmp_path, first).pull("moe")
    # Second run: the listing is re-read (it is the freshness query), the metadata is not.
    second = FakeTransport(_json_response(_listing("a", "b", "c")), _json_response(_meta(title="C")))
    result = _archive(tmp_path, second).pull("moe")
    assert (result.listed, result.fetched, result.cached) == (3, 1, 2)
    assert second.urls[1:] == ["https://archive.org/metadata/c"]


def test_pull_force_rescan_reasks_for_everything(tmp_path):
    first = FakeTransport(_json_response(_listing("a")), _json_response(_meta(title="A")))
    _archive(tmp_path, first).pull("moe")
    second = FakeTransport(_json_response(_listing("a")), _json_response(_meta(title="A2")))
    result = _archive(tmp_path, second).pull("moe", force_rescan=True)
    assert (result.fetched, result.cached) == (1, 0)


def test_pull_reports_a_404_instead_of_caching_a_stub(tmp_path):
    transport = FakeTransport(_json_response(_listing("gone")), Response(404, {}, b""))
    archive = _archive(tmp_path, transport)
    result = archive.pull("moe")
    assert result.missing == ("gone",)
    assert result.fetched == 0
    # Nothing was cached for it, so the next pull tries again rather than believing it has the show.
    assert RawCache(tmp_path).get("archive_org", "gone") is None


def test_pull_refuses_the_placeholder_user_agent(tmp_path):
    from setlistkit.config import SENTINEL_USER_AGENT
    from setlistkit.diagnostics import DiagnosticError

    transport = FakeTransport()
    config = Config(data_root=tmp_path, user_agent=SENTINEL_USER_AGENT,
                    source_path=tmp_path / "slkit.toml", raw={})
    client = PoliteClient(config, RawCache(tmp_path), namespace="archive_org",
                          transport=transport, sleeper=lambda _s: None)
    with pytest.raises(DiagnosticError):
        ArchiveOrgClient(config, RawCache(tmp_path), client=client).pull("moe")
    assert transport.urls == []                      # refused before the listing, not after


def _page(num_found, *identifiers):
    return _json_response({"response": {"numFound": num_found,
                                        "docs": [{"identifier": i} for i in identifiers]}})


def test_pull_counts_distinct_items_when_a_page_boundary_repeats(tmp_path):
    """The stop condition must count DISTINCT items, not returned rows.

    Eight items exist. The page boundary moved between requests, so page 2 repeats c and d.
    Counting rows reaches numFound after page 2 and page 3 is never asked for -- g and h are
    silently gone and every counter says the run was clean. Counting distinct identifiers costs
    one wasted row instead of the tail of the collection.
    """
    transport = FakeTransport(_page(8, "a", "b", "c", "d"), _page(8, "c", "d", "e", "f"),
                              _page(8, "g", "h"),
                              *[_json_response(_meta()) for _ in range(8)])
    result = _archive(tmp_path, transport).pull("moe")
    assert result.listed == 8 and result.fetched == 8
    items = _archive(tmp_path, FakeTransport()).cached_items("moe").items
    assert [item["identifier"] for item in items] == list("abcdefgh")


def test_pull_does_not_hand_the_parser_the_same_show_twice(tmp_path):
    transport = FakeTransport(_page(3, "a", "b"), _page(3, "b", "c"),
                              *[_json_response(_meta()) for _ in range(3)])
    _archive(tmp_path, transport).pull("moe")
    items = _archive(tmp_path, FakeTransport()).cached_items("moe").items
    assert [item["identifier"] for item in items] == ["a", "b", "c"]


def test_every_listed_item_lands_in_exactly_one_counter(tmp_path):
    """listed == fetched + cached + missing + unidentified, with no item falling through.

    The counters are the only thing that would notice a show going past unfetched, so an item
    that is in none of them is an item nobody misses.
    """
    listing = {"response": {"numFound": 4, "docs": [
        {"identifier": "have"}, {"identifier": "new"}, {"identifier": "gone"},
        {"date": "2026-07-04"},                      # no identifier at all
    ]}}
    RawCache(tmp_path).put("archive_org", "have", json.dumps(_meta()).encode("utf-8"))
    transport = FakeTransport(_json_response(listing), _page(4),
                              _json_response(_meta()), Response(404, {}, b""))
    result = _archive(tmp_path, transport).pull("moe")
    assert result.listed == 3 and result.unidentified == 1
    assert result.fetched + result.cached + len(result.missing) == result.listed
    assert result.planned == result.fetched + len(result.missing) + len(result.failed)


def test_pull_reports_a_listing_the_paging_backstop_cut_short(tmp_path):
    # Every page claims there is far more to come, so the walk only ends when the backstop
    # fires. The item is pre-cached so this exercises the paging and nothing else.
    RawCache(tmp_path).put("archive_org", "a", json.dumps(_meta()).encode("utf-8"))
    page = {"response": {"numFound": 99999, "docs": [{"identifier": "a"}]}}
    transport = FakeTransport(*[_json_response(page) for _ in range(40)])
    result = _archive(tmp_path, transport).pull("moe")
    assert result.truncated is True
    assert len(transport.urls) == 40                 # a backstop against a runaway loop


def test_pull_survives_a_listing_the_search_api_does_not_have(tmp_path):
    result = _archive(tmp_path, FakeTransport(Response(404, {}, b""))).pull("moe")
    assert (result.listed, result.fetched, result.truncated) == (0, 0, False)


def test_pull_counts_a_listing_doc_with_no_identifier_instead_of_dropping_it(tmp_path):
    listing = {"response": {"numFound": 2, "docs": [{"identifier": "a"}, {"date": "2026-07-04"}]}}
    transport = FakeTransport(_json_response(listing), _page(2), _json_response(_meta()))
    result = _archive(tmp_path, transport).pull("moe")
    # Unfetchable, so it is not in `listed`; counted, so it is not nowhere.
    assert (result.listed, result.fetched, result.unidentified) == (1, 1, 1)


def test_pull_identifies_its_run_and_its_progress_to_the_host(tmp_path):
    """The listing phase has no denominator; the item phase counts only what will be requested.

    An item already cached costs the host nothing, so it is not in the total. A denominator that
    counted it would promise more traffic than the run is going to send.
    """
    RawCache(tmp_path).put("archive_org", "have", json.dumps(_meta()).encode("utf-8"))
    transport = FakeTransport(_page(2, "have", "new"), _json_response(_meta()))
    seen = []
    archive = _archive(tmp_path, transport)
    archive.pull("moe", announce=seen.append)

    assert len(seen) == 1 and len(seen[0]) == 8            # the id, announced to our side too
    agents = [headers["User-Agent"] for _url, headers in transport.calls]
    assert agents[0] == f"{_UA} (batch {seen[0]}; listing 1)"
    assert agents[1] == f"{_UA} (batch {seen[0]}; item 1/1)"


def test_one_bad_tape_does_not_cost_a_run_that_is_hours_deep(tmp_path):
    """Found in production: a single 502 killed a 4614-item pull 1923 items in.

    502 is retried now, but a permanent failure must not abort either -- the item is recorded and
    the other four thousand still get fetched.
    """
    # 500 is definitive, not transient, so it costs one request and is not retried.
    transport = FakeTransport(_page(3, "a", "bad", "c"),
                              _json_response(_meta()),
                              Response(500, {}, b""),
                              _json_response(_meta()))
    result = _archive(tmp_path, transport).pull("moe")
    assert result.failed == ("bad",)
    assert result.fetched == 2                       # a and c still landed
    assert RawCache(tmp_path).has("archive_org", "c")
    # Nothing cached for the bad one, so the next pull retries it exactly like a 404.
    assert not RawCache(tmp_path).has("archive_org", "bad")


def test_a_run_of_failures_stops_asking(tmp_path):
    """Resilience past this point is thousands of requests at a host already answering badly,
    which is the one behavior the etiquette rules exist to forbid."""
    from setlistkit.sources.client import SourceError
    transport = FakeTransport(_page(20, *[f"i{n}" for n in range(20)]),
                              *[Response(500, {}, b"") for _ in range(200)])
    with pytest.raises(SourceError):
        _archive(tmp_path, transport).pull("moe")
    # Five items attempted, not twenty. Each already exhausted its own retries first.
    metadata_requests = [url for url in transport.urls if "/metadata/" in url]
    assert len(set(metadata_requests)) == 5


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_a_transient_status_is_retried_rather_than_fatal(tmp_path, status):
    """502 and 504 are a gateway answering for a host it could not reach. Neither is a fact
    about the item, and a large host emits both as ordinary weather."""
    transport = FakeTransport(_page(1, "a"), Response(status, {}, b""), _json_response(_meta()))
    result = _archive(tmp_path, transport).pull("moe")
    assert (result.fetched, result.failed) == (1, ())


def test_a_dry_pull_lists_and_then_stops(tmp_path):
    """The listing goes out; not one item request does. That is the expensive half by three
    orders of magnitude, and it is the half a dry run exists to avoid spending."""
    RawCache(tmp_path).put("archive_org", "have", json.dumps(_meta()).encode("utf-8"))
    transport = FakeTransport(_page(3, "have", "new", "newer"))
    result = _archive(tmp_path, transport).pull("moe", dry_run=True)

    assert (result.listed, result.cached, result.planned) == (3, 1, 2)
    assert (result.fetched, result.missing) == (0, ())
    assert transport.urls == [transport.urls[0]]           # the listing, and nothing else
    assert "advancedsearch" in transport.urls[0]
    # And nothing was written, so a real run afterwards still has everything to do.
    assert RawCache(tmp_path).get("archive_org", "new") is None


def test_a_dry_pull_and_the_real_one_agree_about_what_is_left_to_do(tmp_path):
    RawCache(tmp_path).put("archive_org", "have", json.dumps(_meta()).encode("utf-8"))
    dry = _archive(tmp_path, FakeTransport(_page(3, "have", "new", "newer"))).pull(
        "moe", dry_run=True)
    real = _archive(tmp_path, FakeTransport(_page(3, "have", "new", "newer"),
                                            _json_response(_meta()),
                                            _json_response(_meta()))).pull("moe")
    assert dry.planned == real.planned == real.fetched + len(real.missing)


def test_pull_reports_progress(tmp_path):
    transport = FakeTransport(_json_response(_listing("a", "b")),
                              _json_response(_meta()), _json_response(_meta()))
    seen = []
    _archive(tmp_path, transport).pull("moe", progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 2), (2, 2)]


# --- cached_items ----------------------------------------------------------------------------

def test_cached_items_reassembles_without_touching_the_network(tmp_path):
    blob = _meta(title="moe. Live at Northlands", date="2026-07-04T00:00:00Z",
                 description="Set 1: Rebubula", venue="Northlands", coverage="Swanzey, NH")
    transport = FakeTransport(_json_response(_listing("a")), _json_response(blob))
    _archive(tmp_path, transport).pull("moe")

    empty = FakeTransport()                          # any request at all pops from an empty queue
    cached = _archive(tmp_path, empty).cached_items("moe")
    assert empty.urls == []
    item, = cached.items
    assert item["identifier"] == "a"
    assert item["title"] == "moe. Live at Northlands"
    assert item["meta_date"] == "2026-07-04T00:00:00Z"
    assert item["date"] == "2026-07-04"              # the listing half, absent from the metadata
    assert item["list_title"] == "listing a"
    assert item["venue"] == "Northlands"
    assert item["coverage"] == "Swanzey, NH"
    assert item["description"] == "Set 1: Rebubula"


def test_cached_items_reports_a_listed_item_with_no_cached_metadata(tmp_path):
    # A half-finished pull: the listing landed, one item's metadata did not.
    transport = FakeTransport(_json_response(_listing("a", "b")), _json_response(_meta(title="A")),
                              Response(404, {}, b""))
    _archive(tmp_path, transport).pull("moe")
    cached = _archive(tmp_path, FakeTransport()).cached_items("moe")
    assert [item["identifier"] for item in cached.items] == ["a"]
    assert cached.absent == ("b",)


def test_cached_items_is_sorted_by_identifier(tmp_path):
    transport = FakeTransport(_json_response(_listing("c", "a", "b")),
                              *[_json_response(_meta()) for _ in range(3)])
    _archive(tmp_path, transport).pull("moe")
    cached = _archive(tmp_path, FakeTransport()).cached_items("moe")
    assert [item["identifier"] for item in cached.items] == ["a", "b", "c"]


def test_cached_items_is_empty_before_any_pull(tmp_path):
    cached = _archive(tmp_path, FakeTransport()).cached_items("moe")
    assert cached.items == [] and cached.absent == ()


def test_cached_items_joins_a_list_description(tmp_path):
    # archive.org returns `description` as a list when the uploader used multiple fields.
    blob = {"metadata": {"description": ["Set 1: Rebubula", "Encore: ATL"]}, "files": []}
    transport = FakeTransport(_json_response(_listing("a")), _json_response(blob))
    _archive(tmp_path, transport).pull("moe")
    item, = _archive(tmp_path, FakeTransport()).cached_items("moe").items
    assert item["description"] == "Set 1: Rebubula Encore: ATL"


def _cache_page(tmp_path, page, *identifiers, num_found=None):
    response = {"docs": [{"identifier": ident} for ident in identifiers]}
    if num_found is not None:
        response["numFound"] = num_found
    RawCache(tmp_path).put("archive_org", f"advancedsearch/moe/None/p{page}",
                           json.dumps({"response": response}).encode("utf-8"))


def test_cached_items_walks_every_cached_listing_page(tmp_path):
    _cache_page(tmp_path, 1, "a")
    _cache_page(tmp_path, 2, "b")
    for ident in ("a", "b"):
        RawCache(tmp_path).put("archive_org", ident,
                               json.dumps(_meta(title=ident.upper())).encode("utf-8"))
    cached = _archive(tmp_path, FakeTransport()).cached_items("moe")
    assert [item["title"] for item in cached.items] == ["A", "B"]


def test_cached_items_stops_at_an_empty_page(tmp_path):
    _cache_page(tmp_path, 1, "a")
    _cache_page(tmp_path, 2)
    RawCache(tmp_path).put("archive_org", "a", json.dumps(_meta()).encode("utf-8"))
    cached = _archive(tmp_path, FakeTransport()).cached_items("moe")
    assert [item["identifier"] for item in cached.items] == ["a"]
    assert cached.pages == 2 and cached.truncated is False


def test_pull_deletes_listing_pages_the_collection_no_longer_fills(tmp_path):
    """A shrunken collection must not leave its old pages behind for ingest to read.

    Nothing overwrites page 2 once the listing fits on page 1, and the pull reporting cannot
    see it: it counts what it fetched, not what it left in the cache. So `cached_items` would
    keep handing the parser shows the collection no longer lists.
    """
    transport = FakeTransport(_page(3, "a", "b"), _page(3, "c"),
                              *[_json_response(_meta()) for _ in range(3)])
    _archive(tmp_path, transport).pull("moe")
    assert RawCache(tmp_path).has("archive_org", "advancedsearch/moe/None/p2")

    _archive(tmp_path, FakeTransport(_page(1, "a"))).pull("moe")
    assert not RawCache(tmp_path).has("archive_org", "advancedsearch/moe/None/p2")
    items = _archive(tmp_path, FakeTransport()).cached_items("moe").items
    assert [item["identifier"] for item in items] == ["a"]


def test_cached_items_distinguishes_no_pull_from_an_empty_collection(tmp_path):
    """`pages` is what separates the two, and they are not the same fact.

    Without it, an ingest reading a full cache under the wrong min_year gets the identical
    value an actually empty cache returns, publishes nothing, and reports success.
    """
    transport = FakeTransport(_page(1, "a"), _json_response(_meta()))
    _archive(tmp_path, transport).pull("moe", min_year=2020)

    never_pulled = _archive(tmp_path, FakeTransport()).cached_items("moe")   # min_year=None
    assert never_pulled.items == [] and never_pulled.pages == 0

    pulled = _archive(tmp_path, FakeTransport()).cached_items("moe", min_year=2020)
    assert len(pulled.items) == 1 and pulled.pages == 1


def test_cached_items_carries_the_count_the_listing_promised(tmp_path):
    # `expected` travels with the corpus, so a half-finished pull is still detectable days later.
    _cache_page(tmp_path, 1, "a", "b", num_found=9)
    RawCache(tmp_path).put("archive_org", "a", json.dumps(_meta()).encode("utf-8"))
    cached = _archive(tmp_path, FakeTransport()).cached_items("moe")
    assert cached.expected == 9
    assert len(cached.items) == 1 and cached.absent == ("b",)


def test_cached_items_tolerates_a_metadata_blob_of_the_wrong_shape(tmp_path):
    _cache_page(tmp_path, 1, "a")
    RawCache(tmp_path).put("archive_org", "a", b'{"metadata": ["not an object"], "files": []}')
    item, = _archive(tmp_path, FakeTransport()).cached_items("moe").items
    # Valid JSON of an unexpected shape parses to an item with nothing in it, not a TypeError.
    assert item["identifier"] == "a" and item["description"] == ""


def test_cached_items_ignores_a_truncated_payload(tmp_path):
    RawCache(tmp_path).put("archive_org", "advancedsearch/moe/None/p1",
                           json.dumps(_listing("a")).encode("utf-8"))
    RawCache(tmp_path).put("archive_org", "a", b'{"metadata": {"title": "half a')
    cached = _archive(tmp_path, FakeTransport()).cached_items("moe")
    # Half a JSON document is a truncated write, not a fact about the show: absent, not parsed.
    assert cached.items == [] and cached.absent == ("a",)


# --- tracklists ------------------------------------------------------------------------------

def test_tracks_take_the_densest_audio_format(tmp_path):
    blob = {"metadata": {}, "files": [
        {"format": "VBR MP3", "track": "1", "title": "Rebubula", "name": "01.mp3"},
        {"format": "VBR MP3", "track": "2", "title": "ATL", "name": "02.mp3"},
        {"format": "Flac", "track": "1", "title": "Rebubula + ATL", "name": "d1.flac"},
        {"format": "JPEG", "name": "cover.jpg"},     # not audio; never a track
    ]}
    transport = FakeTransport(_json_response(_listing("a")), _json_response(blob))
    _archive(tmp_path, transport).pull("moe")
    item, = _archive(tmp_path, FakeTransport()).cached_items("moe").items
    assert [t["title"] for t in item["tracks"]] == ["Rebubula", "ATL"]


def test_tracks_break_a_density_tie_on_format_name(tmp_path):
    # The LOSING format is listed first, deliberately. dict insertion order puts it first in
    # by_format, and max() returns the first maximal element on a tie -- so without the format
    # name in the sort key this returns "from flac", and the test would pass either way.
    blob = {"metadata": {}, "files": [
        {"format": "Flac", "title": "from flac"},
        {"format": "VBR MP3", "title": "from mp3"},
    ]}
    transport = FakeTransport(_json_response(_listing("a")), _json_response(blob))
    _archive(tmp_path, transport).pull("moe")
    item, = _archive(tmp_path, FakeTransport()).cached_items("moe").items
    # "VBR MP3" > "Flac": an arbitrary winner, but the SAME one on every pull, which is the point.
    assert [t["title"] for t in item["tracks"]] == ["from mp3"]


def test_a_repeated_metadata_field_is_joined_not_stringified(tmp_path):
    # archive.org returns ANY repeated field as a list, not just description.
    blob = {"metadata": {"venue": ["Northlands", "Main Stage"], "title": ["moe. Live at X"]},
            "files": []}
    transport = FakeTransport(_json_response(_listing("a")), _json_response(blob))
    _archive(tmp_path, transport).pull("moe")
    item, = _archive(tmp_path, FakeTransport()).cached_items("moe").items
    assert item["venue"] == "Northlands Main Stage"     # not "['Northlands', 'Main Stage']"
    assert item["title"] == "moe. Live at X"


def test_tracks_are_empty_when_no_audio_file_is_listed(tmp_path):
    blob = {"metadata": {}, "files": [{"format": "JPEG", "name": "cover.jpg"}]}
    transport = FakeTransport(_json_response(_listing("a")), _json_response(blob))
    _archive(tmp_path, transport).pull("moe")
    item, = _archive(tmp_path, FakeTransport()).cached_items("moe").items
    assert item["tracks"] == []


def test_tracks_keep_the_length_archive_org_gives_us():
    """The metadata already carries it and we were dropping it on the floor.

    Every durations question downstream is answered by this field, and the alternative to
    keeping it is 4,614 more requests for a payload already sitting in the cache.
    """
    payload = {"files": [
        {"name": "d1t01.flac", "format": "Flac", "track": "01", "title": "Aurora",
         "length": "575.47"},
        {"name": "d1t02.flac", "format": "Flac", "track": "02", "title": "Wormhole",
         "length": "1103.02"},
    ]}
    tracks = _tracks(payload["files"])
    assert [t["length"] for t in tracks] == ["575.47", "1103.02"]


def test_a_track_with_no_length_gets_an_empty_string_not_a_missing_key():
    """Absent and malformed are different, and a missing key is neither -- it is a KeyError."""
    tracks = _tracks([{"name": "d1t01.flac", "format": "Flac", "track": "01", "title": "Aurora"}])
    assert tracks[0]["length"] == ""


def test_keeping_length_does_not_change_which_format_wins():
    """The tracklist decides 63 of the corpus's shows. This change must not touch it."""
    files = [{"name": "a.flac", "format": "Flac", "track": "01", "title": "Aurora"},
             {"name": "a.mp3", "format": "VBR MP3", "track": "01", "title": "Aurora"},
             {"name": "b.mp3", "format": "VBR MP3", "track": "02", "title": "Wormhole"}]
    # VBR MP3 has more files, so it wins, exactly as before.
    assert [t["name"] for t in _tracks(files)] == ["a.mp3", "b.mp3"]


def test_track_seconds_reads_float_seconds():
    assert track_seconds("575.47") == 575.47


def test_track_seconds_reads_minutes_and_seconds():
    """FLAC carries float seconds, VBR MP3 carries MM:SS. Both are in the corpus."""
    assert track_seconds("09:35") == 575.0


def test_track_seconds_reads_hours_minutes_and_seconds():
    assert track_seconds("1:02:03") == 3723.0


def test_track_seconds_returns_none_for_anything_it_cannot_read():
    for value in ("", "   ", "unknown", None, "1:2:3:4"):
        assert track_seconds(value) is None


def test_track_seconds_rejects_a_negative_length():
    """A negative duration is not a short song, it is a broken record."""
    assert track_seconds("-5") is None

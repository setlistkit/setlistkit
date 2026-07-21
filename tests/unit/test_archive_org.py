# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the archive.org client (built on the polite client, driven by a fake transport)."""

import json

from setlistkit.config import Config
from setlistkit.sources.archive_org import ArchiveOrgClient
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

    def __call__(self, url, headers):
        self.urls.append(url)
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

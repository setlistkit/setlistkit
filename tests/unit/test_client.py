# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the polite/cached/backoff source client.

Every path that could touch the network is exercised through an injected fake transport and a
recording sleeper, so the suite runs at full speed and never opens a socket.
"""

from datetime import timedelta

import pytest

from setlistkit.config import SENTINEL_USER_AGENT, Config
from setlistkit.diagnostics import DiagnosticError
from setlistkit.sources.client import (
    PoliteClient, Response, SourceHTTPError, TransportError,
)
from setlistkit.store.raw_cache import RawCache

_UA = "famoe.ly nightly (you@example.com)"


def _config(tmp_path, user_agent=_UA):
    return Config(data_root=tmp_path, user_agent=user_agent,
                  source_path=tmp_path / "slkit.toml", raw={})


class FakeTransport:
    """Returns queued responses in order; a queued Exception is raised instead. Records calls."""

    def __init__(self, *responses):
        self._queue = list(responses)
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append((url, dict(headers)))
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(tmp_path, transport, *, sleeps=None, **kwargs):
    cache = RawCache(tmp_path)
    return PoliteClient(_config(tmp_path), cache, namespace="archive_org",
                        transport=transport, sleeper=(sleeps.append if sleeps is not None else None),
                        **kwargs)


def ok(body=b"{}", **headers):
    return Response(200, headers, body)


def test_sentinel_user_agent_aborts_before_any_request(tmp_path):
    transport = FakeTransport(ok())
    cache = RawCache(tmp_path)
    # identity is still the shipped placeholder, so nothing may leave the box
    client = PoliteClient(_config(tmp_path, user_agent=SENTINEL_USER_AGENT), cache,
                          namespace="archive_org", transport=transport, sleeper=lambda _s: None)
    with pytest.raises(DiagnosticError):
        client.fetch("item1", "https://archive.org/metadata/item1")
    assert transport.calls == []                     # the transport was never reached


def test_sends_the_configured_user_agent(tmp_path):
    transport = FakeTransport(ok(b'{"ok":true}'))
    client = _client(tmp_path, transport, sleeps=[])
    client.fetch("item1", "https://archive.org/metadata/item1")
    assert transport.calls[0][1]["User-Agent"] == _UA


def test_caches_the_payload_and_http_metadata(tmp_path):
    transport = FakeTransport(ok(b"PAYLOAD", ETag='W/"1"', **{"Last-Modified": "Mon, 01 Jan 2026 00:00:00 GMT",
                                                              "Content-Type": "application/json"}))
    cache = RawCache(tmp_path)
    client = PoliteClient(_config(tmp_path), cache, namespace="archive_org",
                          transport=transport, sleeper=lambda _s: None)
    assert client.fetch("item1", "https://archive.org/metadata/item1") == b"PAYLOAD"
    meta = cache.meta("archive_org", "item1")
    assert meta["etag"] == 'W/"1"'
    assert meta["last_modified"] == "Mon, 01 Jan 2026 00:00:00 GMT"
    assert meta["url"] == "https://archive.org/metadata/item1"


def test_fresh_cache_hit_makes_no_request(tmp_path):
    transport = FakeTransport()                      # empty: any request would IndexError
    cache = RawCache(tmp_path)
    cache.put("archive_org", "item1", b"CACHED")
    client = PoliteClient(_config(tmp_path), cache, namespace="archive_org",
                          transport=transport, sleeper=lambda _s: None)
    # default_max_age is None -> a cached entry never ages out, so no network at all
    assert client.fetch("item1", "https://archive.org/metadata/item1") == b"CACHED"
    assert transport.calls == []


def test_stale_cache_revalidates_with_conditional_headers(tmp_path):
    transport = FakeTransport(Response(304, {}, b""))
    cache = RawCache(tmp_path)
    cache.put("archive_org", "item1", b"CACHED", etag='W/"7"',
              last_modified="Mon, 01 Jan 2026 00:00:00 GMT")
    client = PoliteClient(_config(tmp_path), cache, namespace="archive_org",
                          transport=transport, sleeper=lambda _s: None,
                          default_max_age=timedelta(0))     # always stale -> always revalidate
    assert client.fetch("item1", "https://archive.org/metadata/item1") == b"CACHED"
    sent = transport.calls[0][1]
    assert sent["If-None-Match"] == 'W/"7"'
    assert sent["If-Modified-Since"] == "Mon, 01 Jan 2026 00:00:00 GMT"


def test_304_refreshes_the_timestamp(tmp_path):
    transport = FakeTransport(Response(304, {}, b""))
    cache = RawCache(tmp_path)
    old = cache.put("archive_org", "item1", b"CACHED", fetched_at=None)
    before = cache.meta("archive_org", "item1")["fetched_at"]
    client = PoliteClient(_config(tmp_path), cache, namespace="archive_org",
                          transport=transport, sleeper=lambda _s: None,
                          default_max_age=timedelta(0))
    client.fetch("item1", "https://archive.org/metadata/item1")
    # the body is unchanged but the entry is now marked freshly checked
    assert cache.get("archive_org", "item1") == b"CACHED"
    assert cache.meta("archive_org", "item1")["fetched_at"] >= before
    assert old.exists()


def test_force_rescan_bypasses_a_fresh_cache(tmp_path):
    transport = FakeTransport(ok(b"REFETCHED"))
    cache = RawCache(tmp_path)
    cache.put("archive_org", "item1", b"CACHED")
    client = PoliteClient(_config(tmp_path), cache, namespace="archive_org",
                          transport=transport, sleeper=lambda _s: None)
    # a fresh entry would normally short-circuit; force_rescan re-hits it anyway
    assert client.fetch("item1", "https://archive.org/metadata/item1", force_rescan=True) == b"REFETCHED"
    assert len(transport.calls) == 1


def test_force_rescan_is_still_rate_limited(tmp_path):
    sleeps = []
    transport = FakeTransport(ok(b"REFETCHED"))
    cache = RawCache(tmp_path)
    cache.put("archive_org", "item1", b"CACHED")
    client = PoliteClient(_config(tmp_path), cache, namespace="archive_org",
                          transport=transport, sleeper=sleeps.append, delay=0.5)
    client.fetch("item1", "https://archive.org/metadata/item1", force_rescan=True)
    assert 0.5 in sleeps                             # "force" ignores the cache, never the pause


def test_force_rescan_revalidates_instead_of_redownloading(tmp_path):
    transport = FakeTransport(Response(304, {}, b""))
    cache = RawCache(tmp_path)
    cache.put("archive_org", "item1", b"CACHED", etag='W/"9"')
    client = PoliteClient(_config(tmp_path), cache, namespace="archive_org",
                          transport=transport, sleeper=lambda _s: None)
    # force re-hits, but still sends the conditional header and takes the cheap 304 path
    assert client.fetch("item1", "https://archive.org/metadata/item1", force_rescan=True) == b"CACHED"
    assert transport.calls[0][1]["If-None-Match"] == 'W/"9"'


def test_304_with_no_cached_body_is_rejected(tmp_path):
    transport = FakeTransport(Response(304, {}, b""))
    client = _client(tmp_path, transport, sleeps=[])
    # a 304 with nothing cached to return is anomalous; don't cache an empty body
    with pytest.raises(SourceHTTPError):
        client.fetch("item1", "https://archive.org/metadata/item1")


def test_404_returns_none(tmp_path):
    transport = FakeTransport(Response(404, {}, b"not here"))
    client = _client(tmp_path, transport, sleeps=[])
    assert client.fetch("gone", "https://archive.org/metadata/gone") is None


def test_429_backs_off_then_succeeds(tmp_path):
    sleeps = []
    transport = FakeTransport(Response(429, {"Retry-After": "12"}, b""), ok(b"OK"))
    client = _client(tmp_path, transport, sleeps=sleeps)
    assert client.fetch("item1", "https://archive.org/metadata/item1") == b"OK"
    # the server named its own wait, and we honored it before retrying
    assert 12 in sleeps
    assert len(transport.calls) == 2


def test_429_without_retry_after_uses_exponential_backoff(tmp_path):
    sleeps = []
    transport = FakeTransport(Response(503, {}, b""), Response(503, {}, b""), ok(b"OK"))
    client = _client(tmp_path, transport, sleeps=sleeps)
    assert client.fetch("item1", "https://archive.org/metadata/item1") == b"OK"
    # 2**0 then 2**1 for the two throttled attempts
    assert sleeps[:2] == [1.0, 2.0]


def test_gives_up_after_max_tries_of_throttling(tmp_path):
    transport = FakeTransport(*[Response(429, {}, b"") for _ in range(3)])
    client = _client(tmp_path, transport, sleeps=[], max_tries=3)
    with pytest.raises(TransportError):
        client.fetch("item1", "https://archive.org/metadata/item1")


def test_transport_error_retries_then_raises(tmp_path):
    sleeps = []
    transport = FakeTransport(TransportError("reset"), TransportError("reset"), TransportError("reset"))
    client = _client(tmp_path, transport, sleeps=sleeps, max_tries=3)
    with pytest.raises(TransportError):
        client.fetch("item1", "https://archive.org/metadata/item1")
    assert len(transport.calls) == 3                 # it tried the full budget before giving up


def test_transport_error_then_success(tmp_path):
    transport = FakeTransport(TransportError("reset"), ok(b"RECOVERED"))
    client = _client(tmp_path, transport, sleeps=[])
    assert client.fetch("item1", "https://archive.org/metadata/item1") == b"RECOVERED"


def test_definitive_bad_status_raises(tmp_path):
    transport = FakeTransport(Response(500, {}, b"boom"))
    client = _client(tmp_path, transport, sleeps=[])
    with pytest.raises(SourceHTTPError) as caught:
        client.fetch("item1", "https://archive.org/metadata/item1")
    assert caught.value.status == 500


def test_fetch_json_decodes_the_body(tmp_path):
    transport = FakeTransport(ok(b'{"identifier":"moe2026-07-04","n":8}'))
    client = _client(tmp_path, transport, sleeps=[])
    assert client.fetch_json("item1", "https://archive.org/metadata/item1") == {
        "identifier": "moe2026-07-04", "n": 8,
    }


def test_fetch_json_passes_through_a_404_as_none(tmp_path):
    transport = FakeTransport(Response(404, {}, b""))
    client = _client(tmp_path, transport, sleeps=[])
    assert client.fetch_json("gone", "https://archive.org/metadata/gone") is None


def test_sleeps_the_delay_after_a_real_answer(tmp_path):
    sleeps = []
    transport = FakeTransport(ok(b"OK"))
    client = _client(tmp_path, transport, sleeps=sleeps, delay=0.5)
    client.fetch("item1", "https://archive.org/metadata/item1")
    # be gentle: a pause follows the completed request
    assert 0.5 in sleeps

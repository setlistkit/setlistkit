# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""A polite HTTP client every source inherits: identify, cache, slow down, back off.

The old repo learned this discipline against the Internet Archive after a scraper was deleted
here for spoofing a browser. setlistkit makes it a rule for every source, not a habit of one
client. Four things this client does, in the order they matter:

1. **Identify, or do not run.** Before the first byte leaves, :func:`require_network_identity`
   aborts if ``user_agent`` is still the shipped sentinel. The identity belongs to whoever runs
   the deployment; nothing here disguises itself as a browser or claims to be the project.
2. **Cache.** A cached entry younger than ``max_age`` is returned without a request, so a re-run
   costs the source nothing. ``force_rescan`` is the only path that re-hits data already held --
   and it revalidates rather than blindly re-downloads.
3. **Slow down.** Every path that touches the network sleeps ``delay`` afterwards. "Force" means
   "ignore the cache", never "hammer the host".
4. **Back off.** A 429 or 503 is the server asking us to wait; we honor its ``Retry-After`` and
   otherwise back off exponentially, up to ``max_tries``.

The transport is injectable so the whole client is tested without a socket. The default is a
thin :mod:`urllib` wrapper -- no third-party HTTP dependency, keeping core headless.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import timedelta

from ..config import Config, require_network_identity
from ..store.raw_cache import RawCache

DEFAULT_DELAY = 0.75          # seconds after each request; archive.org bots.html asks for delays
DEFAULT_MAX_TRIES = 5
_BACKOFF_CAP = 60.0           # never sleep more than this on one backoff
_RETRY_STATUSES = frozenset({429, 503})   # "slow down", not "this failed" -- back off and retry


class SourceError(Exception):
    """Any way an upstream source can fail us: no connection, a bad status, an unusable body.

    One base class so the CLI can catch the lot at its boundary and render a diagnostic. A
    source failing is an ordinary Tuesday -- archive.org goes down, a proxy returns an error
    page -- and it should exit with a message, not a traceback.
    """


class TransportError(SourceError):
    """A network-level failure (timeout, refused, reset), distinct from an HTTP status.

    These are retried blindly with exponential backoff. An HTTP status like 429 is a deliberate
    answer from the server and gets handled on its own terms, not treated as a transport fault.
    """


class SourceHTTPError(SourceError):
    """A definitive, non-retryable bad HTTP status (403, 500, ...) from a source."""

    def __init__(self, url: str, status: int) -> None:
        self.url = url
        self.status = status
        super().__init__(f"{url} returned HTTP {status}")


class SourceFormatError(SourceError):
    """A 200 whose body is not what the source promised.

    Its own class because it is the failure a status code cannot describe: a site under
    maintenance, or behind a captive portal or a proxy, serves an HTML page with status 200.
    Every layer above expects JSON, so without this the first thing to notice is
    ``json.JSONDecodeError`` escaping to the top of the process.
    """

    def __init__(self, url: str, detail: str) -> None:
        self.url = url
        super().__init__(f"{url} did not return usable JSON: {detail}")


@dataclass(frozen=True)
class Response:
    """One completed HTTP exchange: whatever the server actually said, 404 and 429 included."""

    status: int
    headers: dict[str, str]
    body: bytes


def _header(headers: dict[str, str], name: str) -> str | None:
    """Case-insensitive header lookup (HTTP header names are not case-sensitive)."""
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def urllib_transport(url: str, headers: dict[str, str], *, timeout: float = 60) -> Response:
    """The default transport: a GET via :mod:`urllib`, returning whatever status came back.

    An HTTP error status (4xx/5xx) is a completed exchange, so it comes back as a
    :class:`Response`, not an exception -- the client decides whether 404 means "absent" and
    whether 429 means "retry". A genuine network failure raises :class:`TransportError`.
    """
    request = urllib.request.Request(url, headers=headers)
    # urlopen flags non-http schemes (file://, ...); every source URL here is a hard-coded
    # https endpoint built in this package, never user input, so the scheme cannot be steered.
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # nosec B310
            return Response(resp.status, dict(resp.headers.items()), resp.read())
    except urllib.error.HTTPError as err:
        body = err.read() if hasattr(err, "read") else b""
        return Response(err.code, dict((err.headers or {}).items()), body)
    except (urllib.error.URLError, OSError) as err:
        raise TransportError(f"{url}: {err}") from err


class PoliteClient:
    """A cached, rate-limited, backing-off GET client for one source namespace.

    ``namespace`` scopes the cache (an archive.org identifier and a setlist.fm id can be the
    same string). ``transport`` and ``sleeper`` are injectable so tests exercise every retry and
    backoff path without a socket or a real pause. ``default_max_age`` of ``None`` means a cached
    entry never ages out on its own -- right for immutable-ish records like show metadata.
    """

    def __init__(self, config: Config, cache: RawCache, *, namespace: str,
                 transport=None, sleeper=None, delay: float = DEFAULT_DELAY,
                 max_tries: int = DEFAULT_MAX_TRIES, default_max_age: timedelta | None = None) -> None:
        self._config = config
        self._cache = cache
        self._namespace = namespace
        self._transport = transport or urllib_transport
        self._sleep = sleeper or time.sleep
        self._delay = delay
        self._max_tries = max_tries
        self._default_max_age = default_max_age

    def fetch(self, key: str, url: str, *, force_rescan: bool = False,
              max_age: timedelta | None = None, accept: str = "application/json") -> bytes | None:
        """Return the payload for ``url``, cached and revalidated. ``None`` when the source 404s.

        A cached entry younger than ``max_age`` is returned without a request. Past that, or with
        ``force_rescan``, a conditional GET revalidates: a 304 refreshes the timestamp and returns
        the cached bytes, so "force" re-checks without re-downloading data that has not changed.
        """
        require_network_identity(self._config)   # sentinel-abort before anything leaves the box
        if max_age is None:
            max_age = self._default_max_age
        cached = self._cache.get(self._namespace, key)
        if cached is not None and not force_rescan and self._fresh(key, max_age):
            return cached

        headers = {"User-Agent": self._config.user_agent, "Accept": accept}
        self._add_conditional(key, headers, cached)
        resp = self._request(url, headers)
        if resp is None:
            return None                          # 404: the source does not have this item
        if resp.status == 304:
            if cached is None:
                # 304 with nothing to fall back on: the body is empty and we have no prior copy,
                # so caching resp.body would poison the entry with b"". Treat it as a bad status.
                raise SourceHTTPError(url, 304)
            self._touch(key)
            return cached
        self._store(key, url, resp)
        return resp.body

    def fetch_json(self, key: str, url: str, **kwargs):
        """Fetch and JSON-decode in one step. ``None`` when the source 404s (see :meth:`fetch`).

        Most sources speak JSON, so decoding lives here rather than being repeated at every call
        site. ``replace`` on decode keeps one stray non-UTF-8 byte from sinking a whole response.

        A body that will not decode is a :class:`SourceFormatError`, not a raw
        ``json.JSONDecodeError``: this is the likeliest real failure of the two, because a site
        under maintenance answers 200 with an HTML page and nothing in the status says so.
        """
        raw = self.fetch(key, url, **kwargs)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError as err:
            raise SourceFormatError(url, str(err)) from err

    def _fresh(self, key: str, max_age: timedelta | None) -> bool:
        """Is the cached entry young enough to skip a request? ``None`` max_age never ages out."""
        if max_age is None:
            return True
        age = self._cache.age(self._namespace, key)
        return age is not None and age <= max_age

    def _add_conditional(self, key: str, headers: dict[str, str], cached: bytes | None) -> None:
        """Attach If-None-Match / If-Modified-Since from the cached sidecar, when we have them.

        This is what makes a revalidation cheap: the server answers 304 and no bytes move.
        """
        if cached is None:
            return
        meta = self._cache.meta(self._namespace, key) or {}
        if meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]
        if meta.get("last_modified"):
            headers["If-Modified-Since"] = meta["last_modified"]

    def _request(self, url: str, headers: dict[str, str]) -> Response | None:
        """Do the GET with backoff. ``None`` for a 404; a Response for anything else usable.

        429/503 back off and retry (honoring Retry-After); a transport failure retries with
        exponential backoff; any other 4xx/5xx is a definitive :class:`SourceHTTPError`.
        """
        for attempt in range(self._max_tries):
            try:
                resp = self._transport(url, headers)
            except TransportError:
                if attempt == self._max_tries - 1:
                    raise
                self._sleep(min(_BACKOFF_CAP, 2.0 ** attempt))    # same cap as the 429/503 path
                continue
            if resp.status in _RETRY_STATUSES:
                self._sleep(self._backoff(resp, attempt))
                continue
            self._sleep(self._delay)             # a real answer: pause before we ask again
            if resp.status == 404:
                return None
            if resp.status >= 400 and resp.status != 304:
                raise SourceHTTPError(url, resp.status)
            return resp
        raise TransportError(f"{url}: still throttled after {self._max_tries} attempts")

    def _backoff(self, resp: Response, attempt: int) -> float:
        """Seconds to wait after a 429/503: the server's Retry-After if given, else exponential."""
        retry_after = _header(resp.headers, "Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return min(_BACKOFF_CAP, 2.0 ** attempt)

    def _store(self, key: str, url: str, resp: Response) -> None:
        """Cache the body plus the HTTP metadata a later conditional re-fetch needs."""
        self._cache.put(
            self._namespace, key, resp.body, url=url,
            content_type=_header(resp.headers, "Content-Type"),
            etag=_header(resp.headers, "ETag"),
            last_modified=_header(resp.headers, "Last-Modified"),
        )

    def _touch(self, key: str) -> None:
        """Refresh ``fetched_at`` after a 304, re-storing the still-current cached bytes."""
        meta = self._cache.meta(self._namespace, key) or {}
        cached = self._cache.get(self._namespace, key) or b""
        self._cache.put(
            self._namespace, key, cached, url=meta.get("url"),
            content_type=meta.get("content_type"),
            etag=meta.get("etag"), last_modified=meta.get("last_modified"),
        )

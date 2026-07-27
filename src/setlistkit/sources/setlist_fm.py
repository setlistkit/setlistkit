# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""setlist.fm as a source: fetch an artist's structured setlists, politely, and cache them.

A source's whole job is to get raw bytes off the network and into the cache and back -- it never
normalizes a song name or knows what the pack vocabulary is. That work belongs to
:mod:`setlistkit.catalog.parse` (:func:`parse_setlist_fm`), the same split archive.org has: this
module produces raw setlist pages, the catalog layer turns them into merge records. The layer test
enforces it -- nothing under ``sources/`` may import ``catalog/``.

setlist.fm hands back a page of already-structured setlists per request (``setlist[].sets.set[]``),
so unlike archive.org there is no per-item second fetch: the listing IS the data. The client pages
through them, authenticating every request with an API key (``x-api-key``) read from its own
perm-checked file.
"""

from __future__ import annotations

import json
from typing import NamedTuple

from ..config import Config
from ..store.raw_cache import RawCache
from .base import PullResult
from .client import Auth, PoliteClient
from .keyfile import read_api_key

# The JSON REST root. A path this package builds, never user input.
API_ROOT = "https://api.setlist.fm/rest/1.0"

# setlist.fm answers this many setlists per page. Read off each response's own ``itemsPerPage`` at
# runtime; this is only the fallback used to estimate a page count before the first page has said
# otherwise, and the backstop below bounds a run whatever the real size turns out to be.
_ITEMS_PER_PAGE = 20
# Paging backstop, mirroring archive_org's: a listing this long is a runaway, not a discography.
# 500 pages * 20 = 10,000 setlists, past any real artist. Reported as truncation if it ever fires.
_MAX_PAGES = 500

# The cache namespace / on-disk directory for this source. Its own id, kept beside the client that
# owns it so a future importer-module contract has one obvious place to read it from.
NAMESPACE = "setlist_fm"


def setlists_in(page: object) -> list[dict]:
    """The raw ``setlist`` objects in one cached page payload, defensively.

    A tiny projection, not a parse: it does not touch a song name, so it stays on the source side
    of the layer line. :func:`setlistkit.catalog.parse.parse_setlist_fm` is what normalizes.
    """
    if not isinstance(page, dict):
        return []
    return [s for s in page.get("setlist", []) if isinstance(s, dict)]


def _year_of(setlist: dict) -> int | None:
    """The year of a setlist's ``eventDate`` (``dd-MM-yyyy``), or ``None`` if unusable.

    Reading a year off a date is not parsing a setlist -- no song name is touched -- so it stays
    on the source side of the layer line. It exists only to let :meth:`SetlistFmClient.pull` stop
    paging once it has reached below the year floor.
    """
    parts = str(setlist.get("eventDate") or "").split("-")
    if len(parts) == 3 and len(parts[2]) == 4 and parts[2].isdigit():
        return int(parts[2])
    return None


def _below_floor(page: object, min_year: int | None) -> bool:
    """True once a page holds any setlist older than ``min_year``.

    setlist.fm returns setlists newest-first, so the first page that dips below the floor is the
    last one worth fetching -- everything after it is older still. The straddling page itself is
    kept (the parser floors the pre-year entries out of it); this only decides when to STOP asking.
    """
    if min_year is None:
        return False
    return any((y := _year_of(s)) is not None and y < min_year for s in setlists_in(page))


class _Plan(NamedTuple):
    """What the first page tells a pull about the shape of the run: how many, how far, capped?"""

    total: int                                     # setlists the artist has, per the API
    pages: int                                     # pages to walk, bounded by the backstop
    truncated: bool                                # the backstop capped it below the real page count


def _plan(first: dict) -> _Plan:
    """Read the pagination shape off the first page. The analog of archive_org's listing math.

    ``itemsPerPage`` is taken from the response rather than assumed, falling back to the documented
    default only if the field is missing or zero, so a change in setlist.fm's page size does not
    silently truncate a walk.
    """
    total = int(first.get("total") or 0)
    per_page = int(first.get("itemsPerPage") or _ITEMS_PER_PAGE) or _ITEMS_PER_PAGE
    full_pages = -(-total // per_page) if total else 1
    return _Plan(total=total, pages=min(_MAX_PAGES, full_pages), truncated=full_pages > _MAX_PAGES)


class SetlistFmClient:
    """Fetch an artist's setlists from setlist.fm through the shared polite client.

    ``mbid`` (a MusicBrainz id) names the artist; the API pages its setlists newest-first. The key
    is read once at construction and sent on every request; a missing or world-readable key file
    fails here, before any network call, with the diagnostic :func:`read_api_key` raises.
    """

    def __init__(self, config: Config, cache: RawCache, *, transport=None, sleeper=None) -> None:
        key = read_api_key(config.section("sources", NAMESPACE)["api_key_file"])
        self._config = config
        self._cache = cache
        self._client = PoliteClient(config, cache, namespace=NAMESPACE,
                                    transport=transport, sleeper=sleeper,
                                    auth=Auth(headers={"x-api-key": key}))

    @staticmethod
    def _page_key(mbid: str, page: int) -> str:
        return f"artist/{mbid}/setlists/p{page}"

    def _page_url(self, mbid: str, page: int) -> str:
        return f"{API_ROOT}/artist/{mbid}/setlists?p={page}"

    def _fetch_page(self, mbid: str, page: int, *, force_rescan: bool) -> tuple[dict | None, bool]:
        """One page payload plus whether it was already cached, for fetched-vs-cached accounting.

        The cache is consulted before the fetch only to COUNT: the fetch itself revalidates and
        returns the same bytes either way, but a run has to be able to say how many pages it
        actually asked setlist.fm for, the same number archive.org reports.
        """
        key = self._page_key(mbid, page)
        was_cached = self._cache.get(NAMESPACE, key) is not None
        payload = self._client.fetch_json(key, self._page_url(mbid, page),
                                          force_rescan=force_rescan)
        return payload, was_cached

    def pull(self, mbid: str, *, min_year: int | None = None, force_rescan: bool = False,
             dry_run: bool = False, progress=None, announce=None) -> PullResult:
        """Fetch ``mbid``'s setlists into the cache, newest-first. Returns a :class:`PullResult`.

        The first page is always fetched: it is how the run learns ``total`` (the number of
        setlists the artist has) and thus how many pages remain -- the same role archive.org's
        listing plays. A dry run stops there and reports what a real run would fetch, having spent
        exactly one request, which is the honest cost of finding out.

        ``min_year`` is a client-side floor, not a query the API supports: the endpoint has no year
        filter, so the run pages from newest until it reaches a setlist older than the floor and
        then stops. The straddling page is kept whole; the parser drops its pre-year entries.
        """
        with self._client.batch() as batch:
            if announce:
                announce(batch.id)
            first, first_cached = self._fetch_page(mbid, 1, force_rescan=force_rescan)
            if first is None:
                # The artist id is unknown to setlist.fm: no setlists, nothing to page.
                return PullResult(listed=0, fetched=0, cached=0, planned=0)
            plan = _plan(first)
            if dry_run:
                cached = len(setlists_in(first)) if first_cached else 0
                # The cost that predicts wall-clock is REQUESTS, not setlists: one request pulls a
                # ~20-setlist page. Pages are contiguous from 1, so the ones cached BEFORE this run
                # are in hand and a real run fetches the rest. This probe just cached page 1, so
                # discount it unless it was already cached beforehand -- the same baseline `planned`
                # uses via `first_cached`, or the two counters would disagree on page 1.
                already = len(self.cached_pages(mbid)) - (0 if first_cached else 1)
                return PullResult(listed=plan.total, fetched=0, cached=cached,
                                  planned=max(0, plan.total - cached),
                                  planned_requests=max(0, plan.pages - already),
                                  probe_requests=0 if first_cached else 1,
                                  truncated=plan.truncated)
            fetched, cached = self._walk(mbid, first, first_cached, plan, min_year=min_year,
                                         force_rescan=force_rescan, progress=progress)
        return PullResult(listed=plan.total, fetched=fetched, cached=cached,
                          planned=fetched, truncated=plan.truncated)

    def _walk(self, mbid, first, first_cached, plan, *, min_year, force_rescan, progress):
        """Walk pages from the first onward until the year floor, tallying fetched vs cached.

        The analog of archive_org's ``_fetch_each``: the accounting loop lives here so ``pull`` reads
        as probe -> plan -> walk. Page 1 is already in hand (that is how ``plan`` was learned), so its
        setlists are tallied here from ``first``/``first_cached`` and the loop carries on from page 2.
        Stops at the first page holding a setlist below ``min_year`` -- newest-first, so the rest are
        older -- keeping that straddling page for the parser to floor.
        """
        fetched = 0 if first_cached else len(setlists_in(first))
        cached = len(setlists_in(first)) if first_cached else 0
        last, page = first, 2
        while page <= plan.pages and not _below_floor(last, min_year):
            payload, was_cached = self._fetch_page(mbid, page, force_rescan=force_rescan)
            if was_cached:
                cached += len(setlists_in(payload))
            else:
                fetched += len(setlists_in(payload))
            if progress:
                progress(page, plan.pages)
            if not setlists_in(payload):            # None or an empty page: the walk is done
                break
            last, page = payload, page + 1
        return fetched, cached

    def cached_pages(self, mbid: str) -> list[dict]:
        """Every cached page payload for ``mbid``, in page order. Never touches the network.

        The read side ``ingest`` uses, the counterpart to archive_org's ``cached_items``: a pure
        cache walk, so re-parsing during development costs setlist.fm nothing. Stops at the first
        page not in the cache, because the pages are contiguous from 1 and a gap means the pull did
        not finish -- reading past it would silently parse a corpus with a hole in it.
        """
        pages: list[dict] = []
        for page in range(1, _MAX_PAGES + 1):
            raw = self._cache.get(NAMESPACE, self._page_key(mbid, page))
            if raw is None:
                break
            try:
                pages.append(json.loads(raw.decode("utf-8", "replace")))
            except json.JSONDecodeError:
                break
        return pages

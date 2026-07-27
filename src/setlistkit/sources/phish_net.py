# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""phish.net as a source: fetch a year of setlists at a time, politely, and cache them.

Like every source, this one only moves raw bytes from the network into the cache and back. It
never normalizes a song name and never decides which artist a row belongs to -- that is
``catalog``'s job, and the layer test enforces it.

Three things about this API shape the client, and each one is a trap the obvious implementation
falls into:

**There is no pagination.** The v5 grammar is ``/v5/{method}/{column}/{value}.json`` plus a
``limit`` that truncates rather than pages. So a bulk pull cannot walk the whole ``setlists``
table; it asks for one YEAR at a time, which is the pattern the official examples demonstrate.
2020 through the current year is a handful of requests, not thousands, and each one is a natural
chunk to cache and re-fetch on its own.

**The key goes in the query string** (``?apikey=``), not a header -- phish.net offers no header
form. That makes it the first source whose credential would otherwise be written to disk by the
cache's own bookkeeping, which is why :class:`~setlistkit.sources.client.PoliteClient` grew
``extra_query``: the key is applied at the transport call and never enters the URL that gets
cached, reported or raised.

**Logical errors arrive as HTTP 200.** An invalid key, a rate limit and an empty result all come
back ``200 OK`` with a non-zero ``error`` in the envelope, so the polite client's status handling
never sees them and would happily cache the error body as though it were a year of setlists. This
module reads the envelope, evicts the poisoned entry, and raises -- because a cached error that
parses to zero shows is indistinguishable from a year the band did not tour.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..config import Config
from ..store.raw_cache import RawCache
from .base import PullResult
from .client import Auth, PoliteClient, SourceError
from .keyfile import read_api_key

# The JSON REST root. A path this package builds, never user input.
API_ROOT = "https://api.phish.net/v5"

# The cache namespace / on-disk directory for this source.
NAMESPACE = "phish_net"

# Phish's first show was in 1983, so there is nothing behind this. Used only when a pull is given
# no ``min_year`` at all; any real run sets one, and the config ships 2020 (see the right-sizing
# measurement in the ingest plan -- a 200-day recency halflife puts ~99.99% of model weight inside
# it, so earlier years cost the host requests to move the fifth decimal place).
FIRST_YEAR = 1983

# Envelope error codes, from the published docs. Only two need distinct handling.
_ERR_OK = 0
# "No data matched" is not a fault: a year the band did not play is an empty year, not a failure.
_ERR_NO_DATA = 11
# What each of the rest means, for a diagnostic a human can act on rather than a bare number.
_ERROR_MEANINGS = {
    2: "invalid API key",
    3: "missing API key",
    4: "rate limited -- the docs warn that apps demanding data too frequently get disabled",
    10: "HTTPS is required",
    12: "access denied for this method",
}


class PhishNetAPIError(SourceError):
    """A non-zero ``error`` in a phish.net envelope that arrived with HTTP status 200.

    Its own class because the polite client cannot see it: every layer below reads the status,
    and the status says the request succeeded. Carrying the code lets a caller tell "your key is
    wrong" (2) from "you are going too fast" (4), which are the same 200 on the wire.
    """

    def __init__(self, url: str, code: int, message: str) -> None:
        self.url = url
        self.code = code
        meaning = _ERROR_MEANINGS.get(code, "unrecognized error code")
        super().__init__(f"{url} returned envelope error {code} ({meaning}): {message}")


def rows_in(payload: object) -> list[dict]:
    """The raw setlist rows in one cached year payload, defensively.

    A projection, not a parse: it unwraps the envelope and hands back whatever rows were in it,
    every artist included. Nothing here touches a song name, so it stays on the source side of
    the layer line.
    """
    if not isinstance(payload, dict):
        return []
    return [row for row in payload.get("data", []) if isinstance(row, dict)]


def show_dates_in(payload: object, artist_slug: str) -> set[str]:
    """The distinct ``showdate`` values in a year payload for one artist.

    Two filters, and both are counting decisions rather than parsing ones -- no song name is
    touched, so this stays source-side:

    * ``artist_slug`` -- the year endpoint returns Trey solo, Mike Gordon, Fishman and guest
      appearances alongside Phish, and they are not a rounding error: across 2020-2026 the
      non-Phish rows OUTNUMBER the Phish ones, 5,336 to 4,668. Without this the corpus is more
      than double and wrong.
    * ``exclude`` -- see below. It does not mean what the endpoint name suggests.

    **What ``exclude`` actually marks, measured rather than assumed.** The obvious reading is
    "a show that was called off", and for the ``shows`` endpoint that is right. It is NOT right
    here, and the reason is structural: a concert nobody played has no songs, so it contributes no
    setlist rows at all and never reaches this function. Every ``exclude=1`` row in 2020-2026 --
    8 dates, 33 rows, all of them checked -- is a performance that really happened but was not a
    concert:

    * NBC Studio 6B (1 song) and NPR headquarters (5 songs) -- a talk-show spot and a Tiny Desk.
    * Moon Palace, Cancun (5 dates, 3-5 songs each) -- the short poolside sets on the Mexico runs,
      as distinct from the full shows at the same venue.

    So this filter separates concerts from appearances, which is the right call for counting
    SHOWS. It is emphatically not the right call for a parser: those are real performances of real
    songs, and dropping them silently loses plays. A parser should keep them and flag them --
    ``showtypes.py``'s reasoning applies unchanged -- and it should take the flag from this field
    rather than re-deriving it from a song count.

    Cross-checked against phish.in, which reaches the same verdict without having the same field:
    of the three of these dates it also lists, every track on all three is flagged
    ``exclude_from_stats`` and titled ``(Check) ...`` or ``Soundcheck``. Two independent editorial
    judgements, two unrelated mechanisms, one answer -- which is the best evidence available that
    the flag means what it appears to.
    """
    dates = set()
    for row in rows_in(payload):
        if str(row.get("artist_slug") or "") != artist_slug:
            continue
        if str(row.get("exclude") or "0") != "0":
            continue
        date = str(row.get("showdate") or "").strip()
        if date:
            dates.add(date)
    return dates


def _current_year() -> int:
    return datetime.now(timezone.utc).year


class PhishNetClient:
    """Fetch a band's setlists from phish.net one year at a time, through the polite client.

    ``artist_slug`` names the band ("phish"); it selects rows within a year payload rather than
    appearing in the setlist URL, because the year endpoint has no artist dimension. The key is
    read once at construction, so a missing or world-readable key file fails here, before any
    network call, with the diagnostic :func:`read_api_key` raises.
    """

    def __init__(self, config: Config, cache: RawCache, *, transport=None, sleeper=None) -> None:
        section = config.section("sources", NAMESPACE)
        # phish.net's key page issues a JSON blob holding both an `apikey` and a `publickey`, so
        # the config names which field to read. Absent, the file is treated as a bare key, which
        # is what every other source here expects.
        key = read_api_key(section["api_key_file"], field=section.get("api_key_field"))
        self._config = config
        self._cache = cache
        self._client = PoliteClient(config, cache, namespace=NAMESPACE,
                                    transport=transport, sleeper=sleeper,
                                    auth=Auth(query={"apikey": key}))

    @staticmethod
    def _year_key(year: int) -> str:
        return f"setlists/showyear/{year}"

    def _year_url(self, year: int) -> str:
        return f"{API_ROOT}/setlists/showyear/{year}.json"

    @staticmethod
    def _shows_key(artist_slug: str) -> str:
        return f"shows/artist/{artist_slug}"

    def _shows_url(self, artist_slug: str) -> str:
        return f"{API_ROOT}/shows/artist/{artist_slug}.json?order_by=showdate"

    def _checked(self, key: str, url: str, payload: object) -> dict | None:
        """Return the envelope, or evict the cache entry and raise if it carries an error.

        The eviction is the point. :meth:`PoliteClient.fetch` has already written these bytes to
        disk by the time we can see the ``error`` field, and an error body left in the cache is
        read back by :meth:`cached_years` as a year holding no shows. That is the worst failure
        available here: silent, and shaped exactly like a true answer.
        """
        if payload is None:
            return None
        if not isinstance(payload, dict):
            self._cache.delete(NAMESPACE, key)
            raise PhishNetAPIError(url, -1, f"expected a JSON object, got {type(payload).__name__}")
        code = int(payload.get("error") or _ERR_OK)
        if code in (_ERR_OK, _ERR_NO_DATA):
            return payload
        self._cache.delete(NAMESPACE, key)
        raise PhishNetAPIError(url, code, str(payload.get("error_message") or ""))

    def _fetch_year(self, year: int, *, force_rescan: bool) -> tuple[dict | None, bool]:
        """One year's payload plus whether it was already cached, for fetched-vs-cached counting."""
        key, url = self._year_key(year), self._year_url(year)
        was_cached = self._cache.get(NAMESPACE, key) is not None
        payload = self._client.fetch_json(key, url, force_rescan=force_rescan)
        return self._checked(key, url, payload), was_cached

    def _fetch_show_metadata(self, artist_slug: str, *, force_rescan: bool) -> None:
        """Cache the artist's show list, which carries the ``updated_at`` the setlist rows lack.

        Fetched for its own sake rather than counted: it is one request, it is not setlists, and
        it is what a later incremental sync reads to notice that history was revised under us --
        phish.net has no ``since`` parameter, so a change feed has to be built from this column.
        """
        key, url = self._shows_key(artist_slug), self._shows_url(artist_slug)
        self._checked(key, url, self._client.fetch_json(key, url, force_rescan=force_rescan))

    def years(self, min_year: int | None) -> list[int]:
        """The years a pull will ask for: ``min_year`` (or 1983) through the current year."""
        return list(range(min_year if min_year is not None else FIRST_YEAR, _current_year() + 1))

    def pull(self, artist_slug: str, *, min_year: int | None = None, force_rescan: bool = False,
             dry_run: bool = False, progress=None, announce=None) -> PullResult:
        """Fetch each year's setlists into the cache. Returns a :class:`PullResult`.

        One request per year, oldest first, which is the officially demonstrated bulk pattern and
        the only one available -- the API has no ``offset`` and no ``page``, so walking the whole
        ``setlists`` table is not something a client can do, and the docs warn against asking.

        Counts follow the setlist.fm convention: ``listed``/``fetched``/``cached`` count SHOWS,
        because that is the unit a corpus is made of, while ``planned_requests`` counts the year
        requests, because that is the unit the host pays for.
        """
        wanted = self.years(min_year)
        if dry_run:
            return self._dry_run(artist_slug, wanted)

        with self._client.batch() as batch:
            if announce:
                announce(batch.id)
            batch.begin("year", total=len(wanted))
            fetched_dates, cached_dates = self._fetch_each(wanted, artist_slug,
                                                           force_rescan=force_rescan,
                                                           progress=progress)
            self._fetch_show_metadata(artist_slug, force_rescan=force_rescan)
        return PullResult(listed=len(fetched_dates | cached_dates), fetched=len(fetched_dates),
                          cached=len(cached_dates), planned=len(fetched_dates),
                          planned_requests=len(wanted) + 1)

    def _fetch_each(self, wanted: list[int], artist_slug: str, *, force_rescan: bool,
                    progress=None) -> tuple[set[str], set[str]]:
        """Fetch each year, returning the show dates this run paid for versus already held."""
        fetched_dates: set[str] = set()
        cached_dates: set[str] = set()
        for index, year in enumerate(wanted, start=1):
            payload, was_cached = self._fetch_year(year, force_rescan=force_rescan)
            dates = show_dates_in(payload, artist_slug)
            (cached_dates if was_cached else fetched_dates).update(dates)
            if progress:
                progress(index, len(wanted))
        # A show can only be counted once, and a date seen in a fetched year outranks the same
        # date seen in a cached one -- the run did pay for it. New Year's runs straddle two year
        # payloads, so this is a real case and not a defensive flourish.
        return fetched_dates, cached_dates - fetched_dates

    def _dry_run(self, artist_slug: str, wanted: list[int]) -> PullResult:
        """What a real run would cost, having spent nothing.

        Cheaper and less informative than a paged source's dry run, and both halves are worth
        saying out loud. Cheaper: the years come from config, so there is no first page to fetch
        to discover the size of the job -- this touches the network zero times where setlist.fm
        spends one request. Less informative: ``planned`` is therefore 0 rather than a show count,
        because how many shows sit in a year nobody has fetched is not knowable without fetching
        it. ``planned_requests`` is the honest number here, and it is the one the host cares about.
        """
        missing_years = [y for y in wanted
                         if self._cache.get(NAMESPACE, self._year_key(y)) is None]
        held = self.cached_show_dates(artist_slug)
        return PullResult(listed=len(held), fetched=0, cached=len(held), planned=0,
                          planned_requests=len(missing_years) + 1, probe_requests=0)

    def cached_years(self, min_year: int | None = None) -> list[dict]:
        """Every cached year payload, oldest first. Never touches the network.

        The read side an importer would parse from, the counterpart to setlist.fm's
        ``cached_pages``. Unlike that one it does NOT stop at the first gap: years are independent
        requests rather than a contiguous walk, so a missing 2022 means 2022 was never fetched, not
        that everything after it is untrustworthy.
        """
        payloads = (self._cache.get_json(NAMESPACE, self._year_key(y))
                    for y in self.years(min_year))
        return [p for p in payloads if p is not None]

    def cached_show_dates(self, artist_slug: str, min_year: int | None = None) -> set[str]:
        """Distinct real show dates held in the cache for ``artist_slug``. A pure cache read."""
        dates: set[str] = set()
        for payload in self.cached_years(min_year):
            dates |= show_dates_in(payload, artist_slug)
        return dates

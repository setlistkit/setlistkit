# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""phish.in as a source: list the shows in a year range, then fetch each one's tracks.

Like every source, this one only moves raw bytes between the network and the cache. It never
normalizes a song name; ``catalog`` does that, and the layer test enforces it.

This is the first source since archive.org with archive.org's TWO-PHASE shape -- a listing that
names the items, then one fetch per item -- and it has it for the same reason: **the index does
not carry the payload.** ``GET /shows?year_range=`` returns show-level fields (venue, tour,
``audio_status``, a whole-show ``duration``) but no ``tracks`` array, so the per-track durations
that are the entire reason to consult phish.in only exist behind ``GET /shows/<date>``. That was
measured against the live API rather than assumed, because the alternative design -- one paged
request for everything -- would have been dramatically politer if it had worked.

What phish.in owns, and what makes the second phase worth its requests:

* **``duration`` in milliseconds, per track, from a database.** Not a string scraped out of a
  taper's tracklist and reconciled across four tapes of one night. phish.net has a ``tracktime``
  field on the same rows and it is empty in every capture anyone has looked at, so this is the
  only clock either source offers.
* **``exclude_from_stats`` per track** -- the stub-track judgement, made upstream by someone with
  better information than a duration threshold has.
* **``audio_status``** per show and per track, so "we hold a complete recording" is a field and
  not an inference.

No key: the API is anonymous, and a key only raises rate caps. Their published terms are three
sentences reserving the right to revoke access, and the API's own description asks callers to
"please be nice" -- so this walks the index at ``per_page`` and then paces itself through the
shows on the polite client's ordinary delay.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..config import Config
from ..store.raw_cache import RawCache
from .base import PullResult
from .client import PoliteClient

# The JSON REST root. A path this package builds, never user input.
API_ROOT = "https://phish.in/api/v2"

# The cache namespace / on-disk directory for this source.
NAMESPACE = "phish_in"

# phish.in is a single-band archive, so its identity is a constant rather than a config key. It is
# still threaded through ``pull`` as one, because the importer contract hands every source an
# identity and inventing a knob that can only hold one value would be worse than saying this.
ARTIST = "phish"

# The index page size. The API caps ``per_page`` at 1000; 100 keeps any one response small enough
# to read in an editor, which is the whole reason raw snapshots are files.
_PER_PAGE = 100
# Paging backstop, mirroring the other sources': a listing this long is a runaway, not a
# discography. 200 pages * 100 = 20,000 shows, past anything phish.in holds. Reported as
# truncation if it ever fires.
_MAX_PAGES = 200

# Phish's first show was in 1983. Used only when a pull is given no ``min_year``.
FIRST_YEAR = 1983


def shows_in(page: object) -> list[dict]:
    """The raw show objects in one cached index page, defensively.

    A projection, not a parse -- it touches no song name, so it stays source-side.
    """
    if not isinstance(page, dict):
        return []
    return [show for show in page.get("shows", []) if isinstance(show, dict)]


def show_dates_in(page: object) -> list[str]:
    """The ``date`` of every show on one index page, in the order the page listed them."""
    return [date for show in shows_in(page) if (date := str(show.get("date") or "").strip())]


def tracks_in(payload: object) -> list[dict]:
    """The raw track objects in one cached show payload, defensively."""
    if not isinstance(payload, dict):
        return []
    return [track for track in payload.get("tracks", []) if isinstance(track, dict)]


def _current_year() -> int:
    return datetime.now(timezone.utc).year


class PhishInClient:
    """Fetch phish.in's shows for a year range, then each show's tracks, through the polite client.

    Anonymous -- no key is read and none is sent. ``identity`` is accepted to satisfy the uniform
    ``pull`` contract and is not used to build a URL, because the API has no artist dimension.
    """

    def __init__(self, config: Config, cache: RawCache, *, transport=None, sleeper=None) -> None:
        self._config = config
        self._cache = cache
        self._client = PoliteClient(config, cache, namespace=NAMESPACE,
                                    transport=transport, sleeper=sleeper)

    @staticmethod
    def _index_key(year_range: str, page: int) -> str:
        return f"shows/index/{year_range}/p{page}"

    def _index_url(self, year_range: str, page: int) -> str:
        return (f"{API_ROOT}/shows?year_range={year_range}"
                f"&per_page={_PER_PAGE}&page={page}&sort=date:asc")

    @staticmethod
    def _show_key(date: str) -> str:
        return f"shows/{date}"

    def _show_url(self, date: str) -> str:
        return f"{API_ROOT}/shows/{date}"

    @staticmethod
    def year_range(min_year: int | None) -> str:
        """The ``year_range`` query value a pull will ask for, e.g. ``2020-2026``."""
        return f"{min_year if min_year is not None else FIRST_YEAR}-{_current_year()}"

    def _list_shows(self, year_range: str, *, force_rescan: bool,
                    progress=None) -> tuple[list[str], bool]:
        """Walk the index and return every show date it names, plus whether the backstop fired.

        The index is re-fetched on every real pull rather than trusted from cache: it is the phase
        that discovers what exists, and a stale index is how a show that landed yesterday goes
        unnoticed forever. It is also cheap -- three requests for a six-year range.
        """
        dates: list[str] = []
        page, truncated = 1, False
        while page <= _MAX_PAGES:
            payload = self._client.fetch_json(self._index_key(year_range, page),
                                              self._index_url(year_range, page),
                                              force_rescan=force_rescan)
            found = show_dates_in(payload)
            dates.extend(found)
            total_pages = int((payload or {}).get("total_pages") or 0)
            if progress:
                progress(page, total_pages or page)
            if not found or page >= total_pages:
                truncated = bool(total_pages) and page < total_pages
                break
            page += 1
        else:
            truncated = True
        return dates, truncated

    def pull(self, identity: str = ARTIST, *,  # pylint: disable=unused-argument
             min_year: int | None = None, force_rescan: bool = False, dry_run: bool = False,
             progress=None, announce=None) -> PullResult:
        """List the shows in the year range and fetch each one's tracks. Returns a PullResult.

        Two phases, the same shape archive.org's pull has: the index names the shows, then each
        show is fetched once and cached, so a re-run costs phish.in nothing. A dry run stops after
        the index and reports what the item phase would cost, having spent only the listing -- the
        honest price of finding out.

        A show the index names but ``/shows/<date>`` 404s lands in ``missing`` rather than failing
        the run. That is a real and *systematic* case here, not a fluke: shows with no circulating
        audience recording are in the catalog with ``audio_status: 'missing'``, and they cluster at
        venues nobody smuggles a rig into -- the Mexico destination runs, most of them.

        ``identity`` is accepted and deliberately unused. phish.in is a single-band archive with no
        artist dimension in any endpoint, so there is nothing to key on; the parameter is here
        because the importer contract hands every source an identity, and taking it while doing
        nothing with it is more honest than inventing a config knob that can only hold one value.
        """
        year_range = self.year_range(min_year)
        with self._client.batch() as batch:
            if announce:
                announce(batch.id)
            batch.begin("index")
            dates, truncated = self._list_shows(year_range, force_rescan=force_rescan)
            held = [d for d in dates if self._cache.get(NAMESPACE, self._show_key(d)) is not None]
            if dry_run:
                # `batch.sent` is incremented past the cache check, so it counts requests that
                # actually left the box -- which is precisely what a dry run owes the operator,
                # and it is zero when the index was already cached.
                return PullResult(listed=len(dates), fetched=0, cached=len(held),
                                  planned=len(dates) - len(held),
                                  planned_requests=len(dates) - len(held),
                                  probe_requests=batch.sent, truncated=truncated)
            batch.begin("show", total=len(dates))
            fetched, cached, missing = self._fetch_each(dates, force_rescan=force_rescan,
                                                        progress=progress)
        return PullResult(listed=len(dates), fetched=fetched, cached=cached,
                          planned=fetched + len(missing), missing=tuple(missing),
                          truncated=truncated)

    def _fetch_each(self, dates: list[str], *, force_rescan: bool,
                    progress=None) -> tuple[int, int, list[str]]:
        """Fetch every listed show, tallying fetched vs already-cached vs absent."""
        fetched, cached, missing = 0, 0, []
        for index, date in enumerate(dates, start=1):
            key = self._show_key(date)
            was_cached = self._cache.get(NAMESPACE, key) is not None
            payload = self._client.fetch_json(key, self._show_url(date), force_rescan=force_rescan)
            if payload is None:
                missing.append(date)
            elif was_cached:
                cached += 1
            else:
                fetched += 1
            if progress:
                progress(index, len(dates))
        return fetched, cached, missing

    def cached_shows(self, min_year: int | None = None) -> list[dict]:
        """Every cached show payload for the year range, oldest first. Never touches the network.

        The read side an importer would parse from. Driven off the cached INDEX pages rather than
        a directory listing, so it returns shows in the order the source listed them and stays
        honest about which range it is answering for.
        """
        payloads = (self._cache.get_json(NAMESPACE, self._show_key(d))
                    for d in self.cached_show_dates(min_year))
        return [p for p in payloads if p is not None]

    def cached_show_dates(self, min_year: int | None = None) -> list[str]:
        """Show dates named by the cached index pages, oldest first. A pure cache read.

        Stops at the first page not in the cache: index pages are contiguous from 1, so a gap
        means the listing phase did not finish, and reading past it would report a show list with
        a hole in it as though it were the whole range.
        """
        year_range = self.year_range(min_year)
        dates: list[str] = []
        for page in range(1, _MAX_PAGES + 1):
            payload = self._cache.get_json(NAMESPACE, self._index_key(year_range, page))
            if payload is None:
                break
            dates.extend(show_dates_in(payload))
        return dates

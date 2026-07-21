# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""archive.org: list a collection's items, then fetch each item's metadata -- politely, cached.

Two endpoints, per https://archive.org/developers/bots.html (read it before changing anything):
``advancedsearch.php`` lists the items in a collection, and ``metadata/<identifier>`` returns one
item's description and file list. Both go through :class:`~setlistkit.sources.client.PoliteClient`,
so the mandatory identifying User-Agent, the cache, the rate limit, and the 429 backoff all come
for free and identically to every other source.

Metadata is treated as immutable-ish and cached without expiry: a taped show's description rarely
changes, and ``force_rescan`` exists for the times it does. A search listing is the opposite -- it
is the "what is new" query -- so it is always revalidated, never served stale from the cache.
"""

from __future__ import annotations

import urllib.parse
from datetime import timedelta

from ..config import Config
from ..store.raw_cache import RawCache
from .client import PoliteClient

NAMESPACE = "archive_org"

_SEARCH_URL = "https://archive.org/advancedsearch.php"
_METADATA_URL = "https://archive.org/metadata/"
# The fields the parser needs off the listing: the identifier to fetch, plus the dates and titles
# used to place and name a show before its metadata is even pulled.
_SEARCH_FIELDS = ("identifier", "date", "title", "year", "addeddate")
_PAGE_ROWS = 500
_MAX_PAGES = 40               # a 20k-item backstop against a runaway loop, not an expected limit


class ArchiveOrgClient:
    """Fetch moe.-collection show data from archive.org through a shared polite client."""

    def __init__(self, config: Config, cache: RawCache, *, client: PoliteClient | None = None) -> None:
        self._client = client or PoliteClient(config, cache, namespace=NAMESPACE)

    def metadata(self, identifier: str, *, force_rescan: bool = False) -> dict | None:
        """One item's metadata blob (description, files, ...), or ``None`` if archive.org 404s it."""
        url = _METADATA_URL + urllib.parse.quote(identifier)
        return self._client.fetch_json(identifier, url, force_rescan=force_rescan)

    def list_items(self, collection: str, *, min_year: int | None = None,
                   force_rescan: bool = False) -> list[dict]:
        """Every item in ``collection``, paged until archive.org's ``numFound`` is exhausted.

        Always revalidated (``max_age`` zero): the listing is the freshness query itself, so a
        cached copy would hide exactly the new uploads the caller is asking about. Still rate-
        limited and identified like every other request.
        """
        docs: list[dict] = []
        for page in range(1, _MAX_PAGES + 1):
            key = f"advancedsearch/{collection}/{min_year}/p{page}"
            payload = self._client.fetch_json(key, self._search_url(collection, min_year, page),
                                              force_rescan=force_rescan, max_age=timedelta(0))
            if payload is None:
                break
            response = payload.get("response", {})
            page_docs = response.get("docs", [])
            docs.extend(page_docs)
            # Stop on an empty page always; stop on the count only when the API gave one. Defaulting
            # a missing numFound to 0 would end the walk after page 1 with docs still coming.
            num_found = response.get("numFound")
            if not page_docs or (num_found is not None and len(docs) >= num_found):
                break
        return docs

    @staticmethod
    def _search_url(collection: str, min_year: int | None, page: int) -> str:
        """Build one advancedsearch page URL. ``fl[]``/``sort[]`` repeat by key, so build by hand."""
        query = f"collection:{collection}"
        if min_year is not None:
            query += f" AND year:[{min_year} TO 9999]"
        parts = [
            "q=" + urllib.parse.quote(query),
            *(f"fl[]={field}" for field in _SEARCH_FIELDS),
            "sort[]=addeddate+asc",
            f"rows={_PAGE_ROWS}",
            f"page={page}",
            "output=json",
        ]
        return _SEARCH_URL + "?" + "&".join(parts)

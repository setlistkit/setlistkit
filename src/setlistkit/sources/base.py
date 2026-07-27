# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The source contract: the shape a fetch client presents and the result it returns.

A neutral home for the two things every source shares, so no source imports them from another. It
used to be that ``PullResult`` lived inside ``archive_org.py`` and ``setlist_fm.py`` reached across
to it -- a sibling standing in for a spec. Here instead: ``PullResult`` is what a pull reports, and
``SourceClient`` is the structural surface ``slkit pull`` drives. Clients do NOT inherit
``SourceClient``; they satisfy it by shape, and a conformance test holds them to it. That keeps each
source a plain class while giving the registry one enforced contract to dispatch against.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PullResult:
    """What one ``slkit pull`` run did, in the terms the CLI reports it. Every source returns this.

    ``fetched`` and ``cached`` are counted separately because the difference is the whole point
    of the cache: a second pull that reports 0 fetched and 900 cached made 0 metadata requests,
    and that is the number archive.org cares about. Under ``force_rescan`` everything is re-asked,
    so ``cached`` is 0 and ``fetched`` counts items that answered 304 as well as ones that sent
    bytes -- "re-asked", not "re-downloaded".

    Every listed item lands in exactly one of ``fetched``, ``cached``, ``missing`` or
    ``unidentified``, and ``listed`` is their sum. That is asserted by a test, because a counter
    an item can fall out of is how a lost show goes unnoticed.
    """

    listed: int = 0                        # DISTINCT identifiers the listing named
    fetched: int = 0
    cached: int = 0
    # Items this run set out to fetch: everything listed that is not already cached. In a real
    # run it equals fetched + missing, which a test asserts. In a dry run it is the whole point,
    # because it is the number that says what the real run would cost the host.
    planned: int = 0
    # Network requests a real run would make, when that differs from `planned`. archive.org fetches
    # one request per item, so it leaves this None and the estimate counts items. setlist.fm fetches
    # a page of ~20 setlists per request, so `planned` setlists cost far fewer requests -- it sets
    # this to the page count so the dry-run time estimate reflects requests, not records.
    planned_requests: int | None = None
    # What THIS dry run itself spent, when the source can say. A dry run is a question about cost
    # that is not free to ask, and how much it costs differs per source: setlist.fm must fetch page
    # one to learn the total, phish.in walks its index, and phish.net spends nothing at all because
    # the years come from config. Reported rather than inferred, because the CLI used to infer it
    # from `planned_requests` and would then tell an operator that one page had been requested on a
    # source that had made no request whatsoever. ``None`` means the source did not say.
    probe_requests: int | None = None
    missing: tuple[str, ...] = ()          # identifiers archive.org 404'd
    # Identifiers whose fetch failed even after the client's own retries. Separate from `missing`
    # because the cause is different -- absent versus unwell -- though the remedy is the same:
    # nothing was cached, so the next pull tries them again.
    failed: tuple[str, ...] = ()
    # Listing docs carrying no identifier at all. Unfetchable and unnameable, so they get a count
    # rather than a list -- but a count, because an item in none of these is an item nobody misses.
    unidentified: int = 0
    # The paging backstop fired, so the listing is a prefix of the collection rather than all of
    # it. Reported because the alternative is an ingest that looks complete and is not.
    truncated: bool = False


@runtime_checkable
class SourceClient(Protocol):  # pylint: disable=too-few-public-methods  # a one-method contract is the point
    """The one method ``slkit pull`` calls, uniform across sources.

    ``pull`` is keyed on the source's identity -- an archive.org collection, a MusicBrainz id --
    which the source's :class:`~setlistkit.importers.base.Importer` resolves from config. The
    keyword shape is deliberately the same for every source so the CLI dispatches without knowing
    which client it holds; only the identity's meaning differs. A new source conforms by presenting
    this shape, checked against the real clients by a conformance test -- no base class to inherit.
    """

    def pull(self, identity: str, *, min_year: int | None = None, force_rescan: bool = False,
             dry_run: bool = False, progress: Callable[[int, int], None] | None = None,
             announce: Callable[[str], None] | None = None) -> PullResult:
        """Fetch ``identity``'s records into the cache; return what was (or would be) fetched."""

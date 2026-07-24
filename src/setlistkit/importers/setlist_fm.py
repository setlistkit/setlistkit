# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The setlist.fm importer: a setlist source.

setlist.fm carries only setlists, no audio, so it feeds no mirror -- it contributes merge
candidates through :func:`_candidates` and ``ingest`` folds them on top of the mirror source. Its
identity is the band's MusicBrainz id, a top-level config key (it names the band, not this source).
"""

from __future__ import annotations

from ..catalog.parse_setlistfm import parse_setlistfm
from ..diagnostics import ERROR, Diagnostic, DiagnosticError
from ..sources.setlist_fm import NAMESPACE, SetlistFmClient
from .base import Importer


def _identity(config) -> str:
    """The band's MusicBrainz id, or a diagnostic naming the top-level key to set.

    setlist.fm keys every request on it, so a run without one cannot even list the artist. It is a
    top-level key (it names the band, not a source), hence the pointed instruction here."""
    if config.mbid:
        return config.mbid
    raise DiagnosticError(Diagnostic(
        severity=ERROR,
        summary="setlist.fm needs the band's MusicBrainz id, and none is configured",
        detail="Add the band's mbid as a TOP-LEVEL key in your config, beside data_root:\n\n"
               '    mbid = "3826a6e0-9ea5-4007-941c-25b9dd943981"\n\n'
               "Find it on musicbrainz.org -- search the artist; the id is the UUID in the\n"
               "/artist/<id> URL.",
    ))


def _candidates(config, cache, *, normalizer, min_year) -> list[dict]:
    """setlist.fm's merge show-records from the cache, or empty if it is not set up here.

    Best-effort and quiet about absence, the way lint's corpus read is: no ``[sources.setlist_fm]``
    table, no top-level ``mbid``, or nothing pulled yet all contribute nothing and the ingest is
    what it was before the source existed. A configured-but-broken key is NOT swallowed --
    :class:`SetlistFmClient` raises for a missing or world-readable key file, and that is a real
    fault a run should stop on.
    """
    if not config.section("sources", NAMESPACE).get("api_key_file") or not config.mbid:
        return []
    pages = SetlistFmClient(config, cache).cached_pages(config.mbid)
    return parse_setlistfm(pages, normalizer=normalizer, min_year=min_year) if pages else []


IMPORTER = Importer(
    id=NAMESPACE,
    display_name="setlist.fm",
    feeds_mirror=False,
    make_client=SetlistFmClient,
    identity=_identity,
    candidates=_candidates,
)

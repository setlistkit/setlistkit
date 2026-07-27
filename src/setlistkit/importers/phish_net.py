# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The phish.net importer: a setlist source, and the one that carries segues.

No audio, so it feeds no mirror -- it contributes merge candidates and ``ingest`` folds them in.
Its identity is the artist slug, which unlike setlist.fm's MusicBrainz id is a per-source key: the
slug is phish.net's own spelling of the band and selects rows inside a year payload, so it belongs
under ``[sources.phish_net]`` rather than beside ``data_root``.
"""

from __future__ import annotations

from ..catalog.parse_phish_net import parse_phish_net
from ..diagnostics import ERROR, Diagnostic, DiagnosticError
from ..sources.phish_net import NAMESPACE, PhishNetClient
from .base import Importer


def _identity(config) -> str:
    """The artist slug whose rows to keep, or a diagnostic naming the table to set it in.

    There is no default. The ``showyear`` endpoint returns every artist phish.net tracks -- Trey
    solo, Mike Gordon, Fishman, guest appearances -- and across 2020-2026 those rows outnumber
    Phish's own. Guessing "phish" for an operator who meant something else would silently build a
    corpus for the wrong act, so the config has to say.
    """
    artist = str(config.section("sources", NAMESPACE).get("artist") or "").strip()
    if artist:
        return artist
    raise DiagnosticError(Diagnostic(
        severity=ERROR,
        summary="[sources.phish_net] artist is not set",
        detail="phish.net's year endpoint returns several artists in one payload, so an ingest\n"
               "has to be told which one is the band. Add it to your config:\n\n"
               '    [sources.phish_net]\n    artist = "phish"\n\n'
               "The value is phish.net's own `artist_slug` -- 'phish', 'trey-anastasio', and so on.",
    ))


def _candidates(config, cache, *, normalizer, min_year) -> list[dict]:
    """phish.net's merge show-records from the cache, or empty if it is not set up here.

    Best-effort about absence and strict about breakage, matching setlist.fm: no
    ``[sources.phish_net]`` table or nothing pulled yet contributes nothing quietly, but a
    configured-and-broken key file still raises, because that is a real fault rather than a
    source sitting out.
    """
    section = config.section("sources", NAMESPACE)
    if not section.get("api_key_file") or not section.get("artist"):
        return []
    client = PhishNetClient(config, cache)
    payloads = client.cached_years(min_year)
    if not payloads:
        return []
    return parse_phish_net(payloads, normalizer=normalizer, min_year=min_year,
                           artist_slug=_identity(config))


IMPORTER = Importer(
    id=NAMESPACE,
    display_name="phish.net",
    feeds_mirror=False,
    make_client=PhishNetClient,
    identity=_identity,
    candidates=_candidates,
)

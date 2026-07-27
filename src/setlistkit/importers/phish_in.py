# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The phish.in importer: a setlist source, and the one that holds the clock.

No segues, so it ranks below phish.net in the merge and loses any date they both carry. It earns
its place on the dates phish.net's filters drop and on the per-track durations nothing else has.

IT IS THE ONLY SOURCE THAT DOES BOTH JOBS. Its setlists fold into the merge and usually lose;
its tapes are the only clock the band has, because archive.org carries no Phish. Those travel on
separate hooks -- ``candidates`` and ``tapes`` -- and :mod:`setlistkit.importers` explains at
length why ``feeds_mirror`` could not have expressed the pair.

Its identity is a constant. phish.in is a single-band archive with no artist dimension in any
endpoint, so there is nothing for config to name and inventing a knob that can only hold one
value would be worse than saying so.
"""

from __future__ import annotations

from ..catalog.parse_phish_in import parse_phish_in, phish_in_tapes
from ..sources.phish_in import ARTIST, NAMESPACE, PhishInClient
from .base import Importer


def _identity(config) -> str:                      # pylint: disable=unused-argument
    """Always ``"phish"``. Takes ``config`` to satisfy the importer contract and reads none of it.

    Every other source resolves an identity from config and raises a diagnostic when it is
    missing. There is deliberately nothing to raise here: a config key the operator cannot get
    wrong is a config key that should not exist.
    """
    return ARTIST


def _candidates(config, cache, *, normalizer, min_year) -> list[dict]:
    """phish.in's merge show-records from the cache, or empty if nothing has been pulled.

    Needs no credential and therefore no configured-but-broken case to distinguish: the source is
    either pulled or it is not. An empty ``[sources.phish_in]`` table is enough to enable it.
    """
    shows = PhishInClient(config, cache).cached_shows(min_year)
    return parse_phish_in(shows, normalizer=normalizer, min_year=min_year) if shows else []


def _tapes(config, cache, *, min_year) -> list[dict]:
    """phish.in's shows as mirror recordings: one official transfer a night, with track lengths.

    Reads the same cached payloads :func:`_candidates` reads, and that is deliberate rather than
    wasteful. One pull, two projections -- setlists into the merge, durations into the mirror --
    so the tape and the setlist for a night can never come from different fetches of a source
    that is still adding audio to old shows.
    """
    shows = PhishInClient(config, cache).cached_shows(min_year)
    return phish_in_tapes(shows, min_year=min_year) if shows else []


IMPORTER = Importer(
    id=NAMESPACE,
    display_name="phish.in",
    feeds_mirror=False,
    make_client=PhishInClient,
    identity=_identity,
    candidates=_candidates,
    tapes=_tapes,
)

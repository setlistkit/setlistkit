# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The importer descriptor: the contract every source's module fills in.

One frozen record per source, so ``cli`` dispatches on data instead of a chain of ``if
args.source == ...``. It is deliberately small -- six fields -- because the heavy code stays where
the layer test wants it (the client under ``sources``, the parser under ``catalog``); an importer
only names them and the identity that ties them together.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..config import Config
from ..sources.base import SourceClient
from ..store.raw_cache import RawCache


@dataclass(frozen=True)
class Importer:
    """Everything ``cli`` needs to pull, cache and ingest one source, keyed by one ``id``.

    ``id`` is the single MACHINE spelling of the source -- the CLI argument, the ``[sources.<id>]``
    table key, and the raw-cache namespace, which used to be three independent strings. It stays in
    the underscore form (``archive_org``, ``setlist_fm``) on purpose: a dotted id would nest as a
    TOML table (``[sources.archive.org]`` parses as ``sources.archive.org``, three tables), forcing
    every operator to quote the header.

    ``display_name`` is the source's real-world name (``archive.org``, ``setlist.fm``) for output a
    human reads. Kept separate from ``id`` so the friendly name is shown without the dot ever
    reaching argparse, a TOML header, or a cache path. Diagnostics that name a config table still
    use ``id`` -- they are telling the operator which ``[sources.<id>]`` to edit.

    ``make_client(config, cache)`` returns the ``sources`` client -- a
    :class:`~setlistkit.sources.base.SourceClient`, whose ``pull(identity, *, min_year, force_rescan,
    dry_run, progress, announce)`` is uniform across sources and returns a ``PullResult``.

    ``identity(config)`` resolves what the pull is keyed on -- an archive collection, a MusicBrainz
    id -- or raises a :class:`~setlistkit.diagnostics.DiagnosticError` naming what to set. It is the
    string handed to ``pull`` and to the cache reads.

    ``candidates(config, cache, *, normalizer, min_year)`` returns this source's merge show-records,
    or ``None`` for the mirror source (archive.org), which ``ingest`` reads explicitly because it
    also needs the raw items for tapes and durations. A setlist source returns its records and
    feeds no mirror.

    ``feeds_mirror`` says whether this source IS the mirror -- the one ``ingest`` reads explicitly,
    because it needs the raw items for tapes, listings and show types as well as for setlists. It
    is what ``ingest`` splits the primary read from the fold-ins on, and exactly one source can be
    it. It does NOT mean "carries audio": see ``tapes``.

    ``tapes(config, cache, *, min_year)`` returns mirror recordings a source contributes WITHOUT
    being the mirror, or is ``None`` for a source with no audio. phish.in is why the two are
    separate. It carries per-track durations and it is a fold-in setlist source at the same time,
    which ``feeds_mirror`` cannot express: setting it there would take phish.in out of the merge
    (``secondary()`` skips mirror sources) to gain the tapes, and the merge is where its setlists
    have to be. So the audio arrives on its own hook, and a source can supply either, both, or
    neither.
    """

    id: str
    display_name: str
    feeds_mirror: bool
    make_client: Callable[[Config, RawCache], SourceClient]
    identity: Callable[[Config], str]
    candidates: Callable | None = None
    tapes: Callable | None = None

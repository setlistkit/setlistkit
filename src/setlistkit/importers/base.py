# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The importer descriptor: the contract every source's module fills in.

One frozen record per source, so ``cli`` dispatches on data instead of a chain of ``if
args.source == ...``. It is deliberately small -- five fields -- because the heavy code stays where
the layer test wants it (the client under ``sources``, the parser under ``catalog``); an importer
only names them and the identity that ties them together.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Importer:
    """Everything ``cli`` needs to pull, cache and ingest one source, keyed by one ``id``.

    ``id`` is the single spelling of the source -- the CLI argument, the ``[sources.<id>]`` table
    key, and the raw-cache namespace, which used to be three independent strings.

    ``make_client(config, cache)`` returns the ``sources`` client, whose ``pull(identity, *,
    min_year, force_rescan, dry_run, progress, announce)`` is uniform across sources.

    ``identity(config)`` resolves what the pull is keyed on -- an archive collection, a MusicBrainz
    id -- or raises a :class:`~setlistkit.diagnostics.DiagnosticError` naming what to set. It is the
    string handed to ``pull`` and to the cache reads.

    ``candidates(config, cache, *, normalizer, min_year)`` returns this source's merge show-records,
    or ``None`` for the mirror source (archive.org), which ``ingest`` reads explicitly because it
    also needs the raw items for tapes and durations. A setlist source returns its records and
    feeds no mirror.

    ``feeds_mirror`` says whether the source carries audio: archive.org yes, setlist.fm no. It is
    what ``ingest`` splits the primary read from the fold-ins on.
    """

    id: str
    feeds_mirror: bool
    make_client: Callable
    identity: Callable
    candidates: Callable | None = None

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Importers: one place that knows a source end to end, above ``sources`` and ``catalog``.

A `source` (under ``sources/``) only fetches raw bytes; a parser (under ``catalog/``) only turns
raw into records; the layer test keeps them apart, so neither can name the other. An **importer**
is the piece that was missing: it sits above both, wires a source's client to its parser, and
declares the one identity every other layer keeps re-deriving -- the id used at the CLI, as the
``[sources.<id>]`` table key, and as the raw-cache namespace.

The point is that a third source becomes a new module here, registered once, instead of another
`if args.source == ...` branch grafted across the CLI. ``cli`` looks a source up in
:data:`IMPORTERS`; it no longer hard-codes which client or parser goes with which name.

Two roles a source can play, distinguished by :attr:`Importer.feeds_mirror`:

* **A mirror source** (archive.org) carries audio -- tapes, durations, show types -- so ``ingest``
  reads it explicitly and builds the mirror from it. Its ``candidates`` is ``None``: it is the
  primary, not a fold-in.
* **A setlist source** (setlist.fm) carries only setlists. It contributes merge candidates through
  :attr:`Importer.candidates` and feeds no mirror. Every such source folds in the same way, so a
  second one is a registry entry, nothing more.
"""

from __future__ import annotations

from . import archive_org, setlist_fm
from .base import Importer

# id -> importer. The single source of truth the CLI dispatches through; ``_SOURCES`` is its keys.
IMPORTERS: dict[str, Importer] = {imp.id: imp for imp in (archive_org.IMPORTER, setlist_fm.IMPORTER)}


def secondary() -> list[Importer]:
    """The importers that fold merge candidates in without feeding the mirror (setlist sources).

    ``ingest`` reads the mirror source itself (it needs the raw items for tapes and durations) and
    then folds every source in this list on top, so adding a setlist source changes what this
    returns and nothing else.
    """
    return [imp for imp in IMPORTERS.values() if not imp.feeds_mirror and imp.candidates]


__all__ = ["Importer", "IMPORTERS", "secondary"]

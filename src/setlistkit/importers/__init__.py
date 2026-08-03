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

Three things a source can do, and they are independent rather than a ladder:

* **Be the mirror** (archive.org). ``ingest`` reads it explicitly, because it needs the raw items
  for tapes, listings and show types as well as setlists. :attr:`Importer.feeds_mirror`. Its
  ``candidates`` is ``None``: it is the primary, not a fold-in.
* **Fold setlists in** (setlist.fm, phish.net, phish.in), through :attr:`Importer.candidates`.
  Every such source folds in the same way, so a second one is a registry entry, nothing more.
* **Contribute tapes** (phish.in), through :attr:`Importer.tapes`, without being the mirror.

phish.in does the last two at once, which is what forced them apart. It is a fold-in setlist
source AND the only thing that holds a clock for a band archive.org has no tapes of, and
``feeds_mirror`` cannot say both -- it selects the single primary. See :class:`Importer`.
"""

from __future__ import annotations

from . import archive_org, phish_in, phish_net, setlist_fm
from .base import Importer

# id -> importer. The single source of truth the CLI dispatches through; ``_SOURCES`` is its keys.
IMPORTERS: dict[str, Importer] = {imp.id: imp for imp in (archive_org.IMPORTER, setlist_fm.IMPORTER,
                                                          phish_net.IMPORTER, phish_in.IMPORTER)}


def secondary() -> list[Importer]:
    """The importers that fold merge candidates in without feeding the mirror (setlist sources).

    ``ingest`` reads the mirror source itself (it needs the raw items for tapes and durations) and
    then folds every source in this list on top, so adding a setlist source changes what this
    returns and nothing else.
    """
    return [imp for imp in IMPORTERS.values() if not imp.feeds_mirror and imp.candidates]


def tape_sources() -> list[Importer]:
    """The importers that contribute recordings to the mirror without being the mirror.

    Separate from :func:`secondary` even though phish.in is currently in both, because the two
    lists answer different questions and will not stay the same. A source can fold setlists in
    and hold no audio (setlist.fm), or hold audio and lose every date on setlists (which is what
    phish.in does against phish.net). Deriving one list from the other would tie those together.
    """
    return [imp for imp in IMPORTERS.values() if not imp.feeds_mirror and imp.tapes]


def configured(config) -> list[Importer]:
    """The registered importers this deployment actually has, in registry order.

    A ``[sources.<id>]`` table IS the opt-in, and its presence is the whole test. Registering an
    importer says the code knows how to talk to that source; writing the table says this operator
    has it. Nothing here reads what is INSIDE the table -- a source that needs a key it was not
    given raises its own diagnostic naming the key, which is a better message than any check here
    could compose, and one that stays right when a future source needs settings this function has
    never heard of.

    That last point is the reason this is a registry query rather than a list: a source added
    after today must work by being registered and configured, without a second place to update.
    """
    return [imp for imp in IMPORTERS.values() if config.section("sources", imp.id)]


def mirror(config) -> Importer | None:
    """The configured mirror source, or ``None`` when this deployment has no audio anywhere.

    ``None`` is an ordinary answer, not a failure. Phish is the worked example: archive.org
    carries no Phish tapes, so a Phish config declares two setlist APIs and no mirror at all, and
    everything the mirror would hold -- tapes, durations, show types -- simply does not exist for
    that band. Callers must PRINT the absence rather than pass over it silently; a band nobody
    taped and an archive.org table with a typo in it produce the same empty result here.
    """
    for importer in configured(config):
        if importer.feeds_mirror:
            return importer
    return None


__all__ = ["Importer", "IMPORTERS", "configured", "mirror", "secondary", "tape_sources"]

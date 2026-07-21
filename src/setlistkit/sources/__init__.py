# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Sources: pluggable ingest from archive.org, setlist.fm, Instagram, and others.

Every source client is polite by construction — cached, rate-limited, backing off on
error, and sending a mandatory identifying User-Agent taken from config. A source is one
input among several and a sanity check, never a hard dependency on a tracker's derived
aggregates.

This layer is the most upstream: it reaches the network and writes raw snapshots into the
``store`` cache, and it imports nothing from ``catalog`` (or below). Parsing those snapshots
into songs is a ``catalog`` concern, and the two layers meet at the cache, not at an import —
the same file-passing seam the old repo already ran on.
"""

from .archive_org import ArchiveOrgClient
from .client import PoliteClient, Response, SourceHTTPError, TransportError
from .keyfile import read_api_key

__all__ = [
    "ArchiveOrgClient",
    "PoliteClient",
    "Response",
    "SourceHTTPError",
    "TransportError",
    "read_api_key",
]

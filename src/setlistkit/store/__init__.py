# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""State store: the SQLite schema, migrations, and the raw file cache.

Raw source snapshots stay as files under ``<data_root>/raw/`` (cacheable, editor-openable
when a parser misbehaves). Everything derived — catalog, model outputs, the picks ledger —
lives in one SQLite database at ``<data_root>/setlistkit.sqlite``. Nothing here lives in
the repository.
"""

from __future__ import annotations

from .db import DB_FILENAME, Store
from .migrations import Migration, schema_version, transaction
from .raw_cache import RawCache

__all__ = ["Store", "RawCache", "Migration", "transaction", "schema_version", "DB_FILENAME"]

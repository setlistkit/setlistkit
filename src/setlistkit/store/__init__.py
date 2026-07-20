"""State store: the SQLite schema, migrations, and the raw file cache.

Raw source snapshots stay as files under ``<data_root>/raw/`` (cacheable, editor-openable
when a parser misbehaves). Everything derived — catalog, model outputs, the picks ledger —
lives in one SQLite database at ``<data_root>/setlistkit.sqlite``. Nothing here lives in
the repository. (Populated in a later phase.)
"""

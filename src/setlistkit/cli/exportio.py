# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""What every export bundle needs and none of them should have to write for itself.

Two things, both pulled out of ``cli/export.py`` once a second bundle needed them: writing the
file (``write_bundle``, moved verbatim -- the atomic-replace behaviour and its reasoning are
unchanged, only its address) and fingerprinting the store that produced it (``fingerprint``, new).

``fingerprint`` exists for the day setlistkit publishes more than one bundle from one store and a
site build wants to refuse a set that disagrees about which corpus it came from -- see the design
document's "One export implementation, three bundles" section. It reads the WHOLE store, never a
window: two bundles covering different windows of the SAME ingest should agree, and a fingerprint
that changed with the window could never say so. It carries no opinion about the pack -- the store
keeps no record of which pack produced it, so a change to ``vocabulary.json`` alone will not move
this number. That is a real gap and it is the honest one: the store is what this module has to
fingerprint, and no more.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Written with a trailing newline and stable key order so the file is diffable and a change to the
# data reads as a change to the data. Indented for the same reason: this is a file people open.
_JSON = {"indent": 2, "sort_keys": False, "ensure_ascii": False, "default": str}


def write_bundle(payload: dict, out: str) -> Path:
    """Serialize the bundle to ``out``, creating its directory.

    Written whole and replaced whole. A consumer polling this file should see the previous bundle
    or the next one, never four megabytes of a bundle that is still being written -- which is what
    a reader hitting a partial file gets, and it fails as a JSON parse error somewhere far from
    here.
    """
    path = Path(out).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + ".partial")
    scratch.write_text(json.dumps(payload, **_JSON) + "\n", encoding="utf-8")
    scratch.replace(path)
    return path


def fingerprint(store) -> str:
    """A short digest of the store state that produced a bundle: how many shows, how recent.

    Two cheap, already-indexed reads (``show_count``, ``show_sources``) rather than the full
    ``shows()`` walk an export already pays for elsewhere -- this runs once more per export and
    has no reason to re-read every setlist entry just to answer "how many nights, how recent".
    """
    count = store.corpus.show_count()
    dates = store.corpus.show_sources().keys()
    basis = f"{count}:{max(dates) if dates else ''}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]

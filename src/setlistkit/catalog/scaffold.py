# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The starter files that make an empty directory a valid pack.

``slkit pack init`` drops these in so ``slkit pack check`` has something to validate. Only
``pack.json`` and ``vocabulary.json`` are strictly required (see :func:`load_pack`); the rest are
optional policy files scaffolded EMPTY, so a fresh pack lints clean and the author fills them in as
the band earns them.

The contents are deliberately minimal, not a copy of ``examples/packs/example`` -- that example
carries real patterns that name example songs (``must_not_match: ["Sound Asleep"]``), which would be
wrong for any other band. An empty ``{"non_song": []}`` is valid and says nothing false.

``band_name`` is left OUT of the scaffold on purpose. It is what drives archive.org band-filtering,
and the pack's own rule is that guessing it is worse than omitting it: no ``band_name`` means no
filter, which is the safe default until a human types the real one.
"""

from __future__ import annotations

import json

VOCABULARY_JSON = "[]\n"
ALIASES_JSON = "{}\n"
PROTECTED_JSON = "[]\n"
CLASSIFIERS_JSON = '{\n  "non_song": []\n}\n'

_README = """\
# {name} pack

A pack teaches setlistkit about one band without touching its code. This directory was
scaffolded by `slkit pack init`; fill it in and run `slkit pack check`.

## The files

- **pack.json** — identity. `name` and `version` are required. Add `band_name` (what the band
  calls *itself*, e.g. "moe.") so archive.org band-filtering can tell its tapes from a side
  project's. Leaving it out runs no band filter, which is the safe default.
- **vocabulary.json** — the canonical song list, your dictionary. `slkit pack draft` can
  draft this from a raw archive.org pull.
- **aliases.json** — normalized spelling -> canonical name, for folds fuzzy matching can't reach.
- **protected.json** — titles a generic shape rule would wrongly delete (real songs that look
  like noise, e.g. "ATL").
- **classifiers.json** — `non_song` patterns that mark an entry as not-music (banter, tuning).

See `examples/packs/example` in the setlistkit source for a tiny worked example of every file.
"""


def starter_pack(name: str) -> dict[str, str]:
    """Filename -> file contents for a minimal, lint-clean pack named ``name``.

    Every value is a complete file body. The caller decides which to write -- :func:`init` never
    clobbers a file that already exists, so this can safely offer the full set alongside a
    vocabulary a draft run already wrote.
    """
    pack_json = json.dumps(
        {
            "name": name,
            "version": "0.1.0",
            "description": f"TODO: describe the {name} pack. Add a band_name to enable "
                           "archive.org band-filtering.",
            "sources": [],
        },
        indent=2,
    ) + "\n"
    return {
        "pack.json": pack_json,
        "vocabulary.json": VOCABULARY_JSON,
        "aliases.json": ALIASES_JSON,
        "protected.json": PROTECTED_JSON,
        "classifiers.json": CLASSIFIERS_JSON,
        "README.md": _README.format(name=name),
    }

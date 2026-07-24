# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `slkit bootstrap` -- the six-step pipeline behind one verb.

Nothing here mocks a step. A cache is populated the way a pull would have left it, then one
`bootstrap --skip-pull` is asked to carry an empty deployment all the way to a drafted, scaffolded
pack and a published corpus -- store init, suggest-vocab, pack init, pack lint and ingest, in that
order, off the real cache. The properties under test are the ones that make it a *pipeline* and not
just a script: the pack it scaffolds lints and is a valid pack, the corpus reaches SQLite, and a
second run does not clobber a vocabulary a human would by then have started editing.
"""

import json
from pathlib import Path

from setlistkit.catalog.pack import load_pack
from setlistkit.cli.main import EXIT_OK, main
from setlistkit.store import Store
from setlistkit.store.raw_cache import RawCache

# Two nights, each present as a titled FLAC derivative (what titles_from_cache mines) and a written
# description (what ingest parses into a setlist). Titles carry the archive noise the pipeline
# exists to fold: a track-number prefix and a spelling variant of "Bridgeless".
ITEMS = {
    "example2025-07-04": {
        "date": "2025-07-04",
        "title": "The Example Live on 2025-07-04",
        "description": "Set 1:\n01. Aurora\n02. Bridgeless >\n03. Ophelia\n",
        "files": [
            {"format": "Flac", "track": "1", "title": "01 Aurora"},
            {"format": "Flac", "track": "2", "title": "Bridgeless >"},
            {"format": "Flac", "track": "3", "title": "Ophelia"},
        ],
    },
    "example2025-07-05": {
        "date": "2025-07-05",
        "title": "The Example Live on 2025-07-05",
        "description": "Set 1:\n01. Aurora\n02. Bridgelesss\n",
        "files": [
            {"format": "Flac", "track": "1", "title": "Aurora"},
            {"format": "Flac", "track": "2", "title": "Bridgelesss"},   # a lone typo, folds in
        ],
    },
}


def _cfg(tmp_path, pack_dir):
    body = (f'data_root = "{tmp_path / "state"}"\n'
            'user_agent = "famoe.ly nightly (you@example.com)"\n'
            f'[catalog]\npack = "{pack_dir}"\n'
            '[sources.archive_org]\ncollection = "example"\n')
    path = tmp_path / "slkit.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def _cache(tmp_path):
    """Populate the raw cache the way a pull under no min_year would have left it."""
    cache = RawCache(tmp_path / "state")
    docs = [{"identifier": ident, "date": meta["date"], "title": meta["title"]}
            for ident, meta in ITEMS.items()]
    cache.put("archive_org", "advancedsearch/example/None/p1",
              json.dumps({"response": {"docs": docs, "numFound": len(docs)}}).encode("utf-8"))
    for ident, meta in ITEMS.items():
        meta = dict(meta)
        files = meta.pop("files")
        cache.put("archive_org", ident,
                  json.dumps({"metadata": meta, "files": files}).encode("utf-8"))
    return cache


def test_bootstrap_drafts_scaffolds_and_ingests(tmp_path, capsys):
    pack_dir = tmp_path / "example-pack"
    cfg = _cfg(tmp_path, pack_dir)
    _cache(tmp_path)

    assert main(["--config", cfg, "bootstrap", "--skip-pull"]) == EXIT_OK
    out = capsys.readouterr().out

    # Every one of the seven steps announced itself.
    for step in range(1, 8):
        assert f"step {step}/7" in out

    # Step 3 drafted a vocabulary off the cached titles, folding the typo into one candidate.
    vocab = json.loads((pack_dir / "vocabulary.json").read_text())
    assert "Aurora" in vocab and "Ophelia" in vocab
    assert "Bridgeless" in vocab and "Bridgelesss" not in vocab      # the fold picked the majority

    # Step 4 scaffolded the rest, leaving the drafted vocabulary untouched, and the result is a
    # pack load_pack accepts.
    assert (pack_dir / "pack.json").is_file()
    assert load_pack(pack_dir).name == "example-pack"

    # Step 6 published a corpus to SQLite.
    with Store(Path(tmp_path / "state")) as store:
        assert store.table_counts().get("shows", 0) >= 2


def test_bootstrap_dry_run_stops_before_writing_a_pack(tmp_path):
    """A dry run initializes the store and reports the pull plan, but drafts and ingests nothing."""
    pack_dir = tmp_path / "example-pack"
    cfg = _cfg(tmp_path, pack_dir)
    _cache(tmp_path)

    assert main(["--config", cfg, "bootstrap", "--skip-pull", "--dry-run"]) == EXIT_OK
    assert not pack_dir.exists()                                     # nothing drafted or scaffolded


def test_bootstrap_second_run_does_not_clobber_edited_vocabulary(tmp_path):
    """The pipeline is safe to re-run: once a human edits vocabulary.json, a re-run refuses to
    overwrite it rather than replacing their work with a fresh draft."""
    pack_dir = tmp_path / "example-pack"
    cfg = _cfg(tmp_path, pack_dir)
    _cache(tmp_path)

    assert main(["--config", cfg, "bootstrap", "--skip-pull"]) == EXIT_OK
    pack_dir.joinpath("vocabulary.json").write_text('["Hand Curated Song"]', encoding="utf-8")

    # A second bootstrap trips the refuse-to-clobber guard and exits non-zero without touching it.
    assert main(["--config", cfg, "bootstrap", "--skip-pull"]) != EXIT_OK
    assert json.loads((pack_dir / "vocabulary.json").read_text()) == ["Hand Curated Song"]

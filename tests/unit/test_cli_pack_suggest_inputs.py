# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests that `slkit pack suggest-vocab` finds its own inputs, and can diff instead of write.

The command once demanded a path for each input. ``bootstrap`` could read the configured sources
and this could not, so the hand work fell on the case that repeats and the automation on the case
that runs one time. A band whose pack is under revision is the one that redrafts.

``--diff`` answers the question that follows a later pull: which published songs does the pack
still not name? ``pack lint`` cannot answer it. Lint reads the corpus and reports titles the
parser KEPT, so a song on no tape at all never reaches it.
"""

import json

import pytest

from setlistkit.cli.main import EXIT_OK, main
from setlistkit.importers import IMPORTERS
from setlistkit.importers.base import Importer

# One song the pack will already know, and one it will not.
RECORD = {"date": "2025-07-04", "year": "2025", "source": "paper_fm", "identifier": "n1",
          "sets": [[{"song": "Aurora", "segue": False, "non_song": False},
                    {"song": "Bridgeless", "segue": False, "non_song": False}]],
          "encore": [], "n_songs": 2}


@pytest.fixture(name="paper_fm")
def _paper_fm(monkeypatch):
    """A setlist source that publishes song names and needs no key."""
    importer = Importer(
        id="paper_fm",
        display_name="paper.fm",
        feeds_mirror=False,
        make_client=lambda config, cache: None,
        identity=lambda config: "the-band",
        candidates=lambda config, cache, *, normalizer, min_year: [dict(RECORD)],
    )
    monkeypatch.setitem(IMPORTERS, importer.id, importer)
    return importer


def _cfg(tmp_path, pack_dir) -> str:
    body = (f'data_root = "{tmp_path / "state"}"\n'
            'user_agent = "famoe.ly nightly (you@example.com)"\n'
            f'[catalog]\npack = "{pack_dir}"\n'
            '[sources.paper_fm]\nkey = 1\n')
    path = tmp_path / "slkit.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def _pack(pack_dir, songs):
    """A minimal pack naming ``songs``, enough for load_pack to accept it."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "pack.json").write_text(
        json.dumps({"name": pack_dir.name, "version": "1", "band_name": "The Band"}),
        encoding="utf-8")
    (pack_dir / "vocabulary.json").write_text(json.dumps(songs), encoding="utf-8")
    (pack_dir / "aliases.json").write_text("{}", encoding="utf-8")
    return pack_dir


def test_suggest_vocab_reads_the_configured_sources_when_from_is_omitted(
        tmp_path, paper_fm, capsys):
    """No --from, no --curated: the draft comes from the same place bootstrap reads."""
    out_dir = tmp_path / "draft"
    cfg = _cfg(tmp_path, tmp_path / "pack")

    assert main(["--config", cfg, "pack", "suggest-vocab", "--out", str(out_dir)]) == EXIT_OK

    vocab = json.loads((out_dir / "vocabulary.json").read_text())
    assert {"Aurora", "Bridgeless"} <= set(vocab)
    # The run says where each input came from, so a redraft is auditable against the draft.
    assert "published song name(s) from paper_fm" in capsys.readouterr().out


def test_suggest_vocab_diff_reports_what_the_pack_lacks_and_writes_nothing(
        tmp_path, paper_fm, capsys):
    """--diff names the candidates the pack does not know, and leaves every file alone."""
    pack_dir = _pack(tmp_path / "pack", ["Aurora"])
    cfg = _cfg(tmp_path, pack_dir)
    before = (pack_dir / "vocabulary.json").read_text()

    assert main(["--config", cfg, "pack", "suggest-vocab", "--diff"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "Bridgeless" in out          # published, and absent from the pack
    assert "1 not in the pack" in out
    # Aurora is already named, so it is not reported as missing.
    assert "\n * Aurora" not in out
    assert (pack_dir / "vocabulary.json").read_text() == before


def test_suggest_vocab_diff_is_quiet_when_the_pack_already_names_everything(
        tmp_path, paper_fm, capsys):
    """A pack that knows every candidate gets a plain statement, not an empty list."""
    cfg = _cfg(tmp_path, _pack(tmp_path / "pack", ["Aurora", "Bridgeless"]))

    assert main(["--config", cfg, "pack", "suggest-vocab", "--diff"]) == EXIT_OK
    assert "already names every candidate" in capsys.readouterr().out


def test_suggest_vocab_matches_a_pack_that_spells_a_song_its_own_way(
        tmp_path, paper_fm, capsys):
    """Comparison is by normalized key. A different spelling of a known song is not missing."""
    cfg = _cfg(tmp_path, _pack(tmp_path / "pack", ["aurora", "BRIDGELESS"]))

    assert main(["--config", cfg, "pack", "suggest-vocab", "--diff"]) == EXIT_OK
    assert "already names every candidate" in capsys.readouterr().out


def test_suggest_vocab_without_out_or_diff_says_which_to_pass(tmp_path, paper_fm):
    """--out is required for a write. The refusal names the two ways forward."""
    cfg = _cfg(tmp_path, tmp_path / "pack")

    assert main(["--config", cfg, "pack", "suggest-vocab"]) != EXIT_OK


def test_suggest_vocab_still_takes_explicit_paths(tmp_path, paper_fm):
    """An explicit --from wins over the config, so the old invocation keeps working."""
    titles = tmp_path / "titles.json"
    titles.write_text(json.dumps(["01 Ophelia", "Ophelia"]), encoding="utf-8")
    out_dir = tmp_path / "draft"
    cfg = _cfg(tmp_path, tmp_path / "pack")

    assert main(["--config", cfg, "pack", "suggest-vocab",
                 "--from", str(titles), "--out", str(out_dir)]) == EXIT_OK

    vocab = json.loads((out_dir / "vocabulary.json").read_text())
    assert "Ophelia" in vocab
    assert "Aurora" not in vocab        # the configured source was not consulted

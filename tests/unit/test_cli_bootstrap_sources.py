# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests that `slkit bootstrap` reads the config's sources rather than one hard-coded name.

The properties here are about GENERALITY, which is why none of them names archive.org except to
prove its absence is survivable. A deployment declares the sources it has by writing a
``[sources.<id>]`` table; bootstrap works from that list and nothing else. The worked example is
Phish -- archive.org carries no Phish tapes, so its config has no archive.org table at all -- but
the tests use a source registered at runtime instead, because the requirement is that an
ARBITRARY future source works, not that the four shipped today do.
"""

import json

import pytest

from setlistkit.cli.main import EXIT_OK, main
from setlistkit.importers import IMPORTERS
from setlistkit.importers.base import Importer
from setlistkit.sources.base import PullResult
from setlistkit.store import Store

# Two nights, as a setlist source publishes them: song names already typed by a human, no audio
# and so no tapes. The spelling variant is deliberate -- it is what proves the draft clustered
# these names rather than passing them through.
RECORDS = [
    {"date": "2025-07-04", "year": "2025", "source": "made_up_fm", "identifier": "n1",
     "sets": [[{"song": "Aurora", "segue": False, "non_song": False},
               {"song": "Bridgeless", "segue": False, "non_song": False},
               {"song": "Ophelia", "segue": False, "non_song": False}]],
     "encore": [], "n_songs": 3},
    {"date": "2025-07-05", "year": "2025", "source": "made_up_fm", "identifier": "n2",
     "sets": [[{"song": "Aurora", "segue": False, "non_song": False},
               {"song": "Bridgeless", "segue": False, "non_song": False}]],
     "encore": [], "n_songs": 2},
]


@pytest.fixture(name="made_up_fm")
def _made_up_fm(monkeypatch):
    """Register a setlist source that exists nowhere in the shipped registry.

    A fake rather than setlist.fm because the claim under test is that bootstrap dispatches
    through the registry. A test written against a source the code could still be naming
    explicitly cannot tell the two apart.
    """
    pulls: list[str] = []

    class _Client:
        def __init__(self, config, cache):
            pass

        def pull(self, identity, **kwargs):
            pulls.append(identity)
            # The real result type, not a stand-in. A stand-in that omits a counter the CLI
            # reports makes the test pass while a real source of this shape fails.
            return PullResult(listed=len(RECORDS), cached=len(RECORDS), planned=len(RECORDS))

    importer = Importer(
        id="made_up_fm",
        display_name="made-up.fm",
        feeds_mirror=False,
        make_client=_Client,
        identity=lambda config: "the-band",
        candidates=lambda config, cache, *, normalizer, min_year: list(RECORDS),
    )
    monkeypatch.setitem(IMPORTERS, importer.id, importer)
    yield pulls
    # monkeypatch.setitem restores the registry; `pulls` is what the test inspects.


def _cfg(tmp_path, pack_dir, *, sources: str) -> str:
    """A config declaring exactly the sources named, and no others."""
    body = (f'data_root = "{tmp_path / "state"}"\n'
            'user_agent = "famoe.ly nightly (you@example.com)"\n'
            f'[catalog]\npack = "{pack_dir}"\n' + sources)
    path = tmp_path / "slkit.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_bootstrap_runs_with_no_archive_org_configured(tmp_path, made_up_fm, capsys):
    """A deployment whose only source publishes setlists bootstraps end to end.

    This is the Phish case. Nothing here is cached under archive.org and no archive.org table
    exists, and the run must neither demand one nor stop at the draft for want of tape titles:
    the names a setlist source publishes are a better vocabulary than mined titles anyway.
    """
    pack_dir = tmp_path / "example-pack"
    cfg = _cfg(tmp_path, pack_dir, sources="[sources.made_up_fm]\nkey = 1\n")

    assert main(["--config", cfg, "bootstrap", "--skip-pull"]) == EXIT_OK

    vocab = json.loads((pack_dir / "vocabulary.json").read_text())
    assert {"Aurora", "Bridgeless", "Ophelia"} <= set(vocab)

    with Store(tmp_path / "state") as store:
        assert store.table_counts().get("shows", 0) >= 2

    # The absence is reported, not inferred. A run that quietly skipped the mirror looks exactly
    # like a run whose archive.org table was mistyped.
    assert "mirror" in capsys.readouterr().out.lower()


def test_bootstrap_pulls_every_configured_source(tmp_path, made_up_fm):
    """Step 2 visits each source the config declares, not one chosen in the source code."""
    pack_dir = tmp_path / "example-pack"
    cfg = _cfg(tmp_path, pack_dir, sources="[sources.made_up_fm]\nkey = 1\n")

    assert main(["--config", cfg, "bootstrap"]) == EXIT_OK
    assert made_up_fm == ["the-band"]


def test_bootstrap_ignores_a_source_with_no_config_table(tmp_path, made_up_fm):
    """Registering a source is not configuring it. Only a ``[sources.<id>]`` table opts one in."""
    pack_dir = tmp_path / "example-pack"
    cfg = _cfg(tmp_path, pack_dir, sources="")

    # No source at all: nothing to draft a vocabulary from, and that is a stated error rather
    # than an empty pack nobody notices.
    assert main(["--config", cfg, "bootstrap", "--skip-pull"]) != EXIT_OK
    assert made_up_fm == []

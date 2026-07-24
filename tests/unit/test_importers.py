# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the importer registry: the contract every source's module fills in.

The registry is what lets the CLI dispatch on data instead of a chain of ``if args.source == ...``.
These lock the shape a new source has to match to be pullable and ingestible, and the one rule the
whole design turns on: a source either feeds the mirror (it has audio) or it contributes merge
candidates, and exactly which decides how ``ingest`` treats it.
"""

import pytest

from setlistkit.cli import main as cli
from setlistkit.config import load_config
from setlistkit.diagnostics import DiagnosticError
from setlistkit.importers import IMPORTERS, secondary
from setlistkit.importers.base import Importer


def test_every_importer_is_keyed_by_its_own_id():
    """The registry key and the descriptor's id are the same string -- the one spelling of a source."""
    assert all(key == imp.id for key, imp in IMPORTERS.items())
    assert set(IMPORTERS) == {"archive_org", "setlist_fm"}


def test_sources_choices_come_from_the_registry():
    """`slkit pull`'s accepted names ARE the registry keys, so the two cannot drift apart."""
    assert cli._SOURCES == tuple(IMPORTERS)


def test_every_descriptor_is_complete():
    for imp in IMPORTERS.values():
        assert isinstance(imp, Importer)
        assert isinstance(imp.id, str) and imp.id
        assert callable(imp.make_client)
        assert callable(imp.identity)
        assert isinstance(imp.feeds_mirror, bool)


def test_a_mirror_source_folds_in_no_candidates_and_a_setlist_source_does():
    """The rule ingest splits on: the mirror source (archive) is read explicitly and has no
    candidates hook; a setlist source (setlist.fm) contributes candidates and feeds no mirror."""
    archive, setlist = IMPORTERS["archive_org"], IMPORTERS["setlist_fm"]
    assert archive.feeds_mirror and archive.candidates is None
    assert not setlist.feeds_mirror and callable(setlist.candidates)
    # `secondary()` is exactly the fold-in sources -- the setlist ones, never the mirror.
    assert [imp.id for imp in secondary()] == ["setlist_fm"]


def _cfg(tmp_path, body):
    path = tmp_path / "slkit.toml"
    path.write_text(body, encoding="utf-8")
    return load_config(str(path))


def test_archive_identity_names_the_table_when_the_collection_is_missing(tmp_path):
    config = _cfg(tmp_path, 'data_root = "s"\nuser_agent = "x (y@z.com)"\n')
    with pytest.raises(DiagnosticError) as exc:
        IMPORTERS["archive_org"].identity(config)
    assert "[sources.archive_org] collection is not set" in exc.value.diagnostic.summary


def test_setlistfm_identity_points_at_the_top_level_mbid_when_missing(tmp_path):
    config = _cfg(tmp_path, 'data_root = "s"\nuser_agent = "x (y@z.com)"\n')
    with pytest.raises(DiagnosticError) as exc:
        IMPORTERS["setlist_fm"].identity(config)
    assert "MusicBrainz id" in exc.value.diagnostic.summary


def test_setlistfm_candidates_are_empty_when_unconfigured(tmp_path):
    """No key file and no mbid: the source sits out, contributing nothing, without raising."""
    from setlistkit.catalog.normalizer import Normalizer
    from setlistkit.store.raw_cache import RawCache
    config = _cfg(tmp_path, 'data_root = "s"\nuser_agent = "x (y@z.com)"\n')
    got = IMPORTERS["setlist_fm"].candidates(
        config, RawCache(tmp_path / "s"), normalizer=Normalizer(vocabulary=[]), min_year=None)
    assert got == []

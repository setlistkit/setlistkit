# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the importer registry: the contract every source's module fills in.

The registry is what lets the CLI dispatch on data instead of a chain of ``if args.source == ...``.
These lock the shape a new source has to match to be pullable and ingestible, and the one rule the
whole design turns on: a source either feeds the mirror (it has audio) or it contributes merge
candidates, and exactly which decides how ``ingest`` treats it.
"""

import inspect

import pytest

from setlistkit.cli import main as cli
from setlistkit.config import load_config
from setlistkit.diagnostics import DiagnosticError
from setlistkit.importers import IMPORTERS, secondary
from setlistkit.importers.base import Importer
from setlistkit.sources.archive_org import ArchiveOrgClient
from setlistkit.sources.base import SourceClient
from setlistkit.sources.phish_in import PhishInClient
from setlistkit.sources.phish_net import PhishNetClient
from setlistkit.sources.setlist_fm import SetlistFmClient


def test_every_importer_is_keyed_by_its_own_id():
    """The registry key and the descriptor's id are the same string -- the one spelling of a source."""
    assert all(key == imp.id for key, imp in IMPORTERS.items())
    assert set(IMPORTERS) == {"archive_org", "setlist_fm", "phish_net", "phish_in"}


def test_sources_choices_come_from_the_registry():
    """`slkit pull`'s accepted names ARE the registry keys, so the two cannot drift apart."""
    assert cli._SOURCES == tuple(IMPORTERS)


def test_every_descriptor_is_complete():
    for imp in IMPORTERS.values():
        assert isinstance(imp, Importer)
        assert isinstance(imp.id, str) and imp.id
        assert isinstance(imp.display_name, str) and imp.display_name
        assert callable(imp.make_client)
        assert callable(imp.identity)
        assert isinstance(imp.feeds_mirror, bool)


def test_id_is_the_machine_key_and_display_name_is_the_real_name():
    """The two spellings are kept apart on purpose: `id` is the underscore form safe as a CLI arg,
    a [sources.<id>] TOML table key, and a cache dir; `display_name` is the dotted real-world name
    shown to humans. A dotted id would nest as a TOML table, so the split is not cosmetic."""
    assert IMPORTERS["archive_org"].display_name == "archive.org"
    assert IMPORTERS["setlist_fm"].display_name == "setlist.fm"
    # The machine ids stay underscore -- never a dot, which argparse tolerates but TOML would nest.
    assert all("." not in imp.id for imp in IMPORTERS.values())
    # And the display name is the one place the dot lives.
    assert all("_" not in imp.display_name for imp in IMPORTERS.values())


def test_a_mirror_source_folds_in_no_candidates_and_a_setlist_source_does():
    """The rule ingest splits on: the mirror source (archive) is read explicitly and has no
    candidates hook; every setlist source contributes candidates and feeds no mirror.

    Three setlist sources now, against one mirror. That ratio is the design working: adding
    phish.net and phish.in was a registry entry apiece, exactly as the module docstring promised,
    and nothing about this rule had to change to hold them."""
    archive = IMPORTERS["archive_org"]
    assert archive.feeds_mirror and archive.candidates is None
    for setlist_id in ("setlist_fm", "phish_net", "phish_in"):
        imp = IMPORTERS[setlist_id]
        assert not imp.feeds_mirror and callable(imp.candidates)
    # `secondary()` is exactly the fold-in sources -- the setlist ones, never the mirror.
    assert [imp.id for imp in secondary()] == ["setlist_fm", "phish_net", "phish_in"]


def _keyword_only_shape(func):
    """The pull contract minus its first positional. `self` and the identity slot (whose NAME is
    allowed to differ -- collection vs mbid) are dropped; what must match is the keyword-only block:
    same params, same kinds, same defaults. That is the part `issubclass` cannot see."""
    params = list(inspect.signature(func).parameters.values())
    return [(p.name, p.kind, p.default) for p in params[2:]]


def test_every_client_is_a_source_client_by_shape_not_inheritance():
    """The pull surface is a structural contract, so a conformance test is what actually enforces
    it -- no base class does. `issubclass` (runtime_checkable) confirms `pull` exists; the shape
    check confirms its keyword block did not drift, which is how a third source would break it."""
    want = _keyword_only_shape(SourceClient.pull)
    for client in (ArchiveOrgClient, SetlistFmClient, PhishNetClient, PhishInClient):
        assert issubclass(client, SourceClient)                  # the method is present...
        assert _keyword_only_shape(client.pull) == want         # ...and its keyword shape matches
        # None inherits the Protocol: they satisfy it structurally, which is the whole point.
        assert SourceClient not in client.__mro__


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

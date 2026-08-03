# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the source that is configured, is cached, and still gives nothing.

This is the failure these tests exist for. An operator writes a ``[sources.<id>]`` table and
pulls it. One required setting is absent or wrong, so the importer can name no band and returns
no records. Every command then runs, publishes and exits 0 on a corpus missing most of its shows.

Measured on the real Umphrey's McGee config, whose MusicBrainz id sat in the wrong table as a
placeholder: the draft recalled 58.5% of the hand-tuned vocabulary instead of 91.0%, and the
corpus held 122 shows instead of 468. Nothing reported a fault.

Two answers, and the tests cover both. ``config check`` asks each configured source whether it
can name its band, so the fault is caught before a pull. The draft step names each source with
its contribution, so a zero is visible while the run is in front of somebody.
"""

import pytest

from setlistkit.cli.main import EXIT_OK, main
from setlistkit.diagnostics import ERROR, Diagnostic, DiagnosticError
from setlistkit.importers import IMPORTERS
from setlistkit.importers.base import Importer

RECORD = {"date": "2025-07-04", "year": "2025", "source": "mute_fm", "identifier": "n1",
          "sets": [[{"song": "Aurora", "segue": False, "non_song": False}]],
          "encore": [], "n_songs": 1}


def _needs_a_setting(config):
    """Refuse to name a band, the way a real importer refuses when its setting is absent."""
    raise DiagnosticError(Diagnostic(
        severity=ERROR,
        summary="mute.fm needs the band's MusicBrainz id, and none is configured",
        detail='Add the band\'s mbid as a TOP-LEVEL key in your config, beside data_root.',
    ))


@pytest.fixture(name="mute_fm")
def _mute_fm(monkeypatch):
    """A source that is configured, holds a cache, and returns nothing because a setting is absent.

    The candidates function returns an empty list rather than raising. That is what the shipped
    setlist sources do, and it is what makes the loss silent.
    """
    importer = Importer(
        id="mute_fm",
        display_name="mute.fm",
        feeds_mirror=False,
        make_client=lambda config, cache: None,
        identity=_needs_a_setting,
        candidates=lambda config, cache, *, normalizer, min_year: [],
    )
    monkeypatch.setitem(IMPORTERS, importer.id, importer)
    return importer


@pytest.fixture(name="loud_fm")
def _loud_fm(monkeypatch):
    """A source with everything it needs, so a test can tell one zero from another."""
    importer = Importer(
        id="loud_fm",
        display_name="loud.fm",
        feeds_mirror=False,
        make_client=lambda config, cache: None,
        identity=lambda config: "the-band",
        candidates=lambda config, cache, *, normalizer, min_year: [dict(RECORD)],
    )
    monkeypatch.setitem(IMPORTERS, importer.id, importer)
    return importer


def _cfg(tmp_path, pack_dir, *, sources: str) -> str:
    body = (f'data_root = "{tmp_path / "state"}"\n'
            'user_agent = "famoe.ly nightly (you@example.com)"\n'
            f'[catalog]\npack = "{pack_dir}"\n' + sources)
    path = tmp_path / "slkit.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_config_check_fails_on_a_source_that_cannot_name_its_band(tmp_path, mute_fm, capsys):
    """The check asks each configured source to resolve its identity, and reports every refusal."""
    cfg = _cfg(tmp_path, tmp_path / "pack", sources="[sources.mute_fm]\nkey = 1\n")

    assert main(["--config", cfg, "check"]) != EXIT_OK
    err = capsys.readouterr().err
    assert "mute.fm is configured but cannot name its band" in err
    # The source's OWN diagnostic reaches the operator. A message composed here would have to
    # guess which setting each source needs, and it would guess wrong for a source added later.
    assert "MusicBrainz id" in err


def test_config_check_reports_every_broken_source_not_only_the_first(
        tmp_path, mute_fm, monkeypatch, capsys):
    """One refusal must not hide the next. An operator fixing one at a time repeats the pull."""
    second = Importer(id="mute2_fm", display_name="mute2.fm", feeds_mirror=False,
                      make_client=lambda config, cache: None, identity=_needs_a_setting,
                      candidates=lambda config, cache, *, normalizer, min_year: [])
    monkeypatch.setitem(IMPORTERS, second.id, second)
    cfg = _cfg(tmp_path, tmp_path / "pack",
               sources="[sources.mute_fm]\nkey = 1\n[sources.mute2_fm]\nkey = 1\n")

    assert main(["--config", cfg, "check"]) != EXIT_OK
    err = capsys.readouterr().err
    assert "mute.fm is configured but cannot name its band" in err
    assert "mute2.fm is configured but cannot name its band" in err


def test_config_check_passes_and_names_the_sources_it_asked(tmp_path, loud_fm, capsys):
    """A config whose sources can all name their band passes, and says which it asked."""
    cfg = _cfg(tmp_path, tmp_path / "pack", sources="[sources.loud_fm]\nkey = 1\n")

    assert main(["--config", cfg, "check"]) == EXIT_OK
    assert "loud_fm" in capsys.readouterr().out


def test_bootstrap_names_a_configured_source_that_gave_nothing(
        tmp_path, mute_fm, loud_fm, capsys):
    """The draft step reports each source's contribution, so a zero cannot pass unseen.

    The run still finishes. A source giving nothing is not fatal on its own, because the other
    sources may carry the whole corpus. What must not happen is silence.
    """
    cfg = _cfg(tmp_path, tmp_path / "pack",
               sources="[sources.mute_fm]\nkey = 1\n[sources.loud_fm]\nkey = 1\n")

    assert main(["--config", cfg, "bootstrap", "--skip-pull"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "0 published song name(s) from mute_fm" in out
    assert "configured, gave nothing" in out
    assert "slkit check" in out
    # The working source is reported by the same line, so the two are comparable at a glance.
    assert "1 published song name(s) from loud_fm" in out

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `slkit pull`: config resolution, the sentinel refusal, and the report.

The fetching itself is tested against a fake transport in test_archive_org.py. What is tested
here is everything the CLI adds around it, and the one thing it must do BEFORE any of it: a
placeholder user_agent stops the run without a client ever being built.
"""

import pytest

from setlistkit.cli import main as cli
from setlistkit.cli.main import EXIT_DIAGNOSTIC, EXIT_OK, main
from setlistkit.config import SENTINEL_USER_AGENT
from setlistkit.sources.archive_org import PullResult
from setlistkit.sources.client import SourceFormatError, SourceHTTPError, TransportError

CONFIG = ('data_root = "state"\n'
          'user_agent = "famoe.ly nightly (you@example.com)"\n'
          '[sources.archive_org]\n'
          'collection = "moe"\n')


def _cfg(tmp_path, body=CONFIG):
    path = tmp_path / "slkit.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


class FakeClient:
    """Stands in for ArchiveOrgClient, recording how the CLI called it."""

    def __init__(self, result=None):
        self.result = result or PullResult(listed=3, fetched=1, cached=2)
        self.calls = []

    def __call__(self, config, cache):
        return self

    def pull(self, collection, *, min_year=None, force_rescan=False, progress=None,
             announce=None):
        self.calls.append({"collection": collection, "min_year": min_year,
                           "force_rescan": force_rescan})
        if announce is not None:
            announce("3f7a9c21")
        if progress is not None:
            progress(1, 1)
        return self.result


@pytest.fixture(name="client")
def _client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(cli, "ArchiveOrgClient", fake)
    return fake


def test_pull_reports_what_it_fetched(tmp_path, capsys, client):
    assert main(["--config", _cfg(tmp_path), "pull", "archive_org"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "3 listed, 1 fetched, 2 already cached" in out


def test_pull_prints_the_batch_id_the_source_will_also_see(tmp_path, capsys, client):
    """A tracking id only one side of the conversation knows is half a tracking id."""
    main(["--config", _cfg(tmp_path), "pull", "archive_org"])
    assert "batch 3f7a9c21" in capsys.readouterr().out


def test_pull_passes_the_configured_collection(tmp_path, client):
    main(["--config", _cfg(tmp_path), "pull", "archive_org"])
    assert client.calls == [{"collection": "moe", "min_year": None, "force_rescan": False}]


def test_pull_force_rescan_reaches_the_client(tmp_path, client):
    main(["--config", _cfg(tmp_path), "pull", "archive_org", "--force-rescan"])
    assert client.calls[0]["force_rescan"] is True


def test_pull_reads_min_year_from_config(tmp_path, client):
    main(["--config", _cfg(tmp_path, CONFIG + "min_year = 2020\n"), "pull", "archive_org"])
    assert client.calls[0]["min_year"] == 2020


def test_pull_flag_overrides_the_configured_min_year(tmp_path, client):
    main(["--config", _cfg(tmp_path, CONFIG + "min_year = 2020\n"),
          "pull", "archive_org", "--min-year", "2015"])
    assert client.calls[0]["min_year"] == 2015


def test_pull_rejects_a_quoted_min_year(tmp_path, capsys, client):
    body = CONFIG + 'min_year = "2020"\n'
    assert main(["--config", _cfg(tmp_path, body), "pull", "archive_org"]) == EXIT_DIAGNOSTIC
    assert "min_year must be a number" in capsys.readouterr().err
    assert client.calls == []


def test_pull_without_a_collection_says_which_table_to_set_it_in(tmp_path, capsys, client):
    body = 'data_root = "state"\nuser_agent = "famoe.ly nightly (you@example.com)"\n'
    assert main(["--config", _cfg(tmp_path, body), "pull", "archive_org"]) == EXIT_DIAGNOSTIC
    err = capsys.readouterr().err
    assert "[sources.archive_org] collection is not set" in err
    assert "[sources.archive_org]" in err
    assert client.calls == []


def test_pull_refuses_the_placeholder_user_agent(tmp_path, capsys, client):
    body = f'data_root = "state"\nuser_agent = "{SENTINEL_USER_AGENT}"\n'
    assert main(["--config", _cfg(tmp_path, body), "pull", "archive_org"]) == EXIT_DIAGNOSTIC
    assert "refusing to touch the network" in capsys.readouterr().err
    assert client.calls == []


def test_pull_names_the_items_the_metadata_api_did_not_have(tmp_path, capsys, monkeypatch):
    fake = FakeClient(PullResult(listed=2, fetched=1, cached=0, missing=("gone",)))
    monkeypatch.setattr(cli, "ArchiveOrgClient", fake)
    main(["--config", _cfg(tmp_path), "pull", "archive_org"])
    out = capsys.readouterr().out
    assert "1 listed item(s) the metadata API does not have" in out
    assert "gone" in out


def test_pull_warns_loudly_when_the_listing_was_truncated(tmp_path, capsys, monkeypatch):
    fake = FakeClient(PullResult(listed=20000, fetched=0, cached=20000, truncated=True))
    monkeypatch.setattr(cli, "ArchiveOrgClient", fake)
    main(["--config", _cfg(tmp_path), "pull", "archive_org"])
    out = capsys.readouterr().out
    # An ingest over a truncated listing looks complete and is not, so this cannot be silent.
    assert "PREFIX of the collection" in out


def test_pull_rejects_an_unknown_source(tmp_path, capsys, client):
    with pytest.raises(SystemExit):
        main(["--config", _cfg(tmp_path), "pull", "myspace"])
    assert "invalid choice" in capsys.readouterr().err


def test_pull_counts_listing_docs_that_carried_no_identifier(tmp_path, capsys, monkeypatch):
    fake = FakeClient(PullResult(listed=1, fetched=1, unidentified=2))
    monkeypatch.setattr(cli, "ArchiveOrgClient", fake)
    main(["--config", _cfg(tmp_path), "pull", "archive_org"])
    assert "2 listed item(s) carried no identifier" in capsys.readouterr().out


def test_pull_reads_settings_from_the_table_named_by_the_source_argument(tmp_path, client):
    # With one source these are the same string. With two, a hard-coded table name makes
    # `slkit pull setlistfm` quietly read the archive.org settings and pull archive.org.
    monkeypatch_sources = getattr(cli, "_SOURCES")
    assert "archive_org" in monkeypatch_sources
    body = CONFIG + '[sources.other]\ncollection = "not-this-one"\n'
    main(["--config", _cfg(tmp_path, body), "pull", "archive_org"])
    assert client.calls[0]["collection"] == "moe"


@pytest.mark.parametrize("error,expected", [
    (SourceHTTPError("https://archive.org/x", 500), "returned HTTP 500"),
    (TransportError("https://archive.org/x: connection reset"), "connection reset"),
    (SourceFormatError("https://archive.org/x", "Expecting value"), "did not return usable JSON"),
])
def test_pull_renders_an_upstream_failure_instead_of_a_traceback(tmp_path, capsys, monkeypatch,
                                                                 error, expected):
    """main() promises it never raises past its boundary; a source having a bad day is the
    likeliest way to find out it does. The JSON case is the realistic one: a site under
    maintenance answers 200 with an HTML page, and no status code says so."""
    class Failing(FakeClient):
        def pull(self, collection, **kwargs):
            raise error

    monkeypatch.setattr(cli, "ArchiveOrgClient", Failing())
    assert main(["--config", _cfg(tmp_path), "pull", "archive_org"]) == EXIT_DIAGNOSTIC
    err = capsys.readouterr().err
    assert "the source could not be read" in err and expected in err
    assert "Nothing was lost" in err

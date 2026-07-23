# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `cli/exportio.py`: the write-the-file and fingerprint-the-store logic every export
sub-verb now shares.

`write_bundle` moved here verbatim from `cli/export.py`'s old `_write` -- its atomic-replace
behaviour is already exercised by every test in `test_cli_export.py` that reads a bundle back off
disk, so what is pinned here is only that the move did not change it. `fingerprint` is new.
"""

import json

from setlistkit.cli.exportio import fingerprint, write_bundle
from setlistkit.store import Store


def test_write_bundle_creates_parent_directories(tmp_path):
    out = write_bundle({"a": 1}, str(tmp_path / "nested" / "dir" / "bundle.json"))
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1}


def test_write_bundle_replaces_whole_rather_than_writing_in_place(tmp_path):
    out = write_bundle({"a": 1}, str(tmp_path / "bundle.json"))
    assert not out.with_name(out.name + ".partial").exists()


def test_write_bundle_is_indented_with_key_order_preserved(tmp_path):
    out = write_bundle({"b": 2, "a": 1}, str(tmp_path / "bundle.json"))
    assert out.read_text(encoding="utf-8") == '{\n  "b": 2,\n  "a": 1\n}\n'


def _show(date):
    return {"date": date, "source": "test", "identifier": date,
            "sets": [[{"song": "Aurora", "segue": False, "non_song": False}]], "encore": []}


def test_fingerprint_is_stable_across_two_calls_over_the_same_store(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        store.corpus.replace_shows([_show("2025-07-04"), _show("2025-07-05")])
        assert fingerprint(store) == fingerprint(store)


def test_fingerprint_changes_when_the_corpus_changes(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        store.corpus.replace_shows([_show("2025-07-04")])
        before = fingerprint(store)
        store.corpus.replace_shows([_show("2025-07-04"), _show("2025-07-05")])
        after = fingerprint(store)
    assert before != after


def test_fingerprint_is_unaffected_by_a_windowed_read_elsewhere(tmp_path):
    """The fingerprint describes the whole store, not whatever window a caller happens to be
    exporting -- two bundles covering different windows of the SAME ingest must agree."""
    with Store(tmp_path) as store:
        store.init()
        store.corpus.replace_shows([_show("2025-07-04"), _show("2025-07-05")])
        before = fingerprint(store)
        store.corpus.shows(since="2025-07-05")            # a windowed read, as a real export makes
        assert fingerprint(store) == before

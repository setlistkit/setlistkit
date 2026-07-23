# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `scripts/songbook_parity.py`'s pure logic -- the intersection arithmetic and the
POC-JSON extraction -- using tiny synthetic fixtures instead of a real POC page or a real corpus.

Loaded by path rather than imported normally: the script lives outside `src/`, is not part of the
installed package, and is not meant to be. It is a one-off verification tool run by hand as the
Slice 1 gate (see the design document's Verification section), not a module anything imports.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "songbook_parity.py"
_spec = importlib.util.spec_from_file_location("songbook_parity", SCRIPT)
songbook_parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(songbook_parity)


def _poc_html(vocab, shows):
    payload = {"meta": {"catalog": len(vocab)}, "vocab": vocab, "shows": shows}
    return (f'<html><body><script id="data" type="application/json">{json.dumps(payload)}'
            "</script></body></html>")


def test_load_poc_extracts_the_embedded_data_block(tmp_path):
    path = tmp_path / "index.html"
    path.write_text(_poc_html(["Aurora"], [{"d": "2020-01-08", "s": [0]}]), encoding="utf-8")
    data = songbook_parity._load_poc(path)
    assert data["vocab"] == ["Aurora"]
    assert data["shows"] == [{"d": "2020-01-08", "s": [0]}]


def test_load_poc_refuses_a_file_with_no_data_block(tmp_path):
    path = tmp_path / "index.html"
    path.write_text("<html><body>no data here</body></html>", encoding="utf-8")
    with pytest.raises(SystemExit):
        songbook_parity._load_poc(path)


def test_plays_by_date_reads_shows_containing_not_performance_counts():
    by_date = songbook_parity._plays_by_date(["Aurora", "Wormhole"],
                                             [{"d": "2020-01-08", "s": [0, 0, 1]}])
    # Even though index 0 appears twice in one show's list, it is one show containing the song.
    assert by_date["2020-01-08"]["Aurora"] == 1
    assert by_date["2020-01-08"]["Wormhole"] == 1


def test_total_plays_sums_shows_containing_across_dates():
    by_date = {"2020-01-08": songbook_parity.Counter({"Aurora": 1}),
               "2020-01-10": songbook_parity.Counter({"Aurora": 1, "Wormhole": 1})}
    total = songbook_parity._total_plays(by_date)
    assert total == {"Aurora": 2, "Wormhole": 1}


def test_main_passes_when_the_intersection_agrees_exactly(tmp_path, capsys):
    poc = tmp_path / "index.html"
    poc.write_text(_poc_html(["Aurora", "Wormhole"],
                             [{"d": "2020-01-08", "s": [0, 1]}]), encoding="utf-8")
    bundle = tmp_path / "songbook.json"
    bundle.write_text(json.dumps({"schema": "setlistkit.songbook/1",
                                  "vocab": ["Aurora", "Wormhole"],
                                  "shows": [{"d": "2020-01-08", "s": [0, 1]}]}),
                      encoding="utf-8")
    code = songbook_parity.main(["--poc", str(poc), "--bundle", str(bundle)])
    assert code == 0
    assert "PASS" in capsys.readouterr().out


def test_main_fails_when_a_shared_date_disagrees(tmp_path, capsys):
    poc = tmp_path / "index.html"
    poc.write_text(_poc_html(["Aurora"], [{"d": "2020-01-08", "s": [0]}]), encoding="utf-8")
    bundle = tmp_path / "songbook.json"
    bundle.write_text(json.dumps({"schema": "setlistkit.songbook/1", "vocab": ["Aurora"],
                                  "shows": [{"d": "2020-01-08", "s": []}]}), encoding="utf-8")
    code = songbook_parity.main(["--poc", str(poc), "--bundle", str(bundle)])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_fails_when_setlistkit_holds_a_night_the_poc_does_not(tmp_path, capsys):
    """The design document's own warning: this direction means 'missing sources' is the wrong
    story, and the gate has to say so rather than pass quietly."""
    poc = tmp_path / "index.html"
    poc.write_text(_poc_html(["Aurora"], [{"d": "2020-01-08", "s": [0]}]), encoding="utf-8")
    bundle = tmp_path / "songbook.json"
    bundle.write_text(json.dumps({"schema": "setlistkit.songbook/1", "vocab": ["Aurora"],
                                  "shows": [{"d": "2020-01-08", "s": [0]},
                                            {"d": "2020-01-09", "s": [0]}]}),
                      encoding="utf-8")
    code = songbook_parity.main(["--poc", str(poc), "--bundle", str(bundle)])
    assert code == 1
    assert "setlistkit holds a night the POC does not" in capsys.readouterr().out

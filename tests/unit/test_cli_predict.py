# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `slkit predict`, `slkit replay`, and `slkit show picks`, driven through ``main``.

Nothing here mocks the model. A corpus is written into a real store, the commands are invoked the
way a person invokes them, and the assertions are on what lands on stdout and in SQLite.

NEITHER COMMAND MAY READ THE CLOCK, and two tests say so by freezing the corpus and running twice.
That is the rule ``catalog/window.py`` states at length: a window anchored on today makes two runs
over the same store disagree, and a replay whose span moved between runs is a replay whose numbers
cannot be compared to the ones published last week.
"""

import sqlite3

import pytest

from setlistkit.cli.main import EXIT_OK, main
from setlistkit.store import Store

CONFIG = ('data_root = "state"\n'
          'user_agent = "famoe.ly nightly (you@example.com)"\n')

REGULARS = ("Aurora", "Rebubula", "Meat", "Zelda", "Buster", "Wormhole", "Gumbo", "Cavern",
            "Timmy", "Moth")


def _entry(song):
    return {"song": song, "segue": False, "non_song": False}


def _show(date, *songs):
    return {"date": date, "source": "description", "identifier": f"t{date}",
            "sets": [[_entry(s) for s in songs]], "encore": []}


def _cfg(tmp_path, stanza="[model]\nrollouts = 25\ntop_n = 10\n"):
    path = tmp_path / "slkit.toml"
    path.write_text(CONFIG + stanza, encoding="utf-8")
    return str(path)


@pytest.fixture(name="ready")
def _ready_fixture(tmp_path):
    """A config path pointing at a store holding twelve identical nights."""
    config = _cfg(tmp_path)
    with Store(tmp_path / "state") as store:
        store.init()
        store.corpus.replace_shows([_show(f"2026-01-{day:02d}", *REGULARS)
                                    for day in range(1, 13)])
    return config


def _run(*argv):
    return main(list(argv))


def test_predict_ranks_the_songs_and_constructs_a_night(ready, capsys):
    assert _run("--config", ready, "predict") == EXIT_OK
    out = capsys.readouterr().out
    assert "predict 2026-01-13" in out
    assert "  set : " in out and "  enc : " in out
    for song in REGULARS:
        assert song in out


def test_predict_defaults_to_the_day_after_the_last_stored_show(ready, capsys):
    """Not today. The earliest date the corpus does not already hold."""
    _run("--config", ready, "predict")
    out = capsys.readouterr().out
    assert "predict 2026-01-13" in out
    assert "through 2026-01-12" in out


def test_predict_says_how_many_shows_it_read(ready, capsys):
    _run("--config", ready, "predict")
    assert "reading 12 show(s)" in capsys.readouterr().out


def test_predict_twice_over_the_same_store_agrees_with_itself(ready, capsys):
    _run("--config", ready, "predict")
    first = capsys.readouterr().out
    _run("--config", ready, "predict")
    assert capsys.readouterr().out == first


def test_predict_can_be_pointed_at_a_night_inside_the_corpus(ready, capsys):
    assert _run("--config", ready, "predict", "--asof", "2026-01-06") == EXIT_OK
    out = capsys.readouterr().out
    assert "predict 2026-01-06" in out
    # Said out loud, because a blinded re-prediction of a played night is not what somebody
    # typing `slkit predict` usually has in mind.
    assert "blinded re-prediction" in out


def test_predict_takes_the_set_length_from_the_flag(ready, capsys):
    _run("--config", ready, "predict", "--set-len", "4")
    out = capsys.readouterr().out
    assert "set of 4" in out
    assert len([line for line in out.splitlines() if line.startswith("  set : ")]) == 1
    setlist = next(line for line in out.splitlines() if line.startswith("  set : "))
    assert len(setlist.split(":", 1)[1].split(",")) == 4


def test_predict_refuses_a_malformed_date(ready, capsys):
    assert _run("--config", ready, "predict", "--asof", "last tuesday") != EXIT_OK
    assert "--asof" in capsys.readouterr().err


def test_predict_refuses_an_empty_corpus(tmp_path, capsys):
    """An empty ranking prints as a prediction that named no songs, which is the wrong report."""
    config = _cfg(tmp_path)
    with Store(tmp_path / "state") as store:
        store.init()
    assert _run("--config", config, "predict") != EXIT_OK
    assert "corpus is empty" in capsys.readouterr().err


def test_replay_grades_every_night_but_the_first(ready, capsys):
    assert _run("--config", ready, "replay") == EXIT_OK
    out = capsys.readouterr().out
    assert "11 night(s)" in out
    assert "1 night(s) skipped" in out


def test_replay_reports_recall_expected_hits_and_calibration(ready, capsys):
    _run("--config", ready, "replay")
    out = capsys.readouterr().out
    assert "mean recall" in out and "expected" in out and "calibration" in out


def test_replay_states_where_its_window_came_from(ready, capsys):
    """A published number nobody can locate the window for is a number nobody can check.

    The anchor prints RESOLVED -- the date it landed on, not the word `last_show` that produced
    it. Both are true; only the date lets a reader check the span against the corpus.
    """
    _run("--config", ready, "replay")
    out = capsys.readouterr().out
    assert "18 calendar months before the anchor, anchored on 2026-01-12" in out


def test_replay_twice_over_the_same_store_agrees_with_itself(ready, capsys):
    """The window slides with the corpus, not with the calendar, so this holds any day."""
    _run("--config", ready, "replay", "--dry-run")
    first = capsys.readouterr().out
    _run("--config", ready, "replay", "--dry-run")
    assert capsys.readouterr().out == first


def test_replay_writes_the_ledger(ready, tmp_path):
    _run("--config", ready, "replay")
    with Store(tmp_path / "state") as store:
        counts = {table: store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                  for table in ("pick_iterations", "picks", "pick_outcomes")}
    assert counts == {"pick_iterations": 11, "picks": 110, "pick_outcomes": 11}


def test_a_dry_run_says_so_and_writes_nothing(ready, tmp_path, capsys):
    assert _run("--config", ready, "replay", "--dry-run") == EXIT_OK
    assert "nothing will be written" in capsys.readouterr().out
    with Store(tmp_path / "state") as store:
        assert store.conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0] == 0


def test_replay_narrows_to_the_dates_given(ready, capsys):
    _run("--since", "2026-01-05", "--until", "2026-01-08", "--config", ready, "replay")
    out = capsys.readouterr().out
    assert "4 night(s)" in out
    assert "given on the command line" in out
    # 01-04 IS expected: it is the cutoff the window's first night is predicted from, which sits
    # outside the window on purpose. 01-03 is the first date nothing should reach.
    assert "2026-01-04" in out and "2026-01-03" not in out


def test_replay_refuses_a_malformed_date(ready, capsys):
    assert _run("--since", "the fall tour", "--config", ready, "replay") != EXIT_OK
    assert "--since" in capsys.readouterr().err


def test_replay_refuses_a_run_that_would_shrink_the_ledger(ready, tmp_path, capsys):
    _run("--config", ready, "replay")
    shrinking = _cfg(tmp_path, "[model]\nrollouts = 25\ntop_n = 3\n")
    assert _run("--config", shrinking, "replay") != EXIT_OK
    err = capsys.readouterr().err
    assert "shrank the picks ledger" in err
    with Store(tmp_path / "state") as store:
        assert store.conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0] == 110


def test_a_shrink_can_be_forced(ready, tmp_path):
    _run("--config", ready, "replay")
    shrinking = _cfg(tmp_path, "[model]\nrollouts = 25\ntop_n = 3\n")
    assert _run("--config", shrinking, "replay", "--force") == EXIT_OK
    with Store(tmp_path / "state") as store:
        assert store.conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0] == 33


def test_dump_picks_shows_the_ledger_and_nothing_else(ready, capsys):
    _run("--config", ready, "replay")
    capsys.readouterr()
    assert _run("--config", ready, "show", "picks") == EXIT_OK
    out = capsys.readouterr().out
    assert "## picks" in out and "## pick_iterations" in out and "## pick_outcomes" in out
    assert "## shows" not in out and "## show_entries" not in out


def test_dump_all_still_shows_the_corpus(ready, capsys):
    _run("--config", ready, "show", "corpus")
    out = capsys.readouterr().out
    assert "## shows" in out
    assert "## picks" in out


def test_dump_picks_narrows_on_the_night(ready, capsys):
    _run("--config", ready, "replay")
    capsys.readouterr()
    _run("--since", "2026-01-11", "--config", ready, "show", "picks")
    out = capsys.readouterr().out
    assert "2026-01-11" in out
    assert "2026-01-05" not in out


def test_a_config_with_an_unknown_model_key_is_refused_by_predict(tmp_path, capsys):
    """The stanza exists so a number can be changed without editing source. A key that is quietly
    skipped is a change somebody believes they made."""
    config = _cfg(tmp_path, "[model]\nrollout = 25\n")
    with Store(tmp_path / "state") as store:
        store.init()
        store.corpus.replace_shows([_show("2026-01-01", *REGULARS)])
    assert _run("--config", config, "predict") != EXIT_OK
    assert "unknown key" in capsys.readouterr().err


def test_a_bad_replay_span_is_reported_against_the_file_that_holds_it(tmp_path, capsys):
    """`PT18M` is eighteen MINUTES and valid ISO 8601. It renders as a diagnostic, not a
    traceback, and the diagnostic names the config file."""
    config = _cfg(tmp_path, '[model]\nreplay_since_back = "PT18M"\n')
    with Store(tmp_path / "state") as store:
        store.init()
        store.corpus.replace_shows([_show(f"2026-01-{d:02d}", *REGULARS) for d in range(1, 4)])
    assert _run("--config", config, "replay") != EXIT_OK
    err = capsys.readouterr().err
    assert "replay_since_back" in err
    assert "slkit.toml" in err


def test_the_ledger_survives_a_re_ingest(ready, tmp_path):
    """The protection worth having a CLI test for as well as a unit one: an ingest that took the
    ledger with it would be silent and unrecoverable."""
    _run("--config", ready, "replay")
    with Store(tmp_path / "state") as store:
        store.corpus.replace_shows([_show(f"2026-02-{day:02d}", *REGULARS)
                                    for day in range(1, 20)])
        assert store.conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0] == 110


def test_the_picks_tables_are_created_by_a_migration_on_an_existing_store(tmp_path):
    """Migration 2, not folded back into 1: two databases already hold real corpora."""
    with Store(tmp_path / "state") as store:
        store.init()
    path = tmp_path / "state" / "setlistkit.sqlite"
    conn = sqlite3.connect(path)
    try:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master "
                                                "WHERE type = 'table'")}
        versions = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    finally:
        conn.close()
    assert {"pick_iterations", "picks", "pick_outcomes"} <= names
    assert 2 in versions

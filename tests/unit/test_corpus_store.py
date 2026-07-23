# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the merged corpus in SQLite.

The thing under test is fidelity. A setlist is an ordered thing, and one read back in the wrong
order is wrong in a way that looks entirely right -- so the round trip is asserted on the whole
record, not on counts.
"""

import sqlite3

import pytest

from setlistkit.store import Store
from setlistkit.store.migrations import MIGRATIONS


def _entry(song, segue=False, non_song=False):
    return {"song": song, "segue": segue, "non_song": non_song}


SHOW = {
    "date": "2026-07-04", "source": "description", "identifier": "moe2026-07-04",
    "sets": [
        [_entry("Rebubula", segue=True), _entry("Timmy Tucker"), _entry("Tuning", non_song=True)],
        [_entry("Plane Crash"), _entry("ATL")],
    ],
    "encore": [_entry("The Faker")],
}


@pytest.fixture(name="store")
def _store(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        yield store


def test_a_show_survives_the_round_trip_exactly(store):
    store.corpus.replace_shows([SHOW])
    read, = store.corpus.shows()
    assert read == {k: v for k, v in SHOW.items()}


def test_set_order_and_song_order_are_both_preserved(store):
    store.corpus.replace_shows([SHOW])
    read, = store.corpus.shows()
    assert [[e["song"] for e in one] for one in read["sets"]] == [
        ["Rebubula", "Timmy Tucker", "Tuning"], ["Plane Crash", "ATL"]]
    assert [e["song"] for e in read["encore"]] == ["The Faker"]


def test_the_encore_is_not_folded_into_the_last_set(store):
    store.corpus.replace_shows([SHOW])
    read, = store.corpus.shows()
    assert len(read["sets"]) == 2
    assert read["encore"] == [_entry("The Faker")]


def test_segue_and_non_song_flags_survive(store):
    store.corpus.replace_shows([SHOW])
    read, = store.corpus.shows()
    assert read["sets"][0][0]["segue"] is True
    assert read["sets"][0][2]["non_song"] is True
    assert read["sets"][0][1] == {"song": "Timmy Tucker", "segue": False, "non_song": False}


def test_a_show_with_no_encore_reads_back_with_an_empty_one(store):
    store.corpus.replace_shows([{"date": "2026-07-05", "source": "tracks", "identifier": "x",
                                "sets": [[_entry("Meat")]], "encore": []}])
    read, = store.corpus.shows()
    assert read["encore"] == []


def test_an_override_keeps_its_reason_and_a_merged_show_has_none(store):
    store.corpus.replace_shows([
        {"date": "2026-07-04", "source": "description", "identifier": "a",
         "sets": [[_entry("Meat")]], "encore": []},
        {"date": "2026-07-05", "source": "override", "identifier": "override-2026-07-05",
         "sets": [[_entry("Meat")]], "encore": [], "reason": "confirmed by ear from the tape"},
    ])
    merged, override = store.corpus.shows()
    assert "reason" not in merged
    assert override["reason"] == "confirmed by ear from the tape"


def test_shows_are_returned_in_date_order(store):
    store.corpus.replace_shows([{"date": d, "source": "s", "identifier": d,
                                "sets": [[_entry("Meat")]], "encore": []}
                               for d in ("2026-07-09", "2026-07-04", "2026-07-06")])
    assert [s["date"] for s in store.corpus.shows()] == ["2026-07-04", "2026-07-06", "2026-07-09"]


def test_replacing_the_corpus_leaves_no_entries_from_the_old_one(store):
    store.corpus.replace_shows([SHOW])
    store.corpus.replace_shows([{"date": "2026-08-01", "source": "s", "identifier": "b",
                                "sets": [[_entry("Wormwood")]], "encore": []}])
    read, = store.corpus.shows()
    assert read["date"] == "2026-08-01"
    # The old show's six entries must not survive as songs attached to nothing, or to whatever
    # reuses the date next.
    assert store.conn.execute("SELECT COUNT(*) FROM show_entries").fetchone()[0] == 1


def test_replace_is_atomic(store):
    store.corpus.replace_shows([SHOW])
    broken = [{"date": "2026-08-01", "source": "s", "identifier": "b",
               "sets": [[_entry("Wormwood")]], "encore": []},
              {"source": "s", "identifier": "no-date"}]        # no "date" key: raises mid-write
    with pytest.raises(KeyError):
        store.corpus.replace_shows(broken)
    # The good snapshot is still there. A delete that committed without its inserts would have
    # left an empty corpus where a complete one had been.
    read, = store.corpus.shows()
    assert read["date"] == "2026-07-04" and len(read["sets"]) == 2


def test_show_count_does_not_read_the_setlists(store):
    store.corpus.replace_shows([SHOW])
    assert store.corpus.show_count() == 1
    store.corpus.replace_shows([])
    assert store.corpus.show_count() == 0


def test_the_corpus_is_empty_before_anything_is_ingested(store):
    assert store.corpus.shows() == [] and store.corpus.show_count() == 0


def test_a_show_with_no_songs_at_all_still_records_that_it_happened(store):
    store.corpus.replace_shows([{"date": "2026-07-04", "source": "s", "identifier": "a",
                                "sets": [], "encore": []}])
    read, = store.corpus.shows()
    assert read["sets"] == [] and read["encore"] == []


def test_the_corpus_dumps_as_reviewable_text(store):
    store.corpus.replace_shows([SHOW])
    dumped = store.dump()
    assert "## shows (1 rows)" in dumped
    assert "## show_entries (6 rows)" in dumped
    assert "Rebubula" in dumped and "The Faker" in dumped


def test_an_orphaned_entry_row_does_not_invent_a_show(tmp_path):
    """The shows table is the truth about which nights exist.

    Reachable exactly as the guard's comment says: the foreign key is only enforced while
    PRAGMA foreign_keys is on, and that is per-connection. Anyone opening the file with the
    plain `sqlite3` CLI, where it defaults OFF, can write an entry for a date no show has.
    """
    with Store(tmp_path) as store:
        store.init()
        store.corpus.replace_shows([SHOW])

    hand_edit = sqlite3.connect(tmp_path / "setlistkit.sqlite")   # no foreign_keys pragma
    hand_edit.execute("INSERT INTO show_entries VALUES('1999-01-01', 0, 1, 0, 'Ghost', 0, 0)")
    hand_edit.commit()
    hand_edit.close()

    with Store(tmp_path) as store:
        assert [s["date"] for s in store.corpus.shows()] == ["2026-07-04"]
        assert "Ghost" not in str(store.corpus.shows())


def test_first_and_last_spans_every_stored_show(store):
    store.corpus.replace_shows([
        {**SHOW, "date": "2020-01-08"},
        {**SHOW, "date": "2026-06-14"},
        {**SHOW, "date": "2023-03-03"},
    ])
    assert store.corpus.first_and_last() == ("2020-01-08", "2026-06-14")


def test_first_and_last_is_none_for_an_empty_corpus(store):
    assert store.corpus.first_and_last() is None


def test_the_migration_is_recorded_and_idempotent(tmp_path):
    # Against MIGRATIONS rather than against a literal [1, 2]: what this test is about is that
    # the corpus tables arrive and that a second open does nothing, and neither of those has an
    # opinion about how many migrations have shipped since. Hardcoding the list makes every
    # later migration fail a test that was never checking for it.
    with Store(tmp_path) as store:
        assert store.init() == [m.version for m in MIGRATIONS]
        assert store.schema_version() == max(m.version for m in MIGRATIONS)
        assert store.init() == []                # nothing pending on a second open
        assert "shows" in store.table_names() and "show_entries" in store.table_names()

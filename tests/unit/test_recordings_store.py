# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The recordings mirror in SQLite: every tape of a night, its tracks in play order, its kind."""

import sqlite3

import pytest

from setlistkit.store import Store


def _tape(identifier, date, *lengths, uploader="taper@example.org", audio_format="Flac"):
    return {"identifier": identifier, "date": date, "uploader": uploader,
            "audio_format": audio_format,
            "duration_tracks": [
                {"idx": idx, "name": f"{identifier}.t{idx:02d}.flac", "title": f"song {idx}",
                 "length_raw": str(length), "seconds": length}
                for idx, length in enumerate(lengths)]}


def test_every_tape_of_a_night_is_kept_not_just_the_winner(tmp_path):
    """The corpus keeps one show per date because a setlist is one fact about a night.

    Durations are the opposite: four tapers timing one performance is four measurements, and the
    disagreement between them is the error bar. A mirror that deduplicated by date would throw
    away the entire basis of the reconciliation.
    """
    with Store(tmp_path) as store:
        store.init()
        assert store.replace_recordings([_tape("a", "2024-01-01", 100.0, 200.0),
                                         _tape("b", "2024-01-01", 101.0, 199.0)]) == (2, 4)
        assert [r["identifier"] for r in store.recordings()] == ["a", "b"]


def test_tracks_come_back_in_the_order_they_were_stored_in(tmp_path):
    """idx is a column and never an accident. rowid order is not promised and is not read."""
    with Store(tmp_path) as store:
        store.init()
        tape = _tape("a", "2024-01-01", 100.0, 200.0, 300.0)
        tape["duration_tracks"].reverse()          # hand them over backwards; idx still decides
        store.replace_recordings([tape])
        stored, = store.recordings()
        assert [t["idx"] for t in stored["tracks"]] == [0, 1, 2]
        assert [t["seconds"] for t in stored["tracks"]] == [100.0, 200.0, 300.0]


def test_an_unreadable_length_survives_the_round_trip_as_null_beside_its_source_string(tmp_path):
    """The whole reason length_raw is a column. A parser bug must be diagnosable from the DB."""
    with Store(tmp_path) as store:
        store.init()
        tape = _tape("a", "2024-01-01", 100.0)
        tape["duration_tracks"].append({"idx": 1, "name": "a.t01.flac", "title": "x",
                                        "length_raw": "unknown", "seconds": None})
        store.replace_recordings([tape])
        stored, = store.recordings()
        assert stored["tracks"][1]["seconds"] is None
        assert stored["tracks"][1]["length_raw"] == "unknown"


def test_the_mirror_is_replaced_whole_and_leaves_no_orphan_tracks(tmp_path):
    """Ingest recomputes it from the whole cache every run; a patched mirror is half of two."""
    with Store(tmp_path) as store:
        store.init()
        store.replace_recordings([_tape("a", "2024-01-01", 100.0, 200.0)])
        assert store.replace_recordings([_tape("b", "2024-02-01", 300.0)]) == (1, 1)
        assert store.recording_count() == 1 and store.track_count() == 1
        assert [r["identifier"] for r in store.recordings()] == ["b"]


def test_a_tape_with_no_readable_durations_is_stored_rather_than_dropped(tmp_path):
    """It is a real recording of a real night. Slice 3 has a table for saying so."""
    with Store(tmp_path) as store:
        store.init()
        assert store.replace_recordings([_tape("a", "2024-01-01", audio_format="")]) == (1, 0)
        stored, = store.recordings()
        assert stored["n_tracks"] == 0 and stored["audio_format"] == ""


def test_the_uploader_is_stored_and_not_looked_up_again(tmp_path):
    """The field whose absence uncredited 425 of 425 tapes in the previous implementation.

    The page read it out of the raw metadata cache, which is gitignored -- so it worked on the
    machine it was written on and came back empty on the publishing server, silently.
    """
    with Store(tmp_path) as store:
        store.init()
        store.replace_recordings([_tape("a", "2024-01-01", 100.0, uploader="nate@example.org")])
        stored, = store.recordings()
        assert stored["uploader"] == "nate@example.org"


def test_deleting_a_recording_takes_its_tracks_with_it(tmp_path):
    """The FK cascade, asserted rather than assumed: it needs a PRAGMA to be on."""
    with Store(tmp_path) as store:
        store.init()
        store.replace_recordings([_tape("a", "2024-01-01", 100.0, 200.0)])
        store.conn.execute("DELETE FROM recordings WHERE identifier = 'a'")
        assert store.track_count() == 0


def test_a_track_cannot_be_stored_for_a_tape_that_does_not_exist(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        with pytest.raises(sqlite3.IntegrityError):
            store.conn.execute(
                "INSERT INTO recording_tracks(identifier, idx, name, title, length_raw, seconds) "
                "VALUES('ghost', 0, 'x.flac', 'x', '1.0', 1.0)")


def test_show_types_are_stored_with_the_tape_the_verdict_was_read_off(tmp_path):
    """`evidence` without an identifier is a claim with no citation.

    "notes describe an acoustic set" names no notes, and re-finding them means re-scanning every
    tape of that date.
    """
    with Store(tmp_path) as store:
        store.init()
        assert store.replace_show_types([
            {"date": "2024-01-01", "kind": "acoustic", "evidence": "'moe.stly' in tape metadata",
             "identifier": "a"},
            {"date": "2024-01-02", "kind": "electric", "evidence": None, "identifier": None},
        ]) == 2
        assert store.show_types() == {"2024-01-01": "acoustic", "2024-01-02": "electric"}
        assert store.show_type_counts() == {"acoustic": 1, "electric": 1}
        row = store.conn.execute(
            "SELECT identifier FROM show_types WHERE date = '2024-01-01'").fetchone()
        assert row["identifier"] == "a"


def test_an_orphaned_track_row_does_not_invent_a_tape(tmp_path):
    """The recordings table is the truth about which tapes exist.

    Reachable exactly as the guard's comment says: the foreign key is only enforced while
    PRAGMA foreign_keys is on, and that is per-connection. Anyone opening the file with the
    plain `sqlite3` CLI, where it defaults OFF, can write a track for a tape nothing has.
    """
    with Store(tmp_path) as store:
        store.init()
        store.replace_recordings([_tape("a", "2024-01-01", 100.0)])

    hand_edit = sqlite3.connect(tmp_path / "setlistkit.sqlite")    # no foreign_keys pragma
    hand_edit.execute("INSERT INTO recording_tracks VALUES('ghost', 0, 'x.flac', 'x', '1.0', 1.0)")
    hand_edit.commit()
    hand_edit.close()

    with Store(tmp_path) as store:
        assert [r["identifier"] for r in store.recordings()] == ["a"]
        assert [t["name"] for t in store.recordings()[0]["tracks"]] == ["a.t00.flac"]

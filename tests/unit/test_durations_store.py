# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The durations tables: what the chain concluded, stored so the export never recomputes it."""

from dataclasses import asdict

import pytest

from setlistkit.catalog.durations import Edge
from setlistkit.catalog.lengths import (ELECTRIC, TAPES_DISAGREE, Consensus, Performance, Sandwich,
                                        Slot, SongStat, as_row, song_stats, withheld_counts)
from setlistkit.store import Store


def _perf(song="Rebubula", date="2024-01-01", set_label="1", position=1, seconds=600.0,
          *, suspect=False, segued=False, show_type=ELECTRIC, excluded=None, sandwich=None,
          n_tapes=3, resolved_by=None):
    return Performance(
        slot=Slot(date, set_label, position, song),
        seconds=seconds,
        consensus=Consensus(n_tapes=n_tapes, n_tapes_seen=n_tapes, n_ballots=n_tapes,
                            spread_seconds=4.0, spread_all_tapes=9.0, suspect=suspect,
                            resolved_by=resolved_by),
        segued=segued, show_type=show_type, excluded=excluded, sandwich=sandwich)


def _stat(song="Rebubula", n=3):
    return SongStat(song=song, n=n, median_seconds=600.0, mean_seconds=610.0, min_seconds=500.0,
                    max_seconds=700.0, p10_seconds=520.0, p90_seconds=690.0, stdev_seconds=80.0,
                    longest_date="2024-01-01")


def _rows(performances):
    return [as_row(performance) for performance in performances]


def _stats(stats):
    return [asdict(stat) for stat in stats]


def _write(store, performances=(), stats=(), **kwargs):
    return store.durations.replace(_rows(performances), _stats(stats), **kwargs)


def test_a_performance_survives_the_round_trip_with_every_column_intact(tmp_path):
    """Nineteen columns in a fixed order. A swapped pair of REALs is wrong and looks right."""
    with Store(tmp_path) as store:
        store.init()
        performance = _perf(seconds=612.5, segued=True, resolved_by="finest_tape")
        _write(store, [performance])
        stored, = store.durations.performances()
        assert stored == as_row(performance)


def test_withheld_is_stored_rather_than_re_derived_from_the_columns_beside_it(tmp_path):
    """The one derived value kept on purpose.

    Its derivation is a priority ordering over four other columns, so re-deriving it in SQL would
    be a SECOND implementation of "why was this held back" -- and two implementations is exactly
    the drift Performance.withheld exists to prevent. A round trip that reconstructs a different
    reason than the one stored means the flattening lost something.
    """
    with Store(tmp_path) as store:
        store.init()
        # Suspect AND acoustic. The property answers with the show type because it is asked first,
        # and a re-derivation that happened to check `suspect` first would answer differently.
        performance = _perf(suspect=True, show_type="acoustic")
        assert performance.withheld == "acoustic"
        _write(store, [performance])
        stored, = store.durations.performances()
        assert stored["withheld"] == "acoustic"
        assert stored["suspect"] is True


def test_a_row_that_forgot_withheld_is_refused_rather_than_stored_as_null(tmp_path):
    """A silent NULL there does not look like a bug -- it looks like a performance that counted.

    The exclusion tally would quietly empty out while every other number in the run stayed
    plausible, which is the failure this whole column exists to make impossible.
    """
    with Store(tmp_path) as store:
        store.init()
        row = as_row(_perf(suspect=True))
        del row["withheld"]
        with pytest.raises(KeyError):
            store.durations.replace([row], [])


def test_a_failed_write_leaves_the_previous_run_untouched(tmp_path):
    """Rows are materialized before the transaction opens, so a malformed one deletes nothing.

    Not incidental. These five tables are five views of ONE run, and the delete is what makes a
    replacement whole -- a crash between the delete and a bad insert would leave the store holding
    no durations at all rather than the perfectly good ones it had a moment earlier.
    """
    with Store(tmp_path) as store:
        store.init()
        _write(store, [_perf()], [_stat()])
        broken = asdict(_stat())
        del broken["median_seconds"]
        with pytest.raises(KeyError):
            store.durations.replace(_rows([_perf(song="Timmy Tucker")]), [broken])
        assert [row["song"] for row in store.durations.performances()] == ["Rebubula"]
        assert [row["song"] for row in store.durations.song_length_stats()] == ["Rebubula"]


def test_a_second_run_replaces_the_tables_rather_than_appending_to_them(tmp_path):
    """The reconciliation is a comparison BETWEEN tapes; there is no incremental update of a vote.

    One tape arriving for one night can change the answer for every performance of that night, so
    a run recomputes the lot. Appending would leave last week's answer beside this week's with
    nothing saying which is current.
    """
    with Store(tmp_path) as store:
        store.init()
        _write(store, [_perf(song="Rebubula")], [_stat("Rebubula")])
        _write(store, [_perf(song="Plane Crash")], [_stat("Plane Crash")])
        assert [row["song"] for row in store.durations.performances()] == ["Plane Crash"]
        assert store.durations.counts()["performance_durations"] == 1


def test_an_ordinary_performance_stores_no_sandwich_columns_at_all(tmp_path):
    """None on the ninety-eight percent of rows that are not a song played twice in one night."""
    with Store(tmp_path) as store:
        store.init()
        _write(store, [_perf()])
        stored, = store.durations.performances()
        assert stored["double_play_parts"] is None
        assert stored["sandwich_total_seconds"] is None
        assert stored["is_longest_part"] is None


def test_a_sandwich_keeps_the_whole_of_what_the_band_played_beside_the_half_that_votes(tmp_path):
    """Moth > [Water > Yellow Tigers] > Moth. The short half is stored, tagged, and does not vote.

    The total is the interesting number and it is nowhere else: the two halves are separate rows
    keyed on separate positions, so a reader adding them up has to know they belong together.
    """
    with Store(tmp_path) as store:
        store.init()
        longest = _perf(song="Moth", position=1, seconds=400.0,
                        sandwich=Sandwich(parts=2, total_seconds=520.0, is_longest_part=True))
        short = _perf(song="Moth", position=5, seconds=120.0,
                      sandwich=Sandwich(parts=2, total_seconds=520.0, is_longest_part=False))
        _write(store, [longest, short])
        first, second = store.durations.performances()
        assert first["is_longest_part"] is True and first["withheld"] is None
        assert second["is_longest_part"] is False
        assert second["withheld"] == "sandwich_short_half"
        assert {row["sandwich_total_seconds"] for row in (first, second)} == {520.0}


def test_a_boolean_comes_back_a_boolean_and_not_a_one(tmp_path):
    """SQLite has no boolean type, so `suspect` round-trips through INTEGER.

    Read back raw, every one of these is truthy -- including the 0 -- so a consumer writing
    `if row["suspect"]` would treat every performance as disputed and never notice.
    """
    with Store(tmp_path) as store:
        store.init()
        _write(store, [_perf(position=1, suspect=True, segued=True),
                       _perf(position=2, suspect=False, segued=False)])
        disputed, settled = store.durations.performances()
        assert disputed["suspect"] is True and disputed["segued"] is True
        assert settled["suspect"] is False and settled["segued"] is False


def test_the_exclusion_tally_is_read_off_the_stored_column_not_recomputed(tmp_path):
    """So the tally and the statistics it explains can only ever come from the same run.

    A performance that DOES vote has NULL and is not counted, which is why this sums to less than
    the table -- the number of rows held back, never the number of rows.
    """
    with Store(tmp_path) as store:
        store.init()
        _write(store, [_perf(position=1),
                       _perf(position=2, suspect=True),
                       _perf(position=3, suspect=True),
                       _perf(position=4, show_type="acoustic")])
        assert store.durations.withheld_counts() == {"acoustic": 1, TAPES_DISAGREE: 2}
        assert store.durations.counts()["performance_durations"] == 4


def test_the_stored_tally_agrees_with_the_one_computed_before_it_was_stored(tmp_path):
    """The drift check, end to end: same reasons, same counts, on both sides of the database."""
    performances = [_perf(position=1), _perf(position=2, suspect=True),
                    _perf(position=3, show_type="acoustic"),
                    _perf(position=4, excluded="truncated")]
    with Store(tmp_path) as store:
        store.init()
        _write(store, performances, song_stats(performances))
        assert store.durations.withheld_counts() == withheld_counts(performances)


def test_statistics_come_back_longest_first_because_that_is_how_they_are_read(tmp_path):
    """Ordered by the query, never by insertion. A bundle whose order depends on rowid order is
    one whose golden-file test passes or fails on something that is not the data."""
    with Store(tmp_path) as store:
        store.init()
        _write(store, stats=[_stat("Rebubula"), _stat("Plane Crash")])
        stats = store.durations.song_length_stats()
        assert [row["song"] for row in stats] == ["Plane Crash", "Rebubula"]


def test_an_edge_keeps_its_reasons_verbatim_and_serializes_them_the_same_way_twice(tmp_path):
    """An edge is never a failure -- it is a decision with its reasons, and the reasons are the
    point. Sorted keys so two runs over one corpus write byte-identical JSON and a dump diff is
    about the findings rather than about dictionary ordering."""
    with Store(tmp_path) as store:
        store.init()
        detail = {"z": 1, "a": [2, 3], "m": "why"}
        edges = [Edge(kind="unclaimed", date="2024-01-01", identifier="tape-a", song="Wormwood",
                      detail=dict(reversed(list(detail.items()))))]
        _write(store, edges=edges)
        stored, = store.conn.execute("SELECT kind, song, detail_json FROM duration_edges")
        assert stored["kind"] == "unclaimed" and stored["song"] == "Wormwood"
        assert stored["detail_json"] == '{"a": [2, 3], "m": "why", "z": 1}'


def test_every_durations_table_is_counted_so_a_stage_that_stopped_is_visible(tmp_path):
    """They fail independently and in ways that look like success. Twenty thousand performances
    and no statistics has aggregated nothing; neither count reveals the other."""
    with Store(tmp_path) as store:
        store.init()
        counts = _write(
            store, [_perf()], [_stat()],
            review=[{"identifier": "tape-b", "date": "2024-01-02", "n_tracks": 17,
                     "n_setlist": 15, "n_desc": 15, "reason": "tracks exceed setlist",
                     "url": "https://archive.org/details/tape-b"}],
            abandoned=[{"identifier": "tape-c", "date": "2024-01-03", "n_tracks": 2,
                        "longest_seconds": 4210.0}],
            edges=[Edge(kind="offset", date="2024-01-01")])
        assert counts == {"performance_durations": 1, "song_length_stats": 1, "duration_review": 1,
                          "duration_abandoned": 1, "duration_edges": 1}
        assert store.durations.counts() == counts


def test_a_review_row_says_which_three_counts_failed_to_reconcile(tmp_path):
    """So the next reader starts from "17 files, 15 songs, 15 in the description" rather than
    from re-running the join by hand to find out what did not line up."""
    with Store(tmp_path) as store:
        store.init()
        _write(store, review=[{"identifier": "tape-b", "date": "2024-01-02", "n_tracks": 17,
                               "n_setlist": 15, "n_desc": 12, "reason": "too many tracks",
                               "url": "https://archive.org/details/tape-b"}])
        row, = store.durations.review()
        assert (row["n_tracks"], row["n_setlist"], row["n_desc"]) == (17, 15, 12)
        assert row["url"].endswith("tape-b")


def test_song_statistics_are_not_narrowed_by_a_date_range(tmp_path):
    """A song's statistics are computed over every year it was played, so a range that reached
    them would print a whole-corpus median under a header claiming it covers one month."""
    with Store(tmp_path) as store:
        store.init()
        _write(store, [_perf(date="2024-01-01")], [_stat()])
        dumped = store.dump(since="1990-01-01", until="1990-12-31")
        assert "## song_length_stats (1 rows, no date axis)" in dumped
        assert "## performance_durations (0 of 1 rows" in dumped

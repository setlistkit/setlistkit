# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the Store: init, provenance, counts, and the plain-text dump."""

import pytest

from setlistkit.store.db import Store, _ident
from setlistkit.store.migrations import MIGRATIONS

# Read off the ledger rather than hard-coded, so adding a migration is one line in migrations.py
# instead of a hunt through the suite for every place a version number was written down.
EXPECTED_MIGRATIONS = [(m.version, m.name) for m in MIGRATIONS]
LATEST_SCHEMA = max(version for version, _ in EXPECTED_MIGRATIONS)


def test_init_applies_every_migration_in_order(tmp_path):
    data_root = tmp_path / "data"
    with Store(data_root) as store:
        assert store.init() == [version for version, _ in EXPECTED_MIGRATIONS]
        assert store.db_path.is_file()
        assert store.schema_version() == LATEST_SCHEMA
        assert "meta" in store.table_names()
        assert "schema_migrations" in store.table_names()


def test_init_is_idempotent(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        # test that a second init applies nothing new
        assert store.init() == []
        assert store.schema_version() == LATEST_SCHEMA


def test_init_stamps_provenance(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        meta = dict(store.conn.execute("SELECT key, value FROM meta").fetchall())
        assert meta["setlistkit_version"]
        assert meta["created_at"]


def test_created_at_is_preserved_across_reinit(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        first = store.conn.execute(
            "SELECT value FROM meta WHERE key='created_at'").fetchone()[0]
        store.init()
        second = store.conn.execute(
            "SELECT value FROM meta WHERE key='created_at'").fetchone()[0]
        assert first == second


def test_table_counts(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        counts = store.table_counts()
        assert counts["schema_migrations"] == LATEST_SCHEMA
        assert counts["meta"] >= 2


def test_dump_is_deterministic_and_drops_volatile_columns(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        first = store.dump()
        second = store.dump()
        assert first == second
        # applied_at is a volatile column and must not appear
        assert "applied_at" not in first
        # real content is present
        assert "setlistkit_version" in first
        assert "baseline" in first


def test_dump_of_empty_db_has_no_tables(tmp_path):
    # Opening without init() creates the file but runs no migrations.
    with Store(tmp_path) as store:
        assert store.table_names() == []
        assert store.dump().strip() == "# setlistkit.sqlite — contents"


def test_ident_rejects_non_identifiers():
    assert _ident("meta") == "meta"
    with pytest.raises(ValueError):
        _ident("meta; DROP TABLE meta")


# --- the ranged dump ---------------------------------------------------------------------
#
# Whole-database dumps stopped being readable somewhere around the seventy-six-thousandth track.
# What a range must never do is make a table LOOK small, which is why every header below carries
# both numbers or says why it carries one.

def _three_nights(store):
    store.init()
    store.replace_recordings([
        {"identifier": ident, "date": date, "uploader": "t@example.org", "audio_format": "Flac",
         "duration_tracks": [{"idx": 0, "name": f"{ident}.flac", "title": "x",
                              "length_raw": "100.0", "seconds": 100.0}]}
        for ident, date in (("a", "2023-06-01"), ("b", "2024-06-01"), ("c", "2025-06-01"))])
    return store


def test_a_range_is_inclusive_at_both_ends(tmp_path):
    """A half-open range is the classic off-by-one, and this is the view someone opens when they
    already suspect something is wrong. One bug at a time."""
    with Store(tmp_path) as store:
        out = _three_nights(store).dump(since="2023-06-01", until="2024-06-01")
        assert "2023-06-01" in out and "2024-06-01" in out
        assert "2025-06-01" not in out


def test_a_range_says_how_much_of_each_table_it_is_not_showing(tmp_path):
    """Otherwise a filtered table is indistinguishable from a table that lost most of its rows."""
    with Store(tmp_path) as store:
        out = _three_nights(store).dump(since="2024-01-01")
        assert "## recordings (2 of 3 rows, from 2024-01-01)" in out


def test_a_table_with_no_date_axis_prints_in_full_and_says_why(tmp_path):
    """meta and schema_migrations are not small because the range excluded them."""
    with Store(tmp_path) as store:
        out = _three_nights(store).dump(since="2024-01-01")
        assert "## meta (2 rows, no date axis)" in out


def test_tracks_are_filtered_through_the_recording_they_belong_to(tmp_path):
    """The case the filter map exists for: no date column, and by a wide margin the largest table.

    An unranged dump of recording_tracks is exactly what a range was reached for to avoid.
    """
    with Store(tmp_path) as store:
        out = _three_nights(store).dump(since="2025-01-01")
        assert "## recording_tracks (1 of 3 rows, from 2025-01-01)" in out
        assert "c.flac" in out and "a.flac" not in out


def test_an_unranged_dump_is_byte_for_byte_what_it_always_was(tmp_path):
    """The existing view and its diffs must not move because a new flag exists."""
    with Store(tmp_path) as store:
        out = _three_nights(store).dump()
        assert "## recordings (3 rows)" in out
        assert "no date axis" not in out
        assert out.startswith("# setlistkit.sqlite — contents\n")


def test_a_date_that_is_not_a_date_is_refused_rather_than_compared(tmp_path):
    """`--until 2023` sorts below every date IN 2023 and would print an empty range as if the
    year held nothing -- the one answer this view exists to make trustworthy."""
    with Store(tmp_path) as store:
        store.init()
        for bad in ("2023", "2023-6-1", "yesterday", "2023-06-01T00:00"):
            with pytest.raises(ValueError):
                store.dump(until=bad)


def test_dump_and_counts_handle_reserved_word_identifiers(tmp_path):
    with Store(tmp_path) as store:
        store.init()
        # A table and column named with SQL keywords must not break dump/table_counts:
        # identifiers are double-quoted, so this is valid SQL rather than a syntax error.
        store.conn.execute('CREATE TABLE "select"("order" TEXT, x TEXT)')
        store.conn.execute('INSERT INTO "select"("order", x) VALUES(?, ?)', ("a", "b"))
        assert store.table_counts()["select"] == 1
        out = store.dump()
        assert "## select (1 rows)" in out
        assert "a | b" in out

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `slkit export tapemeasure`, including the golden file that locks the bundle's shape.

THE GOLDEN FILE IS THE CONTRACT. Everything upstream of here is checked by tests that assert on a
number; this one asserts on the SHAPE, because the bundle is read by another program in another
repository that this test suite cannot see. A field renamed, a unit changed, a list that quietly
became a mapping -- each is a green test suite here and a broken page there, and the only place the
two meet is this file.

So the diff is the review. When the golden file changes, a person reads what changed and decides
whether the consumer survives it; that is the entire mechanism, and it only works if the file stays
small enough to read. Two nights, three tapes, five songs.

Regenerate deliberately, never reflexively:

    SLKIT_UPDATE_GOLDEN=1 pyenv/bin/python -m pytest tests/unit/test_cli_export.py

and then READ THE DIFF before committing it. A golden file regenerated to make a test pass is a
test that has been switched off.
"""

import json
import os
from collections import Counter
from pathlib import Path
from statistics import median

import pytest

from setlistkit.catalog.tapemeasure import SCHEMA
from setlistkit.cli.export import EXIT_NOTHING
from setlistkit.cli.main import EXIT_OK, main

from test_cli_derive import LENGTHS, SETLIST, SONGS, _cache, _cfg, _tape

GOLDEN = Path(__file__).resolve().parent / "golden" / "tapemeasure.json"

# Two nights and three tapes: enough for a second date, a second taper, a consolidated pair of
# uploads and a song with n=2, which is every column the bundle has a way of getting wrong.
TAPES = [
    _tape("example2025-07-04", "2025-07-04", uploader="one@example.org"),
    _tape("example2025-07-04.b", "2025-07-04", (302.0, 478.0, 241.0, 618.0, 181.0),
          uploader="two@example.org"),
    _tape("example2025-07-05", "2025-07-05", (310.0, 470.0, 250.0, 600.0, 190.0),
          uploader="one@example.org"),
]


def _export(tmp_path, tapes=None, *args):
    """Ingest, derive and export, as three commands sharing only the database."""
    _cache(tmp_path, tapes if tapes is not None else TAPES)
    config = _cfg(tmp_path)
    assert main(["--config", config, "ingest"]) == EXIT_OK
    assert main(["--config", config, "derive", "durations"]) == EXIT_OK
    out = tmp_path / "bundle.json"
    code = main(["--config", config, "export", "tapemeasure", "--out", str(out), *args])
    return code, out


def test_the_bundle_matches_the_golden_file(tmp_path):
    """The shape another repository reads. See this module's docstring before regenerating."""
    _code, out = _export(tmp_path)
    produced = json.loads(out.read_text(encoding="utf-8"))
    if os.environ.get("SLKIT_UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(produced, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        pytest.skip("golden file regenerated -- read the diff")
    assert produced == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_the_bundle_carries_its_schema_version(tmp_path):
    """The consumer asserts on this before reading anything else, so it has to be present and it
    has to be the version the code actually wrote."""
    _code, out = _export(tmp_path)
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == SCHEMA


def test_a_taper_is_credited_without_publishing_their_email_address(tmp_path):
    """archive.org's uploader field is an email, and the bundle exists to be published.

    The store keeps the whole address because that is what tells four tapers apart from one taper
    who posted four times. Thanking 499 people by printing their email addresses is not thanking
    them, so the domain stops at this boundary."""
    _code, out = _export(tmp_path)
    credits = json.loads(out.read_text(encoding="utf-8"))["credits"]
    assert {row["uploader"] for row in credits} == {"one", "two"}
    assert not any("@" in row["uploader"] for row in credits)
    # The counts still have to be right, or the masking has eaten the data with the address.
    assert {row["uploader"]: row["n_tapes"] for row in credits} == {"one": 2, "two": 1}


def test_every_song_with_lengths_also_carries_its_structural_profile(tmp_path):
    """The reason features are folded in rather than published beside.

    Two files means two version stamps and a join the consumer has to get right -- and when the
    two halves keyed songs differently, that join silently produced profiles counting only the
    plays that happened to be spelled plainly."""
    _code, out = _export(tmp_path)
    songs = json.loads(out.read_text(encoding="utf-8"))["songs"]
    assert {song["song"] for song in songs} == set(SONGS)
    assert all(song["features"] is not None for song in songs)
    assert all("opener" in song["features"]["rates"] for song in songs)


def test_the_published_column_is_called_set_and_not_set_label(tmp_path):
    """The one place the database's name for a column and the published name for it meet.

    `set` is a Python builtin and reads badly in a WHERE clause; it is also what every person
    discussing a setlist calls it. Both layers are right, and the rename lives in exactly one
    place so it cannot happen twice or not at all."""
    _code, out = _export(tmp_path)
    row = json.loads(out.read_text(encoding="utf-8"))["performances"][0]
    assert "set" in row and "set_label" not in row


def test_exporting_before_deriving_says_so_rather_than_writing_an_empty_bundle(tmp_path):
    """An empty bundle is worse than no bundle: it publishes as a page saying the band has never
    played anything, and nothing upstream looks broken."""
    config = _cfg(tmp_path)
    assert main(["--config", config, "store", "init"]) == EXIT_OK
    out = tmp_path / "bundle.json"
    assert main(["--config", config, "export", "tapemeasure", "--out", str(out)]) == EXIT_NOTHING
    assert not out.exists()


def test_a_dry_run_reports_the_bundle_and_writes_nothing(tmp_path):
    _code, out = _export(tmp_path, None, "--dry-run")
    assert not out.exists()


def test_the_bundle_is_replaced_whole_rather_than_written_in_place(tmp_path):
    """A consumer polling this file sees the previous bundle or the next one, never half of one.

    A partial read fails as a JSON parse error somewhere far away from the exporter."""
    _code, out = _export(tmp_path)
    assert out.exists()
    assert not out.with_name(out.name + ".partial").exists()


def test_totals_count_the_rows_the_bundle_actually_carries(tmp_path):
    """The handful of numbers a reader checks before believing any of the rest. If they are
    computed from anything other than the published rows, they are decoration."""
    _code, out = _export(tmp_path)
    payload = json.loads(out.read_text(encoding="utf-8"))
    totals = payload["totals"]
    assert totals["performances"] == len(payload["performances"])
    assert totals["songs"] == len(payload["songs"])
    assert totals["nights"] == len({row["date"] for row in payload["performances"]})
    assert totals["tapes_queued_for_review"] == len(payload["review"])


def test_a_night_with_one_taper_is_counted_as_resting_on_one_taper(tmp_path):
    """The number that decides how much any of this can be trusted, and the one that cannot be
    seen from the performance count -- twenty thousand timed once and twenty thousand timed four
    times publish the same total."""
    _code, out = _export(tmp_path)
    payload = json.loads(out.read_text(encoding="utf-8"))
    alone = sum(1 for row in payload["performances"] if row["n_ballots"] < 2)
    assert payload["totals"]["single_tape_performances"] == alone == len(SONGS)


def test_the_date_range_describes_the_measurements_and_not_the_ambition(tmp_path):
    """Over the performances rather than the corpus: a range claiming 1992 when the earliest
    timed night is 2004 describes what we wish we had."""
    _code, out = _export(tmp_path)
    generated = json.loads(out.read_text(encoding="utf-8"))["generated"]
    assert generated["date_range"] == ["2025-07-04", "2025-07-05"]


def test_a_song_length_is_published_in_seconds_and_not_reformatted(tmp_path):
    """The bundle carries data, not presentation. Formatting 620 seconds as "10:20" here would
    make the consumer parse it back to draw a histogram.

    Asserted as a UNIT rather than as a value: an exact median across three tapes is the
    reconciliation's business and is tested where the reconciliation lives. What this pins is that
    a ten-minute song arrives as roughly six hundred of something, not roughly ten."""
    _code, out = _export(tmp_path)
    songs = {song["song"]: song for song in json.loads(out.read_text(encoding="utf-8"))["songs"]}
    longest = songs["The Long One"]["median_seconds"]
    assert isinstance(longest, float)
    assert 0.9 * LENGTHS[3] < longest < 1.1 * LENGTHS[3]


def test_a_tape_that_could_not_be_read_reaches_the_bundle_as_a_review_row(tmp_path):
    """The review queue is published because it is the honest half of the number beside it. A
    bundle that carried only what worked would describe a corpus that does not exist."""
    # A tape is only unreadable when BOTH ways of naming its tracks fail. Anonymous filenames
    # alone are not enough -- the taper's own written tracklist rescues those, which is the whole
    # point of reading descriptions at ingest. So: nothing in the filenames, nothing in the
    # description, and a night the OTHER tapes already put in the corpus, or this lands in the
    # different pile marked "no setlist for that night".
    unreadable = _tape("example2025-07-04.c", "2025-07-04",
                       songs=("audio",) * 5, uploader="three@example.org",
                       description="Recorded from the soundboard. Enjoy, and please don't sell.")
    _code, out = _export(tmp_path, TAPES + [unreadable])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert any(row["identifier"] == "example2025-07-04.c" for row in payload["review"])
    assert all(row["url"].startswith("https://archive.org/details/") for row in payload["review"])


# --- the date window ---------------------------------------------------------------------------
#
# The fixture is two nights: 2025-07-04 carries two tapes and 2025-07-05 carries one, so a window
# of just the second night changes every song's n from 2 to 1 and moves every median. A test that
# only checked row counts would pass just as happily against a bundle publishing whole-corpus
# statistics beside windowed performances, which is the failure these are here for.


def test_a_window_narrows_every_dated_section(tmp_path):
    _code, out = _export(tmp_path, None, "--since", "2025-07-05")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert {row["date"] for row in payload["performances"]} == {"2025-07-05"}
    assert payload["generated"]["date_range"] == ["2025-07-05", "2025-07-05"]
    assert all(row["date"] >= "2025-07-05" for row in payload["review"])
    assert all(row["date"] >= "2025-07-05" for row in payload["abandoned"])
    assert all(row["date"] >= "2025-07-05" for row in payload["edges"])


def test_song_statistics_describe_the_window_and_not_the_corpus(tmp_path):
    """THE POINT OF THE WINDOW. The stored statistics are computed over every year a song was
    played, so publishing them beside a narrowed performance list would put two populations in one
    file under one heading -- a median drawn from nights the reader cannot see."""
    # Read each bundle before running the next export: both calls write to the same
    # tmp_path/bundle.json, so holding the two Paths and reading them at the end compares the
    # second bundle with itself -- which passes the interesting assertion for the wrong reason.
    _code, whole = _export(tmp_path)
    whole_n = {s["song"]: s["n"] for s in json.loads(whole.read_text(encoding="utf-8"))["songs"]}
    _code, windowed = _export(tmp_path, None, "--since", "2025-07-05")
    windowed_n = {s["song"]: s["n"]
                  for s in json.loads(windowed.read_text(encoding="utf-8"))["songs"]}
    # Both nights played all five songs, so the corpus counts each twice and the window once.
    assert set(whole_n) == set(windowed_n)
    assert all(whole_n[song] == 2 for song in whole_n)
    assert all(windowed_n[song] == 1 for song in windowed_n)


def test_the_bundle_is_self_consistent_inside_its_window(tmp_path):
    """Every published median is the median of performances published in the SAME file.

    Asserted against the bundle's own rows rather than against expected numbers, because what
    matters is not which median is right in the abstract -- it is that a consumer joining `songs`
    to `performances` cannot find them describing different sets of nights.
    """
    _code, out = _export(tmp_path, None, "--since", "2025-07-05")
    payload = json.loads(out.read_text(encoding="utf-8"))
    counted = Counter(row["song"] for row in payload["performances"] if row["withheld"] is None)
    for stat in payload["songs"]:
        seconds = [row["seconds"] for row in payload["performances"]
                   if row["song"] == stat["song"] and row["withheld"] is None]
        assert stat["n"] == counted[stat["song"]] == len(seconds)
        assert stat["median_seconds"] == pytest.approx(median(seconds), abs=0.05)
        assert stat["min_seconds"] == pytest.approx(min(seconds), abs=0.05)
        assert stat["max_seconds"] == pytest.approx(max(seconds), abs=0.05)


def test_an_unwindowed_bundle_still_reads_the_stored_statistics(tmp_path):
    """The recomputation is for the ranged case only. With no window the export keeps its promise
    to publish exactly what derive stored, and the golden file above is the proof of the shape."""
    _code, out = _export(tmp_path)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["generated"]["window"] == {"since": None, "until": None}


def test_the_window_asked_for_is_recorded_beside_the_range_found(tmp_path):
    """Two fields because they answer different questions: a window opening before the data starts
    is complete, and a consumer with only `date_range` could not tell that from a gap."""
    _code, out = _export(tmp_path, None, "--since", "2020-01-01", "--until", "2025-07-04")
    generated = json.loads(out.read_text(encoding="utf-8"))["generated"]
    assert generated["window"] == {"since": "2020-01-01", "until": "2025-07-04"}
    assert generated["date_range"] == ["2025-07-04", "2025-07-04"]


def test_credits_cover_only_the_tapes_in_the_window(tmp_path):
    """Crediting a taper for a reel the reader cannot see names them for absent work."""
    _code, out = _export(tmp_path, None, "--since", "2025-07-05")
    credited = {row["uploader"] for row in json.loads(out.read_text(encoding="utf-8"))["credits"]}
    # two@example.org taped only 2025-07-04, so the window must not credit them.
    assert len(credited) == 1


def test_an_empty_window_is_not_reported_as_an_empty_store(tmp_path, capsys):
    """`derive` would cheerfully rewrite a correct table to prove a point about a year the band
    did not tour, so the two emptinesses must not share a message."""
    code, _out = _export(tmp_path, None, "--since", "2099-01-01")
    assert code == EXIT_NOTHING
    printed = capsys.readouterr().out
    assert "from 2099-01-01" in printed
    assert "Run `slkit derive" not in printed


def test_a_malformed_window_date_is_refused_rather_than_compared(tmp_path):
    """`--until 2023` sorts below every date IN 2023. Refused for the same reason `dump` refuses
    it: it returns rows, and the rows are wrong."""
    _cache(tmp_path, TAPES)
    config = _cfg(tmp_path)
    assert main(["--config", config, "ingest"]) == EXIT_OK
    assert main(["--config", config, "derive", "durations"]) == EXIT_OK
    out = tmp_path / "bundle.json"
    code = main(["--config", config, "export", "tapemeasure", "--out", str(out), "--until", "2025"])
    assert code != EXIT_OK
    assert not out.exists()

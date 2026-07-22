# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `slkit derive durations`, driven end to end from the cache through to statistics.

Nothing here mocks the reconciliation. A cache is populated with the payload shapes archive.org
returns, ``slkit ingest`` publishes the corpus and the mirror from it, and then derive reads the
STORE -- never the cache -- and the assertions are on what lands in SQLite.

The two-command shape IS the test. Every fixture below is ingested first and derived second, in
separate ``main()`` calls with nothing carried between them but the database, because the failure
this arrangement exists to prevent is exactly the one that only appears when the two halves run at
different times on different machines.
"""

import json
import shutil
from pathlib import Path

from setlistkit.cli.main import EXIT_OK, main
from setlistkit.store import Store
from setlistkit.store.raw_cache import RawCache

PACK = Path(__file__).resolve().parents[2] / "examples" / "packs" / "example"

CONFIG = ('data_root = "state"\n'
          'user_agent = "famoe.ly nightly (you@example.com)"\n'
          f'[catalog]\npack = "{PACK}"\n'
          '[sources.archive_org]\ncollection = "example"\n')

# Five songs, because a tracklist reader will not believe fewer than five lines is a listing
# rather than prose that happens to mention a song.
SONGS = ("Aurora", "Wormhole", "Sound Asleep", "The Long One", "Jamboree")
LENGTHS = (300.0, 480.0, 240.0, 620.0, 180.0)

SETLIST = "Set 1:\n" + "".join(f"{n:02d}. {song}\n" for n, song in enumerate(SONGS, start=1))

# Titles the example pack has never heard of. A tape of these is a tape nothing can name, which is
# the ordinary way a real one fails: a side project's set, or a night whose setlist we do not hold.
UNKNOWN = tuple(f"Unnamed Thing {n}" for n in range(1, 6))

# Filenames that name nothing at all -- "audio", numbered by the taper's rig and nothing else.
# Common enough on real tapes, and the case the taper's WRITTEN tracklist exists to rescue.
ANONYMOUS = ("audio",) * 5


def _cfg(tmp_path):
    path = tmp_path / "slkit.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return str(path)


def _files(lengths, songs):
    """One tape's audio derivatives, in the shape archive.org's ``files`` array arrives in.

    The song name is IN THE FILENAME, the way tapers actually write them, because that is what the
    join reads. A fixture of anonymous "track 1" files would exercise a path no real tape takes,
    and would sit there passing while the thing under test did nothing at all.
    """
    return [{"name": f"example.d1t{idx:02d}.{songs[(idx - 1) % len(songs)]}.flac",
             "title": songs[(idx - 1) % len(songs)],
             "format": "Flac", "track": str(idx), "length": f"{length:.2f}"}
            for idx, length in enumerate(lengths, start=1)]


def _tape(identifier, date, lengths=LENGTHS, *, uploader="taper@example.org",
          songs=SONGS, description=SETLIST):
    return {"identifier": identifier,
            "title": f"The Example Live at Northlands on {date}",
            "date": date, "uploader": uploader, "description": description,
            "files": _files(lengths, songs)}


def _cache(tmp_path, tapes):
    cache = RawCache(tmp_path / "state")
    docs = [{"identifier": t["identifier"], "date": t["date"], "title": t["title"]} for t in tapes]
    cache.put("archive_org", "advancedsearch/example/None/p1",
              json.dumps({"response": {"docs": docs, "numFound": len(docs)}}).encode("utf-8"))
    for tape in tapes:
        meta = dict(tape)
        cache.put("archive_org", meta["identifier"],
                  json.dumps({"metadata": meta,
                              "files": meta.pop("files", [])}).encode("utf-8"))
    return cache


def _run(tmp_path, tapes, *derive_args):
    """Ingest, then derive, as two separate commands sharing only the database."""
    _cache(tmp_path, tapes)
    config = _cfg(tmp_path)
    assert main(["--config", config, "ingest"]) == EXIT_OK
    return main(["--config", config, "derive", "durations", *derive_args])


def _stored(tmp_path):
    with Store(tmp_path / "state") as store:
        return store.durations.performances(), store.durations.song_length_stats()


def test_one_tape_of_one_night_becomes_a_duration_per_song(tmp_path):
    """The whole chain, at its smallest: files -> tracks -> a reading -> a performance."""
    assert _run(tmp_path, [_tape("example2025-07-04", "2025-07-04")]) == EXIT_OK
    performances, stats = _stored(tmp_path)
    assert [(row["song"], row["seconds"]) for row in performances] == list(zip(SONGS, LENGTHS))
    assert {row["song"]: row["median_seconds"] for row in stats} == dict(zip(SONGS, LENGTHS))
    assert all(row["n_ballots"] == 1 for row in performances)


def test_two_tapers_of_one_night_agree_and_the_row_says_both_were_counted(tmp_path):
    """n_tapes sits beside the answer because a well-attested measurement and a lone one look
    identical once they are both just a number of seconds."""
    assert _run(tmp_path, [
        _tape("example2025-07-04", "2025-07-04", uploader="one@example.org"),
        _tape("example2025-07-04.b", "2025-07-04", (302.0, 478.0, 241.0, 618.0, 181.0),
              uploader="two@example.org")]) == EXIT_OK
    performances, _ = _stored(tmp_path)
    assert [row["n_ballots"] for row in performances] == [2] * 5
    assert all(row["suspect"] is False for row in performances)


def test_one_taper_who_posted_four_mixes_still_gets_one_vote(tmp_path):
    """A tape is not a vote -- a TAPER is.

    One person posting a soundboard, a matrix and two mic feeds has published one set of track
    splits four times. Letting each of them vote means the loudest uploader on a night decides
    that night, which is exactly what happened when the uploader field went missing in production.
    """
    prolific = [_tape(f"example2025-07-04.{n}", "2025-07-04", uploader="loud@example.org")
                for n in range(4)]
    quiet = _tape("example2025-07-04.z", "2025-07-04", (305.0, 485.0, 245.0, 625.0, 185.0),
                  uploader="quiet@example.org")
    assert _run(tmp_path, prolific + [quiet]) == EXIT_OK
    performances, _ = _stored(tmp_path)
    # Five tapes seen, two ballots cast. Both counts are stored, because reporting only the second
    # would turn "we consolidated four uploads" into "only two tapes of this night exist".
    assert [row["n_ballots"] for row in performances] == [2] * 5
    assert [row["n_tapes_seen"] for row in performances] == [5] * 5


def test_derive_reads_the_store_even_with_the_raw_cache_deleted(tmp_path):
    """THE POINT OF THE WHOLE ARRANGEMENT.

    The cache is gitignored, so on the server it is simply not there. A derive that read it would
    work perfectly on the machine that pulled and publish nothing anywhere else -- the failure
    that cost the previous implementation its uploader field, and with it the ability to tell four
    tapers apart from one taper who posted four times.
    """
    _cache(tmp_path, [_tape("example2025-07-04", "2025-07-04")])
    config = _cfg(tmp_path)
    assert main(["--config", config, "ingest"]) == EXIT_OK
    shutil.rmtree(tmp_path / "state" / "raw")
    assert main(["--config", config, "derive", "durations"]) == EXIT_OK
    performances, _ = _stored(tmp_path)
    assert [row["song"] for row in performances] == list(SONGS)


def test_a_dry_run_reports_in_full_and_writes_nothing(tmp_path, capsys):
    """The command someone reaches for while diagnosing, so it must do all the work and none of
    the writing. A dry run that skipped the reconciliation would be silent about the half most
    likely to be what they are checking."""
    assert _run(tmp_path, [_tape("example2025-07-04", "2025-07-04")], "--dry-run") == EXIT_OK
    out = capsys.readouterr().out
    assert "5 performance(s) over 1 night(s)" in out
    assert "dry run: nothing written" in out
    assert _stored(tmp_path) == ([], [])


def test_a_bounced_set_is_abandoned_rather_than_timed_as_a_song(tmp_path):
    """One file holding a whole set is not a puzzle to be solved. Listed so the tape count and the
    timed-tape count add up, and never queued for a review nobody could act on."""
    assert _run(tmp_path, [_tape("example2025-07-04", "2025-07-04",
                                 (4200.0, 30.0, 40.0, 50.0, 60.0))]) == EXIT_OK
    with Store(tmp_path / "state") as store:
        rows = store.conn.execute(
            "SELECT identifier, longest_seconds, n_tracks FROM duration_abandoned").fetchall()
        assert [(row["identifier"], row["longest_seconds"]) for row in rows] == [
            ("example2025-07-04", 4200.0)]
        assert store.durations.counts()["duration_review"] == 0
        assert store.durations.counts()["performance_durations"] == 0


def test_a_tape_nothing_can_name_is_queued_with_the_counts_that_disagreed(tmp_path):
    """So the next reader starts from the three numbers rather than re-running the join by hand.

    Which pair disagrees is the diagnosis. Tracks against setlist is a tape of part of a night.
    Tracks against description is a listing we read wrong. Description against setlist is a taper
    who wrote down a different show.

    SEVEN files against a five-song listing, with filenames naming nothing the pack knows. The
    listing is the safety net and it only holds when it lines up with the files, so seven against
    five refuses it -- and with the listing refused there is nothing left but the filenames.

    Note what this is NOT: a tape whose description is unreadable produces no setlist for its date
    at all, so it is a missing night rather than an unreadable tape. The two are counted
    separately because only one of them is a bug on our side.
    """
    assert _run(tmp_path, [_tape("example2025-07-04", "2025-07-04",
                                 (10.0,) * 7, songs=UNKNOWN)]) == EXIT_OK
    with Store(tmp_path / "state") as store:
        row, = store.durations.review()
        assert row["identifier"] == "example2025-07-04"
        # n_desc is 0 rather than 5: an unmatched listing is stored, but it is not handed out to
        # anything that would time a song with it, and this row records what was usable.
        assert (row["n_tracks"], row["n_setlist"], row["n_desc"]) == (7, 5, 0)
        assert row["url"].endswith("example2025-07-04")
        assert store.durations.counts()["performance_durations"] == 0


def test_deriving_twice_replaces_rather_than_doubles(tmp_path):
    """One tape arriving for one night can change the answer for every performance of that night,
    so a run recomputes the lot. Appending would leave last week's answer beside this week's with
    nothing saying which is current."""
    assert _run(tmp_path, [_tape("example2025-07-04", "2025-07-04")]) == EXIT_OK
    assert main(["--config", _cfg(tmp_path), "derive", "durations"]) == EXIT_OK
    performances, stats = _stored(tmp_path)
    assert len(performances) == 5 and len(stats) == 5


def test_deriving_with_nothing_ingested_says_so_rather_than_publishing_an_empty_corpus(tmp_path):
    """An empty result and a result computed from nothing are the same rows and different facts.

    Publishing the second would replace a good set of statistics with none, and report success.
    """
    assert main(["--config", _cfg(tmp_path), "store", "init"]) == EXIT_OK
    assert main(["--config", _cfg(tmp_path), "derive", "durations"]) == EXIT_OK
    assert _stored(tmp_path) == ([], [])


def test_the_report_separates_the_three_ways_a_tape_can_produce_nothing(tmp_path, capsys):
    """They want three different things from a person, and reporting them as one number called
    "skipped" is how a slow drift in any one of them goes unnoticed for a year."""
    assert _run(tmp_path, [
        _tape("example2025-07-04", "2025-07-04"),
        _tape("example2025-07-05", "2025-07-05", (4200.0, 30.0, 40.0, 50.0, 60.0)),
        _tape("example2025-07-06", "2025-07-06", (10.0,) * 7, songs=UNKNOWN),
    ]) == EXIT_OK
    out = capsys.readouterr().out
    assert "'one track holds a whole set': 1" in out
    assert "'could not name enough of its tracks': 1" in out
    assert "'timed': 1" in out
    assert "abandoned as bounced sets" in out
    assert "1 tape(s) queued for review:" in out


def test_the_report_says_how_much_of_the_corpus_rests_on_a_single_taper(tmp_path, capsys):
    """Invisible from the performance count: twenty thousand performances timed once each and
    twenty thousand timed four times each print the same total, and only one of them is
    evidence."""
    assert _run(tmp_path, [
        _tape("example2025-07-04", "2025-07-04", uploader="one@example.org"),
        _tape("example2025-07-04.b", "2025-07-04", (302.0, 478.0, 241.0, 618.0, 181.0),
              uploader="two@example.org"),
        _tape("example2025-07-05", "2025-07-05", uploader="one@example.org"),
    ]) == EXIT_OK
    out = capsys.readouterr().out
    assert "5 performance(s) rest on a single taper (50.0%)" in out


def test_a_tape_with_useless_filenames_is_rescued_by_the_tapers_own_written_tracklist(tmp_path):
    """The reason tracklists are read at all, and the reason they are read at INGEST.

    Plenty of tapers name every file "audio" and write what is on them in the description instead.
    With only the filenames there is nothing to join on and the tape is unreadable; with the
    listing it times five songs. On the real corpus this is the difference between 380 tapes
    joined and 3,266.

    It also pins the storage round trip end to end: the description is read during INGEST and the
    listing is read back from the database during DERIVE, so a listing that is computed and not
    stored fails here exactly as loudly as one that is never computed.
    """
    assert _run(tmp_path, [_tape("example2025-07-04", "2025-07-04",
                                 songs=ANONYMOUS)]) == EXIT_OK
    performances, _ = _stored(tmp_path)
    assert [(row["song"], row["seconds"]) for row in performances] == list(zip(SONGS, LENGTHS))


def test_ingest_stores_the_written_tracklist_so_derive_never_opens_the_cache(tmp_path):
    """Ingest is the one command allowed to read a description, because it is the only moment the
    raw payload is open. Everything after it reads this table."""
    _cache(tmp_path, [_tape("example2025-07-04", "2025-07-04", songs=ANONYMOUS)])
    assert main(["--config", _cfg(tmp_path), "ingest"]) == EXIT_OK
    with Store(tmp_path / "state") as store:
        listings = store.tapes.listings()
        assert [entry["song"] for entry in listings["example2025-07-04"]] == list(SONGS)
        assert store.tapes.listing_readings() == {"numbered": 1}

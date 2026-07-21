# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `slkit ingest`, driven end to end through the shipped example pack.

Nothing here mocks the parser or the merge. The cache is populated with the payload shapes
archive.org actually returns, and the assertions are on what lands in SQLite and on what the
report says about what did not. That is the point: every one of these failures is a show quietly
missing from a corpus that reports a clean run.
"""

import json
from pathlib import Path

import pytest

from setlistkit.cli.main import EXIT_DIAGNOSTIC, EXIT_OK, main
from setlistkit.store import Store
from setlistkit.store.raw_cache import RawCache

PACK = Path(__file__).resolve().parents[2] / "examples" / "packs" / "example"

CONFIG = ('data_root = "state"\n'
          'user_agent = "famoe.ly nightly (you@example.com)"\n'
          f'[catalog]\npack = "{PACK}"\n'
          '[sources.archive_org]\ncollection = "example"\n')

# The three shapes the parser has to tell apart, plus one ordinary night.
ITEMS = {
    "example2025-07-04": {
        "title": "The Example Live at Northlands on 2025-07-04", "date": "2025-07-04",
        "description": "Set 1:\n01. Aurora\n02. Wormhole >\n03. Tuning\nEncore:\n04. Jamboree\n"},
    "other2025-01-01": {                                   # a side project, same collection
        "title": "Some Other Band Live at The Fillmore on 2025-01-01", "date": "2025-01-01",
        "description": "Set 1:\n01. Aurora\n02. Wormhole\n"},
    "example2025-10-31": {                                 # the pack refuses this night
        "title": "The Example Live at The Fillmore on 2025-10-31", "date": "2025-10-31",
        "description": "Set 1:\n01. Aurora\n02. Wormhole\n"},
    "undated": {                                           # no date we can believe
        "title": "The Example Live at Nowhere on ????", "date": "",
        "description": "Set 1:\n01. Aurora\n"},
}


def _cfg(tmp_path, body=CONFIG):
    path = tmp_path / "slkit.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def _cache(tmp_path, items=None, *, num_found=None, page=1):
    """Populate the raw cache the way a pull would have left it."""
    items = ITEMS if items is None else items
    cache = RawCache(tmp_path / "state")
    docs = [{"identifier": ident, "date": meta.get("date", ""), "title": meta.get("title", "")}
            for ident, meta in items.items()]
    response = {"docs": docs}
    if num_found != "omit":                       # "omit" writes a listing that records no count
        response["numFound"] = len(docs) if num_found is None else num_found
    cache.put("archive_org", f"advancedsearch/example/None/p{page}",
              json.dumps({"response": response}).encode("utf-8"))
    for ident, meta in items.items():
        # `files` sits beside `metadata` in an archive.org payload, not inside it. Split here so
        # a fixture can be written as one flat dict.
        meta = dict(meta)
        cache.put("archive_org", ident, json.dumps(
            {"metadata": meta, "files": meta.pop("files", [])}).encode("utf-8"))
    return cache


def _shows(tmp_path):
    with Store(tmp_path / "state") as store:
        return store.shows()


def test_ingest_publishes_the_shows_it_could_parse(tmp_path, capsys):
    _cache(tmp_path)
    assert main(["--config", _cfg(tmp_path), "ingest"]) == EXIT_OK
    shows = _shows(tmp_path)
    assert [show["date"] for show in shows] == ["2025-07-04"]
    assert [entry["song"] for entry in shows[0]["sets"][0]] == ["Aurora", "Wormhole", "Tuning"]
    assert shows[0]["encore"] == [{"song": "Jamboree", "segue": False, "non_song": False}]


def test_a_tagged_non_song_is_kept_and_a_segue_survives_to_sqlite(tmp_path):
    _cache(tmp_path)
    main(["--config", _cfg(tmp_path), "ingest"])
    entries = _shows(tmp_path)[0]["sets"][0]
    assert entries[1] == {"song": "Wormhole", "segue": True, "non_song": False}
    assert entries[2] == {"song": "Tuning", "segue": False, "non_song": True}


def test_ingest_says_why_each_refused_item_is_missing(tmp_path, capsys):
    _cache(tmp_path)
    main(["--config", _cfg(tmp_path), "ingest"])
    out = capsys.readouterr().out
    assert "refused 3 of 4 item(s):" in out
    assert "1: title names a different band" in out
    assert "1: no date we can believe" in out
    assert "1 on 1 dropped date(s):" in out


def test_a_dropped_date_reports_the_reason_the_pack_gave_for_it(tmp_path, capsys):
    """The reason is the only thing a later reader can check the call against, and it lives in a
    file nobody opens during a run. So the run repeats it."""
    _cache(tmp_path)
    main(["--config", _cfg(tmp_path), "ingest"])
    out = capsys.readouterr().out
    assert "2025-10-31 (1 tape(s)): Halloween covers set." in out
    assert "[...]" in out                             # the paragraph is shortened, not dumped


def test_ingest_refuses_to_run_with_nothing_cached(tmp_path, capsys):
    assert main(["--config", _cfg(tmp_path), "ingest"]) == EXIT_DIAGNOSTIC
    err = capsys.readouterr().err
    assert "no usable cached listing" in err
    assert "slkit pull archive_org" in err
    # The min_year trap gets named, because the failure looks identical to an empty collection.
    assert "min_year" in err


def test_a_second_ingest_reports_the_diff_rather_than_the_totals(tmp_path, capsys):
    _cache(tmp_path, {k: ITEMS[k] for k in ("example2025-07-04",)})
    main(["--config", _cfg(tmp_path), "ingest"])
    capsys.readouterr()

    extra = dict(ITEMS)
    extra["example2025-07-05"] = {
        "title": "The Example Live at Northlands on 2025-07-05", "date": "2025-07-05",
        "description": "Set 1:\n01. Aurora\n02. Jamboree\n"}
    _cache(tmp_path, extra)
    main(["--config", _cfg(tmp_path), "ingest"])
    out = capsys.readouterr().out
    assert "(was 1 shows / 3 songs)" in out
    assert "+1 new date(s): ['2025-07-05']" in out


def test_a_long_list_of_new_dates_is_abbreviated_but_says_how_many_it_cut(tmp_path, capsys):
    """A first ingest of a real collection adds hundreds of dates, and printing them all pushes
    every warning above it off the scrollback. Never a silent truncation, though."""
    _cache(tmp_path, _many(20))
    main(["--config", _cfg(tmp_path), "ingest"])
    out = capsys.readouterr().out
    assert "+20 new date(s)" in out
    assert "and 8 more" in out


def test_a_date_leaving_the_corpus_is_never_silent(tmp_path, capsys):
    both = {k: ITEMS[k] for k in ("example2025-07-04",)}
    both["example2025-07-05"] = {
        "title": "The Example Live at Northlands on 2025-07-05", "date": "2025-07-05",
        "description": "Set 1:\n01. Aurora\n02. Jamboree\n03. Wormhole\n"}
    _cache(tmp_path, both)
    main(["--config", _cfg(tmp_path), "ingest"])
    capsys.readouterr()

    _cache(tmp_path, {k: ITEMS[k] for k in ("example2025-07-04",)})
    main(["--config", _cfg(tmp_path), "ingest"])
    assert "-1 removed date(s): ['2025-07-05']" in capsys.readouterr().out


def test_ingest_reports_the_source_that_won_each_date(tmp_path, capsys):
    _cache(tmp_path)
    main(["--config", _cfg(tmp_path), "ingest"])
    assert "winners by source: {'description': 1}" in capsys.readouterr().out


def test_a_date_changing_which_source_it_trusts_is_reported(tmp_path, capsys):
    """Usually a source flip is right; occasionally it is the first sign something broke. It is
    not visible from the corpus alone, so the run that caused it is where it has to be said."""
    thin = {"example2025-07-04": {
        "title": "The Example Live at Northlands on 2025-07-04", "date": "2025-07-04",
        "description": "a lovely evening, thanks all",       # prose, not a setlist
        "files": [{"format": "Flac", "title": t} for t in
                  ("Aurora", "Wormhole", "Jamboree", "Sound Asleep")]}}
    _cache(tmp_path, thin)
    main(["--config", _cfg(tmp_path), "ingest"])
    assert "winners by source: {'tracks': 1}" in capsys.readouterr().out

    _cache(tmp_path, {k: ITEMS[k] for k in ("example2025-07-04",)})
    main(["--config", _cfg(tmp_path), "ingest"])
    out = capsys.readouterr().out
    assert "1 date(s) changed source:" in out
    assert "2025-07-04: tracks -> description" in out


def test_a_refusal_reason_with_no_label_yet_still_gets_a_line(capsys):
    """A module whose whole thesis is "say why a night is missing" must not answer a new kind of
    refusal with silence. The label table grows when the second source lands."""
    from setlistkit.catalog.parse import Skipped
    from setlistkit.cli.ingest import _report_skipped

    class _Pack:
        corpus = type("_C", (), {"drop_dates": {}})()

    _report_skipped((Skipped("mystery-item", "some_new_rule"),), _Pack(), 1)
    out = capsys.readouterr().out
    assert "1: some_new_rule (no label for this reason yet)" in out
    assert "mystery-item" in out


def test_refusals_with_no_dropped_dates_do_not_print_an_empty_heading(tmp_path, capsys):
    _cache(tmp_path, {k: ITEMS[k] for k in ("example2025-07-04", "other2025-01-01")})
    main(["--config", _cfg(tmp_path), "ingest"])
    out = capsys.readouterr().out
    assert "refused 1 of 2 item(s):" in out
    assert "dropped date" not in out


def test_a_truncated_cached_listing_is_flagged_at_ingest_too(tmp_path, capsys):
    """The pull that hit the backstop may have run days ago. The corpus is what is wrong now."""
    cache = RawCache(tmp_path / "state")
    for page in range(1, 41):                    # fills the paging backstop, never ends cleanly
        docs = [{"identifier": f"example{page}", "date": "2025-07-04", "title": "The Example"}]
        cache.put("archive_org", f"advancedsearch/example/None/p{page}",
                  json.dumps({"response": {"docs": docs, "numFound": 9999}}).encode("utf-8"))
        cache.put("archive_org", f"example{page}",
                  json.dumps({"metadata": ITEMS["example2025-07-04"], "files": []}).encode("utf-8"))
    main(["--config", _cfg(tmp_path), "ingest"])
    assert "PREFIX of the\n  collection" in capsys.readouterr().out


# --- the no-shrink guard --------------------------------------------------------------------

def _many(count):
    return {f"example2025-{month:02d}-{day:02d}": {
        "title": f"The Example Live at Northlands on 2025-{month:02d}-{day:02d}",
        "date": f"2025-{month:02d}-{day:02d}",
        "description": "Set 1:\n01. Aurora\n02. Wormhole\n03. Jamboree\n"}
        for month, day in [(1 + n // 28, 1 + n % 28) for n in range(count)]}


def test_a_collapsed_merge_is_refused_rather_than_published(tmp_path, capsys):
    """The failure being caught is upstream, and it produces a small, clean, entirely wrong
    corpus in which every other number looks reasonable."""
    full = _many(10)
    _cache(tmp_path, full)
    main(["--config", _cfg(tmp_path), "ingest"])
    capsys.readouterr()

    _cache(tmp_path, dict(list(full.items())[:3]))     # an upstream that went missing
    assert main(["--config", _cfg(tmp_path), "ingest"]) == EXIT_DIAGNOSTIC
    err = capsys.readouterr().err
    assert "refusing to publish: shows fell from 10 to 3" in err
    assert "--force" in err
    assert len(_shows(tmp_path)) == 10                 # the good corpus is untouched


def test_force_publishes_a_shrink_the_operator_meant(tmp_path, capsys):
    full = _many(10)
    _cache(tmp_path, full)
    main(["--config", _cfg(tmp_path), "ingest"])
    _cache(tmp_path, dict(list(full.items())[:3]))
    assert main(["--config", _cfg(tmp_path), "ingest", "--force"]) == EXIT_OK
    assert len(_shows(tmp_path)) == 3


def test_a_song_leaving_every_show_is_reported_even_though_no_date_moved(tmp_path, capsys):
    """The failure a show count cannot see.

    A junk fragment that reaches a real title deletes it from EVERY night at once. Every date
    survives, every source survives, nothing is added and nothing is removed -- so a report
    counting shows prints the identical line it printed the run before, for a corpus that just
    lost a song from every show in it. `_claimed` protects titles the pack names, so what this
    eats is exactly the songs no source has named yet: the ones nothing else can recover.
    """
    import shutil
    pack = tmp_path / "pack"
    shutil.copytree(PACK, pack)
    items = {f"example2025-07-0{n}": {
        "title": f"The Example Live at Northlands on 2025-07-0{n}", "date": f"2025-07-0{n}",
        "description": "Set 1:\n01. Aurora\n02. Moon Sonnet\n03. Wormhole\n"} for n in range(1, 7)}
    _cache(tmp_path, items)
    main(["--config", _cfg_for(tmp_path, pack), "ingest"])
    capsys.readouterr()

    corpus = json.loads((pack / "corpus.json").read_text(encoding="utf-8"))
    corpus["junk_patterns"].append({"pattern": r"moon\s+sonnet", "why": "we think it is a note"})
    (pack / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    main(["--config", _cfg_for(tmp_path, pack), "ingest"])
    out = capsys.readouterr().out

    assert "6 shows / 12 songs from 6 items (was 6 shows / 18 songs)" in out
    assert "+0 new date(s)" in out                  # every other number is unchanged
    assert "removed date" not in out


def test_a_collapse_in_songs_alone_is_refused(tmp_path, capsys):
    import shutil
    pack = tmp_path / "pack"
    shutil.copytree(PACK, pack)
    # Titles the pack does NOT name, because `_claimed` refuses to let any drop rule delete one
    # it does -- which is the guard working, and exactly why the songs at risk are the unnamed
    # ones. Nothing else can recover those.
    items = {f"example2025-07-0{n}": {
        "title": f"The Example Live at Northlands on 2025-07-0{n}", "date": f"2025-07-0{n}",
        "description": "Set 1:\n01. Aurora\n02. Moon Sonnet\n03. Star Waltz\n"}
        for n in range(1, 7)}
    _cache(tmp_path, items)
    main(["--config", _cfg_for(tmp_path, pack), "ingest"])
    capsys.readouterr()

    # Two of the three titles now match a junk fragment: every date survives, two thirds of the
    # repertoire does not.
    corpus = json.loads((pack / "corpus.json").read_text(encoding="utf-8"))
    corpus["junk_patterns"].append({"pattern": r"moon\s+sonnet|star\s+waltz",
                                    "why": "a bad edit"})
    (pack / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    assert main(["--config", _cfg_for(tmp_path, pack), "ingest"]) == EXIT_DIAGNOSTIC
    err = capsys.readouterr().err
    assert "refusing to publish: songs fell from 18 to 6" in err
    assert "junk or gear fragment" in err
    with Store(tmp_path / "state") as store:
        assert store.song_count() == 18            # the good corpus is untouched


def test_the_report_runs_before_the_guard_refuses(tmp_path, capsys):
    """The guard tells you to go and look at what changed; this run already knows."""
    full = _many(10)
    _cache(tmp_path, full)
    main(["--config", _cfg(tmp_path), "ingest"])
    capsys.readouterr()
    _cache(tmp_path, dict(list(full.items())[:3]))
    main(["--config", _cfg(tmp_path), "ingest"])
    out, err = capsys.readouterr()
    assert "-7 removed date(s):" in out             # the evidence, not withheld
    assert "refusing to publish" in err


def test_dry_run_says_what_would_be_refused_instead_of_refusing(tmp_path, capsys):
    """A dry run cannot touch the stored corpus, so refusing would make the one command that is
    safe to run while diagnosing a shrink the one command that will not tell you about it."""
    full = _many(10)
    _cache(tmp_path, full)
    main(["--config", _cfg(tmp_path), "ingest"])
    capsys.readouterr()
    _cache(tmp_path, dict(list(full.items())[:3]))
    assert main(["--config", _cfg(tmp_path), "ingest", "--dry-run"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "a real run would refuse this (shows fell from 10 to 3)" in out
    assert "-7 removed date(s):" in out
    assert len(_shows(tmp_path)) == 10


def test_a_modest_shrink_is_allowed_through(tmp_path):
    full = _many(10)
    _cache(tmp_path, full)
    main(["--config", _cfg(tmp_path), "ingest"])
    _cache(tmp_path, dict(list(full.items())[:8]))     # 80%, well over the bar
    assert main(["--config", _cfg(tmp_path), "ingest"]) == EXIT_OK
    assert len(_shows(tmp_path)) == 8


def test_the_first_ingest_is_never_a_shrink(tmp_path):
    # Nothing stored means nothing to measure against, and a fresh install must not need --force.
    _cache(tmp_path, {k: ITEMS[k] for k in ("example2025-07-04",)})
    assert main(["--config", _cfg(tmp_path), "ingest"]) == EXIT_OK


@pytest.mark.parametrize("kept, allowed", [(5, True), (4, False)])
def test_the_floor_is_exactly_half(tmp_path, kept, allowed):
    """Both sides of the boundary, because "under 50%" and "at most 50%" are different rules and
    only one of them matches what the constant is documented to mean."""
    full = _many(10)
    _cache(tmp_path, full)
    main(["--config", _cfg(tmp_path), "ingest"])
    _cache(tmp_path, dict(list(full.items())[:kept]))
    code = main(["--config", _cfg(tmp_path), "ingest"])
    assert (code == EXIT_OK) is allowed
    assert len(_shows(tmp_path)) == (kept if allowed else 10)


def test_dry_run_reports_everything_and_writes_nothing(tmp_path, capsys):
    _cache(tmp_path)
    assert main(["--config", _cfg(tmp_path), "ingest", "--dry-run"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "dry run: nothing written" in out
    assert "refused 3 of 4 item(s):" in out            # the whole report still runs
    assert _shows(tmp_path) == []


# --- a cache that is not all there ----------------------------------------------------------

def test_an_item_with_no_readable_metadata_is_named_not_skipped(tmp_path, capsys):
    _cache(tmp_path)
    RawCache(tmp_path / "state").put("archive_org", "example2025-07-04", b'{"metadata": {"tit')
    main(["--config", _cfg(tmp_path), "ingest"])
    out = capsys.readouterr().out
    assert "1 listed item(s) have no readable cached metadata" in out
    assert "example2025-07-04" in out
    # And the way out is named, because an ordinary pull will not fix it.
    assert "--force-rescan" in out


def test_a_listing_that_promised_more_than_it_named_is_flagged(tmp_path, capsys):
    _cache(tmp_path, num_found=99)
    main(["--config", _cfg(tmp_path), "ingest"])
    assert "promised 99; the pull did not finish" in capsys.readouterr().out


def test_a_cached_error_document_is_refused_not_published_as_an_empty_corpus(tmp_path, capsys):
    """Valid JSON of the wrong shape. Every check would have passed it: a page WAS found, and
    `expected` is None because no numFound was readable, which switches off the only count."""
    RawCache(tmp_path / "state").put("archive_org", "advancedsearch/example/None/p1",
                                     json.dumps({"error": "rate limited"}).encode("utf-8"))
    assert main(["--config", _cfg(tmp_path), "ingest"]) == EXIT_DIAGNOSTIC
    err = capsys.readouterr().err
    assert "no usable cached listing" in err
    assert "error document the source returned with status\n  200" in err
    # It refused before opening the database, so there is not even an empty corpus to find.
    assert not (tmp_path / "state" / "setlistkit.sqlite").is_file()


def test_a_listing_with_no_item_count_says_it_cannot_check_completeness(tmp_path, capsys):
    """`expected` is the only completeness check that counts anything, and it is absent exactly
    when the listing payload was damaged. Its absence is itself news."""
    _cache(tmp_path, num_found="omit")
    main(["--config", _cfg(tmp_path), "ingest"])
    assert "recorded no item count, so this run cannot tell whether it is complete" \
        in capsys.readouterr().out


def test_a_cached_listing_doc_with_no_identifier_is_counted(tmp_path, capsys):
    cache = RawCache(tmp_path / "state")
    _cache(tmp_path, {k: ITEMS[k] for k in ("example2025-07-04",)})
    docs = [{"identifier": "example2025-07-04", "date": "2025-07-04", "title": "The Example"},
            {"date": "2025-07-05"}]                # a doc nothing could ever be fetched for
    cache.put("archive_org", "advancedsearch/example/None/p1",
              json.dumps({"response": {"docs": docs, "numFound": 2}}).encode("utf-8"))
    main(["--config", _cfg(tmp_path), "ingest"])
    out = capsys.readouterr().out
    assert "1 cached listing doc(s) carry no identifier" in out


# --- overrides that do nothing --------------------------------------------------------------

def test_an_override_on_a_dropped_date_is_never_silently_discarded(tmp_path, capsys):
    """A hand-confirmed whole show is the highest-evidence input this system takes.

    The drop wins, and it should: a refused date is refused however carefully someone wrote it
    down. But two files in the same pack disagreeing about one night has to be said out loud,
    or someone's evening of listening does nothing and nothing anywhere mentions it.
    """
    import shutil
    pack = tmp_path / "pack"
    shutil.copytree(PACK, pack)
    (pack / "overrides.json").write_text(json.dumps({"overrides": {"2025-10-31": {
        "reason": "Confirmed by ear: they opened with two of their own before the covers set.",
        "sets": [["Aurora", "Wormhole", "Jamboree"]]}}}), encoding="utf-8")
    _cache(tmp_path)
    main(["--config", _cfg_for(tmp_path, pack), "ingest"])
    out = capsys.readouterr().out
    assert "1 override(s) NOT applied, because the date is in drop_dates: ['2025-10-31']" in out
    assert "Remove it from" in out
    assert "2025-10-31" not in [show["date"] for show in _shows(tmp_path)]


def test_a_losing_tape_can_still_raise_an_override_review(tmp_path, capsys):
    """The parse layer used to keep only the richest tape per date, so a losing tape carrying a
    song the override lacks never reached the review that exists to notice exactly that."""
    import shutil
    pack = tmp_path / "pack"
    shutil.copytree(PACK, pack)
    (pack / "overrides.json").write_text(json.dumps({"overrides": {"2025-07-04": {
        "reason": "Confirmed by ear from the soundboard.",
        "sets": [["Aurora", "Jamboree", "Sound Asleep"]]}}}), encoding="utf-8")
    _cache(tmp_path, {
        # the thin tape carries Wormhole and loses the date to the fuller one
        "aaa2025-07-04": {"title": "The Example Live at Northlands on 2025-07-04",
                          "date": "2025-07-04",
                          "description": "Set 1:\n01. Aurora\n02. Wormhole\n03. Jamboree\n"},
        "bbb2025-07-04": {"title": "The Example Live at Northlands on 2025-07-04",
                          "date": "2025-07-04",
                          "description": ("Set 1:\n01. Aurora\n02. Jamboree\n"
                                          "03. Sound Asleep\n04. The Long One\n")},
    })
    main(["--config", _cfg_for(tmp_path, pack), "ingest"])
    out = capsys.readouterr().out
    assert "['Wormhole']" in out                    # only the LOSING tape has it
    assert "aaa2025-07-04" in out


# --- overrides ------------------------------------------------------------------------------

OVERRIDE = json.dumps({"overrides": {"2025-07-04": {
    "reason": "Confirmed by ear against the soundboard; the taper merged two tracks into one.",
    "sets": [["Aurora", "The Long One >", "Wormhole"]],
    "encore": ["Jamboree"]}}})


@pytest.fixture(name="pack_with_override")
def _pack_with_override(tmp_path):
    """A copy of the example pack carrying an overrides.json."""
    import shutil
    pack = tmp_path / "pack"
    shutil.copytree(PACK, pack)
    (pack / "overrides.json").write_text(OVERRIDE, encoding="utf-8")
    return pack


def _cfg_for(tmp_path, pack):
    return _cfg(tmp_path, CONFIG.replace(str(PACK), str(pack)))


def test_an_override_replaces_the_parsed_show(tmp_path, capsys, pack_with_override):
    _cache(tmp_path)
    assert main(["--config", _cfg_for(tmp_path, pack_with_override), "ingest"]) == EXIT_OK
    show, = _shows(tmp_path)
    assert [entry["song"] for entry in show["sets"][0]] == ["Aurora", "The Long One", "Wormhole"]
    assert show["source"] == "override"
    assert show["reason"].startswith("Confirmed by ear")
    assert "1 override(s) applied: ['2025-07-04']" in capsys.readouterr().out


def test_a_source_carrying_a_song_the_override_lacks_is_put_in_front_of_a_person(
        tmp_path, capsys):
    """An override always wins, so nothing else will ever say it went stale. This is the signal.

    The override here is DELIBERATELY short a song the tape carries: it lists Aurora and
    Jamboree, the description also has Wormhole, and the override wins anyway. That is the whole
    scenario -- a correction that was right when it was written and has since been overtaken.
    """
    import shutil
    pack = tmp_path / "pack"
    shutil.copytree(PACK, pack)
    (pack / "overrides.json").write_text(json.dumps({"overrides": {"2025-07-04": {
        "reason": "Confirmed by ear; the taper's track 2 is two songs merged into one.",
        "sets": [["Aurora"]], "encore": ["Jamboree"]}}}), encoding="utf-8")

    _cache(tmp_path)
    main(["--config", _cfg_for(tmp_path, pack), "ingest"])
    out = capsys.readouterr().out
    assert "override review (1):" in out
    assert "2025-07-04: the override has 2 songs" in out
    assert "carries 1 song(s) the override lacks: ['Wormhole']" in out
    assert "the override still won" in out


def test_a_broken_override_file_stops_the_run_with_a_caret(tmp_path, capsys):
    import shutil
    pack = tmp_path / "pack"
    shutil.copytree(PACK, pack)
    (pack / "overrides.json").write_text(
        json.dumps({"overrides": {"2025-07-04": {"reason": "   ", "sets": [["Aurora"]]}}}),
        encoding="utf-8")
    _cache(tmp_path)
    assert main(["--config", _cfg_for(tmp_path, pack), "ingest"]) == EXIT_DIAGNOSTIC
    err = capsys.readouterr().err
    assert "every override needs a non-empty 'reason'" in err
    assert "no reason given" in err                    # the caret caption, so it is positioned
    assert "overrides.json" in err

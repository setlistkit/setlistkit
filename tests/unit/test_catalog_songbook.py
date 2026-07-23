# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the songbook bundle -- a pure function, tested without a database or a pack file.

`bundle()` takes plain dicts shaped like `store.corpus.shows()` and a `Normalizer` built directly
from a vocabulary list, with no `Pack` machinery in between. Unlike `catalog/tapemeasure.py`,
which has never had a test file of its own because its `Concluded` dataclass is awkward to hand
construct with real invariants, `songbook.bundle()`'s inputs are trivial to build directly -- and
building them directly is the ONLY reliable way to pin the exact floor/dedupe/unknown arithmetic
this file exists to test: the archive.org ingest pipeline used by `tests/unit/test_cli_export.py`
has its own five-line heuristic for "is this even a tracklist" that would fight a hand-built
below-floor fixture rather than cooperate with it. CLI-level wiring -- does `slkit export
songbook` call this function with the right arguments, does it write a file -- is tested there
instead; the arithmetic is tested here.
"""

import json

from setlistkit.catalog.normalizer import Normalizer
from setlistkit.catalog.songbook import MIN_SHOW_SONGS, SCHEMA, bundle


def _entry(song, segue=False, non_song=False):
    return {"song": song, "segue": segue, "non_song": non_song}


def _show(date, titles, encore=()):
    """A `store.corpus.shows()`-shaped show from a flat list of titles.

    Each item in ``titles``/``encore`` is either a bare song string, or a ``(title, non_song)``
    pair for an entry that should be tagged not-music.
    """
    def _mk(item):
        if isinstance(item, tuple):
            title, non_song = item
            return _entry(title, non_song=non_song)
        return _entry(item)
    return {"date": date, "sets": [[_mk(t) for t in titles]], "encore": [_mk(t) for t in encore]}


VOCAB = ["Aurora", "Wormhole", "Sound Asleep", "The Long One", "Jamboree"]


def test_the_schema_string_is_versioned():
    assert SCHEMA == "setlistkit.songbook/1"


def test_an_all_in_vocab_show_carries_no_unknown_and_no_dedupe():
    shows = [_show("2025-07-04", VOCAB)]
    payload = bundle(shows, normalizer=Normalizer(vocabulary=VOCAB), since=None, until=None,
                     fingerprint="fp")
    assert payload["vocab"] == sorted(VOCAB)
    assert payload["unknown"] == []
    assert payload["generated"]["deduped"] == 0
    assert payload["generated"]["below_floor"] == 0
    assert payload["generated"]["n_shows"] == 1
    assert payload["shows"] == [{"d": "2025-07-04", "s": list(range(5))}]


def test_a_repeated_song_in_one_show_collapses_to_one_index_and_is_counted_deduped():
    shows = [_show("2025-07-04", ["Aurora", "Wormhole", "Aurora", "Sound Asleep", "Wormhole"])]
    payload = bundle(shows, normalizer=Normalizer(vocabulary=VOCAB), since=None, until=None,
                     fingerprint="fp")
    assert payload["generated"]["deduped"] == 2          # Aurora repeats once, Wormhole once
    assert payload["vocab"] == ["Aurora", "Sound Asleep", "Wormhole"]
    assert payload["shows"] == [{"d": "2025-07-04", "s": [0, 1, 2]}]


def test_a_show_under_the_floor_is_dropped_and_counted_not_silently_discarded():
    shows = [_show("2025-01-01", ["Aurora", "Wormhole", "Sound Asleep"])]     # 3 raw entries
    payload = bundle(shows, normalizer=Normalizer(vocabulary=VOCAB), since=None, until=None,
                     fingerprint="fp")
    assert payload["generated"]["below_floor"] == 1
    assert payload["generated"]["n_shows"] == 0
    assert payload["shows"] == payload["vocab"] == payload["unknown"] == []
    assert payload["generated"]["first"] is None
    assert payload["generated"]["last"] is None


def test_the_floor_counts_non_song_entries_the_same_as_songs():
    """The design doc's own words: count it any other way and parity will not hold."""
    assert MIN_SHOW_SONGS == 5
    five_raw = _show("2025-02-02", ["Aurora", ("Setbreak", True), "Sound Asleep"],
                     encore=[("Tuning", True), ("Tuning", True)])
    payload = bundle([five_raw], normalizer=Normalizer(vocabulary=VOCAB), since=None, until=None,
                     fingerprint="fp")
    assert payload["generated"]["below_floor"] == 0        # kept: 3 songs + 2 non-songs = 5 raw
    assert payload["vocab"] == ["Aurora", "Sound Asleep"]   # the non-songs never enter it


def test_one_fewer_non_song_entry_drops_the_same_show_below_the_floor():
    four_raw = _show("2025-02-02", ["Aurora", ("Setbreak", True), "Sound Asleep"],
                     encore=[("Tuning", True)])
    payload = bundle([four_raw], normalizer=Normalizer(vocabulary=VOCAB), since=None, until=None,
                     fingerprint="fp")
    assert payload["generated"]["below_floor"] == 1
    assert payload["generated"]["n_shows"] == 0


def test_an_unknown_title_is_kept_and_flagged_not_dropped():
    shows = [_show("2025-07-04", ["Aurora", "Wormhole", "Sound Asleep", "The Long One",
                                  "Ghost Story"])]
    payload = bundle(shows, normalizer=Normalizer(vocabulary=VOCAB), since=None, until=None,
                     fingerprint="fp")
    assert "Ghost Story" in payload["vocab"]
    idx = payload["vocab"].index("Ghost Story")
    assert idx in payload["unknown"]
    assert idx in payload["shows"][0]["s"]                 # still counted, not removed


class _AliasedNormalizer(Normalizer):
    """A stand-in for a pack that has since learned an alias the corpus was ingested without."""

    def aliases(self):
        return {"rububula": "Rebubula"}


def test_in_vocab_canonicalizes_before_checking_so_a_freshly_aliased_name_is_recognised():
    """The exact bug this idiom exists to prevent: `normalize(raw) in norm_to_canon` would test
    'rububula' against a vocabulary keyed by 'rebubula' and wrongly call it unknown."""
    shows = [_show("2025-07-04", ["Rububula", "Aurora", "Wormhole", "Sound Asleep",
                                  "The Long One"])]
    normalizer = _AliasedNormalizer(vocabulary=["Rebubula", *VOCAB])
    payload = bundle(shows, normalizer=normalizer, since=None, until=None, fingerprint="fp")
    assert "Rebubula" in payload["vocab"]
    assert "Rububula" not in payload["vocab"]
    assert payload["vocab"].index("Rebubula") not in payload["unknown"]


def test_vocab_is_sorted_canonically_not_by_first_appearance():
    shows = [_show("2025-07-05", ["Wormhole", "Aurora", "Sound Asleep", "The Long One",
                                  "Jamboree"]),
             _show("2025-07-04", ["Jamboree", "The Long One", "Sound Asleep", "Aurora",
                                  "Wormhole"])]
    payload = bundle(shows, normalizer=Normalizer(vocabulary=VOCAB), since=None, until=None,
                     fingerprint="fp")
    assert payload["vocab"] == sorted(VOCAB)


def test_the_bundle_carries_no_build_date():
    shows = [_show("2025-07-04", VOCAB)]
    payload = bundle(shows, normalizer=Normalizer(vocabulary=VOCAB), since=None, until=None,
                     fingerprint="fp")
    assert "today" not in json.dumps(payload)


def test_the_catalog_denominator_comes_from_build_vocab_not_a_bare_vocabulary_length():
    """moe-pack's own lesson: an alias TARGET can sit outside the plain vocabulary list, and the
    denominator has to count it or it undercounts the pack's real reach."""
    normalizer = _AliasedNormalizer(vocabulary=["Aurora"])       # "Rebubula" only via the alias
    payload = bundle([], normalizer=normalizer, since=None, until=None, fingerprint="fp")
    assert payload["generated"]["catalog"] == 2


def test_the_window_and_fingerprint_are_recorded_verbatim_not_recomputed():
    payload = bundle([], normalizer=Normalizer(vocabulary=VOCAB),
                     since="2020-01-01", until="2025-01-01", fingerprint="abc123")
    assert payload["generated"]["window"] == {"since": "2020-01-01", "until": "2025-01-01",
                                              "spec": None}
    assert payload["generated"]["corpus"] == "abc123"


def test_an_empty_shows_list_is_a_valid_empty_bundle():
    payload = bundle([], normalizer=Normalizer(vocabulary=VOCAB), since=None, until=None,
                     fingerprint="fp")
    assert payload == {
        "schema": SCHEMA,
        "generated": {"catalog": 5, "first": None, "last": None, "n_shows": 0,
                      "window": {"since": None, "until": None, "spec": None},
                      "below_floor": 0, "deduped": 0, "corpus": "fp"},
        "vocab": [], "unknown": [], "shows": [],
    }

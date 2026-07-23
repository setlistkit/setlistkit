# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The songbook bundle: every show's songs, sorted into one vocabulary, for a page to tier live.

Mirrors ``catalog/tapemeasure.py`` at the PATTERN level only -- a pure function, a ``SCHEMA``
constant, canonically sorted output -- and NOT at the shape level. The Tape Measure assembles a
dataclass over five tables ``derive`` wrote; there is no analogous table for a single field here.
Everything below is computed straight from ``store.corpus.shows()`` and the pack's vocabulary,
because a song's frequency in a window is a fact about the corpus, not something any derivation
pass has ever stored.

THREE THINGS THIS BUNDLE DELIBERATELY DOES NOT DO, each because a POC it replaces did do them and
each is wrong for a reason worth keeping straight:

- It does not stamp a build date. Two runs over identical data must produce an identical file, or
  the golden-file test that locks this bundle's shape could not exist. A page wanting "today"
  supplies its own wall clock; the bundle supplies ``generated.last``, the last show actually in
  it, which is the honest ceiling for a slider drawn from data that does not know about shows it
  does not have.
- It does not drop a song for being outside the pack's vocabulary. Every name the window contains
  ships in ``vocab``; the ones the pack has never heard of are flagged in ``unknown`` rather than
  removed. A page decides what to draw; this file only decides what is true. Filtering here would
  make "a real song under the wrong spelling" and "taper noise that should never have been kept"
  the same absence, and they want two different repairs.
- It does not number ``vocab`` by first appearance. Sorted canonically, the way
  ``tapemeasure.py`` already sorts its own rows, so ingesting one early show does not renumber
  every index in the file and turn the whole bundle into one diff.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .normalizer import Normalizer, normalize

# Bumped only when a consumer that ignored the change would draw something wrong.
SCHEMA = "setlistkit.songbook/1"

# A show whose RAW entry count -- every line the corpus stored for that date, songs and
# non-songs alike, before dedupe -- falls under this is almost certainly a truncated taper parse,
# not a short set: the POC (``model.py:31``) cites an actual five-song lightning call that still
# has to count. Measured against the intended 2020+ publishing window this rejects nothing at all
# (zero shows fall under it there), so it is protection against a corpus that has not happened
# yet, not a live concern -- and it stays a code constant rather than config for exactly that
# reason: it is a claim about parse integrity, not a presentation choice. The tier thresholds this
# design puts in config are the ones that are actually meant to move.
MIN_SHOW_SONGS = 5


def _raw_entry_count(show: Mapping) -> int:
    """Every entry the corpus stored for this date, songs and non-songs alike, before dedupe.

    The floor tests THIS count, not the count of real songs -- a show padded to five lines by two
    tuning breaks is not a five-song show, but it is not a truncated parse either, and the POC's
    floor was never trying to answer the first question. Count it any other way and a parity
    check against the POC will not hold.
    """
    return (sum(len(one) for one in (show.get("sets") or ()))
            + len(show.get("encore") or ()))


def _song_titles(show: Mapping) -> list[str]:
    """The titles this show actually played, in play order, non-songs left out.

    Non-songs never reach the vocabulary at all -- counted, "Tuning" would be a contender for
    Anchor on some nights, and it is not a song. They still count toward
    :data:`MIN_SHOW_SONGS` above, which is a different question asked by a different function.
    """
    titles = [str(entry.get("song") or "")
              for one in (show.get("sets") or ()) for entry in one
              if not entry.get("non_song")]
    titles += [str(entry.get("song") or "")
               for entry in (show.get("encore") or ()) if not entry.get("non_song")]
    return titles


def _in_vocab(canon: str, norm_to_canon: Mapping[str, str]) -> bool:
    """Is ``canon`` a name the pack's vocabulary already recognises?

    ``canon`` MUST already be a :meth:`Normalizer.canonicalize` result, never a raw title tested
    directly -- see :func:`bundle` for why re-canonicalizing an already-canonical name is not
    redundant. Written exactly as ``catalog/lint.py`` already writes the same test, because a
    second spelling of the same check is how a second, disagreeing answer happens.
    """
    key = normalize(canon) if canon else ""
    return bool(key) and key in norm_to_canon


def _floor_shows(shows: Iterable[Mapping]) -> tuple[list[Mapping], int]:
    """Date-sorted shows, with any below :data:`MIN_SHOW_SONGS` raw entries dropped and counted.

    Split out of :func:`bundle` so the floor pass and the canonicalize-and-dedupe pass each read
    as one job with a small, nameable set of locals -- the two were one function once, and pylint
    was right to say so: a reader has no fewer things to track just because they are not split.
    """
    kept: list[Mapping] = []
    below_floor = 0
    for show in sorted(shows, key=lambda one: one["date"]):
        if _raw_entry_count(show) < MIN_SHOW_SONGS:
            below_floor += 1
            continue
        kept.append(show)
    return kept, below_floor


def _canonical_per_show(kept: list[Mapping],
                        normalizer: Normalizer) -> tuple[list[tuple[str, list[str]]], int]:
    """Each kept show's songs, canonicalized and deduped within the night, plus a dedupe count.

    Canonicalizing here rather than trusting the stored name is not redundant -- see
    :func:`bundle`'s docstring for the aliasing case this exists to catch.
    """
    per_show: list[tuple[str, list[str]]] = []
    deduped = 0
    for show in kept:
        seen: set[str] = set()
        ordered: list[str] = []
        for raw in _song_titles(show):
            canon, _ = normalizer.canonicalize(raw)
            canon = (canon or "").strip()
            if not canon:
                continue
            if canon in seen:
                deduped += 1
                continue
            seen.add(canon)
            ordered.append(canon)
        per_show.append((show["date"], ordered))
    return per_show, deduped


def _assemble(per_show: list[tuple[str, list[str]]],
              norm_to_canon: Mapping[str, str]) -> tuple[list[str], list[int], list[dict], list]:
    """Turn canonicalized per-show titles into the bundle's ``vocab``/``unknown``/``shows`` arrays.

    Split from :func:`bundle` for the same reason as :func:`_floor_shows` and
    :func:`_canonical_per_show` -- one job apiece, each with a small, nameable set of locals.
    """
    names = {name for _, titles in per_show for name in titles}
    vocab = sorted(names)
    index = {name: i for i, name in enumerate(vocab)}
    unknown = sorted(i for name, i in index.items() if not _in_vocab(name, norm_to_canon))
    show_rows = [{"d": date, "s": sorted(index[name] for name in titles)}
                 for date, titles in per_show]
    dates = [row["d"] for row in show_rows]
    return vocab, unknown, show_rows, dates


def bundle(shows: Iterable[Mapping], *, normalizer: Normalizer, since: str | None,
           until: str | None, fingerprint: str) -> dict:
    """The whole songbook as one JSON-ready mapping.

    ``shows`` is the shape :func:`setlistkit.store.corpus.shows` returns, already narrowed to the
    window the caller wants -- ``since``/``until`` are RECORDED here, not APPLIED here, the same
    split :func:`catalog.tapemeasure.bundle` makes for the same reason: a bundle whose window
    opens before the data starts is complete, and a consumer with only what was FOUND could not
    tell that from a gap.

    Every stored title is run back through ``normalizer.canonicalize`` before anything else asks
    what it is. That looks redundant -- ``store.corpus.shows()`` already holds canonicalized
    names, because ``ingest`` canonicalized them once on the way in -- and for almost every title
    it IS a no-op: canonicalizing an already-canonical name returns the same name unchanged. It is
    not redundant for the titles that matter most: a title that was unknown at ingest time and has
    since been added to ``aliases.json`` (see the pack-repair Slice 0 this design describes)
    canonicalizes DIFFERENTLY today than it did when it was stored, and re-running it here is what
    lets that pack fix reach a bundle without a re-ingest. Skipping straight to
    ``normalize(stored_name) in norm_to_canon`` would miss exactly that case -- and would also be
    the bug this project already has a name for: testing a raw string against the vocabulary
    instead of a canonicalized one is how "Hi & Lo" went missing the first time. Do not shortcut
    this.

    ``window.spec`` ships ``None`` in every bundle this function produces. It exists to carry the
    anchor-plus-offset expression that produced ``since``/``until`` once report-window
    configuration lands (see the design document's "Report windows are configuration" section);
    this slice takes literal ``--since``/``--until`` flags, so there is no expression to record.
    """
    canon_list, norm_to_canon = normalizer.build_vocab()
    kept, below_floor = _floor_shows(shows)
    per_show, deduped = _canonical_per_show(kept, normalizer)
    vocab, unknown, show_rows, dates = _assemble(per_show, norm_to_canon)

    return {
        "schema": SCHEMA,
        "generated": {
            "catalog": len(canon_list),
            "first": min(dates) if dates else None,
            "last": max(dates) if dates else None,
            "n_shows": len(show_rows),
            "window": {"since": since, "until": until, "spec": None},
            "below_floor": below_floor,
            "deduped": deduped,
            "corpus": fingerprint,
        },
        "vocab": vocab,
        "unknown": unknown,
        "shows": show_rows,
    }

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The phish.net parser: raw cached year payloads into merge show-records.

phish.net hands back one ROW PER SONG, not a nested setlist, so this parser's job is the one
setlist.fm's does not have: reassembling a show out of a flat table. Everything else -- running
names through the same normalizer, the record shape, pooling encores -- matches ``parse_setlist_fm``
deliberately, so the two setlist sources differ only where the data does.

Four things about the row format that the obvious implementation gets wrong. Each was checked
against the live 2020-2026 corpus (4,668 Phish rows) rather than taken from documentation:

**``position`` counts the SHOW, not the set.** It is a continuous 1-based counter that never
resets: a four-set night runs ``set=1 position=1..9``, ``set=2 position=10..19``, ``set=3
position=20..26``, ``set=e position=27..28``. Several third-party references say the opposite. Use
it as a set-relative index and every set is indexed off its own start, so ``features.py`` reads
openers as mid-set songs. This module never indexes by it -- it groups by ``set`` and SORTS by
``position`` within the group, which makes the re-ranking fall out of the structure instead of
being a step someone can forget.

**``set`` is an open vocabulary.** Observed here: ``1 2 3 e e2``. Documented but unobserved:
``4 e3``. The official renderer's final ``else`` prints "Encore" for anything it does not know, so
an unrecognized label is treated as encore-ish rather than dropped or crashed on.

**``transition`` is a code, and the ``->`` it distinguishes is the signal a rotation model wants.**
Codes and their rendered marks agree perfectly across all 4,668 rows: ``1`` is ``", "`` (a clean
stop), ``2`` is ``" > "`` (a segue), ``3`` is ``" -> "`` (a JAM segue -- the "this song dissolved
into the next" relation that jam vehicles live on), and ``4/5/6`` render empty and encode structure
(last of a set, last before the encore, last of the show). ``7`` is undocumented; it appears 6
times, always on the last song of an encore, always rendering empty.

**``gap`` is editorial and disagrees with itself**, so nothing here does arithmetic on it. We
compute shows-since-last-play ourselves and that is the one the model uses.

A known and deliberate loss, stated because a silent one would be worse: the corpus carries
``segue`` as a boolean, so ``>`` and ``->`` collapse into the same flag and the jam-segue
distinction does not survive into the store. It is preserved in the raw cache. Widening the schema
to hold the code is cheap now and expensive later -- see the ingest plan's recommendation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .normalizer import Normalizer
from .parse import JAM_SEGUE, count_songs, fold_sections, keep, number, text

# Transition codes that mean "this song ran into the next one". 2 renders " > ", 3 renders " -> ".
# Everything else -- a clean stop, or one of the structural end-of-section codes -- is not a segue.
SEGUE_CODES = frozenset({2, 3})

# The code whose mark is " -> ": an improvisational segue rather than a composed one. Defined in
# `parse` beside the record shape, because `features` reads it off a stored entry and must not
# import this module to do so. Re-exported here so a reader of the phish.net code table finds it
# where the rest of that table is.
JAM_SEGUE_CODE = JAM_SEGUE

# Marks a row that happened but was not a concert -- a talk-show spot, a Tiny Desk, a short
# poolside set. Recorded on the show rather than dropped, because those are real performances of
# real songs; see :func:`_is_concert`.
NON_CONCERT = "non_concert"


def _is_encore(set_label: str) -> bool:
    """Whether a ``set`` label names an encore. Anything unrecognized is treated as one.

    ``e``, ``e2``, ``e3`` are encores. A bare number is a regular set. The vocabulary is open, and
    phish.net's own renderer falls through to "Encore" for values it does not know, so an
    unfamiliar label pools into the encore rather than being dropped or becoming a phantom set.
    """
    return not set_label.isdigit()


def _set_order(set_label: str) -> tuple[int, int, str]:
    """Sort key putting numbered sets in order, then encores, then anything unrecognized."""
    if set_label.isdigit():
        return (0, int(set_label), set_label)
    if set_label.startswith("e"):
        # "e" sorts before "e2"; the trailing digits are the encore's number, defaulting to 1.
        return (1, number(set_label[1:], 1), set_label)
    return (2, 0, set_label)


def _entry(row: Mapping, normalizer: Normalizer) -> dict | None:
    """One setlist row into a ``{song, segue, non_song}`` entry, or ``None`` if it names no song.

    ``segue`` comes from the ``transition`` CODE rather than from the rendered ``trans_mark``,
    because the code is the semantic and the mark is its presentation. The two were cross-checked
    across the whole corpus and never disagree, so this is a choice about which to depend on
    rather than a guess about which is right.

    The code itself is carried through as ``transition``, unflattened. That is what keeps the
    ``->`` distinction the boolean cannot hold, and it is omitted rather than defaulted when the
    row has none, so the store records "not told" instead of inventing a clean stop.
    """
    name = text(row.get("song"))
    if not name:
        return None
    canon, name_segue = normalizer.canonicalize(name)
    if not canon:
        return None
    code = number(row.get("transition"), 0)
    entry = {"song": canon, "segue": code in SEGUE_CODES or name_segue,
             "non_song": normalizer.is_non_song(canon)}
    if code:
        entry["transition"] = code
    return entry


def _is_concert(rows: Iterable[Mapping]) -> bool:
    """Whether a show's rows describe a concert, per phish.net's ``exclude`` flag.

    ``exclude`` does NOT mean "this was called off", whatever the name suggests -- a show nobody
    played contributes no song rows and never reaches this parser. Measured across 2020-2026 it
    marks 8 dates that really happened and were not concerts: an NBC talk-show spot (1 song), an
    NPR Tiny Desk (5), and the short poolside sets at Moon Palace.

    They are KEPT and tagged rather than dropped. Those are real plays of real songs, and a
    parser that silently discarded them would quietly lose repertoire -- while a corpus that
    counted a one-song television appearance as a show would corrupt every shows-since-last-play
    gap that steps over it. Tagging is what lets the model make that call instead of this parser.

    phish.in reaches the same verdict by a different route, which is the strongest evidence there
    is that the flag means what it looks like it means. It has no ``exclude`` field, but on the
    three of these dates it also carries, every single track is marked ``exclude_from_stats`` and
    titled ``(Check) ...`` or ``Soundcheck``. Two independent editorial judgements, two unrelated
    mechanisms, one answer.
    """
    return all(text(row.get("exclude") or "0") == "0" for row in rows)


def _record(showdate: str, rows: list[Mapping], normalizer: Normalizer) -> dict | None:
    """One show's rows into a merge show-record, or ``None`` if nothing in it names a song.

    Rows are grouped by ``set`` and sorted by ``position`` WITHIN each group, which is what turns
    phish.net's show-relative counter into set-relative order. Every encore label pools into the
    single ``encore`` list, matching how ``parse_setlist_fm`` treats encore sets.
    """
    by_set: dict[str, list[Mapping]] = {}
    for row in rows:
        by_set.setdefault(text(row.get("set")) or "1", []).append(row)

    sets, encore = fold_sections(
        (_is_encore(label),
         [e for e in (_entry(r, normalizer)
                      for r in sorted(by_set[label], key=lambda r: number(r.get("position")))) if e])
        for label in sorted(by_set, key=_set_order))
    if not sets and not encore:
        return None

    showid = text(rows[0].get("showid")) or showdate
    record = {"date": showdate, "year": showdate[:4], "sets": sets, "encore": encore,
              "n_songs": count_songs(sets, encore), "source": "phish_net",
              "identifier": f"phish_net-{showid}"}
    if not _is_concert(rows):
        record["reason"] = NON_CONCERT
    return record


def parse_phish_net(payloads: Iterable[Mapping], *, normalizer: Normalizer,
                    min_year: int | None = None, artist_slug: str = "phish") -> list[dict]:
    """Cached phish.net year payloads into merge show-record candidates.

    ``artist_slug`` is not optional in practice. The ``showyear`` endpoint returns Trey solo, Mike
    Gordon, Fishman and guest appearances alongside Phish, and across 2020-2026 those rows
    OUTNUMBER the Phish ones 5,336 to 4,668 -- so parsing without it builds a corpus more than
    twice the size of the band's, attributing a sideman's set to the band.

    Sorted by ``(date, identifier)`` so two runs over the same cache agree, matching the other
    parsers. Deduping is the merge's job, not this one's.
    """
    by_date: dict[str, list[Mapping]] = {}
    for payload in payloads:
        for row in (payload.get("data", []) if isinstance(payload, Mapping) else []):
            if not isinstance(row, Mapping) or text(row.get("artist_slug")) != artist_slug:
                continue
            date = text(row.get("showdate"))
            if not date:
                continue
            if min_year is not None and date[:4].isdigit() and int(date[:4]) < min_year:
                continue
            by_date.setdefault(date, []).append(row)

    # ``min_year`` again, though the row loop above already applied it: the parsers all end the
    # same way, and a source pre-filtering for its own reasons should not change what the tail
    # guarantees.
    shows = [rec for rec in (_record(date, rows, normalizer) for date, rows in by_date.items())
             if keep(rec, min_year)]
    return sorted(shows, key=lambda rec: (rec["date"], rec["identifier"]))

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The show-record shape, and the two operations every source's parser needs on it.

Each source gets one parser module named for it -- :mod:`parse_archive_org`,
:mod:`parse_setlist_fm`, :mod:`parse_phish_net`, :mod:`parse_phish_in` -- and they share almost
nothing, because the work is almost all source-specific. archive.org mines a setlist out of prose
a taper typed; phish.net reassembles one from a flat row table; phish.in and setlist.fm read a
structure somebody already built. What they agree on is only the SHAPE they produce::

    {"date": "YYYY-MM-DD", "year": "YYYY",
     "sets":   [[{"song": str, "segue": bool, "non_song": bool}, ...], ...],
     "encore":  [{"song": str, "segue": bool, "non_song": bool}, ...],
     "n_songs": int, "source": str, "identifier": str}

and optionally ``reason``, a per-show annotation that survives into the store.

An entry may also carry ``transition``, an int: the source's own transition code, for a source
that grades transitions more precisely than a boolean can hold. It is OMITTED, never set to a
placeholder, when a source has nothing to say -- absent means "this source does not distinguish",
which is a different claim from "the band stopped" and has to stay different all the way down to
the nullable column. Every other consumer reads ``segue`` and does not care which source it came
from, which is why the boolean stays mandatory.

**Only phish.net sets it, and that is a standing decision, not a queue of sources still to do.**
The bar is a published rule with someone applying it. phish.net has both. A tape description has
neither, so every segue mark a taper writes -- ``>``, ``->``, ``~>`` -- is treated the same and
reported as the boolean alone. See ``parse_archive_org._emit_from_token``.

So this module is that agreement and nothing else. It used to be the archive.org parser as well --
1,000 lines of junk patterns, gear patterns and taper prose heuristics, with ``count_songs`` living
in the middle of them -- which meant every other source's parser imported the archive parser to
count a setlist, and a reader looking for the shared contract found a wall of one source's
problems instead. The archive code moved to the module named after its source, where the other
three already were.

``non_song`` is the flag both functions here turn on, and it is the one piece of the shape worth
explaining twice: tuning, banter, a song played over the PA, a soundcheck. Recorded because it
happened, not counted because it is not repertoire. Every parser sets it from whatever its own
source offers -- a regex over prose, setlist.fm's ``tape``, phish.in's ``exclude_from_stats`` --
and downstream reads one flag without caring which.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# The ``transition`` value meaning "this song dissolved into the next" -- rendered ``->``, as
# against ``>`` for a song that merely ran into the next without a stop.
#
# It lives here rather than in the parser that reads it because the consumer is
# :mod:`setlistkit.catalog.features`, which builds the partner graph and must not have to import
# a source's parser to learn what a stored value means. That is the dependency the archive
# extraction just removed from this module; putting it back in a new direction would be worse for
# being subtler. The numbering is phish.net's, and is documented as part of the record shape above
# for exactly as long as it is the only source grading transitions this finely.
JAM_SEGUE = 3


def text(value: object) -> str:
    """A JSON field as stripped text, whatever type the payload gave it.

    Coerce; never branch on ``isinstance``. The types are not stable even within one source:
    phish.net returns ``showid``, ``position``, ``transition`` and ``gap`` as native ints in newer
    captures and as strings in older ones, mixed inside a single row. A parser that switched on
    the type would work on the cache it was written against and break on the next pull.
    """
    return "" if value is None else str(value).strip()


def number(value: object, default: int = 0) -> int:
    """A JSON field as an int, or ``default`` if it is not one. The counterpart to :func:`text`."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def keep(record: Mapping[str, Any] | None, min_year: int | None) -> bool:
    """Whether a parsed show-record belongs in the corpus: real, in range, and not empty.

    The tail every structured parser runs. A record can fail to be one of these three things for
    reasons that have nothing to do with each other -- no usable date, a year below the floor, a
    night that parsed to nothing but tuning -- and each parser used to spell the same three checks
    out again. The floor is applied HERE, on the parsed year, because none of these APIs offers a
    year filter worth trusting: setlist.fm has none at all, and phish.net's year endpoint returns
    a straddling payload. Whatever a source can pre-filter it still passes through this.
    """
    if record is None:
        return False
    if min_year is not None and record["year"].isdigit() and int(record["year"]) < min_year:
        return False
    return count_songs(record["sets"], record["encore"]) > 0


def fold_sections(sections: Iterable[tuple[bool, list[dict]]]) -> tuple[list[list[dict]], list[dict]]:
    """``(is_encore, entries)`` pairs in play order into a record's ``(sets, encore)`` pair.

    The last step of every structured-source parser: a numbered section becomes one inner list of
    ``sets``, and every encore section pools into the single ``encore`` list -- because the record
    shape has one encore however many the band played. An empty section contributes nothing rather
    than an empty list, which would otherwise read downstream as a set that happened and was
    silent.

    Shared rather than repeated because phish.net and phish.in reach this point by different
    routes -- one reassembling a flat row table, the other reading a nested payload -- and then do
    exactly the same thing. The grouping and ordering differ per source and stay there; only this
    fold is common.
    """
    sets: list[list[dict]] = []
    encore: list[dict] = []
    for is_encore, entries in sections:
        if not entries:
            continue
        if is_encore:
            encore.extend(entries)
        else:
            sets.append(entries)
    return sets, encore


def count_songs(sets: Iterable[Iterable[Mapping[str, Any]]],
                encore: Iterable[Mapping[str, Any]]) -> int:
    """How many actual SONGS were played, ignoring the tagged non-songs.

    Tuning and banter are recorded but they are not repertoire, and this number is what
    downstream completeness checks compare shows by.
    """
    tally = sum(1 for one_set in sets for entry in one_set if not entry.get("non_song"))
    return tally + sum(1 for entry in encore if not entry.get("non_song"))

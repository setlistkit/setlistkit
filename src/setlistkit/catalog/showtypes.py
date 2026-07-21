# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tag each date as electric, acoustic, mixed or alter-ego. Derived, additive, destructive of
nothing.

A band can play two kinds of show that are not the same band. An acoustic night is a duo and
the songs are simply shorter -- one song that runs 18.8 minutes electric runs 6.4 acoustic.
Pooling those is not adding data, it is averaging two distributions that have nothing to do
with each other, and it is how a 5-minute version of a twenty-minute song gets into a length
table.

WHY THIS IS A TAG AND NOT A DELETION
The obvious move is to drop acoustic shows from the corpus. Do not:

  * the corpus feeds the rotation model, the base rates and the backtests. Deleting rows
    silently changes all of their inputs.
  * the band really did play those songs on that night. Whether that should reset a song's
    due-clock for an electric show is a real question worth testing, and deleting the row
    answers it by accident, in the dark, forever.
  * a mixed night breaks deletion outright. An acoustic first set inside a full electric show
    cannot be deleted without throwing away a real electric set. A tag copes with a partial;
    a delete cannot.

So every consumer decides for itself. Length statistics exclude acoustic. Nothing else has to
change at all.

DETECTING ONE
On the brand name, never on the bare word "acoustic". Tapers list their gear in the
description, and "percussion, MalletKat, flute, acoustic guitar" is not an acoustic show.
Matching the bare word swept in New Year's Eve and six other full electric nights.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

ELECTRIC = "electric"
ACOUSTIC = "acoustic"
MIXED = "mixed"
ALTEREGO = "alterego"

# The show brand, dot optional. Unambiguous, unlike "acoustic", which is also a kind of guitar.
_ACOUSTIC_RE = re.compile(r"moe\.?stly", re.I)

# The band occasionally takes the stage as somebody else. An alter-ego night is an improv
# vehicle rather than a normal show, and its long single-track "songs" are most of a set
# wearing one song's coat -- which does to a length distribution exactly what you would expect.
_ALTEREGO_RE = re.compile(r"perform(?:ing|s)\s+as\b", re.I)

# An electric show with an acoustic set inside it. Rare enough to name explicitly: pooling its
# acoustic set would poison the lengths, but the rest of the night is real electric data and
# there is no set-level source to split it with. Flag it, exclude it from lengths, revisit if
# set-level provenance ever arrives.
_MIXED_RE = re.compile(r"set\s*1[^.]{0,40}\bacoustic\b", re.I)

_TAG_RE = re.compile(r"<[^>]+>")

# Strongest evidence wins when a date's tapes disagree, which they do: a night can be taped
# four times over by four people who describe it four ways.
_RANK = {ALTEREGO: 3, ACOUSTIC: 2, MIXED: 1}

_TEXT_FIELDS = ("identifier", "title", "list_title", "description", "venue")


@dataclass(frozen=True)
class ShowType:
    """One date's kind, and the tape that says so. ``evidence`` is None for an ordinary night."""

    date: str
    kind: str
    evidence: str | None
    identifier: str | None


def _text_of(item: Mapping) -> str:
    """Every field a taper might have written it in, HTML stripped."""
    blob = " ".join(str(item.get(field) or "") for field in _TEXT_FIELDS)
    return _TAG_RE.sub(" ", blob)


def _classify(blob: str) -> tuple[str, str] | None:
    """The kind this text is evidence of, with the reason, or None for no evidence at all."""
    if _ALTEREGO_RE.search(blob):
        return ALTEREGO, "the band played this one as somebody else"
    if _ACOUSTIC_RE.search(blob):
        return ACOUSTIC, "'moe.stly' in tape metadata"
    if _MIXED_RE.search(blob):
        return MIXED, "notes describe an acoustic set inside an electric show"
    return None


def show_types(items: Iterable[Mapping]) -> list[ShowType]:
    """One row per dated item date, sorted by date. A date with no evidence is electric."""
    found: dict[str, ShowType] = {}
    dates: set[str] = set()
    for item in items:
        date = str(item.get("meta_date") or "")[:10]
        if not date:
            continue
        dates.add(date)
        verdict = _classify(_text_of(item))
        if verdict is None:
            continue
        kind, why = verdict
        previous = found.get(date)
        if previous is None or _RANK[kind] > _RANK[previous.kind]:
            found[date] = ShowType(date=date, kind=kind, evidence=why,
                                   identifier=str(item.get("identifier") or "") or None)
    return [found.get(date, ShowType(date=date, kind=ELECTRIC, evidence=None, identifier=None))
            for date in sorted(dates)]

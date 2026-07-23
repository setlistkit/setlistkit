# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tag each date as electric, acoustic, mixed or alter-ego. Derived, additive, destructive of
nothing.

A band can play two kinds of show that are not the same band. An acoustic night is a duo and
the songs are simply shorter -- one song that runs 18.8 minutes electric runs 6.4 acoustic.
Pooling those is not adding data, it is averaging two distributions that have nothing to do
with each other, and it is how a 5-minute version of a twenty-minute song gets into a length
table.

TAG OR DELETE IS THE PACK'S CALL, AND IT TURNS ON WHO WAS ON STAGE
There are two different nights hiding under the word "acoustic", and they want opposite
treatment:

  * SAME LINEUP, different instruments. The band as the corpus knows it, playing quieter. Tag
    it. The songs are real evidence about what this band plays, the row belongs in the rotation
    model, and only the length statistics need to leave it out. Deleting it would silently
    change the inputs of the model, the base rates and the backtests at once.
  * DIFFERENT LINEUP under a related name -- a duo, a trio, half the band. That is not the same
    act, and fifteen songs by two of its members are not fifteen plays by the band. The pack
    refuses those outright with ``side_project_patterns``, and they never reach this module.
    moe.stly is that case: fourteen nights, 206 songs, all of them Al and Rob.

A mixed night settles why the first kind cannot simply be deleted. An acoustic set inside a
full electric show cannot be removed without throwing away a real electric set, so a tag copes
with a partial where a delete cannot -- and moe.'s six mixed nights, New Year's Eve among them,
are exactly that.

So this module never deletes anything. What it produces is a tag, every consumer decides for
itself, and length statistics are the consumer that excludes acoustic and mixed.

DETECTING ONE
On the brand name, never on the bare word "acoustic". Tapers list their gear in the
description, and "percussion, MalletKat, flute, acoustic guitar" is not an acoustic show.
Matching the bare word swept in New Year's Eve and six other full electric nights.

WHICH IS WHY THE ACOUSTIC PATTERN COMES FROM THE PACK
That brand name is one band's. "moe.stly" is a fact about moe., exactly like a drop date or an
alias, and it was sitting hardcoded in a layer whose entire premise is that it knows no band --
so this module could never have tagged an acoustic night for anybody else's pack, and would
have reported every one of them electric while running clean.

The two rules that stay in code are the ones that are actually band-agnostic: "performing as"
is how every scene writes an alter-ego billing, and the mixed-set rule is a shape, not a name.
A pack that declares no acoustic pattern simply gets no acoustic tag, which is the same refusal
to guess the band filter makes when it is given no band name.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .parse import item_text

ELECTRIC = "electric"
ACOUSTIC = "acoustic"
MIXED = "mixed"
ALTEREGO = "alterego"

# The band occasionally takes the stage as somebody else. An alter-ego night is an improv
# vehicle rather than a normal show, and its long single-track "songs" are most of a set
# wearing one song's coat -- which does to a length distribution exactly what you would expect.
_ALTEREGO_RE = re.compile(r"perform(?:ing|s)\s+as\b", re.I)

# An electric show with an acoustic set inside it. Rare enough to name explicitly: pooling its
# acoustic set would poison the lengths, but the rest of the night is real electric data and
# there is no set-level source to split it with. Flag it, exclude it from lengths, revisit if
# set-level provenance ever arrives.
_MIXED_RE = re.compile(r"set\s*1[^.]{0,40}\bacoustic\b", re.I)

# Strongest evidence wins when a date's tapes disagree, which they do: a night can be taped
# four times over by four people who describe it four ways.
_RANK = {ALTEREGO: 3, ACOUSTIC: 2, MIXED: 1}


@dataclass(frozen=True)
class ShowType:
    """One date's kind, and the tape that says so. ``evidence`` is None for an ordinary night."""

    date: str
    kind: str
    evidence: str | None
    identifier: str | None


def _classify(blob: str, acoustic: Sequence[re.Pattern]) -> tuple[str, str] | None:
    """The kind this text is evidence of, with the reason, or None for no evidence at all."""
    if _ALTEREGO_RE.search(blob):
        return ALTEREGO, "the band played this one as somebody else"
    for pattern in acoustic:
        if pattern.search(blob):
            # The pattern itself, not a fixed sentence. It is the pack's own words, and it is
            # the only thing that tells a later reader WHICH rule fired on a night they doubt.
            return ACOUSTIC, f"tape metadata matches /{pattern.pattern}/"
    if _MIXED_RE.search(blob):
        return MIXED, "notes describe an acoustic set inside an electric show"
    return None


def show_types(items: Iterable[Mapping], *, dates: Mapping[str, str] | None = None,
               acoustic: Sequence[re.Pattern] = ()) -> list[ShowType]:
    """One row per dated item date, sorted by date. A date with no evidence is electric.

    ``dates`` maps identifier -> the date the show actually happened, and callers that have one
    should pass it. An uploader can type any date they like, so a pack carries corrections; a tag
    computed off the stated date lands on the wrong night for exactly the tapes whose metadata was
    already known to be wrong. Without it this falls back to the item's own ``meta_date``, which
    is honest for a standalone call over raw items and is why the parameter is optional rather
    than required.

    ``acoustic`` is the pack's own patterns for its acoustic billing -- ``moe.stly`` for one band,
    something else entirely for the next. Empty means no acoustic tag is ever produced, which is
    honest: this layer knows no band, and a brand name is not something it can derive.
    """
    lookup = dates or {}
    found: dict[str, ShowType] = {}
    seen: set[str] = set()
    for item in items:
        date = str(lookup.get(str(item.get("identifier") or ""))
                   or item.get("meta_date") or "")[:10]
        if not date:
            continue
        seen.add(date)
        verdict = _classify(item_text(item), acoustic)
        if verdict is None:
            continue
        kind, why = verdict
        previous = found.get(date)
        if previous is None or _RANK[kind] > _RANK[previous.kind]:
            found[date] = ShowType(date=date, kind=kind, evidence=why,
                                   identifier=str(item.get("identifier") or "") or None)
    return [found.get(date, ShowType(date=date, kind=ELECTRIC, evidence=None, identifier=None))
            for date in sorted(seen)]

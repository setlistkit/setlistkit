# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Where a reader goes to hear a tape we hold, per source.

Every page that shows a recording eventually wants to link to it, and every one of them used to
build the URL itself. That was correct exactly as long as archive.org was the only supplier: the
answer was a constant, so a literal prefix and an identifier were indistinguishable from a rule.
phish.in ended that, and it ended it QUIETLY -- ``https://archive.org/details/phish_in-2042`` is a
perfectly well-formed URL that 404s, and nothing in a build would say so.

So the mapping lives once, keyed on ``recordings.source``, which is a column precisely so this
question has an answer that was recorded rather than sniffed off the shape of an identifier.

Two different links, because two different questions:

:func:`item_url` is "this recording" -- what a review row cites when it reports a tape it could
not read. It has to name the one tape.

:func:`night_url` is "recordings of this night" -- what a per-show badge links to. It deliberately
resolves to a SEARCH on archive.org rather than to any tape we hold, because the count beside it
is what we last saw and the reader is being sent to look at what is there now. Tapes land weeks
after the show.

An unknown source returns ``None`` from both, and callers are expected to render nothing rather
than to guess. A dead link is worse than an absent one: it looks like a fact.
"""

from __future__ import annotations

from urllib.parse import quote

ARCHIVE_ORG = "archive_org"
PHISH_IN = "phish_in"


def item_url(source: str, identifier: str) -> str | None:
    """The page for one specific recording, or ``None`` if the source has no such page.

    phish.in is the ``None`` case that matters. Its recordings are addressed by NIGHT and not by
    transfer -- there is one official transfer a night and no per-tape page -- so the honest answer
    for "link me to this tape" is that there is no such link, and the caller should ask for the
    night instead. Returning the night's URL from here would quietly answer a question nobody
    asked and make a per-tape citation point at a page listing something else.
    """
    if source == ARCHIVE_ORG and identifier:
        return f"https://archive.org/details/{quote(identifier, safe='')}"
    return None


def night_url(source: str, date: str, *, collection: str = "") -> str | None:
    """Where to look for recordings of ``date``, or ``None`` when the source cannot be asked.

    archive.org needs the collection, because a date alone matches every band who played that
    night. Without one there is no query worth sending, so this says so rather than linking to a
    search that returns the world.
    """
    if not date:
        return None
    if source == PHISH_IN:
        return f"https://phish.in/{quote(date, safe='')}"
    if source == ARCHIVE_ORG and collection:
        query = f"collection:{collection} AND date:[{date} TO {date}]"
        return f"https://archive.org/search?query={quote(query, safe='')}"
    return None

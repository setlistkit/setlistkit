# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The setlist.fm parser: raw cached pages into merge show-records.

Tiny beside ``parse.py``'s archive parser, because setlist.fm hands back a setlist someone already
structured -- an ordered list of names, split into sets, encores flagged -- so there is nothing to
mine out of prose. What is left is mapping it onto the same show-record shape the merge reads,
running every name through the SAME normalizer so aliases resolve, and encoding the two facts that
set the source apart: it carries no segues (rank 2, below a tape's 3), and a song may be marked
``tape`` (played over the PA, not performed) -- kept, but flagged non-song.

It lives in its own module rather than inside ``parse.py`` on purpose: ``parse.py`` is the
archive.org description/track parser, and a second source's parser has no business swelling it.
This is the first step of the per-source importer split -- see ``importers/`` for where the client
and this parser are wired together.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .normalizer import Normalizer
from .parse import count_songs


def _iso_date(event_date: object) -> str:
    """setlist.fm's ``dd-MM-yyyy`` into the corpus's ``YYYY-MM-DD``, or ``""`` if unusable.

    A date the merge cannot compare sorts and joins wrong, so a string that is not exactly three
    dash-separated numeric parts with a four-digit year is refused here rather than passed on to
    fail somewhere quieter.
    """
    parts = str(event_date or "").split("-")
    if len(parts) != 3:
        return ""
    day, month, year = (p.strip() for p in parts)
    if len(year) != 4 or not (day.isdigit() and month.isdigit() and year.isdigit()):
        return ""
    return f"{year}-{month}-{day}"


def _entry(song: Mapping, normalizer: Normalizer) -> dict | None:
    """One setlist.fm song object into a ``{song, segue, non_song}`` entry, or ``None`` if empty.

    A ``tape`` song -- played over the PA, not performed -- is KEPT but flagged ``non_song``, the
    treatment tuning and banter get: it happened, but it is not a play, so it rides in the record
    for a human to see without counting toward a song's total.
    """
    name = str(song.get("name") or "").strip()
    if not name:
        return None
    canon, segue = normalizer.canonicalize(name)
    if not canon:
        return None
    non_song = normalizer.is_non_song(canon) or bool(song.get("tape"))
    return {"song": canon, "segue": segue, "non_song": non_song}


def _record(setlist: Mapping, normalizer: Normalizer) -> dict | None:
    """One setlist.fm ``setlist`` object into a merge show-record, or ``None`` if it has no date.

    The nesting is ``setlist.sets.set[].song[]``. A ``set`` carrying an ``encore`` marker pools
    into the record's single ``encore`` list; every other set becomes one inner list of ``sets``.
    An empty set contributes nothing rather than an empty list.
    """
    date = _iso_date(setlist.get("eventDate"))
    if not date:
        return None
    sets: list[list[dict]] = []
    encore: list[dict] = []
    for one in (setlist.get("sets") or {}).get("set", []):
        if not isinstance(one, Mapping):
            continue
        entries = [e for e in (_entry(s, normalizer) for s in one.get("song", [])
                               if isinstance(s, Mapping)) if e]
        if not entries:
            continue
        if one.get("encore"):
            encore.extend(entries)
        else:
            sets.append(entries)
    return {"date": date, "year": date[:4], "sets": sets, "encore": encore,
            "n_songs": count_songs(sets, encore), "source": "setlist_fm",
            "identifier": f"setlist_fm-{setlist.get('id') or ''}"}


def parse_setlistfm(pages: Iterable[Mapping], *, normalizer: Normalizer,
                    min_year: int | None = None) -> list[dict]:
    """Cached setlist.fm page payloads into merge show-record candidates.

    One record per setlist that carries a usable date and at least one song, deduped by nothing --
    the merge does the deduping, because two setlist.fm entries for one date is the kind of
    disagreement it exists to resolve. ``min_year`` is applied here, on the parsed ``eventDate``,
    because the artist-setlists endpoint has no server-side year filter. Sorted by
    ``(date, identifier)`` so two runs over the same cache agree, matching ``parse_archive_items``.
    """
    shows: list[dict] = []
    for page in pages:
        for setlist in (page.get("setlist", []) if isinstance(page, Mapping) else []):
            if not isinstance(setlist, Mapping):
                continue
            record = _record(setlist, normalizer)
            if record is None:
                continue
            if min_year is not None and record["year"].isdigit() and int(record["year"]) < min_year:
                continue
            if count_songs(record["sets"], record["encore"]) == 0:
                continue
            shows.append(record)
    return sorted(shows, key=lambda rec: (rec["date"], rec["identifier"]))

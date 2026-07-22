# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""archive.org: list a collection's items, then fetch each item's metadata -- politely, cached.

Two endpoints, per https://archive.org/developers/bots.html (read it before changing anything):
``advancedsearch.php`` lists the items in a collection, and ``metadata/<identifier>`` returns one
item's description and file list. Both go through :class:`~setlistkit.sources.client.PoliteClient`,
so the mandatory identifying User-Agent, the cache, the rate limit, and the 429 backoff all come
for free and identically to every other source.

Metadata is treated as immutable-ish and cached without expiry: a taped show's description rarely
changes, and ``force_rescan`` exists for the times it does. A search listing is the opposite -- it
is the "what is new" query -- so it is always revalidated, never served stale from the cache.

That pairing is also what replaces the old repo's watermark file. It kept an ``addeddate`` mark,
rewound it a week against a boundary race, and held it back whenever a new item's metadata fetch
failed -- three moving parts whose whole job was "do not re-request what we already have, and do
not skip what we missed". Here the cache answers both: a listing is always re-read, so nothing is
skipped, and an identifier already in the cache is not requested again, so nothing is re-fetched.
An item whose metadata fetch fails simply is not cached, and the next run picks it up because the
next run lists everything again.

:meth:`ArchiveOrgClient.cached_items` is the other half of that seam: it reassembles show items
from the cache without touching the network, which is what ``slkit ingest`` reads. Both halves
build their cache keys with the same helpers, deliberately -- a reader that guessed at the key
layout would come back empty on a cache that is in fact full, and say "no data" about it.

The item's ``files`` array is projected TWICE, by :func:`_tracks` and by :func:`duration_tracks`,
because two consumers want incompatible things from it. The parser wants whichever format lists
the most files, in whatever order the payload gave them, because it is mining titles out of a flat
set and density is what completes a setlist. The durations chain wants the most precise format and
a guaranteed play order, because it walks the tape forward mapping track N to song N. One field
cannot serve both: picking FLAC for precision would change which setlist the parser reads, and
leaving the order alone would hand the durations chain a shuffled deck. So they are two functions
over one payload, and :func:`_tracks` is not touched.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta

from ..config import Config, require_network_identity
from ..store.raw_cache import RawCache
from .client import PoliteClient, SourceError

NAMESPACE = "archive_org"

_SEARCH_URL = "https://archive.org/advancedsearch.php"
_METADATA_URL = "https://archive.org/metadata/"
# The fields the parser needs off the listing: the identifier to fetch, plus the dates and titles
# used to place and name a show before its metadata is even pulled.
_SEARCH_FIELDS = ("identifier", "date", "title", "year", "addeddate")
_PAGE_ROWS = 500
_MAX_PAGES = 40               # a 20k-item backstop against a runaway loop, not an expected limit

# The formats archive.org labels an audio derivative with. A show is usually present in several
# of them, and the densest one is the best chance at a complete tracklist: a lossless set is
# sometimes split by disc while the MP3 derivative is not, or the other way round.
_AUDIO_FORMATS = frozenset({"VBR MP3", "MP3", "Flac", "FLAC", "24bit Flac", "Ogg Vorbis",
                            "Shorten", "Apple Lossless Audio"})

# The same formats, ranked by how precisely they state a duration. Lossless derivatives carry
# float seconds ("575.47"); the lossy ones carry "MM:SS", which rounds every track to the nearest
# second. Cross-tape agreement is measured at about five seconds, so rounding at one second is
# error at the scale of the signal -- which is why this list exists and why it is ordered rather
# than counted. A test asserts it covers _AUDIO_FORMATS exactly: a format added to one and not
# the other would make every tape carrying only that format silently unmeasurable.
_DURATION_FORMATS = ("24bit Flac", "Flac", "FLAC", "Apple Lossless Audio", "Shorten",
                     "VBR MP3", "MP3", "Ogg Vorbis")

# Extensions stripped when asking whether one filename nests under another. See _drop_containers.
_TRACK_EXT_RE = re.compile(r"\.(flac|mp3|ogg|shn|m4a|wav|aif{1,2})$", re.I)

_DIGITS_RE = re.compile(r"(\d+)")

# How far a container's stated duration may sit from the sum of its parts and still be called the
# same audio. Tapers' split points are not sample-exact and a lossy derivative rounds, so an exact
# test would never fire; the proportional term covers a 90-minute set where 2 seconds is noise.
_CONTAINER_SLACK_SECONDS = 2.0
_CONTAINER_SLACK_FRACTION = 0.01

# Consecutive item failures before a bulk pull gives up. One failure is a bad tape and is skipped;
# this many in a row is the host having a bad afternoon, and continuing would mean thousands of
# requests at something already answering badly. Each item has ALREADY exhausted the client's own
# retries and backoff by the time it counts here, so this is five failures on top of twenty-five
# attempts -- not a hair trigger.
_MAX_CONSECUTIVE_FAILURES = 5


@dataclass(frozen=True)
class PullResult:
    """What one ``slkit pull`` run did, in the terms the CLI reports it.

    ``fetched`` and ``cached`` are counted separately because the difference is the whole point
    of the cache: a second pull that reports 0 fetched and 900 cached made 0 metadata requests,
    and that is the number archive.org cares about. Under ``force_rescan`` everything is re-asked,
    so ``cached`` is 0 and ``fetched`` counts items that answered 304 as well as ones that sent
    bytes -- "re-asked", not "re-downloaded".

    Every listed item lands in exactly one of ``fetched``, ``cached``, ``missing`` or
    ``unidentified``, and ``listed`` is their sum. That is asserted by a test, because a counter
    an item can fall out of is how a lost show goes unnoticed.
    """

    listed: int = 0                        # DISTINCT identifiers the listing named
    fetched: int = 0
    cached: int = 0
    # Items this run set out to fetch: everything listed that is not already cached. In a real
    # run it equals fetched + missing, which a test asserts. In a dry run it is the whole point,
    # because it is the number that says what the real run would cost the host.
    planned: int = 0
    missing: tuple[str, ...] = ()          # identifiers archive.org 404'd
    # Identifiers whose fetch failed even after the client's own retries. Separate from `missing`
    # because the cause is different -- absent versus unwell -- though the remedy is the same:
    # nothing was cached, so the next pull tries them again.
    failed: tuple[str, ...] = ()
    # Listing docs carrying no identifier at all. Unfetchable and unnameable, so they get a count
    # rather than a list -- but a count, because an item in none of these is an item nobody misses.
    unidentified: int = 0
    # The paging backstop fired, so the listing is a prefix of the collection rather than all of
    # it. Reported because the alternative is an ingest that looks complete and is not.
    truncated: bool = False


@dataclass(frozen=True)
class CachedItems:
    """Show items read back out of the cache, plus everything needed to judge them complete.

    ``absent`` is reported rather than skipped. A listing naming 900 items against 400 cached
    metadata blobs is a half-finished pull, and an ingest that quietly parsed the 400 would
    publish a corpus missing a third of the shows with nothing anywhere saying so.

    ``pages`` and ``expected`` exist for the same reason one level up. Without them
    ``CachedItems([], ())`` means two entirely different things -- "no pull has ever run" and
    "the collection is empty" -- and an ingest reading a full cache under the wrong ``min_year``
    would publish nothing and report success. ``pages`` of 0 says no listing was found at all;
    ``expected`` is the ``numFound`` the cached listing itself recorded, so the count that was
    promised travels with the corpus instead of scrolling past in a pull that ran days ago.
    """

    items: list[dict] = field(default_factory=list)
    absent: tuple[str, ...] = ()           # listed identifiers with no cached metadata
    pages: int = 0                         # listing pages read; 0 means nothing is cached
    expected: int | None = None            # numFound, when the cached listing gave one
    # Cached listing docs carrying no identifier. Counted for the same reason PullResult counts
    # them, and here for a second one: without it they surface only as items != expected, which
    # reads as "the pull did not finish" when the pull finished perfectly well.
    unidentified: int = 0
    truncated: bool = False

    @property
    def listed(self) -> int:
        """How many identified items the cached listing named, readable or not."""
        return len(self.items) + len(self.absent)


@dataclass(frozen=True)
class _Listing:
    """One walk of the search API, deduped. Internal: :class:`PullResult` is what callers see."""

    docs: list[dict] = field(default_factory=list)
    unidentified: int = 0
    pages: int = 0
    expected: int | None = None
    truncated: bool = False


def _search_key(collection: str, min_year: int | None, page: int) -> str:
    """The cache key for one search-listing page. One definition, three readers -- see the module
    docstring. A reader that guessed at this layout would find nothing and call it an empty
    collection."""
    return f"advancedsearch/{collection}/{min_year}/p{page}"


def _page_docs(payload: object) -> tuple[list, int | None]:
    """One search page's docs and its ``numFound``, from a payload of any shape."""
    if not isinstance(payload, Mapping):
        return [], None
    response = payload.get("response")
    if not isinstance(response, Mapping):
        return [], None
    docs = response.get("docs")
    num_found = response.get("numFound")
    return (docs if isinstance(docs, list) else [],
            num_found if isinstance(num_found, int) else None)


def _collect(pages: Iterable[tuple[list, int | None]]) -> _Listing:
    """Fold search pages into one deduplicated listing, stopping the way the API says to.

    Deduplication is not tidiness, it is the stop condition. ``numFound`` counts DISTINCT items,
    so comparing it against a running total of returned ROWS ends the walk early the moment a
    page boundary repeats a document -- and ``sort[]=addeddate`` is not unique, so a bulk upload
    sharing one timestamp is enough to make that happen. The tail of the collection is then never
    requested, `truncated` stays False, and every counter reports a clean run. Counting distinct
    identifiers instead makes a repeat cost one wasted row rather than the rest of the band's
    history, and pages past the real end come back empty, which ends the walk honestly.
    """
    docs: dict[str, dict] = {}             # identifier -> doc, first seen wins
    unidentified = 0
    expected: int | None = None
    read = 0
    for page_docs, num_found in pages:
        read += 1
        if expected is None:
            expected = num_found
        for doc in page_docs:
            identifier = str(doc.get("identifier") or "") if isinstance(doc, Mapping) else ""
            if not identifier:
                unidentified += 1
                continue
            docs.setdefault(identifier, dict(doc))
        # Stop on an empty page always; stop on the count only when the API gave one. Defaulting
        # a missing numFound to 0 would end the walk after page 1 with docs still coming.
        if not page_docs or (num_found is not None and len(docs) >= num_found):
            break
    else:
        # Ran out of pages rather than out of listing -- but only the BACKSTOP means truncated.
        # ``pages`` also ends when a page is simply not there, which is a clean ending, and
        # calling that truncation would fire the warning on every ordinary run instead.
        return _Listing(list(docs.values()), unidentified, read, expected,
                        truncated=read >= _MAX_PAGES)
    return _Listing(list(docs.values()), unidentified, read, expected)


def _tracks(files: object) -> list[dict]:
    """The item's tracklist, taken from whichever audio format has the most files.

    Ties break on the format name so two pulls of the same item agree; without it the tracklist
    the parser falls back to changes between runs, and so does every show it decides with it.

    One format per item, chosen once, so track boundaries within a show stay internally
    consistent. Note that the format with the MOST files is not always the most precise one:
    Flac carries float seconds ("575.47") and VBR MP3 carries "09:35", and rounding thirty
    songs a night to the nearest second is real error when a set's runtime is being estimated.
    Changing the choice here would change which setlist the parser reads, so it is not made
    here on precision grounds -- a consumer that needs better than a second calls
    :func:`duration_tracks`, which projects the same payload the other way.
    """
    by_format: dict[str, list[dict]] = {}
    for entry in files if isinstance(files, list) else []:
        if isinstance(entry, Mapping) and entry.get("format") in _AUDIO_FORMATS:
            by_format.setdefault(str(entry["format"]), []).append(dict(entry))
    if not by_format:
        return []
    best = max(by_format.items(), key=lambda pair: (len(pair[1]), pair[0]))[1]
    return [{"track": str(entry.get("track") or ""), "title": str(entry.get("title") or ""),
             "name": str(entry.get("name") or ""),
             # Kept as the source wrote it. The derived layer decides what a duration MEANS --
             # which format to trust, what to do with a 45-minute "song" -- and it cannot make
             # that call if this layer has already rounded or discarded. Raw layer stays raw.
             "length": str(entry.get("length") or "")} for entry in best]


def track_seconds(value: object) -> float | None:
    """One archive.org ``length`` as seconds, or None when it cannot be read.

    Two notations are in the corpus and they come from different derivatives of the same audio:
    Flac writes float seconds ("575.47"), VBR MP3 writes "MM:SS" ("09:35"), and a long enough
    track writes "H:MM:SS". Returning None rather than 0.0 for the unreadable ones matters --
    a zero-length song is a data point and an unknown length is not, and averaging the two is
    how a nominal length quietly drifts toward zero.
    """
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) > 3:
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    seconds = 0.0
    for number in numbers:                     # sexagesimal, most significant first
        seconds = seconds * 60 + number
    return seconds if seconds >= 0 else None


def _natural_key(name: str) -> list[tuple[int, object]]:
    """Sort a filename the way a human reads it: ``s1t02`` before ``s2t01``, ``moe2`` before
    ``moe10``. Digit runs compare as numbers, everything else as text.

    Each part is tagged with its own kind rather than left bare, so a number never has to compare
    against a string. ``re.split`` on a capture group alternates text and digits, which keeps the
    two aligned for filenames of the same shape -- and one tape uploaded under a different naming
    convention is enough to break that alignment and raise TypeError partway through an ingest of
    four thousand items. The tag makes the comparison total instead: within a shape nothing moves,
    and across shapes the numbers simply sort first.
    """
    return [(0, int(part)) if part.isdigit() else (1, part.lower())
            for part in _DIGITS_RE.split(name)]


def _ordered(entries: list[dict]) -> list[dict]:
    """The files of one item in play order.

    archive.org's ``track`` field looks authoritative and is a trap: it RESTARTS at 1 for each set
    or disc, so on a two-set tape both ``s1t01`` and ``s2t01`` report track 1. Sorting by it
    interleaves the sets like a shuffled deck -- set 1 track 1, set 2 track 1, set 1 track 2 --
    and every downstream mapping then lines the taper's tracklist up against a scrambled tape.
    That silently corrupted 65 of 436 tapes in the previous implementation, and the only reason it
    was caught at all is that two independent sources disagreed about one night.

    So ``track`` is used only where it is genuinely unique across the whole item. Otherwise the
    filename decides, because that is where the taper actually encoded the order.
    """
    numbers = [str(entry.get("track") or "").strip() for entry in entries]
    digits = [number for number in numbers if number.isdigit()]
    if len(digits) == len(entries) and len(set(digits)) == len(entries):
        return sorted(entries, key=lambda entry: int(str(entry.get("track")).strip()))
    return sorted(entries, key=lambda entry: _natural_key(str(entry.get("name") or "")))


def _stem(name: str) -> str:
    """A filename with its directory and audio extension removed, for the nesting test below."""
    return _TRACK_EXT_RE.sub("", name.rsplit("/", 1)[-1])


def _drop_containers(tracks: list[dict]) -> list[dict]:
    """Throw out whole-set files uploaded ALONGSIDE their own split tracks.

    Some tapes ship both::

        moe2023-06-27.s01.flac        62:47      <- the entire first set, in one file
        moe2023-06-27.s01.t01.flac     6:15
        moe2023-06-27.s01.t02.flac     4:37      ... t01..t07 sum to exactly 62:47

    ``.s01.flac`` natural-sorts BEFORE ``.s01.t01.flac``, so the container lands in slot 1 and
    shifts the whole night by one. A song came out at 62 minutes, the next at 102, the next at 78
    -- numbers no listener would believe for a second, and ones that no aggregate flagged.

    This runs here rather than in the derived layer because the stored ``idx`` IS the play order:
    an order recomputed later can differ from the one the join was built against, so a bag left
    sitting in slot 1 would be baked into the mirror.

    The test is arithmetic, not a guess: a file whose duration equals the sum of the files nested
    under its own name is not a track, it is a bag holding them. Two or more children required, so
    a taper who split one song across two files cannot trip it. A candidate or child whose length
    could not be read is left alone -- an unknown does not sum, and guessing it to be zero would
    make every bag look slightly too long and survive.
    """
    stems = [_stem(str(track["name"])) for track in tracks]
    bags: set[int] = set()
    for index, prefix in enumerate(stems):
        whole = tracks[index]["seconds"]
        if whole is None:
            continue
        kids = [other for other, stem in enumerate(stems)
                if other != index and stem.startswith(prefix) and len(stem) > len(prefix)
                and not stem[len(prefix)].isalnum()]
        if len(kids) < 2 or any(tracks[kid]["seconds"] is None for kid in kids):
            continue
        parts = sum(tracks[kid]["seconds"] for kid in kids)
        if abs(parts - whole) <= max(_CONTAINER_SLACK_SECONDS, _CONTAINER_SLACK_FRACTION * whole):
            bags.add(index)
    return [track for index, track in enumerate(tracks) if index not in bags]


def duration_tracks(files: object) -> tuple[str, list[dict]]:
    """The item's tracklist for MEASURING, as ``(format, tracks)``: precise, ordered, debagged.

    The other projection of ``files``; see the module docstring for why there are two. Three
    things differ from :func:`_tracks`, and each of them is a bug that has already been paid for:

    * the format is picked by precision, not by density, so durations arrive as float seconds
      rather than rounded to "MM:SS";
    * the files come back in play order, explicitly computed (:func:`_ordered`);
    * whole-set container files are removed (:func:`_drop_containers`).

    A format present but carrying no readable length at all is passed over for the next one: it
    would otherwise win the preference order and hand back a tracklist that measures nothing. An
    individual unreadable length inside the winning format is KEPT, as ``None`` seconds beside the
    string the source wrote. Dropping it would renumber the tape and hide a parser bug behind a
    tracklist that merely looks short; keeping it means the bug is diagnosable from the database
    instead of needing a re-pull.

    ``("", [])`` for an item with no audio at all -- a photo set, a stub, an artwork-only upload.
    """
    entries = [dict(entry) for entry in (files if isinstance(files, list) else [])
               if isinstance(entry, Mapping)]
    for audio_format in _DURATION_FORMATS:
        chosen = [entry for entry in entries if entry.get("format") == audio_format]
        if not any(track_seconds(entry.get("length")) is not None for entry in chosen):
            continue
        tracks = [{"name": str(entry.get("name") or ""),
                   "title": str(entry.get("title") or ""),
                   "length_raw": str(entry.get("length") or ""),
                   "seconds": track_seconds(entry.get("length"))}
                  for entry in _ordered(chosen)]
        # idx is assigned AFTER the bags come out, so it numbers the tape as it was played
        # rather than as it was uploaded. Numbering first would leave gaps that every later
        # reader would have to know to expect.
        played = _drop_containers(tracks)
        return audio_format, [dict(track, idx=index) for index, track in enumerate(played)]
    return "", []


def _flat(value: object) -> str:
    """One metadata field as a string, joining the list form.

    archive.org returns ANY repeated metadata field as a list, not just ``description``. Left to
    ``str()`` a repeated ``venue`` becomes the literal ``"['Northlands', 'Swanzey']"``, brackets
    and quotes included, and that string then goes on to be compared against song titles. It does
    not currently lose a song -- ``squash`` eats the punctuation -- but it is one filter change
    away from doing so, and there is no reason for four fields to be normalized three ways.
    """
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    return str(value or "")


def _one(value: object) -> str:
    """The FIRST value of a possibly-repeated metadata field, for fields that name a person.

    Not :func:`_flat`. Joining a repeated ``uploader`` would mint an identity nobody has --
    "a@example.org b@example.org" -- and that string then counts as a distinct taper in the
    credits, and as a distinct ballot when tapes are consolidated by who posted them. A wrong
    name is recoverable; an invented one is a person who does not exist being thanked.
    """
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _item_record(doc: Mapping, payload: Mapping) -> dict:
    """One listing doc plus one metadata blob into the flat item shape the parser reads.

    Both halves are needed: the metadata carries ``meta_date``, the description, the venue and
    the files, and the listing carries ``date``. The parser prefers ``meta_date`` and falls back
    to ``date``, so dropping the listing half would silently lose the date of every item whose
    uploader filled in only one of the two.

    ``list_title`` is the listing's own title, which nothing reads today -- the band filter uses
    the metadata ``title``. It is kept because it is raw provenance we have already paid for, and
    because an item with an empty metadata title currently sails through the band filter with no
    title to judge it by. Recorded now so that fix has something to reach for; not claimed as
    load-bearing until it is.

    ``uploader`` is read here and carried, never looked up again later. It is the field whose
    absence disabled ballot consolidation in the previous implementation for 425 of 425 tapes
    while working perfectly on the machine it was developed on: the page read it out of the raw
    metadata cache, and that cache is gitignored, so the publishing server simply had none of it
    and said so nowhere. Captured at ingest, it travels in the database with everything else.
    """
    meta = payload.get("metadata") or {}
    if not isinstance(meta, Mapping):
        meta = {}
    audio_format, measured = duration_tracks(payload.get("files"))
    return {
        "identifier": str(doc.get("identifier") or meta.get("identifier") or ""),
        "date": _flat(doc.get("date")),
        "list_title": _flat(doc.get("title")),
        "meta_date": _flat(meta.get("date")),
        "title": _flat(meta.get("title")),
        "venue": _flat(meta.get("venue")),
        "coverage": _flat(meta.get("coverage")),
        "description": _flat(meta.get("description")),
        "uploader": _one(meta.get("uploader")).strip(),
        "tracks": _tracks(payload.get("files")),
        # The second projection, beside the first. Both are computed once per item here rather
        # than by whoever needs them, so no consumer has to re-derive a tracklist from a payload
        # it would have to go back to the cache for.
        "audio_format": audio_format,
        "duration_tracks": measured,
    }


class ArchiveOrgClient:
    """Fetch moe.-collection show data from archive.org through a shared polite client."""

    def __init__(self, config: Config, cache: RawCache, *, client: PoliteClient | None = None) -> None:
        self._config = config
        self._cache = cache
        self._client = client or PoliteClient(config, cache, namespace=NAMESPACE)

    def metadata(self, identifier: str, *, force_rescan: bool = False) -> dict | None:
        """One item's metadata blob (description, files, ...), or ``None`` if archive.org 404s it."""
        url = _METADATA_URL + urllib.parse.quote(identifier)
        return self._client.fetch_json(identifier, url, force_rescan=force_rescan)

    def list_items(self, collection: str, *, min_year: int | None = None,
                   force_rescan: bool = False) -> list[dict]:
        """Every item in ``collection``, paged until archive.org's ``numFound`` is exhausted.

        Always revalidated (``max_age`` zero): the listing is the freshness query itself, so a
        cached copy would hide exactly the new uploads the caller is asking about. Still rate-
        limited and identified like every other request.
        """
        return self._list_pages(collection, min_year, force_rescan).docs

    def _list_pages(self, collection: str, min_year: int | None, force_rescan: bool) -> _Listing:
        """Walk the search API and fold the pages into one listing. See :meth:`list_items`."""
        def pages():
            for page in range(1, _MAX_PAGES + 1):
                key = _search_key(collection, min_year, page)
                payload = self._client.fetch_json(
                    key, self._search_url(collection, min_year, page),
                    force_rescan=force_rescan, max_age=timedelta(0))
                if payload is None:
                    return
                yield _page_docs(payload)

        listing = _collect(pages())
        self._drop_stale_pages(collection, min_year, listing.pages)
        return listing

    def _drop_stale_pages(self, collection: str, min_year: int | None, kept: int) -> None:
        """Delete cached listing pages past the end of the listing just read.

        A collection that shrinks -- items withdrawn, a takedown, a narrowed query -- leaves the
        pages it no longer fills sitting in the cache, and nothing overwrites them because the
        new walk never reaches page N. :meth:`cached_items` would go on reading them and hand the
        parser shows the collection no longer lists, which the pull reporting has no way to
        notice: it counts what it fetched, not what it left behind.
        """
        page = kept + 1
        while self._cache.delete(NAMESPACE, _search_key(collection, min_year, page)):
            page += 1

    def pull(self, collection: str, *, min_year: int | None = None, force_rescan: bool = False,
             dry_run: bool = False, progress: Callable[[int, int], None] | None = None,
             announce: Callable[[str], None] | None = None) -> PullResult:
        """List ``collection`` and fetch the metadata for every item not already cached.

        The expensive half is one metadata request per item, so an identifier already in the
        cache is not requested again. ``force_rescan`` re-asks for all of them -- and even then
        every request is rate-limited and conditional, so "force" means "ignore the cache", not
        "hammer archive.org". Picking up an edited description costs a 304 per item, not a
        re-download.

        A 404 goes in ``missing`` rather than being cached as an empty item: an identifier the
        listing named and the metadata API does not have is worth saying out loud, and caching a
        stub would make the next run believe it already had the show.

        ``dry_run`` stops after the listing and fetches no item metadata, so ``planned`` says what
        a real run would cost without spending it. It is not a zero-request mode and does not
        pretend to be: the listing is how this learns what is new, so a handful of search requests
        still go out. That is the cheap half by three orders of magnitude, and skipping it would
        mean answering "what would you do" with a guess.
        """
        require_network_identity(self._config)     # before the first byte leaves, not after
        with self._client.batch() as batch:
            if announce is not None:
                announce(batch.id)
            # The listing is what discovers how many items there are, so its own phase carries no
            # denominator. The item phase does, and it counts only what will actually be
            # requested: an item already cached costs the host nothing and is not in the total.
            batch.begin("listing")
            listing = self._list_pages(collection, min_year, force_rescan)
            todo = [doc for doc in listing.docs
                    if force_rescan or not self._cache.has(NAMESPACE, self._identifier(doc))]
            result = PullResult(listed=len(listing.docs), planned=len(todo),
                                cached=len(listing.docs) - len(todo),
                                unidentified=listing.unidentified, truncated=listing.truncated)
            if dry_run:
                return result
            batch.begin("item", total=len(todo))
            fetched, missing, failed = self._fetch_each(todo, force_rescan, progress)
        return replace(result, fetched=fetched, missing=missing, failed=failed)

    def _fetch_each(self, todo: list[dict], force_rescan: bool,
                    progress: Callable[[int, int], None] | None):
        """Fetch every item's metadata, tolerating isolated failures. See :meth:`pull`."""
        fetched, missing, failed, consecutive = 0, [], [], 0
        for index, doc in enumerate(todo, start=1):
            identifier = self._identifier(doc)
            try:
                payload = self.metadata(identifier, force_rescan=force_rescan)
            except SourceError:
                # One bad tape must not cost a run that is hours deep. The client has already
                # retried this item with backoff; recording it and moving on leaves the other
                # four thousand fetchable, and nothing is lost -- it was never cached, so the
                # next pull picks it up exactly like a 404.
                failed.append(identifier)
                consecutive += 1
                if consecutive >= _MAX_CONSECUTIVE_FAILURES:
                    # ...but a run of failures is not a bad tape, it is a bad afternoon for the
                    # host. Past this point "keep going" stops being resilience and turns into
                    # thousands of requests at something already answering badly, which is the
                    # one behaviour the etiquette rules exist to forbid. Stop asking.
                    raise
                continue
            consecutive = 0
            if payload is None:
                missing.append(identifier)
            else:
                fetched += 1
            if progress is not None:
                progress(index, len(todo))
        return fetched, tuple(missing), tuple(failed)

    def cached_items(self, collection: str, *, min_year: int | None = None) -> CachedItems:
        """Reassemble every cached show item. Never touches the network.

        This is what ``slkit ingest`` reads, and it is a pure cache walk on purpose: parsing is
        re-run constantly during development, and re-running it must not cost archive.org a
        single request. The listing pages are read from the cache too, because they carry the
        half of each item -- its listing ``date`` -- that the metadata blob does not.

        The same fold as a live walk, deliberately: the stop rule, the dedupe and the truncation
        test are one function, so the two halves of the seam cannot come to different conclusions
        about what the collection contains.
        """
        def pages():
            for page in range(1, _MAX_PAGES + 1):
                payload = self._cached_json(_search_key(collection, min_year, page))
                if payload is None:
                    return
                yield _page_docs(payload)

        listing = _collect(pages())
        items, absent = [], []
        # Sorted by identifier so a run's output does not depend on the order archive.org
        # happened to page the collection in, and neither does anything computed downstream.
        for doc in sorted(listing.docs, key=self._identifier):
            payload = self._cached_json(self._identifier(doc))
            if payload is None:
                # Cached-but-unreadable lands here too, and the next pull will NOT re-fetch it:
                # it filters on the payload existing, not on it parsing. Reported so the caller
                # can say so and name --force-rescan, which is the only thing that clears it.
                absent.append(self._identifier(doc))
                continue
            items.append(_item_record(doc, payload))
        return CachedItems(items=items, absent=tuple(absent), pages=listing.pages,
                           expected=listing.expected, unidentified=listing.unidentified,
                           truncated=listing.truncated)

    def _cached_json(self, key: str):
        """The cached payload for ``key``, JSON-decoded, or ``None`` when it is not cached.

        A payload that is cached but unreadable comes back ``None`` as well: half a JSON document
        is a truncated write, not a fact about the show. It is NOT deleted, and the next ordinary
        pull will not replace it -- ``pull`` skips an identifier whose payload file exists, and
        this one does. That is deliberate on both counts: the raw cache exists so a human can open
        the exact bytes a parser choked on, and a read path that quietly deletes them takes that
        away. Clearing it is ``--force-rescan``, and :meth:`cached_items` reports the identifier
        so the caller can say so.
        """
        raw = self._cache.get(NAMESPACE, key)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            return None

    @staticmethod
    def _identifier(doc: Mapping) -> str:
        return str(doc.get("identifier") or "")

    @staticmethod
    def _search_url(collection: str, min_year: int | None, page: int) -> str:
        """Build one advancedsearch page URL. ``fl[]``/``sort[]`` repeat by key, so build by hand.

        Two sort keys, not one. ``addeddate asc`` is the right primary -- new uploads append to
        the tail, so no page a previous run already read can shift underneath it. But it is not
        unique: a bulk upload gives a dozen items the same timestamp, and within that run the
        search backend is free to order them differently on each page request. The boundary then
        moves between page N and page N+1, which repeats some items and can skip others.
        ``identifier asc`` is unique, so the pair is a total order and every boundary is stable.

        The range filter matches only items that HAVE an indexed ``year``. See the caveat on
        ``min_year`` in slkit.example.toml: it is off by default for that reason.
        """
        query = f"collection:{collection}"
        if min_year is not None:
            query += f" AND year:[{min_year} TO 9999]"
        parts = [
            "q=" + urllib.parse.quote(query),
            *(f"fl[]={field}" for field in _SEARCH_FIELDS),
            "sort[]=addeddate+asc",
            "sort[]=identifier+asc",
            f"rows={_PAGE_ROWS}",
            f"page={page}",
            "output=json",
        ]
        return _SEARCH_URL + "?" + "&".join(parts)

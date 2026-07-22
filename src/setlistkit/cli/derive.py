# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""``slkit derive``: turn what was ingested into what was concluded.

The first and so far only derivation is ``durations`` -- how long each song ran, from the tapes.
Where ``ingest`` projects raw payloads into tables without judging them, this DECIDES: which tape's
track is which song, which of four tapers to believe, and which performances are too disputed to
publish. Every one of those decisions is the catalog's, in
:mod:`setlistkit.catalog.durations` (one tape) and :mod:`setlistkit.catalog.lengths` (many). What
lives here is the IO and the arithmetic of saying what happened.

IT READS THE STORE AND NEVER THE CACHE. That is the rule the whole shape of this follows. The raw
cache is gitignored, so anything reading it works on the machine that pulled and comes back empty
on the server -- which is exactly how the previous implementation lost the ``uploader`` field for
425 tapes and, with it, the ability to tell four tapers apart from one taper who posted four times.
Descriptions are read once, at ingest, into ``recording_listings``; this reads that table.

WHAT IT REPORTS is every tape that did NOT produce a measurement, in three separate piles, because
they want three different things from a person. A tape whose longest track runs past an hour is a
bounced set and no amount of looking will fix it. A tape whose tracks would not line up with its
night is a review: something is wrong and it might be ours. A night with no setlist is a gap in the
corpus, not in the tapes. Reporting them as one number called "skipped" is how a slow drift in any
one of them goes unnoticed for a year.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..catalog import durations as tapes
from ..catalog import lengths
from ..catalog.normalizer import Normalizer
from ..catalog.pack import load_pack
from ..store import Store
from .common import resolve_pack_dir

EXIT_OK = 0

# Where a reviewer goes to listen. Stored on the row rather than built at read time so a dumped
# review queue is self-contained -- the point of the queue is to be worked through somewhere other
# than here.
_ARCHIVE_ITEM = "https://archive.org/details/"

# Why a tape produced no measurement. Counted under these names so the report and the stored
# review rows use one vocabulary rather than two that drift.
NO_SETLIST = "no setlist for that night"
UNNAMED = "could not name enough of its tracks"
BOUNCED = "one track holds a whole set"

# How many rows of a list the report prints before summarising the rest. Never a silent
# truncation: what was cut is always counted out loud.
_MAX_LISTED = 8


@dataclass
class _Timed:
    """Everything the per-tape reading pass produced, before any tape is compared to another.

    Mutable and accumulated, unlike most of the types in this codebase, because it is genuinely a
    tally being built -- and one type rather than five parallel lists so that a pass which forgets
    to record a refusal cannot typecheck.
    """

    observations: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    review: list = field(default_factory=list)
    abandoned: list = field(default_factory=list)
    rejected_overrides: list = field(default_factory=list)
    outcome: Counter = field(default_factory=Counter)


def derive(config, args) -> int:
    """`slkit derive [durations]`, defaulting to durations.

    Defaulting rather than requiring the noun, the way ``config`` defaults to show and ``store``
    to init. There is one derivation today; when there are three, the default stops being obvious
    and this grows the same action table those two have.
    """
    return _durations(config, args)


def _usable_reading(tape, night, normalizer):
    """One tape's reading of its night, or ``None`` if it did not explain enough to be believed.

    Two ways to qualify, and either is enough, because tapes fail in two directions. A tape may
    cover most of the SETLIST (the ordinary case) or it may not, while still explaining most of
    its own TRACKS -- which is what a single-set tape of a two-set night looks like, and refusing
    it would throw away half the tapes of the 1990s for being short rather than for being wrong.
    """
    reading = tapes.best_reading(tape, night, normalizer)
    covers = len(reading) >= tapes.MIN_MATCH_RATE * max(night.real_songs(normalizer), 1)
    explains = len(reading) >= tapes.MIN_TRACK_RATE * max(len(tape.tracks), 1)
    if len(reading) < tapes.MIN_ROWS or not (covers or explains):
        return None
    return reading


@dataclass(frozen=True)
class _Reader:
    """Everything a tape is read AGAINST, and the one decision that needs all of it.

    A type rather than five parameters threaded through two functions. They travel together at
    every call and always will: reading a tape means asking what the night was (``shows``,
    ``kinds``), what the taper said it was (``listings``), whether a person has overruled him
    (``overrides``), and what the songs are called (``normalizer``). Passing them separately is
    how the fifth one ends up missing at the third call site.
    """

    shows: Mapping
    kinds: Mapping
    listings: Mapping
    overrides: Mapping
    normalizer: Normalizer

    def listing_for(self, tape, out: _Timed):
        """The tracklist to read ``tape`` with, and whether a human wrote it.

        An override outranks the taper, which is the whole point of authoring one -- and it is
        only honored when it has EXACTLY as many labels as the tape has files. That count is the
        witness. An override is a positional list, so one line short or one line long does not
        fail: it slides every song after the gap onto its neighbor's track and produces a full
        set of confident, wrong timings. Refusing on a length mismatch keeps the failure loud.
        """
        override = self.overrides.get(tape.identifier)
        if override is None:
            return self.listings.get(tape.identifier), False
        if len(override) != len(tape.tracks):
            out.rejected_overrides.append({"identifier": tape.identifier, "date": tape.date,
                                           "labels": len(override), "tracks": len(tape.tracks)})
            return self.listings.get(tape.identifier), False
        return tapes.listing_from_labels(override, self.normalizer), True


def _read_tapes(recordings, reader: _Reader) -> _Timed:
    """Every tape in the store, read on its own, into a pile of timings.

    Nothing here compares one tape to another -- that is the next pass, and keeping them apart is
    the same split :mod:`setlistkit.catalog.durations` and :mod:`setlistkit.catalog.lengths` are
    built on. A tape is refused for its own reasons or it votes.
    """
    out = _Timed()
    nights: dict[str, tapes.Night] = {}
    for record in recordings:
        tape = tapes.Tape.of(record)
        show = reader.shows.get(record["date"])
        if tape.longest_seconds > tapes.GIVE_UP_TRACK_SECONDS:
            out.outcome[BOUNCED] += 1
            out.abandoned.append({"identifier": tape.identifier, "date": tape.date,
                                  "n_tracks": len(tape.tracks),
                                  "longest_seconds": tape.longest_seconds})
            continue
        if not show:
            out.outcome[NO_SETLIST] += 1
            continue
        listing, overridden = reader.listing_for(tape, out)
        if listing and len(listing) == len(tape.tracks):
            # An override is authored against the files, in file order, so it needs no re-seating.
            # The taper's own listing does: they write what they remember and number from one.
            if not overridden:
                tape = tapes.reorder_to_listing(tape, listing, reader.normalizer)
            tape = tape.with_description(listing)
        night = nights.setdefault(record["date"],
                                  tapes.Night.of(show, reader.normalizer))
        reading = _usable_reading(tape, night, reader.normalizer)
        if reading is None:
            out.outcome[UNNAMED] += 1
            out.review.append(_review_row(tape, night, listing, reader.normalizer))
            continue
        out.outcome["timed"] += 1
        votes, non_song = lengths.observations_of(
            tape, reading, reader.normalizer,
            reader.kinds.get(record["date"], lengths.ELECTRIC), reading.verdict)
        out.observations.extend(votes)
        out.edges.extend(reading.edges)
        out.edges.extend(non_song)
        out.edges.extend(lengths.unclaimed_songs(tape, reading, night,
                                                 reader.normalizer))
    return out


def _review_row(tape, night, listing, normalizer) -> dict:
    """One unreadable tape, with the three counts that failed to reconcile.

    All three, because which pair disagrees is the diagnosis. Tracks against setlist is a tape of
    part of a night, or a setlist we are missing songs from. Tracks against description is a
    listing we read wrong. Description against setlist is a taper who wrote down a different show.
    """
    return {"identifier": tape.identifier, "date": tape.date,
            "n_tracks": len(tape.tracks), "n_setlist": night.real_songs(normalizer),
            "n_desc": len(listing or ()), "reason": UNNAMED,
            "url": f"{_ARCHIVE_ITEM}{tape.identifier}"}


def _durations(config, args) -> int:
    """Read every stored tape, reconcile the nights, and publish the lengths."""
    pack = load_pack(resolve_pack_dir(getattr(args, "pack", None), args.config))
    with Store(config.data_root) as store:
        store.init()
        shows = {show["date"]: show for show in store.corpus.shows()}
        recordings = store.tapes.recordings()
        if not recordings:
            print("nothing to derive: no tapes are stored. Run `slkit ingest` first.")
            return EXIT_OK
        listings = _corrected(recordings, shows, store.tapes.listings(), pack.normalizer)
        timed = _read_tapes(recordings, _Reader(
            shows=shows, kinds=store.tapes.show_types(), listings=listings,
            overrides=pack.corpus.tape_overrides, normalizer=pack.normalizer))
        performances, disputes = lengths.reconcile(
            timed.observations, shows,
            uploaders=_uploaders(recordings), splits=lengths.track_splits(recordings),
            exclusions=pack.corpus.duration_exclusions)
        timed.edges.extend(disputes)
        stats = lengths.song_stats(performances)
        _report(timed, performances, stats, len(recordings))
        _report_rejected_overrides(timed.rejected_overrides)
        _report_unmatched_exclusions(
            lengths.unmatched_exclusions(performances, pack.corpus.duration_exclusions))
        # getattr, because `slkit derive` with no noun at all reaches here through the default
        # above and argparse never set the flags belonging to the subcommand nobody named.
        if getattr(args, "dry_run", False):
            print("dry run: nothing written")
            return EXIT_OK
        written = store.durations.replace(
            [lengths.as_row(row) for row in performances], [vars(stat) for stat in stats],
            review=timed.review, abandoned=timed.abandoned, edges=timed.edges)
        print(f"  wrote {dict(sorted(written.items()))}")
    return EXIT_OK


def _corrected(recordings, shows, listings, normalizer):
    """Written tracklists, re-seated where a taper's listing is offset from their own files.

    Applied HERE and not at ingest, though both layers could. Ingest stores what the taper wrote;
    this decides what it means, and re-seating a listing is a decision made by comparing it with a
    setlist. Keeping the stored listing verbatim is what makes a wrong correction diagnosable
    rather than permanent.
    """
    corrected, _offsets = tapes.correct_tracklist_offsets(recordings, shows, listings, normalizer)
    return corrected


def _uploaders(recordings) -> dict[str, str]:
    """identifier -> who posted it, lowercased, for tapes that say.

    THE FIELD THE WHOLE VOTE DEPENDS ON. One taper posting a soundboard, a matrix and two mic
    feeds has published one set of track splits four times, and without this every one of them
    votes -- so the loudest uploader on a night decides that night. It went missing entirely in
    the previous implementation (425 of 425 tapes), which is why it is a stored column now and why
    the count of tapes that lack it is printed rather than assumed to be zero.
    """
    return {record["identifier"]: (record.get("uploader") or "").strip().lower()
            for record in recordings if (record.get("uploader") or "").strip()}


def _report(timed: _Timed, performances, stats, n_tapes: int) -> None:
    """What was measured, what was not, and how sure we are about the difference."""
    nights = len({row.slot.date for row in performances})
    print(f"derive durations: {n_tapes} stored tape(s) -> {len(timed.observations)} timing(s)")
    print(f"  {dict(sorted(timed.outcome.items()))}")
    print(f"  {len(performances)} performance(s) over {nights} night(s); "
          f"{len(stats)} song(s) with statistics "
          f"({sum(1 for stat in stats if stat.n >= 3)} at n>=3)")
    _report_confidence(performances)
    withheld = lengths.withheld_counts(performances)
    if withheld:
        # Printed beside the statistics and never on its own. A pool that quietly shrinks is a
        # pool nobody audits, and every reason here is one a person might disagree with.
        print(f"  held back from the per-song pools: {dict(sorted(withheld.items()))}")
    if timed.edges:
        print(f"  {len(timed.edges)} edge(s) recorded: "
              f"{dict(sorted(Counter(edge.kind for edge in timed.edges).items()))}")
    _report_unmeasured(timed)


def _report_confidence(performances) -> None:
    """How much of the corpus rests on one taper, and how much survived a real disagreement.

    The single-tape share is the number that decides how much any of this can be trusted, and it
    is invisible from the performance count -- twenty thousand performances timed once each and
    twenty thousand timed four times each print the same total.
    """
    if not performances:
        return
    alone = sum(1 for row in performances if row.consensus.n_ballots < 2)
    resolved = Counter(row.consensus.resolved_by for row in performances
                       if row.consensus.resolved_by)
    suspect = sum(1 for row in performances if row.consensus.suspect)
    print(f"  {alone} performance(s) rest on a single taper "
          f"({100 * alone / len(performances):.1f}%)")
    if resolved:
        print(f"  disputes settled: {dict(sorted(resolved.items()))}; "
              f"{suspect} left suspect")


def _report_rejected_overrides(rejected) -> None:
    """Hand-written tracklists that did not fit the tape they name.

    Named and never counted. Somebody sat with a tape and wrote out sixteen songs by ear; an
    override that silently stops applying leaves the taper's own wrong labels in charge, which is
    the state the override was written to end.
    """
    if not rejected:
        return
    print(f"  {len(rejected)} tape override(s) refused:")
    for row in rejected:
        print(f"    {row['identifier']} ({row['date']}): {row['labels']} label(s) "
              f"for {row['tracks']} track(s)")


def _report_unmatched_exclusions(unmatched) -> None:
    """Ledger entries that ruled nothing out, named one per line.

    LOUD, and never a count on its own. Somebody sat and listened to each of these and decided a
    measurement was wrong; an entry that stops applying means that judgement has silently stopped
    being honored and the performance is back in the statistics. Printing "2 exclusions did not
    match" tells a reader there is a problem and nothing about where, which for a file of three
    entries is the entire content of the answer.
    """
    if not unmatched:
        return
    print(f"  {len(unmatched)} duration exclusion(s) ruled nothing out:")
    for row in unmatched:
        found = row["found"]
        at = f"found {found!r}" if found else "no performance at that slot"
        print(f"    {row['date']} set {row['set']} position {row['position']}: "
              f"wanted {row['song']!r} ({row['reason']}), {at}")


def _report_unmeasured(timed: _Timed) -> None:
    """The tapes that produced nothing, named rather than counted.

    Named because a review queue nobody can act on is a review queue nobody reads. The abandoned
    list is only counted: a bounced set is not a puzzle, it is one file holding a whole set, and
    there is nothing for a person to do with a list of them.
    """
    if timed.abandoned:
        print(f"  {len(timed.abandoned)} tape(s) abandoned as bounced sets (one track over "
              f"{tapes.GIVE_UP_TRACK_SECONDS // 60} minutes)")
    if not timed.review:
        return
    print(f"  {len(timed.review)} tape(s) queued for review:")
    for row in timed.review[:_MAX_LISTED]:
        print(f"    {row['date']} {row['identifier']}: {row['n_tracks']} track(s), "
              f"{row['n_setlist']} song(s) in the setlist, {row['n_desc']} in the description")
    if len(timed.review) > _MAX_LISTED:
        print(f"    ... and {len(timed.review) - _MAX_LISTED} more (all of them are stored)")

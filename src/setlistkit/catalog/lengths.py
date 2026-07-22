# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Reconcile the several tapes of one night into one duration per performance.

:mod:`durations` reads each tape on its own and says which track is which song. This module
takes those readings and answers the question nobody's tape can answer alone: *how long was
that song?* Several strangers recorded the same night, and where they agree we have a
measurement; where they disagree, one of them is usually wrong in a way we can name.

The split between the two modules is where the reasoning changes kind. Everything there is
about READING a document a stranger wrote. Nothing here parses text at all -- it counts votes,
measures spread, and decides who is allowed to speak for a performance.

WHY DISAGREEMENT IS INFORMATION AND NOT NOISE
Two tapers who both caught a standalone song put it a median of 12.4 seconds apart -- measured
here, over the 4,647 performances on this corpus that more than one of them timed. Two strangers,
two rigs, two sets of notes, twelve seconds apart.

So when two tapes of one night differ by MINUTES they are not two opinions to be averaged. One
of them is measuring something else, and the usual something else is a segue lumped into one
file and named after the first song:

    11-track tape:  "Mar-De-Ma"  25:19
    17-track tape:  "Mar-De-Ma"   6:10  +  "George"  19:11   =  25:21

25:19 is not the length of Mar-De-Ma. It is the length of Mar-De-Ma AND George, and no amount of
averaging makes it otherwise. Averaging is what a pipeline does when it has decided in advance
that its inputs are interchangeable; these are not.

WHAT COUNTS AS A VOTE
A tape is not a vote. A TAPER is. One person who posts three mic feeds and a matrix of them has
published one set of track splits four times, and letting it vote four times means the loudest
uploader on a night decides that night. Ballots are consolidated before anything is counted --
see :func:`ballots`, and see the design doc for how the absence of the ``uploader`` field
disabled exactly this in a previous implementation, for 425 tapes out of 425, while working
perfectly on the machine it was written on.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from .durations import FILENAME as D_FILENAME
from .durations import MIN_PERFORMANCE_SECONDS, Edge, Night, Reading, Row, Tape, basename, \
    touches_segue
from .normalizer import Normalizer

# How far two tapes of ONE performance may disagree before we stop taking them on trust.
#
# Measured on this corpus rather than inherited. Over the 9,764 performances more than one BALLOT
# timed, the pairwise disagreement between tapers runs:
#
#     standalone   median 12.4s   p75  45.4s   p90 105.3s   (n=4,647)
#     segued       median 33.2s   p75 106.4s   p90 236.7s   (n=5,117)
#
# 45 seconds lands within half a second of the standalone third quartile, and that is the useful
# place for this line. Crossing it is not a verdict -- it is where a performance stops being taken
# on trust and goes to the resolution below. A quarter of multi-taper performances get that
# scrutiny (2,579) and three quarters of those come out settled (1,935), leaving 644 genuinely
# disputed out of 22,214.
#
# A SEGUED song is a different object. There is no objective boundary inside a segue: where the
# Bring You Down jam stops being Bring You Down and starts being Brent Black is the taper's
# aesthetic call, and two tapers routinely put it half a minute apart while agreeing almost
# exactly on where the PAIR begins and ends. Holding those to the standalone tolerance would throw
# away good data for failing to answer a question that has no answer. 150 sits between the segued
# p75 and p90.
#
# The era matters and these are whole-corpus figures. Segued disagreement more than halves from
# the 2000s (median 52.1s) to the 2020s (23.8s) as gear and taper conventions improved, so a
# tolerance fitted only to recent tapes would call a great deal of the 2000s suspect. The 2020s
# figure reproduces the proof-of-concept's 23.8s exactly, on a corpus ten times the size, which is
# the strongest evidence available that this is the same chain and not merely a similar one.
TAPE_DISAGREE_SECONDS = 45
TAPE_DISAGREE_SECONDS_SEGUED = 150

# What settled a disputed performance, where anything did.
OUTLIER_DROPPED = "outlier_dropped"
FINEST_TAPE = "finest_tape"

# Why a measured performance does not vote for its song's nominal length. The show kinds
# (acoustic, mixed, alterego) are withholding reasons too, and appear here under their own names.
TAPES_DISAGREE = "tapes_disagree"
SANDWICH_SHORT_HALF = "sandwich_short_half"
ELECTRIC = "electric"


@dataclass(frozen=True, order=True)
class Slot:
    """What every tape of a night is trying to time: one night, one place in the setlist.

    A type rather than a four-tuple because it is threaded through the whole module -- it keys
    the grouping, it identifies the performance, and it is the primary key of the stored table.
    A bare tuple would be positional at every one of those, and the two middle fields are a
    string and an int that both mean "where in the night".

    Ordered on ``(date, set_label, position)`` before ``song``, so sorting a pile of these gives
    PLAY order. The encore sorts after the numbered sets because "E" sorts after the digits.
    """

    date: str
    set_label: str
    position: int
    song: str


@dataclass(frozen=True)
class Observation:
    """One tape's reading of one performance: a single timing, and who timed it.

    ``identifier`` is carried because every question asked of an observation later is really a
    question about its tape -- who posted it, and how many boundaries they drew.
    """

    slot: Slot
    identifier: str
    seconds: float
    show_type: str = ELECTRIC
    # Which source named this track -- a filename, or the taper's own written tracklist. Nothing
    # in the reconciliation reads it and no stored column holds it, which is exactly why it is
    # easy to drop: the tolerances below are justified by how closely those TWO INDEPENDENT
    # sources agree, and without this the justification cannot be re-measured from the data. A
    # constant whose evidence you cannot reproduce is a magic number with a story attached.
    named_by: str = D_FILENAME
    # Other songs sharing this file. A segued pair in one track times the RUN, not either song,
    # so these never reach a performance -- see :func:`reconcile`.
    combined_with: tuple[str, ...] = ()

    @classmethod
    def of(cls, tape: Tape, row: Row, show_type: str, named_by: str = D_FILENAME) -> Observation:
        """One row of one tape's reading, as a vote."""
        return cls(slot=Slot(tape.date, row.set_label, row.position, row.song),
                   identifier=tape.identifier, seconds=row.seconds, show_type=show_type,
                   named_by=named_by, combined_with=row.combined_with)


@dataclass(frozen=True)
class Consensus:
    """How many tapes spoke for a performance, how far apart they were, and what settled it.

    ``n_tapes`` is what we KEPT and ``n_tapes_seen`` is what the night actually offered. Reporting
    only the first turns "we discarded three tapes" into "only one tape exists", which is how a
    resolved dispute ends up on the page as an uncorroborated measurement.
    """

    n_tapes: int
    n_tapes_seen: int
    n_ballots: int
    spread_seconds: float
    spread_all_tapes: float
    suspect: bool
    resolved_by: str | None = None


@dataclass(frozen=True)
class Sandwich:
    """A song played more than once in one night, and what that does to its parts.

    ``None`` on an ordinary performance rather than four columns reading False/None/None/True on
    the ninety-eight percent of rows that are not sandwiches.
    """

    parts: int
    total_seconds: float
    is_longest_part: bool


@dataclass(frozen=True)
class Performance:
    """One song, one night, one place in the setlist, and how long it took.

    Grouped rather than flat because the groups are the shape of the reasoning. ``consensus``
    is the whole answer to "how sure are we"; ``sandwich`` is the whole answer to "was this the
    only time they played it tonight". Both are asked as units and neither means much a field
    at a time.
    """

    slot: Slot
    seconds: float
    consensus: Consensus
    segued: bool
    show_type: str = ELECTRIC
    excluded: str | None = None
    sandwich: Sandwich | None = None

    @property
    def withheld(self) -> str | None:
        """Why this does not vote for its song's nominal length, or ``None`` if it does.

        One place says why, so the published statistic and the tally of what was left out cannot
        drift apart. The previous implementation dropped suspect performances at the point of
        aggregation without counting them anywhere, so the exclusion tally was silently missing
        its largest category.
        """
        if self.show_type != ELECTRIC:
            return self.show_type
        if self.excluded:
            return self.excluded
        if self.sandwich is not None and not self.sandwich.is_longest_part:
            return SANDWICH_SHORT_HALF
        if self.consensus.suspect:
            return TAPES_DISAGREE
        return None


@dataclass(frozen=True)
class SongStat:
    """The nominal-length pool for one song: what it usually runs, and how much it varies.

    Flat, unlike :class:`Performance`, because a statistics record IS a flat record -- every
    field is one number about the same set of performances, and grouping them would be filing
    by arithmetic rather than by meaning.
    """

    song: str
    n: int
    median_seconds: float
    mean_seconds: float
    min_seconds: float
    max_seconds: float
    p10_seconds: float
    p90_seconds: float
    stdev_seconds: float
    longest_date: str


def track_splits(recordings: Iterable[Mapping]) -> dict[str, int]:
    """How many REAL boundaries each tape drew, by identifier.

    "The finest tape" has to mean the one that drew the most boundaries inside the music, and a
    raw track count is not that number. A taper who leaves two 18-second false starts at the head
    of the reel has a 17-track tape with 15 boundaries in it, and counting the stubs promotes the
    sloppiest tape of the night to arbiter of the whole night.

    2024-07-29 is the case that found this: a 17-track reel opening with two 18-second stubs
    outranked three good tapes and booked Big World at 0:18 against their 5:14, 5:32 and 6:14.
    """
    return {
        str(record["identifier"]): sum(1 for track in record.get("tracks") or ()
                                       if (track["seconds"] or 0.0) >= MIN_PERFORMANCE_SECONDS)
        for record in recordings
    }


def observations_of(tape: Tape, reading: Reading, normalizer: Normalizer,
                    show_type: str = ELECTRIC,
                    named_by: str = D_FILENAME) -> tuple[list[Observation], list[Edge]]:
    """One tape's reading as votes, with the tracks that are not music set aside.

    An MC introduction consumed a track and has to stay consumed -- dropping it earlier would
    slide every song after it up one slot -- but it is not a length observation, and a repertoire
    where "Intro" runs 40 seconds every night is a repertoire with a fictional song in it.

    A ZERO-LENGTH TRACK IS THE SAME KIND OF THING, and it is absent data rather than a short
    performance. archive.org states some tracks as 0:00 -- a placeholder, a derivative that never
    finished, a metadata stub -- and the parser reads exactly what it is told. Left as a vote it
    is not merely noise: it is the smallest possible number, so it lands as the song's minimum and
    drags the low end of every chart drawn from it. Moth published a floor of 0:00 across 350
    performances this way. Consumed like a non-song, for the same reason: the track exists and
    the alignment after it has to stay honest.
    """
    votes: list[Observation] = []
    edges: list[Edge] = []
    for row in reading.rows:
        if not row.seconds:
            edges.append(Edge("zero_length_track", tape.date, tape.identifier, row.song,
                              {"track": basename(row.track_name),
                               "note": "the source states this track as 0:00, which is missing "
                                       "data rather than a performance of zero seconds; it "
                                       "consumed its track but casts no vote"}))
            continue
        if normalizer.is_non_song(row.song):
            edges.append(Edge("non_song_excluded", tape.date, tape.identifier, row.song,
                              {"seconds": row.seconds, "track": basename(row.track_name),
                               "note": "not music (MC intro, announcements); it consumed its "
                                       "track so the alignment stays honest, but it is not a "
                                       "length observation"}))
            continue
        votes.append(Observation.of(tape, row, show_type, named_by))
    return votes, edges


def unclaimed_songs(tape: Tape, reading: Reading, night: Night,
                    normalizer: Normalizer) -> list[Edge]:
    """Songs nobody claimed on a tape we otherwise believed.

    Either the taper spelled it in a way we do not recognise yet, or it was folded into a
    neighbor's file. Worth a look either way: this is where the next alias comes from.
    """
    return [Edge("song_not_found_on_named_tape", tape.date, tape.identifier, song,
                 {"note": "tape named most songs but not this one"})
            for index, (_, _, song) in enumerate(night.setlist)
            if index not in reading.claimed and not normalizer.is_non_song(song)]


def ballots(observations: Iterable[Observation],
            uploaders: Mapping[str, str]) -> dict[str, list[Observation]]:
    """Group observations by who published them. One uploader, one ballot, however many tapes.

    A vote is meant to be an INDEPENDENT reading of where a song starts and stops, and it is not
    one when the same person posts three mic feeds and a matrix of them: that is one cue sheet
    published four times, outvoting everybody else on the night.

    2023-04-27 is the case in point -- one taper's two mic feeds and the matrix built from them
    are three archive.org items with a single set of track splits between them.

    A tape whose uploader we do not know votes as itself, which is the behaviour we would have
    had anyway. That is a weaker consolidation, not a wrong one, and the caller is expected to
    say how often it happens rather than let it degrade quietly.
    """
    by_who: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_who[uploaders.get(observation.identifier) or observation.identifier].append(observation)
    return dict(by_who)


def largest_cluster(observations: Sequence[Observation], limit: float,
                    uploaders: Mapping[str, str]) -> list[tuple[float, Observation]] | None:
    """The biggest group of BALLOTS that agree with each other, or ``None`` if there is no majority.

    Walk the ballot times in order and break the chain at every gap wider than the tolerance. The
    longest run is the night's consensus; anything outside it is a reel that failed, not a reading
    with a case.

    A TIE IS NOT A CONSENSUS. Two against two means we genuinely cannot say which pair is right,
    so the performance stays disputed rather than being decided by whichever tape happens to sort
    first -- which would be a coin flip published as a measurement.
    """
    votes = [(statistics.median([o.seconds for o in tapes]), tapes)
             for tapes in ballots(observations, uploaders).values()]
    if len(votes) < 2:
        return None                  # one ballot cannot outvote itself
    votes.sort(key=lambda vote: vote[0])
    runs, run = [], [votes[0]]
    for previous, current in zip(votes, votes[1:]):
        if current[0] - previous[0] > limit:
            runs.append(run)
            run = []
        run.append(current)
    runs.append(run)
    if len(runs) == 1:
        return None                  # every ballot already agrees; there is no outlier to drop
    runs.sort(key=len, reverse=True)
    if len(runs[0]) == len(runs[1]):
        return None
    return [(seconds, o) for seconds, tapes in runs[0] for o in tapes]


def _spread(seconds: Sequence[float]) -> float:
    return max(seconds) - min(seconds) if seconds else 0.0


def _finest_tape(observations: Sequence[Observation],
                 splits: Mapping[str, int]) -> Observation | None:
    """The one tape that drew strictly more boundaries than any other, where that settles it.

    The coarse-tape premise predicts a SHORTER but still musical value -- a boundary drawn inside
    real music. A sub-minute answer is not a finer reading of the song, it is a broken track, and
    no track count earns the right to overrule every other tape of the night with one.

    Ties return ``None``: equal granularity means the tapers themselves could not tell us.
    """
    ranked = sorted(observations, key=lambda o: -splits.get(o.identifier, 0))
    finest = splits.get(ranked[0].identifier, 0)
    # Defaulting the runner-up to the leader's own count is what makes a lone tape fail the strict
    # comparison below, so there is no length guard to get wrong and no way to index past the end.
    runner_up = max((splits.get(o.identifier, 0) for o in ranked[1:]), default=finest)
    if finest > runner_up and ranked[0].seconds >= MIN_PERFORMANCE_SECONDS:
        return ranked[0]
    return None


@dataclass(frozen=True)
class _Verdict:
    """Which timings of one performance survived the argument, and what settled it.

    ``suspect`` is derived rather than stored, so the published duration and the flag warning you
    not to trust it can never be computed from different sets of numbers.
    """

    limit: float
    kept: tuple[Observation, ...]
    seconds: tuple[float, ...]
    resolved_by: str | None = None

    @property
    def spread(self) -> float:
        """How far apart the surviving timings are."""
        return _spread(self.seconds)

    @property
    def suspect(self) -> bool:
        """Do the survivors still disagree by more than this kind of song is allowed to?"""
        return self.spread > self.limit


def _resolve(observations: Sequence[Observation], segued: bool, uploaders: Mapping[str, str],
             splits: Mapping[str, int]) -> _Verdict:
    """Narrow a disputed performance to the timings that still have a case.

    Weight of evidence gets first crack and the finest-tape tie-break only decides what the tapers
    themselves could not. Measured over 126 disputed performances the two orders are within 2% of
    each other on accuracy -- a coin flip -- so this is settled on which is the more defensible
    thing to say out loud: we go with what most independent tapers heard, and fall back to track
    splits only when there is no majority to go with.
    """
    limit = TAPE_DISAGREE_SECONDS_SEGUED if segued else TAPE_DISAGREE_SECONDS
    kept = tuple(observations)
    seconds = tuple(sorted(o.seconds for o in kept))
    verdict = _Verdict(limit, kept, seconds)
    if len(kept) < 2 or not verdict.suspect:
        return verdict

    cluster = largest_cluster(kept, limit, uploaders)
    if cluster and len(cluster) < len(kept):
        # One taper, one vote, all the way through: an uploader who posted four mic feeds should
        # not also get four times the weight in the median we finally publish. What survives here
        # is one time per ballot, not one per tape.
        verdict = _Verdict(limit, tuple(o for _, o in cluster),
                           tuple(sorted({secs for secs, _ in cluster})), OUTLIER_DROPPED)
        if not verdict.suspect:
            return verdict

    finest = _finest_tape(verdict.kept, splits)
    if finest is not None:
        return _Verdict(limit, (finest,), (finest.seconds,), FINEST_TAPE)
    return verdict


def _performance_of(slot: Slot, observations: Sequence[Observation], show: Mapping, *,
                    uploaders: Mapping[str, str], splits: Mapping[str, int],
                    exclusions: Mapping[tuple[str, str, int], Mapping[str, str]]
                    ) -> tuple[Performance, Edge | None]:
    """One slot's timings, reconciled into one duration and whatever a human should see."""
    segued = touches_segue(show, slot.set_label, slot.position)
    verdict = _resolve(observations, segued, uploaders, splits)
    edge = None
    if verdict.suspect:
        edge = Edge("tapes_disagree", slot.date, song=slot.song,
                    detail={"spread_seconds": round(verdict.spread, 1),
                            "n_tapes": len(verdict.kept),
                            "values": [round(s, 1) for s in verdict.seconds],
                            "note": f"tapes disagree by more than {verdict.limit:g}s; "
                                    "excluded from the statistics"})
    consensus = Consensus(n_tapes=len(verdict.kept), n_tapes_seen=len(observations),
                          n_ballots=len(ballots(observations, uploaders)),
                          spread_seconds=round(verdict.spread, 2),
                          spread_all_tapes=round(_spread([o.seconds for o in observations]), 2),
                          suspect=verdict.suspect, resolved_by=verdict.resolved_by)
    return Performance(slot=slot, seconds=round(statistics.median(verdict.seconds), 2),
                       consensus=consensus, segued=segued,
                       show_type=observations[0].show_type,
                       excluded=_excluded_reason(slot, exclusions)), edge


def _excluded_reason(slot: Slot, exclusions: Mapping[tuple[str, str, int], Mapping[str, str]]
                     ) -> str | None:
    """The ledger's reason for this slot, but ONLY if the ledger meant this song.

    THE ENTRY HAS TO PROVE IT HIT WHAT IT WAS WRITTEN FOR. An exclusion is identified by
    (date, set, position), and a position is a number somebody counted off a page -- so it is
    exactly the kind of key that can be one out and still land on a real row. When it does, the
    ledger silently deletes a performance nobody examined and leaves the one somebody did.

    That is not hypothetical: the first three entries this project seeded were written against a
    different implementation's numbering, and "the 2:17 Moth reprise" landed on an eighteen-minute
    Pit. Both halves failed quietly -- a good measurement stopped voting, and the bad one kept
    voting -- and the run reported a tidy tally of exclusions applied.

    So the song is carried in the ledger as a WITNESS rather than as part of the key. Part of the
    key, a renamed song would stop excluding anything and say nothing; as a witness, a mismatch is
    a refusal that :mod:`setlistkit.cli.derive` reports by name.
    """
    entry = exclusions.get((slot.date, slot.set_label, slot.position))
    if entry is None or entry.get("song") != slot.song:
        return None
    return entry.get("reason")


def unmatched_exclusions(performances: Iterable[Performance],
                         exclusions: Mapping[tuple[str, str, int], Mapping[str, str]]
                         ) -> list[dict]:
    """Ledger entries that did not rule anything out, with what sits at the slot instead.

    Reported rather than merely skipped. An exclusion exists because a person listened to a
    performance and judged it; if it stops applying -- the setlist was re-parsed, a song was
    renamed, the position was miscounted -- then that judgement has silently stopped being
    honored, and the performance it was written about is back in the statistics.
    """
    at_slot = {(p.slot.date, p.slot.set_label, p.slot.position): p.slot.song
               for p in performances}
    out = []
    for (date, set_label, position), entry in sorted(exclusions.items()):
        found = at_slot.get((date, set_label, position))
        if found == entry.get("song"):
            continue
        out.append({"date": date, "set": set_label, "position": position,
                    "song": entry.get("song"), "reason": entry.get("reason"),
                    "found": found})
    return out


def _mark_sandwiches(performances: Sequence[Performance]) -> tuple[list[Performance], list[Edge]]:
    """Find the songs played more than once in a night and rank their parts.

    moe. opens a song, wanders off into two or three others, and comes back to finish it:
    Moth > [Water > Yellow Tigers] > Moth. Neither half is a performance of the song, any more
    than the first half of a sentence is a short sentence -- and the short halves were dragging
    every jam vehicle's median down. This is the same shape as the segue-boundary problem: the
    split point is real music but an arbitrary place to measure from.

    The LONGER half is kept as the song's length for that night and the shorter is set aside. The
    SUM is recorded too, because "how much of the night did this song eat" is an honest question
    with a real answer -- it is simply a different question from "how long is this song", and the
    one a runtime budget will eventually want. Nothing is discarded: the short halves stay in the
    table, tagged, and every one is listed for a human to overrule.
    """
    by_night_song: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, performance in enumerate(performances):
        by_night_song[(performance.slot.date, performance.slot.song)].append(index)

    out = list(performances)
    edges: list[Edge] = []
    for (date, song), indexes in sorted(by_night_song.items()):
        if len(indexes) < 2:
            continue
        indexes.sort(key=lambda index: -performances[index].seconds)
        total = round(sum(performances[index].seconds for index in indexes), 2)
        for rank, index in enumerate(indexes):
            out[index] = replace(out[index],
                                 sandwich=Sandwich(len(indexes), total, rank == 0))
        edges.append(Edge("song_played_twice", date, song=song, detail={
            "parts": [{"set": out[index].slot.set_label, "position": out[index].slot.position,
                       "seconds": out[index].seconds} for index in indexes],
            "total_seconds": total,
            "note": "same song more than once in one night -- a sandwich, a reprise, or a "
                    "genuine repeat play. The longest part is kept as the song's length and "
                    "the rest are set aside; overrule in the pack's durations ledger."}))
    return out, edges


def _by_slot(observations: Iterable[Observation]) -> dict[Slot, list[Observation]]:
    """Every timing of each performance, gathered under the performance it is a timing of."""
    per_slot: dict[Slot, list[Observation]] = defaultdict(list)
    for observation in observations:
        # A segued pair's time belongs to the run, not to either song in it, and there is no rule
        # for dividing it between them that is not an invention.
        if observation.combined_with:
            continue
        per_slot[observation.slot].append(observation)
    return dict(per_slot)


def reconcile(observations: Iterable[Observation], shows: Mapping[str, Mapping], *,
              uploaders: Mapping[str, str], splits: Mapping[str, int],
              exclusions: Mapping[tuple[str, str, int], Mapping[str, str]] | None = None
              ) -> tuple[list[Performance], list[Edge]]:
    """Every tape's timings for a night, reconciled into one duration per performance.

    ``exclusions`` is the pack's ledger of performances a human has ruled out by listening --
    a tape that cut off mid-song, a two-minute reprise of a song played in full earlier. Neither
    is detectable from metadata, because both look exactly like a genuinely unusual performance,
    which is what a statistic cannot tell them from. Excluded rows are still measured and still
    returned, tagged with their reason. They simply do not vote.

    Each entry names the song it was written about, and an entry that lands on a different song
    is REFUSED rather than applied -- see :func:`_excluded_reason`. Ask
    :func:`unmatched_exclusions` for the ones that did not take.
    """
    ruled_out = exclusions or {}
    performances: list[Performance] = []
    edges: list[Edge] = []
    for slot, timings in sorted(_by_slot(observations).items()):
        performance, edge = _performance_of(slot, timings, shows.get(slot.date) or {},
                                            uploaders=uploaders, splits=splits,
                                            exclusions=ruled_out)
        performances.append(performance)
        if edge is not None:
            edges.append(edge)

    performances, sandwich_edges = _mark_sandwiches(performances)
    return performances, edges + sandwich_edges


def _percentile(seconds: Sequence[float], fraction: float) -> float:
    """Nearest-rank on a sorted list, clamped at both ends so n=1 is its own p10 and p90.

    THE RANK IS A CEILING, NOT A FLOOR. ``int(fraction * (n - 1))`` reads like nearest-rank and
    is not: for n=2 it puts p90 at index 0, so the ninetieth percentile came back as the SHORTEST
    of the two, and for n=3 it lands on the middle one, so p90 was the median exactly. On the real
    corpus that made p90 <= median for 106 of the 389 songs with more than one timing -- Vocal Jam
    published a 19.6-minute median beside a 7.4-minute p90, on a page whose whole subject is how
    long this band plays things.

    The clamp stays: it is what lets a song timed once be its own p10 and p90 rather than an
    error, and a song with one good timing is the ordinary case here, not the edge.
    """
    index = math.ceil(fraction * len(seconds)) - 1
    return seconds[min(max(index, 0), len(seconds) - 1)]


def song_stats(performances: Iterable[Performance]) -> list[SongStat]:
    """The nominal length of each song, from the performances entitled to speak for it.

    ELECTRIC ONLY. An acoustic Lazarus is a real six-minute performance and it stays in the
    performance table, tagged -- but it is not evidence about how long the electric band plays
    Lazarus, and averaging the two produces a number describing neither.

    Median rather than mean is the headline for the same reason: one 27-minute Recreational
    Chemistry is a real night, not a correction to be applied to every other one.
    """
    by_song: dict[str, list[Performance]] = defaultdict(list)
    for performance in performances:
        if performance.withheld is None:
            by_song[performance.slot.song].append(performance)

    stats = []
    for song, rows in by_song.items():
        seconds = sorted(row.seconds for row in rows)
        longest = max(rows, key=lambda row: row.seconds)
        stats.append(SongStat(
            song=song, n=len(seconds),
            median_seconds=round(statistics.median(seconds), 1),
            mean_seconds=round(statistics.fmean(seconds), 1),
            # Rounded to the same place as the median beside them, and not because a tenth of a
            # second matters. Rounding the computed three and leaving the observed four raw made
            # p90 land 0.04s BELOW the median for songs whose every timing was identical -- and a
            # consumer drawing a bar from median to p90 gets a negative width out of that. Six
            # numbers published side by side are compared to each other, so they share a precision.
            min_seconds=round(seconds[0], 1), max_seconds=round(seconds[-1], 1),
            p10_seconds=round(_percentile(seconds, 0.10), 1),
            p90_seconds=round(_percentile(seconds, 0.90), 1),
            stdev_seconds=round(statistics.stdev(seconds), 1) if len(seconds) > 1 else 0.0,
            longest_date=longest.slot.date))
    stats.sort(key=lambda stat: (-stat.median_seconds, stat.song))
    return stats


def withheld_counts(performances: Iterable[Performance]) -> dict[str, int]:
    """Why performances were left out of the per-song pool, tallied by reason.

    Reported rather than merely computed. A pool that quietly shrinks is a pool nobody audits,
    and every reason in here is one a human might disagree with.
    """
    return dict(Counter(reason for reason in (p.withheld for p in performances) if reason))


def as_row(performance: Performance) -> dict:
    """One performance as a flat mapping: the form it is stored and published in.

    This lives HERE, next to the type, rather than in the store or the exporter, and the reason is
    ``withheld``. It is a property -- a priority ordering over four other fields -- so any consumer
    flattening a performance for itself would either drop it or reimplement it, and a second
    implementation of "why was this held back" is exactly the drift the property was introduced to
    prevent. Flattening is a question about a Performance, so the Performance answers it.

    Nesting is dropped because storage and JSON are both flat, but nothing is renamed on the way
    out: the keys here are the column names, so the mapping between the two is the identity and
    there is no third place where a field can be misfiled.
    """
    slot, consensus, sandwich = performance.slot, performance.consensus, performance.sandwich
    return {
        "date": slot.date, "set_label": slot.set_label, "position": slot.position,
        "song": slot.song, "seconds": performance.seconds,
        "n_tapes": consensus.n_tapes, "n_tapes_seen": consensus.n_tapes_seen,
        "n_ballots": consensus.n_ballots, "spread_seconds": consensus.spread_seconds,
        "spread_all_tapes": consensus.spread_all_tapes, "suspect": consensus.suspect,
        "resolved_by": consensus.resolved_by, "segued": performance.segued,
        "show_type": performance.show_type, "excluded": performance.excluded,
        "withheld": performance.withheld,
        "double_play_parts": None if sandwich is None else sandwich.parts,
        "sandwich_total_seconds": None if sandwich is None else sandwich.total_seconds,
        "is_longest_part": None if sandwich is None else sandwich.is_longest_part,
    }


def from_row(row: Mapping) -> Performance:
    """A stored row back as a :class:`Performance`. The exact inverse of :func:`as_row`.

    Beside its inverse so the two are read together, because they are only correct as a pair and
    every field of a Performance is a stored column precisely so this can exist.

    ``export`` needs it to answer a ranged request: song statistics are computed over every year a
    song was played, so the stored ones are the wrong answer for a window and there is nothing in
    the database to read instead. The alternative was to re-aggregate the stored rows directly,
    which would mean a second reading of ``withheld`` -- a priority ordering over four columns --
    and two implementations of "why was this held back" is the drift that property exists to
    prevent. Rebuilding the type and handing it to :func:`song_stats` keeps one of each.

    ``withheld`` is deliberately NOT read off the row. It is recomputed by the property from the
    fields it derives from, which makes the stored column redundant here on purpose: if the two
    ever disagree, the stored one is stale, and a test asserts they do not.
    """
    sandwich = None
    if row["double_play_parts"] is not None:
        sandwich = Sandwich(parts=row["double_play_parts"],
                            total_seconds=row["sandwich_total_seconds"],
                            is_longest_part=bool(row["is_longest_part"]))
    return Performance(
        slot=Slot(date=row["date"], set_label=row["set_label"],
                  position=row["position"], song=row["song"]),
        seconds=row["seconds"],
        consensus=Consensus(
            n_tapes=row["n_tapes"], n_tapes_seen=row["n_tapes_seen"],
            n_ballots=row["n_ballots"], spread_seconds=row["spread_seconds"],
            spread_all_tapes=row["spread_all_tapes"], suspect=bool(row["suspect"]),
            resolved_by=row["resolved_by"]),
        segued=bool(row["segued"]),
        show_type=row["show_type"],
        excluded=row["excluded"],
        sandwich=sandwich,
    )

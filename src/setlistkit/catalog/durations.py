"""Read one tape against one night's setlist, and say which track is which song.

This is the recognition half of the measuring chain. It answers a single question per tape --
*which of tonight's songs does each track hold?* -- and hands the answer to :mod:`lengths`,
which reconciles the several tapes of a night into one duration per performance.

The split is where the reasoning changes kind. Everything here is about READING a document
somebody else wrote: filenames, a taper's description, the conventions of one uploader. Nothing
here is statistical, and nothing there parses text.

HOW A TRACK IS MATCHED TO A SONG
Not by parsing the filename. The setlist is already known, so a song does not have to be
DISCOVERED, only RECOGNISED. Both sides are squashed to bare ``[a-z0-9]`` and the question is
whether tonight's song appears in this filename:

    "moe. 2023-01-19 Neumann AK40/01 Stranger Than Fiction.flac" -> strangerthanfiction
    "moe2023-06-10 06 Worm Wood.flac"                            -> wormwood
    "moe2026-02-05t04.flac"                                      -> nothing; goes to review

Squashing makes every disagreement about spacing, punctuation, case and curly quotes evaporate,
and stops the taper's separator conventions being our problem. The alternative -- a regex that
knows every way a stranger might format a filename -- is an arms race against the internet, and
each miss silently misattributes a duration rather than failing loudly.

Squashing does reopen substring collisions ("bud" sits inside "buddy"), so short names must line
up with whole words. Every such call is recorded as an edge rather than trusted quietly.

NAMELESS TAPES ARE NOT GUESSED AT
A taper who numbers tracks without naming them gives nothing to recognise. Zipping tracks to
songs by position is available and is refused: tapers also put a 40-second tuning track at t01,
so ``track[i]`` becomes ``song[i-1]`` and the whole night shifts by one, quietly teaching the
model that every song is the length of its neighbour. A 5-minute jam vehicle is the tell, and it
is indistinguishable from a real observation once it is in the table. Nameless tapes go to the
review queue with a link, for a human to read.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .normalizer import Normalizer

# Songs this short cannot be matched as a substring: "bud" sits inside "buddy", "four" inside
# "fourteen". They have to line up with whole words instead.
SHORT_NAME = 4

# "Moth (w/ Daniel Donato)" is a Moth. The guest is an annotation on the performance, not a
# different song, and leaving it attached files a real Moth under its own name with n=1.
_GUEST_SUFFIX = re.compile(r"\s*\(\s*w(?:ith|/)[^)]*\)\s*$", re.I)
_QUOTES = re.compile(r"[‘’“”\"]")
_AUDIO_EXT = re.compile(r"\.(flac|mp3|ogg|shn|m4a)$", re.I)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# The longest run of consecutive words considered as one candidate name. Six covers every real
# title; beyond that the runs are combinatorial padding that can only slow the match down.
_MAX_RUN_WORDS = 6

# When do we believe a tape?
#
# The obvious test -- "did it name 60% of tonight's setlist?" -- is the wrong question, and it
# silently threw away every PARTIAL recording. A taper who only caught set two names half the
# night and fails, even though every track they have is correctly labelled and unambiguous.
#
# The right question is how much of THE TAPE was explained. Six tracks, six songs found, is a
# tape read completely. A tape of bare indexes explains nothing and is still rejected, and a tape
# whose names belong to some other night still matches nothing and is still rejected.
MIN_MATCH_RATE = 0.6          # ...of the setlist  (a full show)
MIN_TRACK_RATE = 0.6          # ...or of the tape's own tracks (a partial one)
MIN_ROWS = 3                  # but never on the strength of one or two lucky hits

# How many more songs the taper's description must explain before it outranks the filenames.
# Two, not one: an off-by-one description can beat the correct reading by a single row. See
# :func:`best_reading`, which is where this is argued in full.
DESCRIPTION_MARGIN = 2

# A track this long is not a song, it is a taper who bounced a whole set into one file. The tape
# is given up on rather than building machinery to take it apart.
#
# The evidence is in the filenames, which say so outright: "09 Puebla into Timmy Tucker" (36.9m),
# "13. Kyle's Song > Kids" (30.6m), "08 Big World Ricky Martin Cant Seem To Find Drums Bass Brent
# Black" (33.5m -- five songs). The tell corroborates: every one of these tapes ships far fewer
# files than the night had songs (6 tracks for a 13-song setlist, 11 for 16).
#
# There is no honest way to split them. The durations are real but they belong to a RUN of songs,
# and any rule dividing 36.9 minutes between two of them is inventing the boundary -- which is the
# guessing this chain exists to refuse. An earlier truncation hack made the counts line up and
# published a 78:55 song.
#
# Set at 60 minutes, deliberately conservative. Bands really do play a 28-minute jam, so the
# threshold has to clear the longest genuine performance by a wide margin; the tapes this catches
# are 45 minutes and up. Dropping it to 30 would catch seven more, all merged the same way, but 30
# minutes is close enough to a real jam that the rule would stop being obviously correct -- and a
# rule you have to argue about on every tape is not a blanket rule.
GIVE_UP_TRACK_SECONDS = 3600

# Nothing in this repertoire is a minute and a quarter long. Under this, a track is a false start,
# a tuning stub, a fragment of stage business, or a taper's test recording -- never a performance.
MIN_PERFORMANCE_SECONDS = 75

# How many songs must carry a length prior before an offset test means anything, and how much
# better the shifted reading has to be. The real cases come in around 4x better, so there is a
# wide gap between signal and noise -- see :func:`correct_tracklist_offsets`.
MIN_OFFSET_EVIDENCE = 5
OFFSET_MARGIN = 0.6

# What a reading of a tape SAID, not merely which source was used. "agreed" is counted separately
# because otherwise "filename won" swallows every tape where there was nothing to win, and the
# tally stops meaning anything.
AGREED = "agreed"
DESCRIPTION = "description"
FILENAME = "filename"


@dataclass(frozen=True)
class Tape:
    """One recording: who it is, when it is, and the tracks in play order.

    A type rather than three parameters because the three are never separable -- a track list
    with no identifier cannot be reported on, and a date with no tracks cannot be read. Passing
    them together also keeps the edge records honest: every edge can cite its tape because the
    tape is always in scope.
    """

    identifier: str
    date: str
    tracks: tuple[Mapping, ...] = ()

    @classmethod
    def of(cls, record: Mapping) -> Tape:
        """Build one from a stored recording, whose tracks may carry a description name."""
        return cls(identifier=str(record.get("identifier") or ""),
                   date=str(record.get("date") or ""),
                   tracks=tuple(record.get("tracks") or ()))

    def with_description(self, listing: Sequence) -> Tape:
        """The same tape with the taper's own track names attached, one per track.

        Only called where the listing already matches the file count, which is the check that
        licenses the zip at all.
        """
        return Tape(self.identifier, self.date,
                    tuple({**track, "desc_song": _entry_song(listing[i])}
                          for i, track in enumerate(self.tracks)))

    def reordered(self, tracks: Sequence[Mapping]) -> Tape:
        """The same tape with its tracks put back in play order."""
        return Tape(self.identifier, self.date, tuple(tracks))

    @property
    def longest_seconds(self) -> float:
        """The longest single track, which is how a bounced-set tape gives itself away."""
        return max((t["seconds"] or 0.0 for t in self.tracks), default=0.0)


@dataclass(frozen=True)
class Night:
    """Tonight's setlist, flattened, with the spellings each slot can be recognised by.

    The two are computed together and used together on every track of every tape of the night.
    Keeping them in one object is what stops the forms being rebuilt per tape -- the
    proof-of-concept recomputed them per song per track, which is the same answer tens of millions
    of times over.
    """

    setlist: tuple[tuple[str, int, str], ...] = ()
    forms: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def of(cls, show: Mapping, normalizer: Normalizer) -> Night:
        """Flatten a stored show and compute its recognisable spellings once."""
        setlist = tuple(flatten_setlist(show))
        return cls(setlist, setlist_forms(setlist, normalizer))

    def real_songs(self, normalizer: Normalizer) -> int:
        """How many slots are music.

        A taper who named all fourteen songs but not the MC's introduction has named the whole
        setlist as far as this chain is concerned.
        """
        return sum(1 for _, _, song in self.setlist if not normalizer.is_non_song(song))

    def __len__(self) -> int:
        """How many slots the night has, music or not."""
        return len(self.setlist)


@dataclass(frozen=True)
class Row:
    """One track, assigned to one slot of tonight's setlist."""

    set_label: str
    position: int
    song: str
    seconds: float
    track_name: str
    # Other songs named in the same file. A segued pair in one track: the time is real but it
    # belongs to the run, not to either song, so these rows never reach the per-song pool.
    combined_with: tuple[str, ...] = ()

    @property
    def shape(self) -> tuple:
        """What the reading claimed, for comparing two readings of one tape."""
        return (self.set_label, self.position, self.song, self.track_name)


@dataclass(frozen=True)
class Edge:
    """A fuzzy call worth showing a human. Never a failure -- a decision with its reasons."""

    kind: str
    date: str
    identifier: str = ""
    song: str = ""
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Reading:
    """What one tape yielded, and how it was arrived at."""

    rows: tuple[Row, ...] = ()
    claimed: frozenset[int] = frozenset()
    edges: tuple[Edge, ...] = ()
    verdict: str = FILENAME

    def __len__(self) -> int:
        return len(self.rows)


def basename(name: str) -> str:
    """The track's own name. A leading directory is the taper's mic rig, not the song."""
    return _AUDIO_EXT.sub("", str(name).rsplit("/", 1)[-1])


def clean_song(song: str) -> str:
    """Canonical form of a setlist entry.

    Strips a guest annotation ("Moth (w/ Daniel Donato)" is a Moth) and the stray curly quotes a
    description parse drags in ('Moth"' is also a Moth). Both otherwise file a real performance
    under its own name with n=1, which is how one song ends up appearing three times in the
    vocabulary with a sample size of one in each.
    """
    return _QUOTES.sub("", _GUEST_SUFFIX.sub("", str(song))).strip()


def _entry_song(entry) -> str:
    """The song of a tracklist entry, whether it is a TrackEntry or a plain mapping."""
    if isinstance(entry, Mapping):
        return str(entry.get("song") or "")
    return str(getattr(entry, "song", "") or "")


def flatten_setlist(show: Mapping) -> list[tuple[str, int, str]]:
    """Show -> ordered ``[(set_label, position, song)]``."""
    out: list[tuple[str, int, str]] = []
    for set_no, songs in enumerate(show.get("sets") or [], start=1):
        for position, entry in enumerate(songs, start=1):
            out.append((str(set_no), position, clean_song(entry["song"])))
    for position, entry in enumerate(show.get("encore") or [], start=1):
        out.append(("E", position, clean_song(entry["song"])))
    return out


def touches_segue(show: Mapping, set_label: str, position: int) -> bool:
    """Does this slot segue into the next song, or the previous song into it?

    Either way its boundary was drawn by ear rather than by silence, and two tapers will draw it
    differently. Which is why a segued song is allowed to wander further before being called
    suspect -- see :mod:`lengths`.
    """
    if set_label == "E":
        arr = show.get("encore") or []
    else:
        sets = show.get("sets") or []
        index = int(set_label) - 1
        arr = sets[index] if 0 <= index < len(sets) else []
    here = position - 1
    if here >= len(arr):
        return False
    if arr[here].get("segue"):
        return True
    return here > 0 and bool(arr[here - 1].get("segue"))


def song_forms(song: str, normalizer: Normalizer,
               by_canon: Mapping[str, Sequence[str]] | None = None) -> set[str]:
    """Every squashed spelling meaning this song.

    The name itself, its normalized form (which drops a leading "the", so a taper writing "Road"
    still lands on "The Road"), and any alias pointing at it.
    """
    forms = set()
    aliases = (by_canon or {}).get(song, ())
    for amp in ("and", ""):                   # both spellings of "&" -- see squash()
        forms.add(normalizer.squash(song, amp))
        forms.add(normalizer.squash(normalizer.normalize(song), amp))
        for alias in aliases:
            forms.add(normalizer.squash(alias, amp))
    return {form for form in forms if form}


def _aliases_by_canon(normalizer: Normalizer) -> dict[str, list[str]]:
    """canonical title -> every alias pointing at it.

    Built once per night rather than scanned per song per track. The proof-of-concept walked the
    whole alias table inside the innermost loop; with a 70-alias pack and 75,000 tracks that is
    tens of millions of comparisons to reach the same answer.
    """
    out: dict[str, list[str]] = defaultdict(list)
    for alias, canon in normalizer.aliases().items():
        out[canon].append(alias)
    return out


def setlist_forms(setlist: Sequence[tuple[str, int, str]],
                  normalizer: Normalizer) -> tuple[tuple[str, ...], ...]:
    """The candidate spellings of each slot, longest first, computed once for the night.

    Longest first because :func:`claims_in` takes the first form that hits and stops: a longer
    form is the more specific claim, so it must be tried before a short one that is a substring
    of it.
    """
    by_canon = _aliases_by_canon(normalizer)
    return tuple(tuple(sorted(song_forms(song, normalizer, by_canon), key=len, reverse=True))
                 for _, _, song in setlist)


def file_forms(text: str, normalizer: Normalizer) -> tuple[set[str], set[str]]:
    """(whole squashed text, every squashed run of consecutive words in it).

    Long song names are looked up in the first. Short ones are looked up in the second, where a
    run is delimited by word boundaries -- so "bud" matches the word "Bud" but not "Buddy".
    """
    wholes: set[str] = set()
    runs: set[str] = set()
    for amp in ("and", ""):                   # both spellings of "&" -- see squash()
        wholes.add(normalizer.squash(text, amp))
        words = _NON_ALNUM.sub(" ", str(text).lower().replace("&", f" {amp} ")).split()
        runs |= {"".join(words[i:j])
                 for i in range(len(words))
                 for j in range(i + 1, min(len(words), i + _MAX_RUN_WORDS) + 1)}
    return wholes, runs


def claims_in(text: str, night: Night, normalizer: Normalizer) -> list[tuple[int, str, bool]]:
    """Which of tonight's songs does this one piece of text name?

    Returns ``[(setlist_index, matched_form, was_short_rule)]``. Usually one hit; two when the
    taper put a segue pair in one string; none for a bare index.
    """
    if not str(text).strip():
        return []
    wholes, runs = file_forms(text, normalizer)
    hits: list[tuple[int, str, bool]] = []
    for index, candidates in enumerate(night.forms):
        for form in candidates:
            short = len(form) <= SHORT_NAME
            if (form in runs) if short else any(form in whole for whole in wholes):
                hits.append((index, form, short))
                break

    # One song's title can sit INSIDE another's. A band with a song called "Time" makes "time" a
    # whole word in "Deep This Time" -- so a track named "Deep This Time" claimed both, the cursor
    # jumped past Time's slot near the end of the night, and the ten songs in between became
    # unreachable. The night then looked like a tape that simply could not be read.
    #
    # The whole-word rule stops "bud" matching "buddy"; it cannot stop "Time" matching a real word
    # in a longer title. When two matched songs overlap this way, the longer title is the one the
    # taper meant. (Genuinely distinct songs in one file -- "Time > Breathe Reprise" -- do not
    # overlap, so a real segue pair is untouched.)
    if len(hits) > 1:
        matched = [form for _, form, _ in hits]
        hits = [hit for hit in hits
                if not any(hit[1] != other and hit[1] in other for other in matched)]
    return hits


def claims(track: Mapping, night: Night, normalizer: Normalizer,
           prefer: str = DESCRIPTION) -> list[tuple[int, str, bool]]:
    """Which of tonight's songs does this track name?

    The filename and the taper's description line are NOT concatenated. Reading them as one string
    let a wrong entry in either invent a claim the other flatly contradicted, and :func:`match_tape`
    cannot tell an invented claim from a real one -- it just advances its cursor past the song, and
    everything between here and there becomes unreachable.

    So one source leads and the other is a fallback, and ``prefer`` says which. Neither order is
    right on its own; see :func:`best_reading`, which picks per tape.

    The fallback is what keeps this honest: a source that names nothing defers to the other rather
    than silencing it. A taper who writes "t01" in their notes and puts the song name on the file
    is still read.
    """
    described = track.get("desc_song")
    filename = basename(track["name"])
    first, second = ((described, filename) if prefer == DESCRIPTION else (filename, described))
    if first:
        hits = claims_in(first, night, normalizer)
        if hits:
            return hits
    return claims_in(second, night, normalizer) if second else []


def tape_named_slots(tape: Tape, night: Night, normalizer: Normalizer) -> set[int]:
    """Setlist entries appearing in EVERY filename -- so they name the TAPE, not a track.

    One night had no durations at all for exactly this reason. Its setlist carried an entry naming
    the show's format (promoted to a song by a parser upstream), and every file on the tape was
    named after that format. So every single track claimed that entry: track one matched it, the
    cursor jumped to the last slot of the night, and the remaining 22 tracks had nothing left to
    match forward into. One song matched, tape rejected, night dark.

    This is the collision :func:`claims_in` handles, one level up. That rule can only compare two
    titles that both matched; it cannot notice that one of them matched literally every file and is
    therefore part of the naming convention rather than a claim about any particular track.

    Deliberately computed from FILENAMES ONLY. The tape's name is a property of how the files were
    named, and a taper's description is the thing being protected here -- folding it in would let a
    good description mask a bad convention.

    Requires five tracks. On a two- or three-track fragment "the same song names every file" is a
    plausible accident (a song, its reprise, its jam) rather than evidence of a convention, and
    dropping the song would cost real data to defend against a case that does not arise.
    """
    if len(tape.tracks) < 5:
        return set()
    per_track = [{index for index, _, _ in claims_in(basename(t["name"]), night, normalizer)}
                 for t in tape.tracks]
    return set.intersection(*per_track) if per_track else set()


def reorder_to_listing(tape: Tape, listing: Sequence, normalizer: Normalizer) -> Tape:
    """Put a misnumbered file back where it belongs.

    Tapers miscount. One tape has TWO files numbered 02 and no 12::

        01 Understand.flac      02 It.flac      02 She.flac      03 Threw It All Away.flac

    So "02 She.flac" natural-sorts into third place -- but She was played TENTH, as both the
    taper's own tracklist and the setlist agree. Zipping the listing onto that order glues the
    third song's name to the tenth song's audio, and every track after it inherits the error. The
    night came out with eight songs instead of sixteen, and it would have been far worse had it
    quietly come out with sixteen WRONG ones.

    Two independent things are known here and they repair each other: the FILENAME says which song
    a file contains, and the LISTING (corroborated by the setlist) says what order they were played
    in. So each listing entry is paired with the file that names it, and the listing's order taken.

    Only done when EVERY entry pairs with a distinct file. A partial pairing would mean guessing
    about the rest, and this chain does not guess: the tape is returned untouched instead.
    """
    if len(tape.tracks) != len(listing):
        return tape

    forms = [file_forms(basename(t["name"]), normalizer) for t in tape.tracks]
    used: set[int] = set()
    paired: list[Mapping] = []
    for entry in listing:
        want = normalizer.squash(_entry_song(entry))
        if not want:
            return tape
        short = len(want) <= SHORT_NAME       # "It", "She" -- whole words only, never substrings
        hit = _pair_with(want, short, forms, used)
        if hit is None:
            return tape                       # not every song names a file: leave it alone
        used.add(hit)
        paired.append(tape.tracks[hit])
    return tape.reordered(paired)


def _pair_with(want: str, short: bool, forms: Sequence[tuple[set[str], set[str]]],
               used: set[int]) -> int | None:
    """The first unused file whose name carries ``want``."""
    for index, (wholes, runs) in enumerate(forms):
        if index in used:
            continue
        if (want in runs) if short else any(want in whole for whole in wholes):
            return index
    return None


def _first_slot_per_song(hits: Sequence[tuple[int, str, bool]],
                         night: Night) -> list[tuple[int, str, bool]]:
    """Keep only the first eligible slot per distinct song.

    A song can be played TWICE in a night (a reprise, or a jam sandwiched between two halves of
    one song). :func:`claims_in` reports every slot whose song appears in this filename, so a track
    called "Moth" on a night with two Moths reports BOTH -- and treating that as a segue pair
    advances the cursor past the later one, skipping the entire setlist in between and collapsing
    the night. A second slot for the same song belongs to a later track.
    """
    first: dict[str, tuple[int, str, bool]] = {}
    for index, form, short in sorted(hits, key=lambda h: h[0]):
        first.setdefault(night.setlist[index][2], (index, form, short))
    return sorted(first.values(), key=lambda h: h[0])


def _assign(track: Mapping, hits: Sequence[tuple[int, str, bool]], tape: Tape,
            night: Night) -> tuple[Row, list[Edge]]:
    """One track and its resolved hits into a row, plus whatever a human should see about it."""
    primary, form, short = hits[0]
    set_label, position, song = night.setlist[primary]
    others = tuple(night.setlist[i][2] for i, _, _ in hits if i != primary)
    edges: list[Edge] = []
    if short:
        edges.append(Edge("short_name_match", tape.date, tape.identifier, song,
                          {"matched_form": form, "track": basename(track["name"]),
                           "note": f"matched on a <= {SHORT_NAME}-char name "
                                   "via the whole-word rule"}))
    if others:
        edges.append(Edge("one_file_many_songs", tape.date, tape.identifier, song,
                          {"also": list(others), "track": basename(track["name"]),
                           "seconds": track["seconds"],
                           "note": "segue pair in a single file; the time cannot be split, "
                                   "so it is excluded from the per-song statistics"}))
    return Row(set_label=set_label, position=position, song=song, seconds=track["seconds"],
               track_name=track["name"], combined_with=others), edges


def match_tape(tape: Tape, night: Night, normalizer: Normalizer,
               prefer: str = DESCRIPTION) -> Reading:
    """Assign this tape's tracks to tonight's songs by what the filenames name.

    Walks tracks forward and only ever advances through the setlist, so a filename mentioning a
    song from earlier in the night cannot drag the alignment backwards. Tracks naming nothing
    (tuning, banter, a tease) are skipped -- which is the entire reason to match on names.
    """
    rows: list[Row] = []
    claimed: set[int] = set()
    edges: list[Edge] = []
    next_song = 0
    # Entries the filename convention names on every track. Never a claim about a track, and
    # ruinous if treated as one -- see tape_named_slots().
    tape_named = tape_named_slots(tape, night, normalizer)

    for track in tape.tracks:
        hits = [hit for hit in claims(track, night, normalizer, prefer)
                if hit[0] >= next_song and hit[0] not in tape_named]
        if not hits:
            continue
        hits = _first_slot_per_song(hits, night)
        row, row_edges = _assign(track, hits, tape, night)
        rows.append(row)
        edges.extend(row_edges)
        claimed.update(index for index, _, _ in hits)
        next_song = max(index for index, _, _ in hits) + 1

    return Reading(tuple(rows), frozenset(claimed), tuple(edges), prefer)


def best_reading(tape: Tape, night: Night, normalizer: Normalizer) -> Reading:
    """Read the tape both ways and keep whichever the SETLIST says is the better reading.

    Whether the filename or the description is the more reliable source is a property of the TAPE,
    not of the pipeline, and there is no ordering right for both of these:

      One tape's track 07 is called "07 Plane Crash" on disk while the taper's notes call it
      Recreational Chemistry. The notes are right -- the real Plane Crash is track 14. Trusting the
      filename skips six songs and the tape is rejected as unreadable.

      Another has a description whose first six numbered lines are the taper's GEAR ("Neumann
      ak50(ortf)-lc3-km1", "SD Mixer 10ii on board matrix"), which the description parser read as
      tracklist rows. Every song name in it is therefore shifted six places, and it still counts 16
      rows against 16 files so the count check waves it through. Trusting the description books a
      5:47 jam vehicle -- the length of the last track on the tape -- over a 27:26 one, silently.

    Neither has to be guessed at, because what was played that night is already known. The setlist
    is the referee: read the tape both ways, and keep the reading accounting for more of the night.
    A description offset by six matches almost nothing and loses; a description correcting one bad
    filename matches nearly everything and wins.

    The description has to win by a MARGIN, not a nose, and that is the whole subtlety.

    A description offset by a block still matches nearly everything. The setlist runs in the same
    order as the files, so shifting it by one lines song N up against track N+1 and scores almost
    as well as the truth -- sometimes better, when the shift happens to skate past a song whose
    filename is unrecognised. One tape's description opens with a date header, so every song sits
    one track late, and the shifted reading beat the correct one 13 to 12. Taking the winner on raw
    count booked a song at 22:18 -- the length of an entirely different pair -- and moved five
    others with it.

    So a near-tie is not weak evidence for the description. It is the SIGNATURE of an offset one,
    and the tie-break has to run the other way. Filenames are per-track artifacts: each is attached
    to the audio it names, and a taper who mistypes one has mistyped one. A description is a single
    document whose alignment to the files is inferred, and it fails in blocks. When the two
    readings are close, the filenames are the safer of the two.

    MARGIN of 2 rather than 1 because an off-by-one description can beat the truth by one row.
    Where the description is genuinely the better source it does not win narrowly -- the real cases
    win 13 to 7 and 18 to 1.
    """
    described = match_tape(tape, night, normalizer, DESCRIPTION)
    named = match_tape(tape, night, normalizer, FILENAME)
    if len(described) - len(named) >= DESCRIPTION_MARGIN:
        return described
    same = [row.shape for row in described.rows] == [row.shape for row in named.rows]
    return Reading(named.rows, named.claimed, named.edges, AGREED if same else FILENAME)


# ---------------------------------------------------------------------------------------------
# Re-seating a description whose first row is not a song
# ---------------------------------------------------------------------------------------------

def provisional_medians(recordings: Iterable[Mapping], shows: Mapping[str, Mapping],
                        normalizer: Normalizer) -> dict[str, float]:
    """Per-song medians built ONLY from tapes whose filenames name their own songs.

    These are the priors the offset test is scored against, and where they come from is the whole
    reason it is allowed to exist. Feeding a pipeline's output back into its input makes it reason
    from data it invented and converge on its own mistakes.

    That loop is broken here by construction. A tape can only contribute a prior if its FILENAMES
    explain the night on their own, and a tape can only be CORRECTED if its filenames explain
    nothing (they are bare indexes, so the description is the only reading). The two sets cannot
    overlap: nothing is ever scored against a prior it helped produce.
    """
    by_song: dict[str, list[float]] = defaultdict(list)
    nights: dict[str, Night] = {}
    for record in recordings:
        show = shows.get(record["date"])
        if not show:
            continue
        tape = Tape.of(record)
        if tape.longest_seconds > GIVE_UP_TRACK_SECONDS:
            continue
        night = nights.setdefault(record["date"], Night.of(show, normalizer))
        reading = match_tape(tape, night, normalizer, FILENAME)
        if len(reading) < MIN_MATCH_RATE * max(night.real_songs(normalizer), 1):
            continue                          # the filenames do not explain this tape
        for row in reading.rows:
            if not normalizer.is_non_song(row.song):
                by_song[row.song].append(row.seconds)
    return {normalizer.squash(song): statistics.median(values)
            for song, values in by_song.items() if len(values) >= 3}


def alignment_error(names: Sequence[str], seconds: Sequence[float], shift: int,
                    medians: Mapping[str, float],
                    normalizer: Normalizer) -> tuple[float, int]:
    """Mean ``|log(observed / usual)|`` for a listing read ``shift`` places off. Lower is better.

    Log-ratio rather than raw seconds so a 2x error on a 4-minute song weighs the same as a 2x
    error on a 20-minute one. Raw seconds would let the jam vehicles decide every alignment.
    """
    errors = []
    for i, name in enumerate(names):
        j = i + shift
        if not 0 <= j < len(seconds) or not seconds[j] or seconds[j] <= 0:
            continue
        usual = (medians.get(normalizer.squash(name))
                 or medians.get(normalizer.squash(normalizer.normalize(name))))
        if not usual:
            continue
        errors.append(abs(math.log(seconds[j] / usual)))
    return (sum(errors) / len(errors) if errors else 99.0), len(errors)


def shift_listing(names: Sequence[str], shift: int) -> list[str]:
    """Re-seat a listing read ``shift`` places off, padding the end that falls off.

    A dropped row is replaced with an empty name rather than deleted, because the listing has to
    stay the same length as the tape -- that equal-count check is what licenses using it at all.
    The track that loses its name simply goes unmatched, which is the right trade: one track
    unidentified beats nineteen tracks confidently mislabelled.
    """
    names = list(names)
    if shift < 0:
        k = -shift
        return names[k:] + [""] * k
    return [""] * shift + names[:len(names) - shift]


def _offset_of(record: Mapping, listing: Sequence, medians: Mapping[str, float],
               normalizer: Normalizer) -> tuple[int, float, float, list[str]] | None:
    """The shift this listing should be read at, or None to leave it alone.

    Returns ``(shift, error_before, error_after, names)``. A shift only wins if enough songs
    carried a prior to score it and it beats the status quo by the whole margin.
    """
    seconds = [track["seconds"] for track in record["tracks"]]
    names = [_entry_song(entry) for entry in listing]
    base, scored = alignment_error(names, seconds, 0, medians, normalizer)
    if scored < MIN_OFFSET_EVIDENCE:
        return None
    best_shift, best_error = 0, base
    for shift in (-2, -1, 1, 2):
        error, k = alignment_error(names, seconds, shift, medians, normalizer)
        if k >= MIN_OFFSET_EVIDENCE and error < best_error:
            best_shift, best_error = shift, error
    if best_shift and best_error < base * OFFSET_MARGIN:
        return best_shift, base, best_error, names
    return None


def correct_tracklist_offsets(recordings: Sequence[Mapping], shows: Mapping[str, Mapping],
                              tracklists: Mapping[str, Sequence],
                              normalizer: Normalizer) -> tuple[dict[str, list], list[dict]]:
    """Re-seat description tracklists whose first row is not a song.

    Tapers open a description with something that looks like a numbered row and is not: a lineage
    ("Channel Mix Lineage: MixPre-10 II -> USB Transfer -> ..."), an encoding note ("bit 96 kHz
    FLAC Level 8 with MBIT dither"), a date header ("FEB-2025", "March 2026 (Friday)"). The listing
    then names every track one place late, and it still has exactly as many rows as the tape has
    files, so the equal-count check waves it through.

    :func:`best_reading` catches this when the filenames offer a competing reading. It cannot catch
    it when the filenames are bare indexes, because then there is nothing to compete: the offset
    listing wins unopposed, and the night is published confidently wrong. One such tape had no
    names on its files and produced a 25:37 song that is really a different one entirely, a 13:22
    that is really 5:09, and a 9:24 "reprise" that is really 1:46. Sixteen songs, all
    misattributed, all in the published medians.

    Only the DURATIONS can see it. The song names are the same whichever track they are pinned to,
    so neither the setlist nor the vocabulary can tell -- but a four-minute song is not 25 minutes
    and banter is not 10, and that is known because other tapes said so. Hence the offset test,
    scored against priors those other tapes produced (see :func:`provisional_medians` for why that
    is not circular).

    Deliberately conservative: at least ``MIN_OFFSET_EVIDENCE`` songs must carry a prior, and the
    shifted reading has to land under ``OFFSET_MARGIN`` of the current error. Measured on the tapes
    this fires on, the winning shift scores 3-5x better than the status quo, so the threshold sits
    in open space rather than on top of the data.
    """
    medians = provisional_medians(recordings, shows, normalizer)
    fixed: dict[str, list] = dict(tracklists)
    notes: list[dict] = []
    for record in recordings:
        listing = tracklists.get(record["identifier"])
        if not listing or len(listing) != len(record["tracks"]):
            continue
        found = _offset_of(record, listing, medians, normalizer)
        if found is None:
            continue
        shift, before, after, names = found
        moved = shift_listing(names, shift)
        fixed[record["identifier"]] = [{"song": name, "index": i + 1}
                                       for i, name in enumerate(moved)]
        notes.append({"identifier": record["identifier"], "date": record["date"],
                      "shift": shift, "error_before": round(before, 3),
                      "error_after": round(after, 3),
                      "dropped_row": names[0][:80] if names else ""})
    return fixed, notes


__all__ = [
    "AGREED", "DESCRIPTION", "FILENAME",
    "DESCRIPTION_MARGIN", "GIVE_UP_TRACK_SECONDS", "MIN_MATCH_RATE", "MIN_OFFSET_EVIDENCE",
    "MIN_PERFORMANCE_SECONDS", "MIN_ROWS", "MIN_TRACK_RATE", "OFFSET_MARGIN", "SHORT_NAME",
    "Edge", "Night", "Reading", "Row", "Tape",
    "alignment_error", "basename", "best_reading", "claims", "claims_in", "clean_song",
    "correct_tracklist_offsets", "file_forms", "flatten_setlist", "match_tape",
    "provisional_medians", "reorder_to_listing", "setlist_forms", "shift_listing", "song_forms",
    "tape_named_slots", "touches_segue",
]

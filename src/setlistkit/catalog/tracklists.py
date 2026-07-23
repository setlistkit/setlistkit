# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Read the taper's own tracklist out of a tape's description: one entry per FILE, in play order.

The tapers wrote down what every track is. The job here is not to discover it, only to RECOGNIZE
it -- a lesson this pipeline has had to learn three separate times, once for filenames, once for
numbered listings and once here, because each time the reflex was to invent a pattern the taper
was supposed to have followed instead of reading what they actually wrote.

THE SECOND PROJECTION OF A DESCRIPTION, and the direct analog of ``duration_tracks`` beside
``_tracks`` one layer down. :mod:`setlistkit.catalog.parse` reads the same text into a SETLIST --
what was played, grouped into sets and an encore -- and that is a different object from a
TRACKLIST, which is what is on the tape. A tracklist has exactly as many entries as the tape has
files, it keeps the tuning and the crowd noise because those occupy a file each, and it does not
care which set a song was in. Neither output can be derived from the other, so the description is
read twice.

TWO WAYS TO READ ONE, because tapers use both.

  NUMBERED   the listing carries indexes, in whatever style the taper felt like that day::

               01. Happy Hour Hero        3) Mar-De-Ma ->        12 - Rebubula
               d1t01. Space Truckin'>     10.Wurm                E01 Plane Crash
               01 Intro and Tuning        s2t08. (E) The Faker

  BARE       no numbers at all. Just the songs, one per line, with section headers and footnotes
             around them::

               SET ONE                 |   moe. - 02/24/23
               Happy Hour Hero         |   Billy Goat >
               Gone                    |   Bearsong
               Jazz Cigarette          |   Crushing
               ENCORE                  |   * = with Suke Cerulo on guitar

A bare listing cannot be found by pattern, because there is no pattern -- it is just words on
lines. It is found by RECOGNITION: a line naming a song this band actually plays is a track, and
so is a line naming something tapers put between songs and give a track to. Everything else --
the mic rig, the venue, the lineage, the footnotes, the thanks -- is neither, and falls away.

WHAT DECIDES BETWEEN THE READINGS IS THE FILE COUNT, never a hunch. Every defensible reading is
produced and the first one holding exactly as many entries as the tape has files wins. That count
check is the entire safety net: a listing that does not line up with the files is not used,
because using it would mean guessing where the discrepancy is, and a misplaced guess writes a
wrong duration that looks exactly like a right one.

WHAT THIS MODULE DOES NOT OWN. It does not decide what a gear line is, what counts as a segue,
what a footnote looks like, or whether something is music -- :mod:`setlistkit.catalog.parse` and
the pack's normalizer already answer all four, and the answers have to be the SAME ones. A song
name cleaned differently here than in the parser is a song name that will not join to the corpus
in the slice that consumes this, and the failure looks like a tape nobody could read rather than
like two modules disagreeing.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .normalizer import Normalizer, normalize, squash
from .parse import (CHECKSUM_TAIL, E_PREFIX, FOOT, SEG_SPLIT, gear_pattern,
                    unescape_description)

# Every tag is a line break here. See :func:`_lines` for why that differs from the setlist parser.
_TAG_RE = re.compile(r"<[^>]+>")

# A numbered line. The index may carry a disc/set prefix (d1t01, s2t08) or an encore prefix
# (E01), and the separator may be punctuation, whitespace, or -- because one taper wrote
# "10.Wurm" -- nothing at all. That single missing space once hid an entire fully-labeled show.
_NUMBERED = re.compile(
    r"^\s*(?:([ds])(\d{1,2}))?\s*(e)?\s*t?(\d{1,3})\s*(?:[.):\-]\s*|\s+)(.+?)\s*$", re.I)

# "Set I:Tuning", "Set II - Fearless", "Enc:Seat Of My Pants", "Encore: Bearsong". The marker is
# glued to the front of a real title, so it is peeled rather than used to reject the line.
_SET_PREFIX = re.compile(r"^\s*(?:set\s*[ivx\d]+|encore|enc|e)\s*[:\-]\s*", re.I)

# A line that is ONLY a section header. It sits inside the listing and is not a track.
_SET_HEADER = re.compile(r"^\s*(set\s*(one|two|three|four|1|2|3|4|i{1,3}|iv)|encore|enc|e|"
                         r"disc\s*\d+)\s*:?\s*$", re.I)

# Encore indexes restart at 1 ("E01 Plane Crash"), so they are pushed past any main-set index
# rather than interleaved with them. Discs are spaced the same way one order of magnitude down.
_ENCORE_BAND, _DISC_BAND = 100000, 1000

# Fewer lines than this and we are not looking at a tracklist, we are looking at prose that
# happens to contain a song name.
_MIN_LINES = 5

# How close a typo has to be. Far safer than it sounds: the candidate pool is one band's actual
# repertoire, not the English language, so "In Strikde", "Revovling Door" and "Mar-Dema" have
# very little to be confused WITH.
_FUZZY = 0.86

# How many times the corpus must have seen a title before it may vouch for a tracklist line.
# Two, because a parser accident happens once and repertoire repeats. See song_vocabulary for
# the measurement, and for what a floor of one costs.
MIN_PLAYS = 2

# A taper's annotation hung off the end of a title: "(nh)", "(UM)", "(NH)". Short and
# parenthesised, which is what keeps it away from a real title -- plenty of songs carry a
# parenthesised subtitle, and none of them is three letters long. The normalizer already knows
# this shape as a whole ENTRY (see BARE_NOTE); the difference here is that it arrives welded to
# the end of a song the taper did want to name.
_TRAILING_NOTE = re.compile(r"\s*\([^A-Za-z]*[A-Za-z]{1,3}[^A-Za-z]*\)\s*$")

# A tracklist line is a title, not a sentence. Past this it is a thank-you note or a paragraph of
# the taper's prose that happens to mention a song.
_MAX_TITLE = 60

# Which reading produced a listing, carried so a mismatch can be argued with rather than guessed
# at. See :func:`read_tracklist` for what each one is.
AS_WRITTEN = "numbered"
NUMBERED_TRUNCATED = "numbered-truncated"
NUMBERED_LEADING = "numbered-leading"
NUMBERED_SPLIT = "numbered-split"
BARE = "bare"
BARE_TRUNCATED = "bare-truncated"
BARE_SPLIT = "bare-split"
UNMATCHED = "unmatched"


@dataclass(frozen=True)
class TrackEntry:
    """One file on the tape, and what the taper said is on it."""

    index: int
    song: str
    segue: bool


@dataclass(frozen=True)
class Tracklist:
    """A tape's listing, which reading produced it, and whether it lines up with the files.

    ``matched`` is the only thing a consumer may join on. A listing that did not line up is still
    returned, because the best failed attempt is what makes a mismatch diagnosable -- but it is
    returned SAYING so, so that the decision to use it anyway has to be taken on purpose.
    """

    entries: tuple[TrackEntry, ...] = ()
    reading: str = UNMATCHED
    matched: bool = False

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class SongVocabulary:
    """What a tracklist line is recognized against, in the two forms recognition needs.

    ``normalized`` is the pack's own normalized key for every song it knows -- articles dropped,
    punctuation flattened, aliases folded in. Recognition tries this FIRST and it is the reason
    a taper writing "The Faker" is understood when the pack's canonical name is "Faker": the
    normalizer already answers "are these the same song", and asking it beats re-deciding the
    question with a blunter comparison.

    ``squashed`` is the same vocabulary reduced to bare ``[a-z0-9]``, sorted, and exists only for
    the fuzzy pass -- difflib needs a sequence, and a typo has usually damaged the very
    punctuation the normalizer keys on.
    """

    normalized: frozenset[str] = frozenset()
    squashed: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.normalized)


def song_vocabulary(normalizer: Normalizer, shows: Iterable[Mapping] = (),
                    min_plays: int = MIN_PLAYS) -> SongVocabulary:
    """Every squashed string that names a song this band plays, sorted.

    Drawn from the pack's vocabulary and its aliases on both sides, plus songs the corpus has
    seen at least ``min_plays`` times. The corpus half earns its place the way the previous
    implementation argued: a pack cannot list a cover the band played twice in 1998, our own
    setlists can, and a description naming it is still a tracklist.

    WHY THERE IS A FLOOR AT ALL, since the obvious reading is to take every song the corpus
    holds. The corpus is parser OUTPUT, not a curated list, and the titles it holds that no human
    has triaged are exactly the ones ``slkit pack lint`` reports as unknown. Feeding all of them
    back in makes the vocabulary a laundering channel: ``(moe.)``, ``Set List`` and one taper's
    ``= with Suke Cerulo on guitar`` are all stored as songs, so a band-name line, a section
    header and a footnote each become "a line naming a song" and get counted as tracks.

    Measured against the previous implementation's own published listings, over the 392 tapes
    both hold: pack plus EVERY corpus song agrees on 82.9% of them, and a floor of two agrees on
    91.8%. Junk is overwhelmingly a singleton -- a parser accident happens once -- while real
    repertoire repeats.

    The cost is real and is the right way round. A song actually played once and never again is
    not recognized, so a tape whose listing depends on it fails to line up and is not used. That
    is the safe failure: an unmatched listing is refused, while a junk-inflated one MATCHES and
    writes a wrong duration that looks exactly like a right one.

    SORTED, and that is not cosmetic. :func:`difflib.get_close_matches` returns matches in the
    order it finds them, so handing it a set makes the fuzzy branch depend on set iteration
    order, which changes between runs. A tape would read one way today and another tomorrow from
    identical inputs, and the durations computed from it would move with it.
    """
    names, norm_to_canon = normalizer.build_vocab()
    aliases = normalizer.aliases()
    # norm_to_canon is already keyed by normalized form and already has the aliases folded in;
    # the alias keys are added again because an alias may be written in a form that normalizes
    # to something the canonical side never produces.
    known = set(norm_to_canon) | {normalize(key) for key in aliases}
    known |= {normalize(name) for name in names}
    known |= {normalize(value) for value in aliases.values()}
    # Tagged non-songs are left out rather than counted. They are not repertoire, and a bare
    # listing recognizes them through the pack's classifiers anyway -- see _is_interstitial --
    # so putting them here would only give tuning a second, less careful way in.
    played: Counter[str] = Counter(
        str(entry.get("song") or "")
        for show in shows
        for section in list(show.get("sets") or ()) + [show.get("encore") or ()]
        for entry in section if not entry.get("non_song"))
    known |= {normalize(song) for song, seen in played.items() if seen >= min_plays}
    known = {key for key in known if key}
    return SongVocabulary(frozenset(known), tuple(sorted({squash(key) for key in known} - {""})))


def _lines(description: object) -> list[str]:
    """The description as stripped, non-empty lines, with the checksum table cut off the end.

    EVERY tag is a line break here, which is where this parts company with the setlist parser.
    That one turns most tags into a space, correctly, because it is reading prose in which a
    ``<b>`` sits mid-sentence. A tracklist is one entry per line and tapers mark those lines with
    whatever markup came to hand -- ``<li>``, ``<p>``, ``<div>``, a bare ``<br>`` -- so a tag
    that becomes a space fuses two tracks into one and the tape comes up one file short.

    The unescape IS shared, because "what did the taper actually type" has one answer. The
    checksum cut is the parser's too, and its line anchor is worth not rediscovering: a loose
    match on ``ffp`` also hits the taper's own lineage, which sits ABOVE the setlist, and cutting
    there decapitated 23 descriptions and threw away the very listings this module exists to find.
    """
    text = _TAG_RE.sub("\n", unescape_description(description))
    cut = CHECKSUM_TAIL.search(text)
    if cut:
        text = text[:cut.start()]
    return [line.strip() for line in text.split("\n") if line.strip()]


def _clean_title(raw: str, normalizer: Normalizer) -> tuple[str, bool]:
    """One line down to a song name, plus whether it segued into the next.

    The segue peel and the footnote strip are the parser's and the normalizer's, not this
    module's. A title cleaned differently here would not join to the corpus entry made from the
    same words, and the symptom would be a tape that looks unreadable rather than two modules
    that disagree about an asterisk.
    """
    song = _SET_PREFIX.sub("", raw)
    song = E_PREFIX.sub("", song)
    # Notes come off BEFORE the segue, which is the whole reason this is ordered rather than a
    # pile of substitutions. A taper writing "Buster >(nh)" puts the annotation outside the
    # arrow, so peeling the segue first finds "(nh)" at the end, peels nothing, and leaves a
    # song called "Buster >" -- an arrow welded to a title that then joins to nothing.
    song = _TRAILING_NOTE.sub("", song)
    song = FOOT.sub("", song)
    song, segued = normalizer.strip_segue(song)
    return song.strip(" .-*_:"), segued


def _pieces(song: str) -> list[str]:
    """A line split on its internal segue markers, separators dropped, each piece tidied.

    ``SEG_SPLIT`` captures its separator so the parser can see it; here only the songs matter.

    The annotation strip runs again per piece, and has to. A taper who writes
    "Billy Goat (NH) > Bearsong" puts the note in the MIDDLE of the line, where the trailing-note
    rule in :func:`_clean_title` cannot reach it -- it is only trailing once the line has been
    split. Left alone it survives into a song called "Billy Goat (NH)", which joins to nothing.
    """
    out = []
    for piece in SEG_SPLIT.split(song):
        piece = piece.strip()
        if not piece or SEG_SPLIT.fullmatch(piece):
            continue
        out.append(_TRAILING_NOTE.sub("", piece).strip(" .-*_:"))
    return [piece for piece in out if piece]


def _recognised(song: str, vocab: SongVocabulary) -> bool:
    """Is this line naming a song the band plays?

    The pack's normalization first, which is what makes "The Faker" and "Faker" one song, along
    with every difference of article, apostrophe, ampersand and punctuation the normalizer
    already knows how to ignore.

    Then fuzzy on the squashed form, because tapers type in the dark and a typo usually damages
    the very punctuation normalization keys on: "In Strikde", "Revovling Door", "Mar-Dema".

    Finally a line may name SEVERAL songs at once ("Crushing > Silver Sun > BJ Pizza") because
    the taper kept a whole segue run in one file; if its first piece is a song, the line is a
    track.
    """
    if not song.strip():
        return False
    if normalize(song) in vocab.normalized:
        return True
    squashed = squash(song)
    if (len(squashed) >= 6
            and difflib.get_close_matches(squashed, vocab.squashed, n=1, cutoff=_FUZZY)):
        return True
    pieces = _pieces(song)
    return len(pieces) > 1 and _recognised(pieces[0], vocab)


def _is_interstitial(song: str, normalizer: Normalizer) -> bool:
    """Something tapers put between songs and give a track to: tuning, crowd, a set break.

    Not songs -- but they ARE tracks, and dropping them breaks the count that is the only safety
    net here. The pack already owns this list, so it is asked rather than re-stated.

    ``firing_non_song_patterns`` rather than ``is_non_song``, and the difference matters. The
    latter is also true for a taper's guest note ("w/ Andy Frasco") and for a bare parenthesised
    annotation, neither of which is a file on the tape -- they are lines ABOUT the show. Those are
    caught by shape rules that belong to the normalizer rather than to any pack, so they fire no
    pack pattern, and asking which pattern fired excludes them for free.
    """
    return bool(normalizer.firing_non_song_patterns(song))


def _read_numbered(lines: Iterable[str], normalizer: Normalizer) -> list[TrackEntry]:
    """The taper numbered the tracks. Read the indexes they wrote."""
    rows: list[TrackEntry] = []
    for line in lines:
        match = _NUMBERED.match(line)
        if not match:
            continue
        disc = int(match.group(2)) if match.group(2) else 0
        encore = 1 if match.group(3) else 0
        index = int(match.group(4))
        song, segued = _clean_title(match.group(5), normalizer)
        if not song or not re.search(r"[A-Za-z]", song):
            continue
        rows.append(TrackEntry((encore * _ENCORE_BAND) + (disc * _DISC_BAND) + index,
                               song, segued))
    return rows if len(rows) >= _MIN_LINES else []


def _read_bare(lines: Iterable[str], vocab: SongVocabulary, normalizer: Normalizer,
               gear: re.Pattern) -> list[TrackEntry]:
    """No numbers -- just the songs. Found by recognizing them."""
    hits: list[TrackEntry] = []
    for line in lines:
        if _SET_HEADER.match(line) or len(line) > _MAX_TITLE or gear.search(line):
            continue
        song, segued = _clean_title(line, normalizer)
        if not song or not re.search(r"[A-Za-z]", song):
            continue
        if _recognised(song, vocab) or _is_interstitial(song, normalizer):
            hits.append(TrackEntry(len(hits) + 1, song, segued))
    return hits if len(hits) >= _MIN_LINES else []


def _truncate_at_restart(rows: Sequence[TrackEntry]) -> list[TrackEntry]:
    """The listing up to the point its numbering goes backwards.

    A restart is really ambiguous and must not be guessed at::

        19. Godzilla            |    08. Kyle's Song
         1. Part of the Lowell  |    01. Along For The Ride     <- set two, renumbered
         2. Fade in added       |    02. Head

    The first is footnotes and the run should be cut. The second is SET TWO, and cutting it
    throws away half the show -- which an earlier implementation did, silently, to every taper who
    numbers per set. So both readings are produced and the file count decides which was right.
    """
    if not rows:
        return []
    kept = [rows[0]]
    for row in rows[1:]:
        if row.index <= kept[-1].index:
            break
        kept.append(row)
    return kept


def _split_runs(rows: Sequence[TrackEntry]) -> list[TrackEntry]:
    """Break a segue run into one entry per song.

    Some tapers write a SETLIST rather than a tracklist, with the songs grouped by segue on one
    line, while the TAPE splits them into separate files. Eleven lines, sixteen tracks, and the
    night is refused for a mismatch that is not one.

    Offered as another candidate reading, with the file count deciding. If the taper kept the run
    in ONE file the unsplit reading matches instead and this is discarded, so we never have to
    know which kind of taper we are dealing with.
    """
    out: list[TrackEntry] = []
    for row in rows:
        pieces = _pieces(row.song)
        if len(pieces) < 2:
            out.append(row)
            continue
        for position, piece in enumerate(pieces):
            last = position == len(pieces) - 1
            # Every piece but the last segued by construction -- that is what put them on one
            # line. The last one carries whatever the line itself ended with.
            out.append(TrackEntry(0, piece, row.segue if last else True))
    return [TrackEntry(number, row.song, row.segue) for number, row in enumerate(out, start=1)]


def _renumbered(rows: Sequence[TrackEntry]) -> tuple[TrackEntry, ...]:
    """Sort keys replaced by 1..n. The disc/encore arithmetic is a sort key, never an index."""
    return tuple(TrackEntry(number, row.song, row.segue)
                 for number, row in enumerate(rows, start=1))


def read_tracklist(description: object, n_tracks: int, *, vocab: SongVocabulary,
                   normalizer: Normalizer, gear_patterns: Sequence[str] = ()) -> Tracklist:
    """Read one tape's description into a listing of ``n_tracks`` entries, if any reading fits.

    Every defensible reading is produced and the FILE COUNT picks the winner. Nothing here
    chooses between them on a hunch, because the readings disagree exactly where the description
    is ambiguous, and an ambiguity resolved by preference is a wrong duration that looks right.

    The order is by how much each reading is trusted, and the first exact fit wins:

    1. the numbered listing as written
    2. ...cut where its numbering restarts, if that restart was footnotes rather than a new set
    3. ...or just its leading run, for a listing that carries on past the tape
    4. the bare listing, recognized by name
    5. ...cut at a restart in the same way
    6. ...with segue runs broken into one entry per song, for a taper who wrote a setlist
    7. ...and the same for the numbered listing

    A tape whose listing fits none of them comes back with the best attempt and ``matched``
    False, which is what makes the mismatch visible instead of silently absent.
    """
    lines = _lines(description)
    numbered = _read_numbered(lines, normalizer)
    bare = _read_bare(lines, vocab, normalizer, gear_pattern(tuple(gear_patterns)))
    readings = (
        (AS_WRITTEN, numbered),
        (NUMBERED_TRUNCATED, _truncate_at_restart(numbered)),
        (NUMBERED_LEADING, numbered[:n_tracks]),
        (BARE, bare),
        (BARE_TRUNCATED, _truncate_at_restart(bare)),
        (BARE_SPLIT, _split_runs(bare)),
        (NUMBERED_SPLIT, _split_runs(numbered)),
    )
    for name, reading in readings:
        if len(reading) == n_tracks and len(reading) >= _MIN_LINES:
            return Tracklist(_renumbered(reading), name, matched=True)
    return Tracklist(_renumbered(numbered or bare), UNMATCHED, matched=False)

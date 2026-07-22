# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Song-name normalization: the mechanism, band-agnostic.

Ported from the old repo's ``songnorm.py``, which was already cleanly split between
mechanism and policy — it just did not know it. The mechanism (squash, normalize, the
attribution/segue/timecode/encore shape rules, the fuzzy vocabulary match) lives here and
never mentions a band. The policy (aliases, non-song patterns, protected titles) is
supplied by a pack through the hooks on :class:`Normalizer`, which default to empty so the
class works with no pack at all — just less precisely.

The pure functions stay at module level so ``normalize``'s cache is process-wide: the
prediction hot path calls it on the same fixed vocabulary hundreds of thousands of times,
and a per-instance cache would throw that away.
"""

from __future__ import annotations

import difflib
import functools
import re

# tokens that mark a segue when trailing a song
_SEG = ("~>", "->", ">", "→")

# Precompiled once, not recompiled per call. normalize() runs its four substitutions on the same
# ~650 song names hundreds of thousands of times per prediction run (every night of every
# Monte-Carlo rollout re-checks the whole vocabulary), and passing a string pattern to re.sub()
# pays a re._compile cache lookup on each of those millions of calls. The `\s+` collapse is the
# most expensive of the four (it scans and rebuilds the whole string); the other three are cheaper
# character-class passes. Profiled 2026-07-14: these four were ~40% of a forward_sim run.
_P_APOS = re.compile(r"[‘’']")                     # apostrophes -> ''
# Hyphens and slashes become spaces along with the rest. They are pure formatting in a song title
# and tapers disagree about them constantly: "Mar-De-Ma" / "Mar De Ma" / "Mar-Dema" are one song
# that was sitting in the vocabulary three times (n=17, 4, 2), and "High-Heeled" / "High Heeled"
# and "3-Eyed" / "3 Eyed" split the same way. Nothing in the canonical vocabulary depends on a
# hyphen to tell it apart from another song.
_P_PUNC = re.compile(r"[\.\,\!\?\(\)\[\]\"“”\:;\-/]")   # punctuation -> space
_P_WS = re.compile(r"\s+")                          # collapse whitespace
_P_THE = re.compile(r"^the ")                       # strip a leading 'the '

# Who WROTE the song, tacked onto the end of what we were told the song IS.
#
# Tapers annotate covers, and they do it in a dozen shapes: "Ophelia [The Band]", "The Carpet
# Crawlers [Genesis]", "Do It Again [Steely Dan]", "No Rain (Blind Melon cover)", "Corduroy (UM)",
# "Würm [YES, from Starship Trooper]". The annotation is about provenance, not identity -- an
# Ophelia is an Ophelia -- but left attached it files each one as its own song with n=1, which is
# how one cover ends up in the vocabulary three times under three spellings.
#
# Handled here as a RULE rather than as alias entries, because the set of bands moe. cover is
# open-ended and a hand list would need a new line every time they play something new.
#
# Square brackets are stripped unconditionally: no song in the canonical setlist.fm vocabulary
# uses them (checked -- zero of 313). Parentheses are NOT, because six canonical titles END in a
# real parenthesised phrase that is part of the name -- "Z0Z (Zed Nought Z)", "Breathe (In the
# Air)", "Breathe (Reprise)", "Escape (The Piña Colada Song)". So parens are only removed when
# what is inside them announces itself as an attribution: the words "cover"/"version", or a bare
# 2-3 letter band initialism like (UM) or (NH). That case-sensitive initialism rule is why this
# runs BEFORE the string is lowercased.
_P_ATTRIB_BRACKET = re.compile(r"\s*\[[^\]]*\]\s*$")
_P_ATTRIB_COVER = re.compile(r"\s*\([^)]*\b(?:cover|version)\b[^)]*\)\s*$", re.I)
_P_ATTRIB_INITIAL = re.compile(r"\s*\([A-Z]{2,3}\)\s*$")

# An encore marker welded onto the front of the song.
#
# Tapers who number a set straight through still need to say where the encore starts, so they
# flag it inline: moe2023-09-16.dpa numbers set two 01-07 and writes "06. e. Threw It All Away",
# "07. e. Ophelia". The description parser strips the "07." and leaves the "e.", which arrives
# here as a song title -- and "e. Ophelia" is then a different song from "Ophelia" forever.
#
# Requires the trailing dot or colon. A bare leading "e " is not stripped, because that is a
# word, and a rule that eats it would eat the first word of anything.
#
# The full word "encore" is held to the same rule, though it collides with nothing in the
# vocabulary and could safely be loosened. Loosening it made "^encorebreak$" a dead rule -- the
# marker came off before the classifier ever saw the title -- which is a live change to a shared
# path in exchange for letting one hand-written override say "encore okayalright" instead of
# "e. okayalright". Not worth it. An override author writes the marker the way a taper does.
_P_ENCORE_MARK = re.compile(r"^(?:e|enc|encore)\s*[.:]\s*", re.I)

# A guest credit welded onto the song, unparenthesised. Emma Derhak (Rob's daughter) sings with
# them regularly, and the 2026-03-14 tapes write it inline: "St. Augustine -with emma d". That
# is a St. Augustine, but it arrives here as a song nobody has ever played.
#
# THE TRAP: "Don't Fuck With Flo" is a real song containing the word "with". So the bare word is
# only treated as a credit when something separates it from the title -- a dash, a hash, a star,
# an open paren. The SLASHED form "w/" needs no separator: it is never part of a song name.
_P_GUEST = re.compile(r"(?:\s*[-#*+^~(]\s*with\b|\s*[-#*+^~(]?\s*w/\s*).*$", re.I)

# A leading timestamp from a taper who indexes their listing: "(09:39) Tamborine".
_P_TIMECODE = re.compile(r"^\s*\(?\s*\d{1,2}:\d{2}(?::\d{2})?\s*\)?\s*")

# "W/ Andy Frasco", "w/ Haley Jane", "(w/ BRONCO)", "& Emma D" -- a note about who sat in,
# standing alone in the setlist as if it were a song. The "&" spelling is here because it is the
# same annotation: with it living in the parser instead, one spelling got tagged and the other
# got deleted.
#
# The word boundary sits INSIDE the alternation, on "with" alone. Written as `w(?:ith|/)\b` it
# could never fire for the slashed form: "/" and " " are both non-word characters, so there is no
# boundary between them, and every one of the examples above -- the regex's OWN documented
# targets -- was read as a song. "with" still needs the boundary or it eats the first word of
# "Within Your Reach". "w/" needs nothing, because no song title has ever started with it.
GUEST_NOTE = re.compile(r"^\(?\s*=?\s*(?:&\s*|w(?:ith\b|/))", re.I)

# A setlist entry that is nothing but a PARENTHESISED note -- "(NH)", "(UM)" -- is a taper's
# annotation the description parser promoted to a song. Left alone it is also a collision
# hazard, because a two-letter "song" sits inside half the vocabulary.
#
# The parentheses are doing the real work here. An earlier version matched any 1-3 letter all-caps
# entry and promptly deleted ATL and NYC, both real songs. Junk with n=1 is survivable; a real
# song silently vanishing from the vocabulary is not.
#
# The comment this was ported from named TLH alongside them as a third "real song". It is not one.
# TLH is Trader's Little Helper, the FLAC checksum tool tapers run, it appears nowhere in the
# corpus as a setlist entry, and the old parser listed `tlh` in its gear words at the same time it
# was being protected as a song. It got grouped with the other two because it LOOKED like them:
# a bare three-letter all-caps token. Shape is what this whole rule is about, so that is exactly
# the mistake to expect.
BARE_NOTE = re.compile(r"^\([^A-Za-z]*[A-Za-z]{1,3}[^A-Za-z]*\)$")

# An entry wrapped in quotes end to end: '"thank you very much everybody..."', '"Penguin Joke"',
# '“Band Interview”'. Quoting is how a setlist writes down a spoken moment, and the entries that
# use it say so plainly -- announcements, dedications, jokes, a call-back to something said an
# hour earlier. Left as songs they take a slot in a song's structural profile and, worse, get
# TIMED: "thank you very much everybody..." published a median length of 14 seconds.
#
# The pack already caught eight of these by vocabulary ("Tuning", "Banter", "Al-nouncements")
# and missed twenty-six that differ only in wording. A shape rule catches the ones nobody thought
# to list, which is the whole reason shape rules live in core.
#
# TWO TRAPS. It must wrap the WHOLE entry: 'Wind it Up "False Start"' is a Wind it Up with a note
# attached and keeps its slot. And a rule with no vocabulary behind it will eventually fire on a
# real title someone chose to quote -- which is what ``protected`` is for, checked first.
QUOTED_NOTE = re.compile(r'^["“”].*["“”]$', re.S)

# "Moth (w/ Daniel Donato)" is a Moth. The guest is an annotation on the performance, not a
# different song, and leaving it attached files a real Moth under its own name with n=1.
_GUEST_SUFFIX = re.compile(r"\s*\(\s*w(?:ith|/)[^)]*\)\s*$", re.I)

# Stray quote marks a description parse drags in: 'Moth"' is also a Moth.
#
# ONLY WHERE A QUOTE IS ACTUALLY A QUOTE. An earlier form of this deleted every one of these
# characters wherever it stood, and the curly apostrophe is in the set, so "Hey, It’s Christmas"
# was published as "Hey, Its Christmas" -- a real song under a name it does not have, and one
# that no longer matched the same song coming from anywhere else. A quote character between two
# letters is an apostrophe and is left alone.
_STRAY_QUOTE = re.compile(r"(?<![A-Za-z])[‘’“”\"]|[‘’“”\"](?![A-Za-z])")


def strip_attribution(text: str) -> str:
    """Drop a trailing cover credit and a leading encore marker.

    Loops: "No Rain [Blind Melon] (cover)" carries two annotations, and "e. Ophelia [The Band]"
    carries one at each end."""
    for _ in range(3):
        before = text
        text = _P_TIMECODE.sub("", text)
        text = _P_ENCORE_MARK.sub("", text)
        for pattern in (_P_ATTRIB_BRACKET, _P_ATTRIB_COVER, _P_ATTRIB_INITIAL):
            text = pattern.sub("", text)
        # Only after the brackets are gone, so "[The Band]" cannot hide the credit behind it.
        # Never applied to a string that IS the credit ("w/ Haley Jane") -- that leaves nothing,
        # and an empty name matches every song rather than none.
        cut = _P_GUEST.sub("", text).strip()
        if re.search(r"[A-Za-z0-9]", cut):
            text = cut       # must leave an actual NAME behind, not "=" or a stray bracket
        if text == before:
            break
    return text.strip()


def clean_song(song: str) -> str:
    """Canonical form of a setlist entry: the name a performance is FILED under.

    Here rather than beside the tape-reading code that first needed it, because it is the answer
    to "what is this song called", and more than one pass has to give the same answer. The length
    chain cleaned and :func:`setlistkit.catalog.features.song_features` did not, so the two halves
    of an export keyed the same performance under two spellings -- the join silently produced a
    song whose structural profile counted only the plays that happened to be spelled plainly.

    A SPOKEN MOMENT KEEPS ITS QUOTES, because they are not decoration on a name -- they are the
    only thing saying it is not a song. Everything downstream asks :meth:`Normalizer.is_non_song`
    about the CLEANED entry, so stripping them here hands the classifier '"Penguin Joke"' with
    nothing left to recognise, and twenty-six announcements, dedications and jokes get timed and
    published with median lengths.
    """
    text = str(song).strip()
    if QUOTED_NOTE.match(text):
        return text
    return _STRAY_QUOTE.sub("", _GUEST_SUFFIX.sub("", text)).strip()


def strip_segue(text: str) -> tuple[str, bool]:
    """Peel any trailing segue token off ``text``, reporting whether one was there."""
    text = text.strip()
    seg = False
    changed = True
    while changed:
        changed = False
        for marker in _SEG:
            if text.endswith(marker):
                text = text[: -len(marker)].strip()
                seg = True
                changed = True
        if text.endswith("~"):
            text = text[:-1].strip()
            changed = True
    return text, seg


# Cached: normalize() is a pure function of its input, and the prediction hot path calls it on the
# same fixed vocabulary over and over (measured 2026-07-14: 669,676 calls resolving to just 650
# distinct strings in one 60-rollout forward_sim -- a ~1030x redundancy). The cache collapses that
# to one computation per distinct string. Bounded so a full rebuild that streams thousands of raw
# taper-note fragments through here can't grow it without limit; the working set is far smaller.
@functools.lru_cache(maxsize=16384)
def normalize(text: str) -> str:
    """Reduce a raw song string to its canonical normalized key."""
    text = text.strip()
    text, _ = strip_segue(text)
    text = strip_attribution(text)     # before lower(): the (UM) rule is case-sensitive
    text = text.lower()
    text = text.replace("&", "and")
    text = _P_APOS.sub("", text)
    text = _P_PUNC.sub(" ", text)
    text = _P_WS.sub(" ", text).strip()
    text = _P_THE.sub("", text)
    return text


def squash(text: str, amp: str = "and") -> str:
    """Down to bare [a-z0-9]. Every disagreement about spacing, punctuation, capitalisation and
    curly quotes evaporates: the taper's "Worm Wood", "worm-wood" and our "Wormwood" all become
    `wormwood`; the alias map's "rec chem" becomes `recchem`. This is precisely why we never
    need to know how a given taper formats a filename.

    "&" has no single right answer, so `amp` says which spelling to make. Tapers write one title
    every possible way: 2025-01-09 has "Ups & Downs" against a setlist saying "Ups and Downs",
    and 2023-10-22 has "Good Guys, Bad Guys" against a setlist saying "Good Guys & Bad Guys".
    Deleting the ampersand fixes the second and breaks the first; expanding it does the reverse.
    So we generate BOTH forms and let either one match."""
    return re.sub(r"[^a-z0-9]", "", text.lower().replace("&", f" {amp} "))


def _build_vocab(names, aliases: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """Return (canon_list, norm_to_canon) from a name list plus the alias targets.

    The alias targets fold in because an alias value ("Recreational Chemistry") is a canonical
    name whether or not the caller listed it. sorted() so that when two display names share a
    normalized key the winner is deterministic -- set iteration order is hash-seed-randomized
    across processes, which otherwise flips e.g. "Faker" <-> "The Faker" run to run.
    """
    pool = set(names)
    pool.update(aliases.values())
    norm_to_canon: dict[str, str] = {}
    for name in sorted(pool):
        norm_to_canon.setdefault(normalize(name), name)
    return sorted(pool), norm_to_canon


def _canonicalize(raw: str, norm_to_canon: dict[str, str], aliases: dict[str, str],
                  cutoff: float = 0.9) -> tuple[str | None, bool]:
    """Map a raw song string to a canonical display name + segue flag."""
    # The display form gets the same annotations stripped as the normalized one. Otherwise a song
    # we do NOT recognise falls through to `disp` still carrying its credit -- "Rebubula II (UM)"
    # entered the vocabulary with the (UM) attached even though normalize() had already removed
    # it, so the fallback re-created exactly the split this is here to prevent.
    disp, seg = strip_segue(raw)
    disp = strip_attribution(disp)
    norm = normalize(raw)
    if not norm:
        return None, seg
    if norm in aliases:
        return aliases[norm], seg
    if norm in norm_to_canon:
        return norm_to_canon[norm], seg
    # fuzzy against known vocabulary
    match = difflib.get_close_matches(norm, list(norm_to_canon.keys()), n=1, cutoff=cutoff)
    if match:
        return norm_to_canon[match[0]], seg
    # unknown -> the cleaned display form
    return disp.strip(), seg


class Normalizer:
    """Generic mechanism. Every hook defaults to empty, so the class works with no pack at
    all, just less precisely.

    A ``vocabulary`` is the canonical song list the pack owns (see the spec: "the pack is the
    dictionary"). It plays the role setlist.fm's clean name list used to play, with no live
    tracker dependency. The policy hooks — :meth:`aliases`, :meth:`non_song_patterns`,
    :meth:`protected_titles` — are where a band-specific pack subclass injects its knowledge.
    """

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        self._vocabulary = list(vocabulary or [])
        # Built lazily and cached: build_vocab is deterministic, and canonicalize hits it on
        # every call.
        self._canon: list[str] | None = None
        self._norm_to_canon: dict[str, str] | None = None
        # The protected set squashed once, not per is_non_song call.
        self._protected_squashed: set[str] | None = None

    # -- mechanism (thin wrappers over the module-level pure functions) --------------------

    def squash(self, text: str, amp: str = "and") -> str:
        """Reduce ``text`` to bare ``[a-z0-9]``; ``amp`` picks the ``&`` spelling."""
        return squash(text, amp)

    def normalize(self, text: str) -> str:
        """Reduce ``text`` to its canonical normalized key (process-wide cached)."""
        return normalize(text)

    def strip_attribution(self, text: str) -> str:
        """Drop trailing cover credits and a leading encore marker from ``text``."""
        return strip_attribution(text)

    def strip_segue(self, text: str) -> tuple[str, bool]:
        """Peel a trailing segue token off ``text``, reporting whether one was there."""
        return strip_segue(text)

    def build_vocab(self) -> tuple[list[str], dict[str, str]]:
        """Return (canon_list, norm_to_canon) from this normalizer's vocabulary + aliases."""
        if self._norm_to_canon is None:
            self._canon, self._norm_to_canon = _build_vocab(self._vocabulary, self.aliases())
        return self._canon, self._norm_to_canon

    def canonicalize(self, raw: str, cutoff: float = 0.9) -> tuple[str | None, bool]:
        """Map a raw song string to a canonical display name + segue flag.

        Alias first, then exact normalized match, then difflib fuzzy within ``cutoff``;
        anything else falls through as its own cleaned display form.
        """
        _, norm_to_canon = self.build_vocab()
        return _canonicalize(raw, norm_to_canon, self.aliases(), cutoff)

    def synonym_map(self, names) -> dict[str, str]:
        """Map every name in ``names`` to the one name that IS that song.

        The vocabulary here is the input names themselves (plus alias targets), not this
        normalizer's own vocabulary: this collapses spelling variants *within* a given set, so
        "Ricky", "Ricky Marten" and "Ricky Martin" become one song rather than three. The
        survivor is the canonical spelling, because the play history joins on canonical names.
        """
        aliases = self.aliases()
        _, norm_to_canon = _build_vocab(names, aliases)
        out: dict[str, str] = {}
        for name in names:
            canon, _ = _canonicalize(name, norm_to_canon, aliases)
            out[name] = canon or name
        return out

    # -- policy hooks (empty by default; a pack subclass supplies these) -------------------

    def aliases(self) -> dict[str, str]:
        """Normalized-key -> canonical display name. Empty in the base class."""
        return {}

    def non_song_patterns(self) -> list[re.Pattern]:
        """Compiled patterns that, matched against a squashed entry, mark it as not-a-song."""
        return []

    def protected_titles(self) -> set[str]:
        """Titles a generic shape rule would wrongly delete. Always songs. See ``is_non_song``."""
        return set()

    def is_non_song(self, entry: str) -> bool:
        """Is this setlist entry NOT music? Shape rules plus the pack's non-song patterns.

        Core owns the rule, the pack owns the exceptions: a protected title is ALWAYS a song,
        even if a pattern would match it. That guard is what stopped an earlier all-caps rule
        from deleting ATL and NYC, both real songs.
        """
        entry = entry.strip()
        if self.is_protected(entry):
            return False
        if GUEST_NOTE.match(entry) or BARE_NOTE.match(entry) or QUOTED_NOTE.match(entry):
            return True
        squashed = squash(entry)
        return any(pattern.search(squashed) for pattern in self.non_song_patterns())

    def firing_non_song_patterns(self, entry: str) -> list[str]:
        """Which pack patterns actually mark ``entry`` as not-music, as pattern strings.

        The same question :meth:`is_non_song` answers, asked so the answer can be attributed to
        a rule. Empty for a protected title, because the guard runs first and no pattern gets to
        fire -- which is the honest answer and the one a "does this rule ever do anything" check
        needs: a rule shadowed by ``protected.json`` never fires, however well it matches.

        Empty too when the shape rules did the work: ``GUEST_NOTE``, ``BARE_NOTE`` and
        ``QUOTED_NOTE`` belong to this module, not to any pack, so no pack rule earns the credit.
        """
        entry = entry.strip()
        if self.is_protected(entry):
            return []
        squashed = squash(entry)
        return [pattern.pattern for pattern in self.non_song_patterns()
                if pattern.search(squashed)]

    def is_protected(self, entry: str) -> bool:
        """True when ``entry`` squashes to a protected title. Compared squashed on both sides so
        formatting ("A.T.L", "ATL") cannot slip a protected song past the guard.

        Public because ``is_non_song`` is not the only rule that can delete a real song by
        accident: any shape gate wants to ask this first."""
        if self._protected_squashed is None:
            self._protected_squashed = {squash(title) for title in self.protected_titles()}
        return bool(self._protected_squashed) and squash(entry) in self._protected_squashed

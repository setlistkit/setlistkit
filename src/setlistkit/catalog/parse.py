# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
r"""Turn an archive item's taper description into a structured setlist.

The mechanism is all here: strip the HTML, cut the machine output off the end, find where the
setlist starts, split it into sets and an encore, and clean each token down to a song name.
What is NOT here is knowledge of any particular band -- the dates to skip, the items filed
under the wrong date, the cover artists a taper writes in place of a song. Those arrive in an
:class:`ArchivePolicy` the caller builds from a pack.

The one real change from the port is the seam. The old parser kept its own non-song set, its
own banter set, and its own banter-combo check, and it DROPPED whatever they matched. The
normalizer kept a separate answer to the same question in ``is_non_song``. Two answers means
the corpus disagrees with itself depending on which door an entry came through -- and both
answers were wrong in exactly the same way, because "w/ Andy Frasco" slipped past a copy of the
same regex on each side. Now there is one answer, ``Normalizer.is_non_song``, and this parser
TAGS what it finds instead of dropping it:

    {"song": "Tuning", "segue": False, "non_song": True}

Keeping it makes the corpus a faithful record of the tape, and lets each consumer decide for
itself: the vocabulary and the rotation model skip the non-songs, a completeness count skips
them, and someone reading a parsed show still sees that the band spent two minutes tuning
between the second song and the third.

The line between TAG and DROP, since everything below is on one side of it:

    TAG      something that happened on the stage and is not music. Tuning, banter, a drum
             solo, a note about who sat in. It belongs in the record, labelled.
    DROP     text that was never a setlist entry at all. The gear lineage, the venue name, a
             credit line, a checksum table, a URL. Nothing happened; a parser just found words.

A rule that DROPS is only allowed to answer "is this an entry", never "is this music". The
second question has exactly one owner.
"""

from __future__ import annotations

import functools
import html
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .normalizer import Normalizer, squash

# A description parse this thin is not a setlist, it is a taper who wrote prose. Below it we
# ask the tracklist for a second opinion.
_WEAK_PARSE = 4

# The shape a show date has to have to be believed. Shape only -- the pack schema is where a
# month of 13 gets rejected -- but anchored at both ends, which is the half that matters here.
_DATE_SHAPE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")

# Recording gear, in every spelling tapers use for it. The preamble to a description is a
# lineage -- the mics, the deck, the software it passed through on the way to a file -- and
# none of those words belong to a song title. This is the single widest filter in the module.
#
# What is here is gear any taper writes. What is NOT here any more is `kcy`, `nbob`, `pfa`,
# `ela` and `cf`, which came over with the port carrying no explanation in the source they came
# from and none anywhere else. Each is a two-or-three letter token, which is exactly the shape
# that deleted ATL and NYC, and a token I cannot read is a token I cannot say is safe for a band
# I have never heard of. They live in the pack now, via ``ArchivePolicy.gear_patterns``, where
# an author who knows their own taper scene writes down what each one is and lint holds it
# against their vocabulary.
_GEAR_BASE = (r"mic|mics|flac|wav|wave|sbd|aud|akg|schoeps|neumann|mk4|km1|matrix|khz|"
              r"recorded|transfer|transferred|taped|taper|dac|nbox|mixpre|sound\s*devices|"
              r"24bit|16bit|32bit|cd\s*wave|cdwav|processed|audacity|fob|loc|hypers|cardioid|"
              r"lineage|source|edit|edited|normalize|mastered|sector|samplitude|"
              r"macbook|xact|telefunken|tagging|pcm|amadeus|aes|ebu|"
              r"xlrs?|firewire|k?ables|preamp|phantom\s*power|gain|sd\s*card|"
              # the lineage line names the machines it passed through
              r"imac|reaper|foobar|checksums?|shntool|ffp|md5|tlh|render\w*")

# A band-member credit LINE: "Al Schnier - guitar, vocals", "Rob Derhak: bass". A name, a
# separator, then the instruments.
#
# The shape is the whole point. Matching the bare instrument word anywhere -- which is what the
# old parser did, and what its non-song word list did AGAIN a few lines later -- deleted "Drums",
# "Bass Solo" and "Percussion". Those are things that happened on the stage, so they are the
# normalizer's question, not this one's. A credit line is not a setlist entry at all, and that
# IS this one's question.
CREDIT = re.compile(r"^[^:\-]{2,}\s*[:\-]\s*.*\b(vocals|keyboards?|guitars?|drums|bass|"
                    r"percussion|mallet|flute|malletkat|vibes)\b", re.I)

# Leading track numbering in any of the shapes a taper uses: "01. ", "d1t05. ", "t3 ", "7) ".
PREFIX = re.compile(r"^(d\d+\s*t\d+|disc\s*\d+|t\d+|\d{1,2})\s*[\.\):]?\s+", re.I)

# "(E) The Faker" -- an encore marker the taper puts in front of the title. It is not the song.
E_PREFIX = re.compile(r"^\(\s*e\s*\)\s*", re.I)

# Taper credit URLs that leak in as track titles.
URLISH = re.compile(r"://|www\.|\.(?:org|com|net)\b", re.I)

# Footnote and annotation marks that are never part of a real title. Note the absence of '&',
# which is legitimate -- plenty of songs are "X & Y".
FOOT = re.compile(r"[\*\#\^\@\%\+]")

# Internal segue markers that join two distinct songs inside one token ("A > B", "A -> B").
SEG_SPLIT = re.compile(r"(~?->|~?>|→|<)")

# A set HEADER: "Set 1" / "Set 01" (zero-padded) / "Set II" / "1st Set" / "Disc 2". Marks where
# a section begins and, more importantly, where the taper's gear preamble ends. Kept separate
# from the encore markers below so an encore can never be mistaken for the setlist's start.
SETHEADER = re.compile(r"set\s*(?:0?[1-9]|one|two|three|four|i{1,3}|1st|2nd|3rd)\b|"
                       r"\b(?:1st|2nd|3rd|first|second|third)\s+set\b|"
                       r"\bdisc\s*(?:\d|i{1,3})\b", re.I)
ENCMARK = re.compile(r"encore|\benc\b|\be:", re.I)
# Any section marker (header OR encore) -- used to split the setlist region into segments.
SETMARK = re.compile(SETHEADER.pattern + r"|encore|\benc\b|\be:\s", re.I)
# First numbered track ("01. ", "1) "). The dot-or-paren plus whitespace requirement avoids
# matching version numbers ("1.2.1") and durations. This finds where the setlist starts when a
# taper numbered the songs but wrote no "Set N" header -- without it, a later "Enc:" marker
# becomes the first thing we recognise and the whole main set is thrown away.
FIRST_NUMTRACK = re.compile(r"(?:^|[\n\s])0*[1-9]\d?[\.\)]\s+\S")

# Tapers paste a checksum verification table at the end of the description:
#
#     SHNTOOL OUTPUT
#     length   expanded size   cdr  fmt  ratio  filename
#     6:08.007   211971914 B   cxx   --   ---xx   flac  0.5850  08 No Rain.flac
#
# The old parser read the column headers ("length", "expanded size", "ratio", "filename") and
# the table's own flag columns ("cxx", "xx") as ENCORE SONGS. 35 shows had an encore made of a
# checksum table, which means every encore rate ever computed was measured against one.
#
# Everything from this marker to the end is machine output, never a setlist. Anchored to the
# START OF A LINE: a loose \bffp\b also matches the taper's lineage -- "XAct(FLAC 8,ffp,tagging)"
# -- which sits ABOVE the setlist, so cutting there ate the setlists of 23 shows. A checksum
# table announces itself with a header on its own line.
CHECKSUM_TAIL = re.compile(r"^[ \t]*(shntool|shn\s*tool|md5|ffp|sha1|checksums?)\b", re.I | re.M)

# The other thing tapers paste after the setlist: credits and the band's press bio. Same failure
# as the checksum table, different boilerplate. One item puts the whole setlist on a single line
# and then pastes the label's one-sheet underneath, and the parser read the prose as an encore --
# twenty "songs" including "Lauded by American", "brace of songs" and "painfully complicated
# life". Another ends with a credit roll and contributed "Total Time", "FOH" and "Poster Artist".
#
# Line-anchored, for the same reason as CHECKSUM_TAIL. These two are band-agnostic; the ones
# built from the band's own name are added by _credit_tail_pattern.
#
#   total time   "Total Time:  [02:40:38]" -- the tape's runtime, and reliably the first line of
#                the credit block.
#   band:        a lineup block introduced by a heading on its own line. Band-agnostic: every
#                scene writes one, and _credit_tail_pattern's band-name form below only fires
#                when the taper types the BAND'S name ("moe.:"), while most of them type the
#                word. Found on real data, where it cost a "Band" song plus "Guitar", "Bass",
#                "Drums", "Percussion" and a member's name, once per tape that formats this way.
#
# The lineup heading is anchored to a line holding ONLY the heading, because "A Band In The Sky"
# is a real song and a pattern that cut at "Band:" anywhere is one taper's colon away from eating
# a setlist. And the trailing class is [^\S\n], not [ \t]: tapers paste from web pages, so the
# character after the colon is routinely a non-breaking space, and a [ \t]*$ tail compiles
# perfectly and then never fires -- the worst way for a defence to be wrong.
_CREDIT_TAIL_BASE = (r"total\s+time\s*[:\[]", r"check\s+out\s+tour\s+dates",
                     r"(?:band|band\s+members|line-?up|personnel|musicians)\s*:[^\S\n]*$")

# Backstop for a credit line that survives the cut because its block had no header: a crew role.
# "Steve Young: FOH", "Poster Artist". Roles, not names -- a name list would never end, and these
# are the words that make a line a credit rather than a title.
CREW_ROLE = re.compile(r"\b(foh|lds?|monitors?|lighting|poster\s+artist|tour\s+manager|"
                       r"stage\s+manager|production\s+manager|front\s+of\s+house|"
                       r"total\s+time)\b", re.I)

# Recurring track titles that are annotations rather than entries: taper notes,
# acknowledgements, support-act and guest-roster credits. An opener or a list of who else was
# on the bill is a note about the performance, never a thing that was performed. Kept tight to
# avoid nuking real titles. Anything band-specific -- the cover artists a taper names in place
# of the song, the members' surnames they write beside a sit-in -- comes from the pack via
# ``ArchivePolicy.junk_patterns``.
#
# Everything here answers "is this an entry", which is the only question a DROP rule is allowed
# to ask. `greeting` used to be in this list and does not belong: "Greeting By Al" is Al
# greeting the crowd, which HAPPENED, and a thing that happened on stage and is not music is a
# TAG. It moved to the pack's classifiers, where the one owner of "is this music" can answer it.
# `a\s+team`, `home\s+team` and `inverted\s+version` moved to ``junk_patterns``: they are drops,
# but they read as one corpus's artifacts and nothing generic justifies them.
_JUNK_BASE = (r"seeded|assistant|notes|footnotes|"
              r"all\s+members|acknowledg\w*|opened\s+for|opened$|website")

# leading "BAND DATE Venue City, ST" header. re.S so it spans the newlines clean_html leaves
# between the venue line and the gear block.
VENUE_HEADER = re.compile(r".{0,160}?,\s*[A-Z]{2}\b", re.S)

# Words that appear in the name of a room and never in the name of a song. The taper's header is
# often four or five lines deep -- festival, venue, stage, city, date -- and matching the item's
# own `venue` string is not enough to catch them: archive.org says "Peach Festival" while the
# description says "The Peach Music Festival", and we ended up with THAT, plus "Toyota Pavilion
# at Montage Mountain", plus "Peach Stage", plus "Setlist", all filed as songs the band played.
VENUE_WORDS = re.compile(
    r"\b(festival|pavilion|amphitheat(?:er|re)|theat(?:er|re)|arena|ballroom|stadium|"
    r"casino|coliseum|auditorium|fairgrounds|racetrack|speedway|winery|brewery|"
    r"opera\s*house|music\s*hall|setlist|main\s*stage|stage)\b", re.I)

# archive.org's own titles name the band: "moe. Live at Northlands on 2026-06-14" against
# "bob. Live at Ophelia's on 2024-11-07". We never have to infer it from the setlist.
#
# Bounded and lazy, and compared squashed rather than letter by letter. The version this
# replaces captured `[A-Za-z][A-Za-z.]{0,10}?`, which is the shape of "moe." and of very little
# else: for any band whose name holds a space, a digit or an apostrophe the regex could read
# neither its own items NOR the side projects it exists to reject, so it fell through to "accept
# everything" and did so silently. A filter that cannot fail loudly should at least fail rarely.
BAND_IN_TITLE = re.compile(r"^\s*(.{0,60}?)\s+Live\s+at\b", re.I)


# Why an item did not become a show. Three rules can refuse one, and from downstream all three
# look identical: the date is simply not in the corpus. A night that was deliberately dropped and
# a night nobody taped are very different facts about a band, so the parser says which.
NOT_THIS_BAND = "not_this_band"      # the title names a different act; a side project's tape
NO_DATE = "no_date"                  # no date we can believe, so nothing can be joined to it
DROPPED_DATE = "dropped_date"        # the pack refuses this night, and says why in corpus.json


@dataclass(frozen=True)
class Skipped:
    """One item the parser refused, and which rule refused it.

    ``date`` is filled in only for :data:`DROPPED_DATE`, because that is the only case where a
    date was successfully read before the item was turned away -- and it is the one the caller
    needs, to look the reason up in the pack that asked for the drop.
    """

    identifier: str
    reason: str
    date: str = ""


@dataclass(frozen=True)
class Census:
    """Every cleaned token the parser met, and what became of it.

    This is what the corpus-aware checks in ``slkit pack lint`` need, and it is emphatically NOT
    the stored corpus. A junk or gear rule DROPS what it matches, so by the time a show reaches
    the database every token those rules ever removed is gone from it. A check that asked "does
    this pattern match anything in the corpus" against the stored shows would find nothing for
    every rule, on every pack, forever -- and would report all of them dead while running clean.
    That is the same shape as a filter test written above the set header: it does not fail, it
    just never fires.

    So the parser records what it actually saw. Counters rather than lists, so a full collection
    stays small and a finding can say how many times a rule earned its place rather than only
    that it did.

    ``seen`` is every token BEFORE canonicalization, which is the form the drop rules and the
    alias keys are both written against.
    """

    seen: Counter = field(default_factory=Counter)
    dropped: Counter = field(default_factory=Counter)     # tokens no rule let through
    tagged: Counter = field(default_factory=Counter)      # tokens a classifier called not-music


@dataclass(frozen=True)
class ParseResult:
    """Every show a source yielded, every item it refused, and everything it met on the way."""

    shows: list[dict] = field(default_factory=list)
    skipped: tuple[Skipped, ...] = ()
    census: Census = field(default_factory=Census)


@dataclass(frozen=True)
class ArchivePolicy:
    """The band-specific inputs the mechanism refuses to invent for itself.

    Everything here is a fact about one band's corpus that no amount of parsing can derive, so
    it is supplied rather than guessed. The CLI builds this from the pack and config; the
    defaults are all empty, which parses honestly and just less precisely.

    ``drop_dates``
        Shows that happened but are not evidence about this band: a tribute night, an all-star
        improv jam, a costume show where the "songs" were bits. Their setlists poison the
        vocabulary for every ordinary show.
    ``date_overrides``
        identifier -> the date the show actually happened. An uploader can type anything, and a
        well-formed lie is indistinguishable from the truth to every parser downstream.
    ``band_filter``
        Decides whether an item is this band at all. Side projects land in the same collection.
        See :func:`title_band_filter`.
    ``junk_patterns``
        Extra regex fragments for the credit/annotation filter -- cover artists, member
        surnames. Each is grouped and bounded by non-word lookarounds, so a fragment may
        contain its own alternation and may begin or end with punctuation.
    ``gear_patterns``
        Extra regex fragments for the lineage filter, in the same shape -- the tape-utility
        shorthand one scene uses and the next has never heard of.
    ``band_name``
        Used to recognise the press-bio and credit block the band's own name introduces
        ("About moe", "thanks to moe", "moe.:").
    """

    drop_dates: frozenset[str] = frozenset()
    # hash=False because a dict is not hashable and frozen=True synthesises __hash__ from every
    # comparable field. Equality still reads it; only hashing skips it.
    date_overrides: Mapping[str, str] = field(default_factory=dict, hash=False)
    band_filter: Callable[[Mapping[str, Any]], bool] | None = None
    junk_patterns: tuple[str, ...] = ()
    gear_patterns: tuple[str, ...] = ()
    band_name: str | None = None


@dataclass(frozen=True)
class _Rules:
    """Policy resolved into the compiled shape the parse functions actually use.

    Built once per run so the pack's patterns are compiled once rather than per item. ``vocab``
    is the normalizer's normalized-key -> canonical-name map, and only membership is read from
    it: the question being asked is "does the band play anything by this name".
    """

    normalizer: Normalizer
    # Filled as the parse runs. On _Rules rather than threaded through four signatures, and
    # always collected rather than switched on by a flag: a census gathered only in "lint mode"
    # is a census that is empty exactly when someone forgot, and an empty one reports every rule
    # in the pack as dead.
    census: Census
    vocab: Mapping[str, str]
    # The pack's normalized-spelling -> canonical map, read for membership only. An alias key
    # is the pack declaring "this taper spelling IS that song", which makes it exactly as much
    # a real title as a vocabulary entry -- and it has to be resolved here rather than left to
    # canonicalize(), because every DROP rule runs BEFORE canonicalization.
    aliases: Mapping[str, str]
    junk: re.Pattern
    gear: re.Pattern
    credit_tail: re.Pattern


def fragment_pattern(*fragments: str) -> re.Pattern:
    r"""One or more filter fragments as the filter actually applies them.

    Each fragment is grouped on its own, and the edges are lookarounds rather than ``\b``. A
    ``\b`` demands a word character on the INSIDE of the match, so a perfectly reasonable pack
    fragment like ``\(cover\)`` compiles clean and then matches nothing at all -- which is the
    worst way for a pack to be wrong, because there is nothing to see.

    Public because the pack loader compiles each fragment through it too. A lint check that
    approximates the filter instead of running it can be wrong in both directions, and the
    check exists to be believed.
    """
    body = "|".join(f"(?:{fragment})" for fragment in fragments)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.I)


@functools.lru_cache(maxsize=32)
def _junk_pattern(extra: tuple[str, ...]) -> re.Pattern:
    """The credit/annotation filter, with the pack's fragments folded into the alternation."""
    return fragment_pattern(_JUNK_BASE, *extra)


@functools.lru_cache(maxsize=32)
def _gear_pattern(extra: tuple[str, ...]) -> re.Pattern:
    """The lineage filter, with the pack's gear shorthand folded into the alternation.

    ``_gear_pattern(())`` -- no pack, the gear every taper writes and nothing else -- is a
    legal and useful call, but it is not kept as a module constant. One built filter per run,
    reached the same way whether or not a pack had anything to add to it.
    """
    return fragment_pattern(_GEAR_BASE, *extra)


@functools.lru_cache(maxsize=32)
def _credit_tail_pattern(band_name: str | None) -> re.Pattern:
    """Where the setlist ends and the credit roll begins.

    The band's own name is the reliable tell -- a press bio always introduces itself -- so when
    we are told the name we add those three shapes to the generic ones.
    """
    alternatives = list(_CREDIT_TAIL_BASE)
    stated = (band_name or "").strip()
    core = stated.rstrip(".")
    if core:
        # (?!\w) rather than \b, which cannot fire at the end of a line when the name ends in
        # punctuation. "About !!!" and "About Sunn O)))" both sail past a \b and take the whole
        # press bio into the setlist with them.
        alternatives += [rf"about\s+{re.escape(core)}(?!\w)",
                         rf"thanks\s+to\s+{re.escape(core)}(?!\w)",
                         # [^\S\n], not [ \t] -- see the note on _CREDIT_TAIL_BASE. A pasted
                         # description puts a non-breaking space after the colon and this never
                         # fired for it.
                         rf"{re.escape(stated)}\s*:[^\S\n]*$"]
    return re.compile(r"^[ \t]*(?:" + "|".join(alternatives) + r")", re.I | re.M)


def _rules_for(normalizer: Normalizer, policy: ArchivePolicy) -> _Rules:
    """Compile ``policy`` against ``normalizer`` once."""
    _, norm_to_canon = normalizer.build_vocab()
    return _Rules(normalizer=normalizer,
                  census=Census(),
                  vocab=norm_to_canon,
                  aliases=normalizer.aliases(),
                  junk=_junk_pattern(tuple(policy.junk_patterns)),
                  gear=_gear_pattern(tuple(policy.gear_patterns)),
                  credit_tail=_credit_tail_pattern(policy.band_name))


def title_band_filter(band_name: str) -> Callable[[Mapping[str, Any]], bool]:
    """Accept only items whose title names ``band_name`` -- or names no band at all.

    Members of a band play in other projects, and those tapes land in the same collection. Five
    items in one corpus were a Dylan covers band with two of the members in it, and its eleven
    Dylan songs went into the vocabulary as things the band might play next.

    We should not try to infer the band from the setlist, and we do not have to: archive.org
    puts it in the title of every item. Only a title that names a DIFFERENT band is rejected --
    an unrecognised title shape is left alone, because being unable to read a title is not
    evidence that a show is fake.
    """
    wanted = squash(band_name)

    def _filter(item: Mapping[str, Any]) -> bool:
        match = BAND_IN_TITLE.match(str(item.get("title") or ""))
        if not match:
            return True
        # startswith, not equality: plenty of tapers write "moe. 2026-06-14 Live at ..." and the
        # date is not part of the band's name.
        return squash(match.group(1)).startswith(wanted)

    return _filter


def clean_html(value: object) -> str:
    """Flatten an item's description field to plain text with the line breaks preserved.

    Unescaped twice because archive.org descriptions are routinely double-encoded.
    """
    if isinstance(value, list):
        value = " ".join(str(part) for part in value)
    text = html.unescape(html.unescape(str(value or "")))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p>|</div>|<li>", "\n", text, flags=re.I)
    return re.sub(r"<[^>]+>", " ", text)


def _claimed(token: str, rules: _Rules) -> bool:
    """Does the pack say this token IS a song? The one guard every DROP rule defers to.

    A whole cleaned token that the vocabulary or ``protected_titles`` recognises is a song the
    band is known to play, and no rule in this module is allowed to delete one. That principle
    was already stated twice -- on shape, and on venue words -- and applied in only those two
    places, so the annotation, credit and crew-role filters could each still delete a real
    title. "Notes", "Lighting" and "Total Time" are all plausible song names, and a pack had no
    way to defend one: ``protected.json`` is consulted here and in ``is_non_song``, and both ran
    BEFORE the filters that did the deleting.

    Note how narrow this is. It fires only when the ENTIRE token normalizes onto a known song,
    so it cannot rescue "Al Schnier - guitar, vocals" or "Umphrey's McGee" -- neither of which
    is a title. It costs a false keep (a taper's literal "Notes", for a band that plays a song
    called Notes) and buys back a class of silent, untraceable song loss. That trade is the one
    this catalog was rebuilt to make.

    The aliases are read here and not left to ``canonicalize``, which runs later: an alias key
    is the pack saying "this spelling IS that song", so a rule that deletes one deletes the song
    just as surely as if it had matched the canonical name.
    """
    norm = rules.normalizer.normalize(token)
    return norm in rules.vocab or norm in rules.aliases or rules.normalizer.is_protected(token)


def _looks_songlike(token: str, rules: _Rules) -> bool:
    """Shape gate: right length, no gear vocabulary, mostly letters.

    The gear filter is the widest in this module and it cannot tell a song from a tape utility.
    "Wave", "Matrix", "Source", "Edit", "Gain" and "Reaper" are all gear words, and every one of
    them is a perfectly plausible song title -- so callers ask :func:`_claimed` first, which is
    what keeps the pack's own gear words from deleting the pack's own songs.

    Deliberately does NOT ask whether the token is banter or a guest credit. That question has
    exactly one answer and it lives in ``Normalizer.is_non_song`` -- see the module docstring.
    """
    words = token.split()
    if not 1 <= len(words) <= 8:
        return False
    if rules.gear.search(token):
        return False
    letters = sum(char.isalpha() for char in token)
    return letters >= max(2, 0.5 * len(token.replace(" ", "")))


def _emit_from_token(token: str, rules: _Rules, songs: list[dict]) -> None:
    """Split a raw token on internal segues, clean each piece, and append it to ``songs``.

    A piece followed by a segue marker is flagged ``segue``. A piece the normalizer says is not
    music is flagged ``non_song`` and kept -- the tape said it happened, so the corpus records
    that it happened.
    """
    pieces = SEG_SPLIT.split(token)
    for index in range(0, len(pieces), 2):
        piece = E_PREFIX.sub("", pieces[index])          # "(E) The Faker" -> "The Faker"
        piece = FOOT.sub("", piece).strip(" .,-:")
        seg_after = index + 1 < len(pieces)              # a segue marker followed this piece
        # trailing '&' footnote, while the interior one in "Hi & Lo" stays. A piece that IS an
        # "& Someone" sit-in note survives this and goes on to be tagged: it is the same
        # annotation as "w/ Someone", so it gets the same answer from the same place.
        piece = re.sub(r"\s*&+\s*$", "", piece).strip(" .,-:")
        if not piece:
            continue
        # Recorded before anything can remove it. This is the only place the raw token exists;
        # every branch below either drops it or rewrites it into a canonical name.
        rules.census.seen[piece] += 1
        # One guard, ahead of every rule that DROPS. Asked once and reused, so a title the pack
        # claims cannot be deleted by the shape gate, the gear words, the annotation filter, a
        # credit line or a crew role -- previously only the first two deferred to it.
        claimed = _claimed(piece, rules)
        if not claimed:
            if not _looks_songlike(piece, rules):
                rules.census.dropped[piece] += 1
                continue
            if CREDIT.search(piece) or CREW_ROLE.search(piece):
                rules.census.dropped[piece] += 1
                continue
            if rules.junk.search(piece) or URLISH.search(piece):
                rules.census.dropped[piece] += 1
                continue
        canon, canon_seg = rules.normalizer.canonicalize(piece)
        if not canon:
            rules.census.dropped[piece] += 1
            continue
        non_song = rules.normalizer.is_non_song(canon)
        # An unknown title we are keeping AS A SONG has to look like one: short, no digits.
        # Non-songs skip this gate because they are already labelled for what they are.
        if not non_song and rules.normalizer.normalize(canon) not in rules.vocab:
            if len(canon.split()) > 5 or re.search(r"\d", canon):
                rules.census.dropped[piece] += 1
                continue
        if non_song:
            # Against the RAW piece, not the canonical name: a classifier is matched against the
            # squashed entry, and the census is what lint holds its patterns against.
            rules.census.tagged[piece] += 1
        songs.append({"song": canon, "segue": bool(seg_after or canon_seg), "non_song": non_song})


def _split_songs(chunk: str) -> list[str]:
    """One chunk of setlist text into one token per song, whatever the taper's layout."""
    parts = re.split(r"\n|\s{2,}", chunk)
    parts = [part.strip(" .,-") for part in parts if part.strip(" .,-")]
    # collapsed to one big blob -- try single '>' / ',' splits instead
    if len(parts) <= 2 and (">" in chunk or "," in chunk):
        parts = re.split(r"\s*>\s*|,", chunk)
        parts = [part.strip(" .,-") for part in parts if part.strip(" .,-")]
    return parts


def _place_terms(item: Mapping[str, Any], normalizer: Normalizer) -> set[str]:
    """Every way this item names its own venue and city, normalized.

    archive.org hands us `venue` ("Northlands") and `coverage` ("Swanzey, NH") on the item
    itself. The taper writes those same strings at the top of their description, and when the
    header regex fails to eat them -- it wanted "City, ST" with a comma, and this one wrote
    "Swanzey NH" without -- they become SONGS.

    Tightening the header regex is the same arms race we lost on filenames. We do not have to
    guess at the venue: the item tells us what it is. So we take it at its word and refuse to
    let it be a song. Nothing here can hide a real song, because no band has a song named after
    the room they are standing in.
    """
    terms = set()
    for key in ("venue", "coverage"):
        raw = str(item.get(key) or "").strip()
        if not raw:
            continue
        terms.add(normalizer.normalize(raw))            # "Swanzey, NH" -> "swanzey nh"
        for part in raw.split(","):                     # ...and each half on its own
            part = normalizer.normalize(part)
            if len(part) > 2:
                terms.add(part)
    return {term for term in terms if term}


def _is_place(song: str, places: set[str], rules: _Rules) -> bool:
    """Is this 'song' actually a room?

    Two signals: it overlaps the venue or city the item itself declares, or it contains a word
    that belongs to buildings rather than to songs.

    The vocabulary guard is what makes this safe. Anything the band is known to play is never
    dropped, no matter what it is called -- so a real song could be named "The Ballroom" and
    survive. What cannot survive is a string that is both unknown to the repertoire AND shaped
    like an address.
    """
    norm = rules.normalizer.normalize(song)
    if not norm or _claimed(song, rules):
        return False
    if norm in places:
        return True
    tokens = set(norm.split())
    for place in places:               # "the peach music festival" vs venue "peach festival"
        parts = set(place.split())
        if len(parts) >= 2 and parts <= tokens:
            return True
    return bool(VENUE_WORDS.search(song))


def _cut_tails(text: str, rules: _Rules) -> str:
    """Drop the machine output, the credit roll, and the venue header from a description."""
    for tail in (CHECKSUM_TAIL, rules.credit_tail):
        cut = tail.search(text)
        if cut:
            text = text[:cut.start()]
    # Strip a leading "BAND DATE Venue City, ST" header so the venue and city don't leak in as
    # songs on descriptions with no set or track markers. Bounded to the first ~160 characters
    # and only fires when it lands on a "City, ST" pattern, so it cannot eat into a setlist.
    header = VENUE_HEADER.match(text)
    return text[header.end():] if header else text


def _setlist_region(text: str) -> str:
    """Where the setlist starts, past the taper's gear preamble.

    Prefer an explicit set header. Failing that, the first numbered track, so an implicit
    "01. .. 02. .." set is not lost. Only if neither exists do we parse the whole text -- songs
    written as a bare "A -> B -> C" run. The preamble is dropped by the filters downstream,
    and an "Enc:" marker still splits the encore, whereas starting the region AT that marker
    would swallow the entire main set.
    """
    header = SETHEADER.search(text)
    if header:
        return text[header.start():]
    numbered = FIRST_NUMTRACK.search(text)
    return text[numbered.start():] if numbered else text


def _segments(region: str) -> list[tuple[str, str]]:
    """Split the setlist region into (marker, chunk) pairs.

    Text BEFORE the first marker is the implicit opening set. The old version discarded it,
    which dropped whole main sets from tapers who numbered the songs and then wrote only an
    "Enc:" marker with no "Set 1" header above it.
    """
    marks = list(SETMARK.finditer(region))
    if not marks:
        return [("set 1", region)]
    segments = []
    head = region[:marks[0].start()]
    if head.strip():
        segments.append(("set 1", head))
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(region)
        segments.append((mark.group(0), region[mark.end():end]))
    return segments


def _parse_description(desc: object, rules: _Rules,
                       places: Iterable[str] = ()) -> tuple[list[list[dict]], list[dict]]:
    """A taper's description into (sets, encore)."""
    text = _cut_tails(clean_html(desc), rules)
    sets: list[list[dict]] = []
    encore: list[dict] = []
    for label, chunk in _segments(_setlist_region(text)):
        songs: list[dict] = []
        for token in _split_songs(chunk):
            # Numbering comes off FIRST. It is the only thing separating "02. w/ Andy Frasco",
            # which is entirely a guest note, from "Meat w/ Jake", which is a song with one
            # appended. Splitting on "w/" first truncated the former to "02." and threw the
            # entry away before anything could ask what it was -- a third accidental answer to
            # a question that is supposed to have exactly one.
            token = PREFIX.sub("", token)                         # leading 01 / d1t05. / t3
            token = re.split(r"\s+w/", token)[0]                  # trailing "w/ guest..." note
            token = re.sub(r"\([^)]*\d:\d\d[^)]*\)", "", token)   # (7:43) durations
            token = token.strip(" .,-:")
            if token:
                _emit_from_token(token, rules, songs)
        if not songs:
            continue
        if ENCMARK.search(label):
            encore.extend(songs)
        else:
            sets.append(songs)
    known = set(places)
    sets = [[entry for entry in one_set if not _is_place(entry["song"], known, rules)]
            for one_set in sets]
    return ([one_set for one_set in sets if one_set],
            [entry for entry in encore if not _is_place(entry["song"], known, rules)])


def _parse_tracks(tracks: Iterable[Mapping[str, Any]] | None,
                  rules: _Rules) -> tuple[list[list[dict]], list[dict]]:
    """The item's tracklist into (sets, encore) -- the fallback when the description is prose.

    A tracklist has no set structure to read, so everything lands in one set. Deduped by
    normalized name because the same show is often uploaded in two formats or split by disc.
    """
    raw: list[dict] = []
    for track in tracks or []:
        title = str(track.get("title") or "").strip()
        if not title:
            continue
        _emit_from_token(PREFIX.sub("", title), rules, raw)
    seen = set()
    songs = []
    for entry in raw:
        key = rules.normalizer.normalize(entry["song"])
        if key in seen:
            continue
        seen.add(key)
        songs.append(entry)
    return ([songs], []) if songs else ([], [])


def _show_date(item: Mapping[str, Any], overrides: Mapping[str, str]) -> str:
    """The date the show actually happened -- which is not always the date on the item.

    An uploader can type any date they like. One item is dated 2024, describes itself as June
    2025, ships 2025 artwork, and was uploaded in July 2025; no such show exists on the date it
    claims. Believing its metadata invented a show that never happened AND buried half of a real
    one. Corrections come from the pack, each one carrying the evidence that earned it.
    """
    stated = str(item.get("meta_date") or item.get("date") or "")[:10]
    date = overrides.get(str(item.get("identifier") or ""), stated)
    # \Z, not the default re.match prefix behaviour and not $. An override is not sliced to ten
    # characters the way `stated` is, so a trailing newline or a time component would otherwise
    # sail through and become the show's date -- and `$` would have let the newline through too.
    return date if _DATE_SHAPE.match(date) else ""


def count_songs(sets: Iterable[Iterable[Mapping[str, Any]]],
                encore: Iterable[Mapping[str, Any]]) -> int:
    """How many actual SONGS were played, ignoring the tagged non-songs.

    Tuning and banter are recorded but they are not repertoire, and this number is what
    downstream completeness checks compare shows by.
    """
    tally = sum(1 for one_set in sets for entry in one_set if not entry.get("non_song"))
    return tally + sum(1 for entry in encore if not entry.get("non_song"))


def _parse_item(item: Mapping[str, Any], rules: _Rules,
                policy: ArchivePolicy) -> dict | Skipped:
    """One item into a show record, or a :class:`Skipped` saying which rule refused it."""
    identifier = str(item.get("identifier") or "")
    if policy.band_filter is not None and not policy.band_filter(item):
        return Skipped(identifier, NOT_THIS_BAND)
    date = _show_date(item, policy.date_overrides)
    if not date:
        return Skipped(identifier, NO_DATE)
    if date in policy.drop_dates:
        return Skipped(identifier, DROPPED_DATE, date)
    sets, encore = _parse_description(item.get("description", ""), rules,
                                      places=_place_terms(item, rules.normalizer))
    source = "description"
    if count_songs(sets, encore) < _WEAK_PARSE:
        track_sets, track_encore = _parse_tracks(item.get("tracks"), rules)
        if count_songs(track_sets, track_encore) > count_songs(sets, encore):
            sets, encore, source = track_sets, track_encore, "tracks"
    return {"date": date, "year": date[:4], "sets": sets, "encore": encore,
            "n_songs": count_songs(sets, encore), "source": source,
            "identifier": str(item.get("identifier") or "")}


def parse_archive_item(item: Mapping[str, Any], *, normalizer: Normalizer,
                       policy: ArchivePolicy | None = None) -> dict | None:
    """Parse a single archive item. None when the policy rules it out or it carries no date.

    Use :func:`parse_archive_items` when you need to know WHY an item was refused; this one
    collapses all three reasons back to a bare ``None``.
    """
    policy = policy or ArchivePolicy()
    record = _parse_item(item, _rules_for(normalizer, policy), policy)
    return None if isinstance(record, Skipped) else record


def parse_archive_items(items: Iterable[Mapping[str, Any]], *, normalizer: Normalizer,
                        policy: ArchivePolicy | None = None) -> ParseResult:
    """Parse every item this source has, and report every item it refused.

    A show can be taped four times over, and ALL FOUR come back. This used to keep only the
    richest parse per date, which was a second answer to a question
    :func:`~setlistkit.catalog.merge.pick_show` already owns -- and the two answered it
    differently, breaking a tie toward the lowest identifier here and the highest there. Worse,
    the tapes discarded here never reached the merge, so ``override_disagreements`` could not see
    them: a losing tape carrying a song an override lacks is exactly the signal that review
    exists to raise, and it was being thrown away one layer too early.

    Sorted by (date, identifier) so two runs over the same items agree whatever order the source
    returned them in, and so does everything computed downstream.

    ``skipped`` is why this returns a result object rather than a list. An item that is refused
    leaves no trace anywhere downstream: the date is simply not in the corpus, and a night that
    is missing looks exactly like a night nobody taped. Three different rules can do that -- the
    band filter, an unreadable date, and a date the pack refuses -- and being able to say which
    one, for which identifier, is the difference between "this show is absent" and "this show is
    absent BECAUSE".
    """
    policy = policy or ArchivePolicy()
    rules = _rules_for(normalizer, policy)
    shows: list[dict] = []
    skipped: list[Skipped] = []
    for item in items:
        record = _parse_item(item, rules, policy)
        if isinstance(record, Skipped):
            skipped.append(record)
        else:
            shows.append(record)
    return ParseResult(shows=sorted(shows, key=lambda rec: (rec["date"], rec["identifier"])),
                       skipped=tuple(sorted(skipped, key=lambda s: s.identifier)),
                       census=rules.census)

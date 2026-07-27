# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Bootstrap a band pack from a bag of raw track titles, and from any curated titles alongside them.

A fresh pull of a band's tapes hands back thousands of ``files[].title`` strings, and a first
vocabulary has to be carved out of them by hand. This turns that bag into a much shorter, clustered
worklist a person with band knowledge can trim into a real pack -- it is a REVIEW AID, not a
classifier. It never decides what a song IS; it groups the near-duplicates so a human decides once
per group instead of once per string.

WHERE THE TITLES COME FROM MATTERS, and for a long time this module acted as though it did not.
The mirror's titles are what a taper typed into a filename; a setlist source publishes the names
the band uses. Both are clustered here, but only the second is allowed to NAME a cluster -- see
``curated`` in :func:`suggest`. Drafting from tapes alone had a specific, measurable cost: on UM it
left 226 songs out of the vocabulary entirely and welded track numbers onto several that made it in
(``14 Sad Clint Eastwood``), because the electing rules can only ever guess which spelling is real
from the spellings themselves, and tapers agreeing with each other is not the band agreeing.

Two properties follow from "review aid, not classifier", and both are essential:

* **Nothing is silently dropped.** A title that looks like banter is *segregated* into its own
  section, not deleted, and every raw variant that folds into a cluster is listed beside the
  candidate. Over-merging is cheap here -- the human sees the members and splits them back -- but a
  real song vanishing without a trace is not recoverable from the output. (This is the pack's DATA
  DOCTRINE applied to bootstrapping: keep-and-flag, count the losses.)

* **The aggressive cleanup only feeds the merge, never the label.** A bare leading number is
  stripped to compute a clustering key, because "10 All In Time" and "All In Time" are one song --
  but the displayed candidate is chosen from the raw members, so "25 or 6 to 4" is never shown as
  "or 6 to 4".

The heavy lifting is the band-agnostic mechanism in :mod:`setlistkit.catalog.normalizer`:
``normalize`` for the canonical key, ``squash`` for the fuzzy-similarity form, ``strip_segue`` for
trailing segue tokens, and ``Normalizer.is_non_song`` for the banter shapes. What lives here is the
part that machinery never had: the archive.org FLAC-title noise (leading track numbers, taper
sigils, ``(E)`` encore flags) that the setlist-description parser used to strip, and a greedy
within-set clustering step -- ``synonym_map`` resolves unknowns against a *known* vocabulary, and
there is no vocabulary yet, so it cannot cluster a bag of unknowns against itself.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .normalizer import Normalizer, normalize, squash, strip_attribution, strip_segue

# -- preclean: the archive.org FLAC-title noise the normalizer never sees -----------------
#
# On the setlist path these came off in the description parser (see parse.py); a title pulled
# straight from files[].title skips that parser, so the noise arrives welded to the name.

# "(E) Africa", "(E) Helplessly Hoping" -- an encore flag on the front. The core normalizer knows
# the dotted "e." / "encore:" form but not this parenthesised one.
_P_ENCORE_PAREN = re.compile(r"^\(E\)\s*", re.I)

# "#@+$Ophelia", "*Der Bluten Kat", "^Under Pressure", "#Jack Straw" -- a taper's private notation
# stacked in front of the title. None of these characters is ever part of a song name.
_P_SIGIL_LEAD = re.compile(r"^[#*^+$@%~!]+\s*")
# The same stray notation trailing a title: "Wappy Sprayberry#", "Glory*".
_P_SIGIL_TRAIL = re.compile(r"[#*^+$@%~]+$")

# An UNAMBIGUOUS track index: a leading zero ("01 1348", "08 Den") or a 1-2 digit number carrying a
# dot/paren ("02. Domino Theory", "10) Foo"). This is how a ripper numbers a disc; no song title is
# spelled this way, so it is safe to strip even for display.
_P_TRACK_SURE = re.compile(r"^(?:0\d|\d{1,2}[.)])\s+(?=\S)")
# A BARE 1-2 digit lead ("10 All In Time", "20 The Triple Wide"). AMBIGUOUS: a real song can open
# with a number ("25 or 6 to 4", "32 Things"). Stripped only into the clustering KEY, never trusted
# as a display label. Restricted to 1-2 digits so four-digit titles ("1348", "1999") are untouched.
_P_TRACK_BARE = re.compile(r"^\d{1,2}\s+(?=\S)")

# A setlist entry that is nothing but a parenthesised note the parser promoted to a "song":
# "(Andy Farag Intro)", "(Outro)", "(Aron Speech)". is_non_song already catches the short-token
# BARE_NOTE form; this catches the long prose ones it was written to leave alone.
_P_PARENS_ONLY = re.compile(r"^\(.*\)$", re.S)

# Pairs whose squashed lengths differ by more than this can't clear the similarity cutoff, so the
# quadratic scan skips them. 4 measured against the real UM data: no true fold spans a wider gap.
_LEN_GUARD = 4


def preclean_display(raw: str) -> str:
    """A human-facing cleaned title: sigils, ``(E)``, sure track numbers, cover/encore/timecode
    annotations and trailing segue/sigils removed, but a BARE leading number LEFT IN -- it might be
    part of the name.

    ``strip_attribution`` is the core's own display cleanup: it takes the ``Encore:`` off
    "Encore: Cemetery Walk", the ``[Blind Melon]`` off a cover, the ``(09:39)`` timecode off an
    indexed listing. Reusing it keeps the label a human reads consistent with the key the machine
    files under, since ``normalize`` runs the same strip."""
    text = _P_ENCORE_PAREN.sub("", raw)
    text = _P_SIGIL_LEAD.sub("", text)
    text = _P_TRACK_SURE.sub("", text)
    text = strip_attribution(text)
    text, _ = strip_segue(text)
    text = _P_SIGIL_TRAIL.sub("", text).strip()
    return text.strip()


def preclean_key(raw: str) -> str:
    """The clustering-key form: :func:`preclean_display` plus the bare leading number dropped, so
    "10 All In Time" keys with "All In Time". Aggressive on purpose -- this only feeds the merge."""
    return _P_TRACK_BARE.sub("", preclean_display(raw)).strip()


@dataclass(frozen=True)
class Cluster:
    """One candidate song: the chosen display label and every raw title that folded into it."""

    canonical: str
    members: tuple[str, ...]
    #: True when the label came from a curated source rather than being elected from tape spellings.
    curated: bool = False
    #: True when at least one member came from the raw bag -- i.e. a tape spelled this song too.
    #: Tracked rather than inferred from ``size``, because when both sources spell a title
    #: identically the members dedupe to one string and a size test would call it tape-less.
    on_tape: bool = True

    @property
    def size(self) -> int:
        """How many raw titles folded into this candidate."""
        return len(self.members)


@dataclass(frozen=True)
class Suggestion:
    """The whole worklist: clustered song candidates, segregated non-songs, and the raw count."""

    songs: tuple[Cluster, ...]
    non_songs: tuple[str, ...]
    raw_count: int

    @property
    def curated(self) -> tuple[Cluster, ...]:
        """The candidates whose label a curated source supplied. See :func:`suggest`."""
        return tuple(c for c in self.songs if c.curated)

    @property
    def backfilled(self) -> tuple[Cluster, ...]:
        """Curated candidates NO tape ever spelled -- songs the mirror alone would have missed.

        Worth counting on its own: these are not a cleaner spelling of something already known,
        they are repertoire that existed only in the structured source.
        """
        return tuple(c for c in self.songs if c.curated and not c.on_tape)

    def vocabulary(self) -> list[str]:
        """The draft ``vocabulary.json``: canonical candidates, sorted, deduped."""
        return sorted({c.canonical for c in self.songs}, key=str.casefold)

    def aliases(self) -> dict[str, str]:
        """The draft ``aliases.json``: variant normalized-key -> canonical display.

        Only the fuzzy folds earn an entry. Within a single normalized family every spelling
        already collapses under ``normalize`` at runtime, so it needs no alias; an alias is only
        useful where two DIFFERENT normalized keys were merged (``cemetary walk`` vs the canonical
        ``Cemetery Walk``). The canonical's own key is excluded so a cluster never aliases a name
        to itself.
        """
        out: dict[str, str] = {}
        for cluster in self.songs:
            canon_key = normalize(preclean_key(cluster.canonical))
            for member in cluster.members:
                key = normalize(preclean_key(member))
                if key and key != canon_key:
                    out[key] = cluster.canonical
        return dict(sorted(out.items()))


def _pick_label(members: list[str], curated: frozenset[str] = frozenset()) -> str:
    """The cleanest, most representative display form to offer a human as the candidate name.

    Prefers, in order:

    * a CURATED member -- a name from a source that publishes song titles rather than one typed
      onto a tape. This outranks every heuristic below because those heuristics are all guesses at
      "which of these spellings is the real one" from the spellings alone, and a curated source
      answers that question directly. Without it the medoid rule elects the MAJORITY spelling,
      which is the wrong answer whenever tapers agree with each other and not with the band: 9
      tapes spelling ``Ja Junk`` would outvote setlist.fm's ``JaJunk``, and 3 spelling
      ``14 Sad Clint Eastwood`` would keep the track number. Ties inside the curated set fall
      through to the same rules as before.
    * a form whose display carries no stripped-number artifact -- "1348" over "11 1348", because
      the bare "11 " is a track index the key dropped but the display kept. A clean sibling wins;
      when every member is number-led (the "25 or 6 to 4" case) this dimension ties and the number
      stays, which is the honest answer.
    * a mixed-case form over ALL-CAPS or all-lower ("Fenced In" over "FENCED IN").
    * the MEDOID -- the spelling most similar to the rest of the cluster. This is what down-weights
      a lone typo: when nine members squash to ``bridgeless`` and one to ``bridgelesss``, the common
      spelling has the higher summed similarity and wins. A plain "prefer more letters" rule did the
      opposite, electing the doubled-letter typo as canonical.
    * alphabetical, as a deterministic final tiebreak independent of set iteration order.
    """
    displays = {m: (preclean_display(m) or m) for m in members}
    squashed = {m: squash(displays[m]) for m in members}

    def numbered(raw: str) -> bool:
        # The display kept a bare leading number that the clustering key stripped -- a track index,
        # not part of the name.
        return normalize(preclean_display(raw)) != normalize(preclean_key(raw))

    def centrality(raw: str) -> float:
        return sum(difflib.SequenceMatcher(None, squashed[raw], squashed[other]).ratio()
                   for other in members if other != raw)

    def score(raw: str) -> tuple:
        disp = displays[raw]
        letters = [c for c in disp if c.isalpha()]
        mixed = any(c.islower() for c in letters) and any(c.isupper() for c in letters)
        return (raw not in curated, numbered(raw), not mixed, -centrality(raw), disp.casefold(),
                disp)

    return displays[min(members, key=score)]


def _cluster(keys: list[str], cutoff: float,
             anchored: frozenset[str] = frozenset()) -> dict[str, list[str]]:
    """Greedy union-find over normalized family keys, merging on squashed difflib similarity.

    ``anchored`` are the keys of curated names, and two anchored keys NEVER merge into each other.
    The similarity test cannot tell a typo from a sequel, and on this data the difference is huge:
    ``Cemetery Walk``, ``Cemetery Walk I`` and ``Cemetery Walk II`` are three separate UM pieces
    carrying 55, 59 and 41 plays, and they squash to strings 0.96 similar -- above any cutoff loose
    enough to still fold ``cemetary``. Same for ``Soul Food I``/``II``, ``Rebubula``/``Rebubula I``
    and ``Another Brick in the Wall`` parts 1 and 2.

    The asymmetry is the argument, not a special case: clustering exists to fold a TAPER'S noise
    into a real name. When a source that publishes titles lists two names, it has already made that
    distinction, and re-deciding it from spelling alone can only lose information a human would
    then have to notice was missing.
    """
    parent = {k: k for k in keys}
    holds_anchor = {k: k in anchored for k in keys}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:            # path compression, so repeated finds stay cheap
            parent[x], x = root, parent[x]
        return root

    def union(x: str, y: str) -> None:
        """Merge x's group into y's, unless both already speak for a curated name."""
        root_x, root_y = find(x), find(y)
        if root_x == root_y or (holds_anchor[root_x] and holds_anchor[root_y]):
            return
        parent[root_x] = root_y
        holds_anchor[root_y] = holds_anchor[root_x] or holds_anchor[root_y]

    sq = {k: squash(k) for k in keys}
    for i, a in enumerate(keys):
        sa = sq[a]
        for b in keys[i + 1:]:
            sb = sq[b]
            if abs(len(sa) - len(sb)) > _LEN_GUARD:
                continue
            if difflib.SequenceMatcher(None, sa, sb).ratio() >= cutoff:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    return groups


def suggest(titles: list[str], cutoff: float = 0.88,
            curated: Iterable[str] = ()) -> Suggestion:
    """Turn a bag of raw titles into a clustered, human-trimmable worklist.

    ``cutoff`` is the difflib ratio at which two normalized families merge; higher is more
    conservative. The default 0.88 was tuned on real UM data to fold obvious variants
    (cemetary/cemetery, believe/belive) without pulling in unrelated titles.

    ``curated`` is song names from a source that PUBLISHES titles -- setlist.fm, phish.net -- as
    opposed to the mirror's ``titles``, which are what a taper typed into a filename. They are
    added to the same bag and clustered by the same rules, so a taper's spelling still folds in and
    still earns an alias. What changes is who gets to name the cluster: a curated member wins the
    canonical slot outright (see :func:`_pick_label`).

    This is why the parameter exists at all rather than the caller just concatenating the two
    lists. Concatenated, a curated name is one vote among many and loses every majority; the whole
    value of the source is that it is not a vote.

    A curated name no tape ever carried still becomes a candidate, which is the second half of the
    point: it BACKFILLS repertoire the mirror never recorded. On UM that is 226 titles the tapes
    alone never produced.

    Two curated names never fold into EACH OTHER, at any cutoff -- see :func:`_cluster`. Without
    that rule the seeded UM draft merged ``Cemetery Walk``, ``Cemetery Walk I`` and ``Cemetery
    Walk II`` into one entry and took 155 plays of three separate pieces with them.

    Nothing here trusts the curated source blindly. A curated name still goes through
    ``is_non_song`` and the empty-key check, so the annotations a structured source also carries
    (a quoted improv name, a spoken-moment note) segregate exactly as a taper's would -- see
    :attr:`Suggestion.non_songs`. A source being better is not the same as a source being clean.
    """
    from_tapes = set(titles)
    curated = frozenset(curated)
    titles = list(titles) + sorted(curated - from_tapes, key=str.casefold)

    nz = Normalizer(vocabulary=[])
    song_raw: list[str] = []
    non_songs: list[str] = []
    for title in titles:
        entry = title.strip()
        if not entry:
            continue
        if nz.is_non_song(entry) or _P_PARENS_ONLY.match(entry):
            non_songs.append(entry)
            continue
        # A title that is nothing BUT noise (all sigils, a bare number) leaves no name once
        # precleaned. Segregate it rather than mint an empty-keyed "song".
        if not normalize(preclean_key(entry)):
            non_songs.append(entry)
            continue
        song_raw.append(entry)

    families: dict[str, list[str]] = {}
    for entry in song_raw:
        families.setdefault(normalize(preclean_key(entry)), []).append(entry)

    anchored = frozenset(normalize(preclean_key(t)) for t in curated) - {""}
    clusters: list[Cluster] = []
    for group_keys in _cluster(sorted(families), cutoff, anchored).values():
        members = sorted({m for key in group_keys for m in families[key]}, key=str.casefold)
        label = _pick_label(members, curated)
        clusters.append(Cluster(canonical=label, members=tuple(members),
                                curated=label in curated,
                                on_tape=any(m in from_tapes for m in members)))

    # Biggest clusters first -- that is where a human's attention pays off -- then alphabetical.
    clusters.sort(key=lambda c: (-c.size, c.canonical.casefold()))
    return Suggestion(
        songs=tuple(clusters),
        non_songs=tuple(sorted(set(non_songs), key=str.casefold)),
        raw_count=len(titles),
    )


def titles_from_cache(items: Iterable[Mapping]) -> list[str]:
    """Every distinct track title in a set of cached archive.org items, sorted.

    The programmatic stand-in for the ``dirty-stripper`` jq filter: it hands :func:`suggest` the
    same bag of raw ``files[].title`` strings, but sourced through the pull cache the pipeline
    already has rather than a hand-run jq over the blob directory. Each item is the flat record
    :func:`setlistkit.sources.archive_org.cached_items` produces, whose ``tracks`` were already
    projected off whichever AUDIO format listed the most files -- so the cover-art / checksum
    sidecars the jq blacklisted by format never reach here at all.

    Deduped and sorted for the same reason the jq ended in ``unique``: :func:`suggest` counts what
    it is handed, and a title that appears on forty tapes is one vocabulary decision, not forty.
    """
    seen: set[str] = set()
    for item in items:
        for track in item.get("tracks") or ():
            title = str(track.get("title") or "").strip()
            if title:
                seen.add(title)
    return sorted(seen, key=str.casefold)


def render_markdown(suggestion: Suggestion) -> str:
    """The ``SUGGESTIONS.md`` audit worklist: every cluster and every segregated non-song."""
    lines = [
        "# Vocabulary suggestions",
        "",
        f"{len(suggestion.songs)} song candidates and {len(suggestion.non_songs)} probable "
        f"non-songs, distilled from {suggestion.raw_count} raw titles.",
        "",
        "This is a worklist, not a finished pack. Delete junk, split any cluster that merged two "
        "real songs, fix the canonical spelling, then run `slkit pack lint`.",
        "",
        "## Song candidates",
        "",
    ]
    for cluster in suggestion.songs:
        if cluster.size == 1:
            lines.append(f"- **{cluster.canonical}**")
        else:
            variants = ", ".join(m for m in cluster.members if m != cluster.canonical)
            lines.append(f"- **{cluster.canonical}** ({cluster.size}) ← {variants}")
    lines += ["", "## Probable non-songs (banter, intros, spoken moments)", ""]
    lines += [f"- {entry}" for entry in suggestion.non_songs]
    return "\n".join(lines) + "\n"

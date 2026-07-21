# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
r"""Pick one setlist per date out of everything every source knows, then let a human overrule it.

This is the ONE place cross-source precedence lives. Each source parser produces its own records
and knows nothing about the others; this module groups them by date and picks exactly one winner
per date with a single global rule.

Why one global pick rather than a chain of pairwise merges: "the highest-ranked source that is
complete enough" cannot be computed two sources at a time. With three candidates of 12, 15 and 20
songs at ranks 3, 2 and 1 and a threshold of 0.75, folding them together pairwise picks the
20-song rank-1 record, while the rule as stated picks the 15-song rank-2 one (12 is under 75% of
20, so it is disqualified as partial, and the best-ranked survivor wins). You have to see every
candidate for a date at once, so they arrive here together.

The rule:

    rank order, BUT a candidate must carry at least ``complete_frac`` of the best-known song
    count for its date to win on rank alone. A thin parse from a good source drops out and a
    fuller record from a worse one takes the date. Empty stubs are dropped before the pick, so
    they can never win -- which is what makes "the best source is final" safe while still
    letting a lesser source cover a date the best one never taped.

It self-corrects: the day a full recording of that date finally lands, it is both complete and
top-ranked, and it reclaims the date along with its segues.

Manual overrides sit downstream of all of it. See the block comment above
:func:`overrides_from_mapping` for why they are whole shows, why they always win, and why the
merge reports its own disagreements with them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType

from .jsonpos import JSONSource
from .normalizer import Normalizer
from .parse import count_songs

# Source rank, highest first. A recording of the show is the ground truth; a structured setlist
# service is tidier but carries no segues and is sometimes partial; a hand-transcribed graphic is
# a last resort. The names are the ``source`` field each parser writes, and packs with different
# sources supply their own map.
# A proxy, not a dict: this is a module global, and a caller who took the default and then
# adjusted one rank would be adjusting it for the whole process.
DEFAULT_RANKS = MappingProxyType({"description": 3, "tracks": 3,   # a tape of the show
                                  "setlistfm": 2,                  # structured, no segues
                                  "instagram": 1})                 # decoded from a picture

# How complete a candidate has to be, against the best count known for its date, before it is
# allowed to win on rank. Below it the candidate is partial and yields to a fuller record from a
# lower-ranked source.
COMPLETE_FRAC = 0.75

# The rank given to a source nobody has ranked. Zero, not "middling": an unranked source can
# still win a date no ranked source carries, but it never outranks one that was configured.
_UNRANKED = 0


@dataclass(frozen=True)
class MergePolicy:
    """How to choose between sources, and which dates to refuse outright.

    ``drop_dates`` is applied to EVERY source, which is the entire reason it lives here as well
    as in the archive parser. Refusing a date in one parser was not enough: the other sources
    carry the same night, so the merge quietly picked one of those copies up instead and the
    show came back. A date is one we want or it is not; which website told us about it does not
    enter into it.
    """

    # hash=False: a dict is unhashable and frozen=True builds __hash__ from every compared field.
    ranks: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_RANKS), hash=False)
    complete_frac: float = COMPLETE_FRAC
    drop_dates: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        # Above 1.0 nothing can ever clear the bar, and the rule silently inverts into "the
        # thinnest top-ranked candidate wins" -- the precise outcome this module exists to
        # prevent. Below 0 it means nothing at all.
        if not 0.0 <= self.complete_frac <= 1.0:
            raise ValueError(f"complete_frac must be between 0 and 1, got {self.complete_frac}")


@dataclass(frozen=True, eq=False)
class MergeResult:
    """The merged corpus, plus what it took to get there.

    ``candidates`` is kept because the caller needs it to report anything useful: which sources
    lost, and what an override is quietly disagreeing with.
    """

    shows: list[dict]
    applied: list[str]                       # dates an override replaced or added
    candidates: dict[str, list[dict]]        # date -> every record that was in the running
    # Overrides that were NOT applied because the date is in drop_dates. Reported rather than
    # dropped: an override is the highest-evidence input this system takes, someone sat and
    # listened to write it, and two files in the same pack disagreeing about one night is a
    # thing to say out loud rather than resolve in silence.
    refused: tuple[str, ...] = ()


def _entries(record: Mapping) -> list[dict]:
    """Every setlist entry in a record, sets then encore, order preserved.

    ``or ()`` rather than a default, because a key that is present and null is exactly what a
    truncated or hand-edited intermediate produces, and a default only covers a missing key.
    """
    return [entry for one_set in record.get("sets") or () for entry in one_set] + \
        list(record.get("encore") or ())


def _n_songs(record: Mapping) -> int:
    """The record's song count, counted from its entries every time.

    Deliberately not read from the record's own ``n_songs`` field. The stub filter below is what
    makes "the best source is final" safe, and a record that states a count nothing backs up
    walks straight through it: an empty record claiming 99 songs clears the stub filter, sets
    the bar every honest candidate is then measured against, disqualifies all of them as
    partial, and enters the corpus with no setlist in it. A count is derived from the songs, so
    it gets derived from the songs.
    """
    return count_songs(record.get("sets") or (), record.get("encore") or ())


def pick_show(candidates: Iterable[Mapping], policy: MergePolicy | None = None) -> dict:
    """Choose one record from every candidate for a single date.

    Ties break on song count and then on identifier, so two runs over the same candidates agree
    regardless of the order the sources were read in. Candidates sharing an identifier are
    genuinely indistinguishable to this rule, and fall back to the order they arrived in.

    Every candidate is expected to have at least one song; ``merge_shows`` drops the stubs
    before it gets here.
    """
    policy = policy or MergePolicy()
    pool = list(candidates)
    if not pool:
        raise ValueError("pick_show needs at least one candidate")
    best = max(_n_songs(record) for record in pool)
    # complete_frac is bounded to 0..1, so whichever record holds `best` always clears its own
    # bar and this list can never come back empty.
    complete = [record for record in pool if _n_songs(record) >= policy.complete_frac * best]
    return max(complete,
               key=lambda record: (policy.ranks.get(record.get("source", ""), _UNRANKED),
                                   _n_songs(record), str(record.get("identifier") or "")))


# What an override record has to carry to BE a show. Not a re-validation of the override file --
# `overrides_from_mapping` does that, and says it far better -- but a contract check on the only
# input to this module that gets to overwrite a real record without arguing for it.
#
# It exists because of a bug that shipped this far: a pack holds two unrelated kinds of override,
# one that moves a show to a different night and one that says what was played, and a single
# rebound local name fed the first kind in here as the second. The result was a "show" made of a
# date and a paragraph of prose, which replaced the genuine record for that night and entered the
# corpus with no setlist, no source and no identifier. Nothing downstream could tell.
_OVERRIDE_KEYS = ("date", "sets", "encore", "source", "identifier")


def apply_overrides(shows: Iterable[Mapping],
                    overrides: Mapping[str, Mapping]) -> tuple[list[dict], list[str]]:
    """Replace (or add) overridden dates in a merged show list. Never looks at a song count.

    That it never counts is the point: an override wins because it was written down by someone
    who listened, not because it argued its way past a threshold. Which is exactly why each one
    is checked for the shape of a show first: this is the one path into the corpus that outranks
    every parser, so something that is not a setlist must not travel down it quietly.

    Raises :class:`ValueError`, not a diagnostic. A malformed override FILE is a user error and
    is reported as one by :func:`overrides_from_mapping`; anything malformed by the time it
    reaches here got past that, which makes it a bug in the caller.
    """
    for date, record in overrides.items():
        missing = [key for key in _OVERRIDE_KEYS if key not in record]
        if missing:
            raise ValueError(
                f"override for {date!r} is not a show record: missing {', '.join(missing)}. "
                f"Overrides come from overrides_from_mapping; corpus.json's date_overrides are a "
                f"different thing and do not belong here.")
        # The mapping key is what gets replaced; record["date"] is what gets STORED. Let them
        # disagree and the override deletes one night and invents another, while `applied`
        # reports a date that ends up in neither. Same failure as the one above, one field in.
        if record["date"] != date:
            raise ValueError(
                f"override filed under {date!r} says its date is {record['date']!r}. It would "
                f"delete {date!r} and create {record['date']!r}.")
    by_date = {show["date"]: deepcopy(dict(show)) for show in shows}
    by_date.update({date: deepcopy(dict(record)) for date, record in overrides.items()})
    merged = sorted(by_date.values(),
                    key=lambda show: (show["date"], str(show.get("identifier") or "")))
    return merged, sorted(overrides)


def merge_shows(records: Iterable[Mapping], *, overrides: Mapping[str, Mapping] | None = None,
                policy: MergePolicy | None = None) -> MergeResult:
    """Every source's records into one show per date.

    Overrides are folded in HERE rather than by the caller, so an override for a date no source
    carries grows the corpus instead of looking like one that shrank.
    """
    policy = policy or MergePolicy()
    candidates: dict[str, list[dict]] = {}
    for record in records:
        date = str(record.get("date") or "")
        if not date or date in policy.drop_dates:
            continue
        # A stub with nothing in it can never win, so it never gets to set the bar that
        # complete_frac is measured against either.
        if _n_songs(record) > 0:
            candidates.setdefault(date, []).append(deepcopy(dict(record)))
    shows = [pick_show(pool, policy) for pool in candidates.values()]
    # A refused date is refused however carefully someone wrote it down. Without this, the
    # override quietly puts the date back and the diagnostic that tells people to use
    # drop_dates to delete a date is a lie.
    wanted = {date: record for date, record in (overrides or {}).items()
              if date not in policy.drop_dates}
    refused = tuple(sorted(set(overrides or {}) - set(wanted)))
    merged, applied = apply_overrides(shows, wanted)
    return MergeResult(shows=merged, applied=applied, candidates=candidates, refused=refused)


# ==================================================================================================
# MANUAL OVERRIDES
#
# Every parser can be wrong in a way no parser can detect, and one night they all were at once. A
# taper merged two songs into a single track, so the second had no token to parse and vanished;
# they also wrote down seven non-song lines, which padded the record enough to look complete. The
# structured service carried the PRINTED setlist, which included a song the weather cut from the
# show. The truth existed in no source.
#
# Precedence cannot fix that. The tape outranks the service and clears the completeness bar
# whether you count raw tokens or real songs, so the wrong record wins the night under every
# tuning of the rule. Hence an escape hatch, with three properties it is built to have:
#
#   1. An override is a WHOLE SHOW, not a patch. A patch ("insert after the fourth song") cannot
#      be read on its own: what it produces depends on whichever record currently wins, and that
#      changes the day a new tape lands. A whole show says exactly what the corpus gets.
#   2. It ALWAYS wins, applied after the pick rather than as a top rank tier, because a rank tier
#      would still be filtered by the completeness bar before rank is ever consulted, and an
#      honest 8-song override would lose to a junk-padded 14-token parse. Replacement makes
#      "always wins" true by construction rather than true by careful ranking.
#   3. It is NOT self-correcting, so the merge reports its own disagreements instead. Precedence
#      gets "a better source reclaims the date" for free; a hard override throws that away. See
#      override_disagreements.
#
# ``reason`` is mandatory. Nothing goes in on a hunch.
# ==================================================================================================


def _canon_entries(raws: object, normalizer: Normalizer, date: str, field_name: str,
                   *, src: JSONSource, at: tuple) -> list[dict]:
    """One list of raw song strings into canonical entries, or a diagnostic."""
    if not isinstance(raws, list):
        src.fail(f"{date}: {field_name!r} must be a list", at=at,
                 detail=f"Got {type(raws).__name__}. Each set is a list of song names, in the "
                        f"order they were played.")
    entries = []
    for index, raw in enumerate(raws):
        if not isinstance(raw, str):
            src.fail(f"{date}: {field_name!r} entries must be strings", at=at + (index,),
                     detail=f"Got {raw!r}. A song is written as its name, with a trailing '>' if "
                            f"it segued into the next one.")
        song, segue = normalizer.canonicalize(raw)
        if not song:
            src.fail(f"{date}: {field_name!r} contains an empty song name", at=at + (index,),
                     detail=f"{raw!r} normalizes to nothing at all.")
        entries.append({"song": song, "segue": segue,
                        "non_song": normalizer.is_non_song(song)})
    return entries


def overrides_from_mapping(data: object, normalizer: Normalizer,
                           *, src: JSONSource | None = None) -> dict[str, dict]:
    """``{"overrides": {date: entry}}`` into show records, or raise :class:`DiagnosticError`.

    Song names run through the same normalizer as every other source, so aliases resolve and a
    trailing '>' sets the segue flag. They do NOT go through the parser's shape gates, and that
    is deliberate: an override is a person saying what was played, which includes the case where
    a song is new and no source has ever named it. Refusing an unfamiliar name here would defeat
    the point of having the escape hatch.

    Raises rather than skipping a bad entry. A loader that swallowed its own error would hand the
    date back to whatever the sources say, silently reverting the correction someone just made,
    which is the one outcome worse than refusing to run.

    ``src`` is where the data came from, and every diagnostic is anchored through it. An empty
    :class:`JSONSource` is legitimate and means "a caller handed me a dict": the errors are the
    same errors, rendered without a file or a caret. One code path, degraded, rather than two.
    """
    src = src or JSONSource()
    if not isinstance(data, Mapping) or not isinstance(data.get("overrides"), Mapping):
        src.fail("expected a top-level {\"overrides\": {date: entry}} object",
                 detail="The file holds one entry per date, keyed by the date the show happened.")
    out: dict[str, dict] = {}
    for date, entry in sorted(data["overrides"].items(), key=lambda item: str(item[0])):
        at = ("overrides", date)
        if not isinstance(date, str) or not date.strip():
            src.fail(f"{date!r} is not a usable date", at=at,
                     detail="Every override is keyed by the date the show happened.")
        if not isinstance(entry, Mapping):
            src.fail(f"{date}: entry must be an object", at=at,
                     detail="Expected sets, encore and reason.")
        reason = entry.get("reason")
        # isinstance first: str(None) is "None", which is not empty, so stringifying before the
        # test lets null, false, 0 and [] all pass for a reason.
        if not isinstance(reason, str) or not reason.strip():
            src.fail(f"{date}: every override needs a non-empty 'reason'", at=at + ("reason",),
                     caption="no reason given",
                     detail="Say how the setlist was confirmed. An override outranks every source "
                            "we have, so the next person to read it has nothing else to go on. "
                            "Nothing goes in on a hunch.")
        raw_sets = entry.get("sets", [])
        if not isinstance(raw_sets, list):
            src.fail(f"{date}: 'sets' must be a list of lists", at=at + ("sets",),
                     detail="One list per set, even when there was only one set.")
        sets = [_canon_entries(one, normalizer, date, "sets",
                               src=src, at=at + ("sets", index))
                for index, one in enumerate(raw_sets)]
        encore = _canon_entries(entry.get("encore", []), normalizer, date, "encore",
                                src=src, at=at + ("encore",))
        if count_songs(sets, encore) == 0:
            src.fail(f"{date}: override has no songs", at=at,
                     detail="Either it lists nothing, or the pack classes everything it lists as "
                            "a non-song (tuning, banter, a guest note). Such an override would "
                            "silently hand the date back to the sources it was written to "
                            "correct. To refuse a date outright, add it to the merge policy's "
                            "drop_dates instead.")
        out[date] = {"date": date, "year": date[:4], "sets": sets, "encore": encore,
                     "n_songs": count_songs(sets, encore), "source": "override",
                     "identifier": f"override-{date}", "reason": reason.strip()}
    return out


def _real_songs(record: Mapping, normalizer: Normalizer) -> set[str]:
    """The in-vocabulary songs of a record.

    The vocabulary test is against the NORMALIZED KEYS, not the display-name list. Testing a
    normalized name against display names returns False for everything, whatever the truth, and
    it does it quietly.

    ``is_non_song`` handles the rest: a setlist service will happily hand back a bare "Intro" as
    though it were a song, and being in the vocabulary is not the same as being music.

    The entries' own ``non_song`` tags are not consulted. They are what the same
    ``is_non_song`` already said about the same names, so reading them would be asking the
    question twice and getting two answers the day a source tags something it should not have.
    """
    _, norm_to_canon = normalizer.build_vocab()
    found = set()
    for entry in _entries(record):
        name = entry.get("song", "")
        if normalizer.normalize(name) in norm_to_canon and not normalizer.is_non_song(name):
            found.add(name)
    return found


def override_disagreements(candidates: Mapping[str, list[dict]],
                           overrides: Mapping[str, Mapping],
                           normalizer: Normalizer) -> list[dict]:
    """Sources carrying a real song the override does not have.

    The safety net for the third property above. An override always wins, so nothing else will
    ever tell us it went stale or was wrong in the first place. A newly landed tape holding a
    song the override lacks is exactly the signal worth putting in front of a person.

    Filtered through the vocabulary deliberately. The record an override replaces usually has
    junk in it -- that is generally why it was overridden -- and if that junk counted, the date
    would print a review line on every run forever, which teaches the reader to skip them.
    """
    out = []
    for date, record in sorted(overrides.items()):
        have = _real_songs(record, normalizer)
        for candidate in candidates.get(date, []):
            missing = sorted(_real_songs(candidate, normalizer) - have)
            if missing:
                out.append({"date": date,
                            "identifier": str(candidate.get("identifier") or "?"),
                            "source": candidate.get("source", ""),
                            "n_override": record.get("n_songs", 0),
                            "missing": missing})
    return out

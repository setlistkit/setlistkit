# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The tape measure bundle: one file that carries everything a page needs to draw itself.

THE POINT IS THAT THERE IS ONE FILE. The implementation this replaces published seven -- statistics
and observations and features and a review queue and edge cases and an abandoned list, each written
by a different pass, each read separately by the dashboard builder, none of them stamped with which
run they came from. Nothing stopped six of them being current and the seventh being from Tuesday,
and nothing would have said so. A bundle cannot be half-fresh.

It also carries what the consumer would otherwise have to hold for itself. ``credits`` is here
because the previous dashboard read the raw archive.org cache to find out who to thank -- a
gitignored directory the publishing machine did not have, so the credits page was empty in
production and complete on a workstation. A consumer is never expected to hold data the exporter
did not give it; that is the same rule ``slkit derive`` follows one layer down, applied outward.

``schema`` carries a major version and the consumer asserts on it. The version changes when a
consumer that ignored the change would draw something WRONG, not when a field is added.

This module knows nothing about SQLite and nothing about files. It takes the mappings the store
hands back and returns the mapping the exporter writes, so the shape can be tested without a
database and the golden file is a test of the shape rather than of the query.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .features import SongFeature

# Bumped only when a consumer that ignored the change would draw something wrong. Adding a field
# is not that; renaming one, changing a unit, or changing what a row MEANS is.
SCHEMA = "setlistkit.tapemeasure/1"

# The one place the database's name for a column and the published name for it meet.
#
# The store spells it `set_label` because `set` is a Python builtin and a SQL-adjacent word that
# reads badly in a WHERE clause. The bundle spells it `set` because that is what it is called by
# everyone who has ever discussed a setlist. Both are right in their own layer, and the mapping
# between them is written down HERE, once, rather than being a rename that happens in whichever
# consumer remembers to do it.
_RENAMED = {"set_label": "set"}

SECONDS_PER_HOUR = 3600.0


def _credited(uploader: str) -> str:
    """How a taper is NAMED in public, from the identity the store knows them by.

    archive.org's ``uploader`` is an email address. The store keeps it whole because that is what
    tells four tapers apart from one taper who posted four mixes, and getting that wrong is what
    lets the loudest uploader on a night decide that night. But the bundle is written to be
    published, and publishing 499 email addresses to thank people is not thanking them.

    So the domain is dropped and the local part -- which is the handle they chose -- is what gets
    the credit. The masking happens HERE, at the boundary where data stops being ours and starts
    being everyone's, rather than in the store, because every pass before this one has a real use
    for the whole address.
    """
    return uploader.split("@", 1)[0].strip() or uploader


# A performance backed by fewer ballots than this rests on one taper's opinion of where the track
# boundaries were. Counted in the totals because it is invisible from the performance count.
_ALONE = 2


@dataclass(frozen=True)
class Concluded:
    """Everything one ``slkit derive durations`` run stored, as the store hands it back.

    One argument rather than five, for the same reason the store writes those five tables in one
    transaction: they are five views of ONE run, and any two of them taken from different runs are
    a contradiction that nothing downstream would report. Passing them separately is what makes
    mixing them possible in the first place.
    """

    performances: Sequence[Mapping] = ()
    stats: Sequence[Mapping] = ()
    review: Sequence[Mapping] = ()
    abandoned: Sequence[Mapping] = ()
    edges: Sequence[Mapping] = field(default_factory=tuple)


def _published(row: Mapping) -> dict:
    """One stored row under its published names, in the order the store gave it."""
    return {_RENAMED.get(key, key): value for key, value in row.items()}


def _feature_of(feature: SongFeature | None) -> dict | None:
    """One song's structural profile as JSON, or ``None`` for a song the corpus never saw played.

    ``rates`` stays nested rather than being flattened into the song. The six of them share a
    denominator and that shared denominator is the only thing making them comparable -- which is
    exactly why :mod:`setlistkit.catalog.features` grouped them, and flattening here would be
    re-deciding that from a layer that knows less.
    """
    if feature is None:
        return None
    return {"n_plays": feature.n_plays,
            "rates": {"opener": feature.rates.opener,
                      "set_closer": feature.rates.set_closer,
                      "show_closer": feature.rates.show_closer,
                      "encore": feature.rates.encore,
                      "segue_out": feature.rates.segue_out,
                      "segue_in": feature.rates.segue_in},
            "mean_slot": feature.mean_slot,
            "top_partners": [list(pair) for pair in feature.top_partners],
            "first_seen": feature.first_seen,
            "last_seen": feature.last_seen}


def _songs(stats: Iterable[Mapping], features: Iterable[SongFeature]) -> list[dict]:
    """Every song's lengths with its structural profile folded in.

    Folded in rather than published beside, because the page draws them on one axis pair and two
    files means two version stamps and a join the consumer has to get right. A song with lengths
    but no profile keeps a null: it means a song was timed off a tape whose night the corpus does
    not have it in, which is a real disagreement and is not improved by hiding it behind a zero.
    """
    by_song = {feature.song: feature for feature in features}
    return [{**dict(stat), "features": _feature_of(by_song.get(stat["song"]))}
            for stat in stats]


def _date_range(performances: Sequence[Mapping]) -> list[str] | None:
    """First and last night that produced a measurement, or ``None`` if none did.

    Over the PERFORMANCES rather than over the corpus: this is the range the page can draw, and a
    range that claims 1992 when the earliest timed night is 2004 describes an ambition.
    """
    if not performances:
        return None
    dates = [row["date"] for row in performances]
    return [min(dates), max(dates)]


def _totals(concluded: Concluded) -> dict:
    """The handful of numbers a reader checks before believing any of the rest.

    ``single_tape_performances`` is here because it is the number that decides how much of this can
    be trusted and it cannot be seen from any other total -- twenty thousand performances timed
    once each and twenty thousand timed four times each publish the same performance count.
    """
    performances = concluded.performances
    seconds = sum(row["seconds"] or 0 for row in performances)
    return {"performances": len(performances),
            "nights": len({row["date"] for row in performances}),
            "songs": len(concluded.stats),
            "songs_at_n3": sum(1 for stat in concluded.stats if stat["n"] >= 3),
            "hours": round(seconds / SECONDS_PER_HOUR, 2),
            "single_tape_performances": sum(1 for row in performances
                                            if (row["n_ballots"] or 0) < _ALONE),
            "tapes_queued_for_review": len(concluded.review),
            "tapes_abandoned": len(concluded.abandoned),
            "withheld": _withheld(performances)}


def _withheld(performances: Iterable[Mapping]) -> dict[str, int]:
    """reason -> how many measured performances did not vote for their song's nominal length.

    Published rather than left as an internal detail, because every reason in it is one a reader
    might disagree with, and a pool that quietly shrinks is a pool nobody audits.
    """
    out: dict[str, int] = {}
    for row in performances:
        if row["withheld"]:
            out[row["withheld"]] = out.get(row["withheld"], 0) + 1
    return dict(sorted(out.items()))


def bundle(concluded: Concluded, features: Iterable[SongFeature],
           tapers: Mapping[str, int], *,
           corpus_shows: int, recordings_read: int) -> dict:
    """The whole tape measure as one JSON-ready mapping.

    Row order is whatever the store handed over, which is sorted by content rather than by rowid
    for exactly this reason: the bundle is diffed between runs by a person, and a diff that moves
    ten thousand rows because insertion order changed is a diff nobody reads.
    """
    return {
        "schema": SCHEMA,
        "generated": {"corpus_shows": corpus_shows,
                      "recordings_read": recordings_read,
                      "date_range": _date_range(concluded.performances)},
        "songs": _songs(concluded.stats, features),
        "performances": [_published(row) for row in concluded.performances],
        "review": [dict(row) for row in concluded.review],
        "abandoned": [dict(row) for row in concluded.abandoned],
        "edges": [dict(row) for row in concluded.edges],
        "credits": [{"uploader": _credited(uploader), "n_tapes": n}
                    for uploader, n in tapers.items()],
        "totals": _totals(concluded),
    }

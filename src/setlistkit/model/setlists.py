# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Turn a ranked list of songs into something shaped like a night: an order, and segue marks.

:mod:`setlistkit.model.scores` says WHICH songs. This says what they look like laid end to end --
which one opens, what follows what, where the band probably does not stop. It is the half of the
prediction a reader actually recognizes as a setlist.

NONE OF IT IS SCORED, AND THAT IS DELIBERATE. Hitting the order of a set is close to impossible,
and a metric nobody can move is a metric that gets quoted anyway. So the ordering here is
presentation: it is recorded and shown, never graded. If positional metrics (opener hit rate,
adjacent-pair accuracy) are ever added, this machinery should have to earn its place the way the
cooldown did in :mod:`setlistkit.model.scores` -- by measuring better than not having it.

THE AS-OF BOUNDARY LIVES IN :func:`transitions`, ON PURPOSE
:func:`setlistkit.catalog.features.song_features` has no as-of parameter and no leak guard of its
own, correctly: it is a descriptive catalog function and the Songbook asks it about the whole
corpus. So the filter sits here, at the model boundary where the same obligation already lives for
``song_scores``, and it happens INSIDE :func:`transitions` rather than in its callers.

The trap that closes: :func:`setlistkit.store.corpus.shows` takes an INCLUSIVE ``until``, so the
naive way for a caller to pre-filter -- ``shows(until=D)`` -- hands back the very night being
predicted and leaks the whole answer into the partner graph. There is exactly one implementation
of "strictly before" reachable from set construction, and it is not a date string a caller
assembles.

WHAT CHANGED FROM THE VERSION THIS WAS PORTED FROM
The original built its own tallies (``build_transitions``, 24 lines) over the same setlists
``song_features`` already walks, and ranked on raw COUNTS: the most opener-ish song in a pool was
whichever had opened the most sets. This reads rates instead, because a count answers "which of
these is played most" and a rate answers "which of these opens", and only the second is the
question. The pool always arrives pre-ranked by ``song_scores``, so play frequency has already had
its say by the time anything gets here; letting it decide the ordering too counts it twice.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ..catalog.features import SongFeature, song_features
from .scores import SongScore, _as_date

# The base rate stood in for a song the scorer never rated, when ranking encore candidates. Small
# enough to sort below every real rate and non-zero so the encore count still breaks the tie --
# a song the corpus has only ever seen in an encore slot is exactly the case this is for.
_UNRATED_BASE = 0.0001


@dataclass(frozen=True)
class PredictConfig:
    """The construction knobs, kept SEPARATE from :class:`~setlistkit.model.scores.ScoreConfig`.

    Deliberately a second class rather than six more fields on the first one. ``ScoreConfig``'s
    docstring records what each of its knobs measured out of sample -- which of them earn their
    place, which are arguable, and which do nothing at all. None of the knobs here has been
    measured that way, and folding them in would let them inherit credibility they have not
    earned from the ones that have.

    ``rollouts`` is the one number here with a stated basis, and it is inherited: 500 comes from
    the version this was ported from, whose own convergence study measured "rank-30 probability
    around 0.05 gives a standard error around 0.01" -- against a different corpus, and a config
    that still carried a base-rate blend this project has since dropped. So it is re-measured on
    our corpus rather than trusted; see ``scripts/rollout_convergence.py``.

    ``set_len`` is the count a live prediction has to GUESS, and here it is only a default: the
    replay hands the real number in from the night being graded. A set-length prior is the open
    question that has to be answered before anything publishes a prediction for a night that has
    not happened.

    ``top_n`` and ``set_len`` are different numbers and conflating them is easy. ``set_len`` is how
    many songs one draw takes, so it controls what a plausible night looks like. ``top_n`` is how
    far down the resulting ranking gets recorded, and it is larger: a song can be worth listing
    without ever being likely enough to make a typical night.
    """

    set_len: int = 10
    encore_len: int = 1
    rollouts: int = 500
    seed: int = 42
    segue_threshold: float = 0.45
    top_n: int = 30


@dataclass(frozen=True)
class Entry:
    """One slot in a constructed setlist: a song, and whether it runs into the next one.

    The same two keys :func:`setlistkit.store.corpus.shows` hands back for a real entry, so a
    predicted night and a played one can be printed by the same code. No ``non_song``: the scorer
    never rates stage business, so nothing here can construct any.
    """

    song: str
    segue: bool


@dataclass(frozen=True)
class Transitions:
    """What the corpus BEFORE a date says about how one song follows another.

    Wraps the features rather than copying four dicts out of them, so the play count that every
    rate rests on stays reachable beside the rate. ``asof`` is carried for the same reason
    :class:`~setlistkit.catalog.window.ResolvedWindow` carries its anchor: a caller printing or
    storing a construction should be able to state the boundary it was built under, rather than
    having to trust that whoever called :func:`transitions` passed the date they meant.
    """

    features: Mapping[str, SongFeature]
    asof: str

    def opener_rate(self, song: str) -> float:
        """How often this song opened a set, over every time it was played."""
        feature = self.features.get(song)
        return feature.rates.opener if feature else 0.0

    def encore_rate(self, song: str) -> float:
        """How often this song turned up in an encore, over every time it was played."""
        feature = self.features.get(song)
        return feature.rates.encore if feature else 0.0

    def segue_rate(self, song: str) -> float:
        """How often this song ran into the next one without a stop.

        Over ``n_plays``, which includes the times it CLOSED a set and could not have segued. So
        it reads slightly low against the version this was ported from, which divided by
        transitions-to-a-next-song instead. Kept as the catalog computes it: one definition of
        "does this song run into things", shared with the Songbook, beats two that nearly agree.
        """
        feature = self.features.get(song)
        return feature.rates.segue_out if feature else 0.0

    def partners(self, song: str) -> Mapping[str, int]:
        """song -> how many times it followed ``song`` without a stop, whole graph."""
        feature = self.features.get(song)
        return dict(feature.top_partners) if feature else {}


def transitions(shows: Iterable[Mapping], *, asof: str | dt.date) -> Transitions:
    """Structural features over the shows STRICTLY BEFORE ``asof``.

    The filter is here and not in the caller. See the module docstring: ``corpus.shows(until=D)``
    is inclusive, and a caller assembling its own boundary is a caller that can leak the night
    being predicted into the graph that predicts it.

    Asks for the whole partner graph rather than the profile's top three -- a greedy walk needs to
    know what follows each song out of the eleven others in the pool, and three answers "nothing"
    for nearly every pair of them.
    """
    when = _as_date(asof)
    if when is None:
        raise ValueError(f"as-of date is not a date: {asof!r}")
    past = []
    for show in shows:
        date = _as_date(show.get("date") or "")
        if date is not None and date < when:
            past.append(show)
    features = song_features(past, top_partners=None)
    return Transitions(features={feature.song: feature for feature in features},
                       asof=when.isoformat())


def order_set(songs: Iterable[str], trans: Transitions,
              config: PredictConfig) -> list[Entry]:
    """A chosen bag of songs, walked into an order, with segue marks.

    The greedy walk, kept as it arrived: start from the most opener-ish song in the pool, then
    repeatedly take the highest-count learned transition out of the current song, falling back to
    opener-ishness when the corpus has never seen anything follow it.

    Ties break on the song name, and that is not tidiness. Rates and counts tie constantly in a
    pool of ten, and a dict's iteration order is hash-seed randomized, so without a total order
    the same seed and the same corpus would produce a different setlist process to process --
    which is the determinism the whole replay rests on.
    """
    pool = list(songs)
    if not pool:
        return []
    order = [min(pool, key=lambda song: (-trans.opener_rate(song), song))]
    remaining = [song for song in pool if song != order[0]]
    while remaining:
        counts = trans.partners(order[-1])
        nxt = min(remaining, key=lambda song: (-counts.get(song, 0), song))
        if not counts.get(nxt, 0):
            nxt = min(remaining, key=lambda song: (-trans.opener_rate(song), song))
        order.append(nxt)
        remaining.remove(nxt)
    last = len(order) - 1
    return [Entry(song=song,
                  segue=index < last and trans.segue_rate(song) >= config.segue_threshold)
            for index, song in enumerate(order)]


def pick_encore(scores: Sequence[SongScore], trans: Transitions, config: PredictConfig,
                used: set[str]) -> list[Entry]:
    """The encore, drawn from songs the corpus has actually encored and the set has not used.

    Ranked on encore rate times base rate: how encore-ish the song is, weighted by how present it
    is in the rotation at all. Either alone picks badly -- rate alone hands it to something
    encored once in 1998, and base alone hands it to the most-played song in the band's history
    whether or not it has ever closed a night.

    ``used`` is MUTATED, so a caller constructing several nights gets no-repeat across them for
    free. That is how the version this came from worked and the replay depends on it not changing:
    a night whose encore is already in its own set is not a night.

    NO ``rng``. The original took one to feed a "wildcard" variant that drew randomly from ranks
    two through six; per-night variants are not ported (see the design's "Not ported"), so the
    parameter would have been an argument nothing could reach.
    """
    base = {row.song: row.base for row in scores}
    ranked = sorted(
        (song for song, feature in trans.features.items()
         if feature.rates.encore > 0 and song not in used),
        key=lambda song: (-trans.encore_rate(song) * base.get(song, _UNRATED_BASE), song))
    chosen = ranked[:max(0, config.encore_len)]
    used.update(chosen)
    return [Entry(song=song, segue=False) for song in chosen]


def draw_set(pool: Sequence[str], weights: Mapping[str, float], n: int, rng) -> list[str]:
    """``n`` distinct songs drawn from ``pool`` with probability proportional to ``weights``.

    Without replacement, so one draw is one plausible night rather than a night that plays its
    best song four times. A song with no weight, or a weight of zero, cannot be drawn at all --
    which is what a cooldown of 0.0 means, and it has to survive to here rather than being
    normalized back into the running.

    ``rng`` is a :class:`random.Random`, seeded by the caller. Passing it in rather than reaching
    for the module-level generator is what makes a whole replay reproducible from one number.
    """
    avail = [song for song in pool if weights.get(song, 0.0) > 0.0]
    chosen: list[str] = []
    for _ in range(min(n, len(avail))):
        total = sum(weights[song] for song in avail)
        target = rng.random() * total
        # The last song is the fallback rather than an error case: floating-point summation can
        # leave `target` a hair above the accumulated total, and falling off the end of the loop
        # with nothing picked would silently shorten the set.
        pick = avail[-1]
        running = 0.0
        for song in avail:
            running += weights[song]
            if running >= target:
                pick = song
                break
        chosen.append(pick)
        avail.remove(pick)
    return chosen

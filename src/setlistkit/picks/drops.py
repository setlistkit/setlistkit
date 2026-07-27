# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Predict one night repeatedly, from successively later cutoffs, and watch the list move.

:mod:`setlistkit.picks.replay` predicts each night exactly once, from the show immediately before
it, which is the only cutoff that can be GRADED -- it is the most informed honest one. That
answers "was the model right". It cannot answer the other question a reader asks, which is whether
the model knew it early or only worked it out at the last minute.

A DROP is the same night predicted from an earlier cutoff. Predicting 2026-07-18 from the show
five nights before it is exactly as honest as predicting it from the show one night before: both
train on strictly earlier data. The two differ only in how much they were allowed to know, and
lining them up is what turns a single verdict into a story about when the model made up its mind.

The picks schema was built for this and says so. ``picks`` is keyed ``(iteration_id, night_date,
rank)`` rather than by song, and the migration comment gives the reason outright: "what did we
have at rank 3 that night" is the question a churn view asks. ``ledger.record`` already takes a
MAPPING of night to ranked list rather than one night, for the same anticipated reason. So this
module adds a walk, not a schema.

TWO THINGS ARE HELD FIXED ACROSS A NIGHT'S DROPS, and both matter for the comparison to mean
anything.

* **The config**, for the reason :mod:`~setlistkit.picks.replay` gives: a config that moves during
  a run produces differences nobody can attribute.
* **``set_len``**, which is the night's own played count. It is the same one-integer lookahead the
  replay takes, applied identically to every drop of a night. Letting it vary would put movement
  into the churn that came from the draw length rather than from the data, which is precisely the
  thing being measured.

HOW THE CUTOFF IS APPLIED. The corpus is filtered to shows on or before the cutoff BEFORE it
reaches the model, and ``asof`` is still the night being predicted. That looks redundant and is
not: the pre-filter is what makes the drop a drop, and the ``asof`` boundary inside
:func:`~setlistkit.model.rollout.rollout_night` is the guarantee that stays true whatever a caller
hands in. Pre-filtering ALONE would be the leak the design warns about -- ``corpus.shows(until=D)``
is inclusive, so a filter written to the night rather than to a strictly earlier cutoff would hand
the model the answer. Every cutoff here is strictly before its night, so the filter cannot reach it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ..model.backtest import gradeable
from ..model.rollout import SongChance, rollout_night
from ..store.migrations import transaction
from . import churn, ledger
from .ledger import Outcome, guard_no_shrink
from .replay import ReplaySpec

# How many earlier cutoffs a night is predicted from, when the caller names none. Twelve is a
# DEFAULT and not a finding: it is roughly two weeks of a touring run, which is long enough for a
# list to visibly move and short enough that the walk stays a page-build rather than a batch job.
# The cost is nights x depth predictions, so this is the knob that decides whether the build takes
# seconds or minutes.
DEFAULT_DEPTH = 12


@dataclass(frozen=True)
class Drop:
    """One night's shortlist as it stood at one cutoff."""

    cutoff: str
    chances: tuple[SongChance, ...]

    @property
    def songs(self) -> tuple[str, ...]:
        """The ranked song names alone, which is what the churn metrics compare."""
        return tuple(chance.song for chance in self.chances)


@dataclass(frozen=True)
class NightDrops:
    """Every drop for one night, oldest cutoff first, plus how the last one scored.

    ``outcome`` grades the LAST drop and nothing else, because that is the only one a replay would
    have published. The earlier drops are the story of how it got there; scoring them all would
    invite reading the best of them as the model's performance, which is picking a winner after
    seeing the answers.
    """

    night: str
    played: tuple[str, ...]
    drops: tuple[Drop, ...]
    outcome: Outcome | None

    @property
    def drift(self) -> tuple[float, ...]:
        """Membership distance from the first drop, per drop. Starts at 0.0."""
        return churn.series([drop.songs for drop in self.drops])[0]

    @property
    def step(self) -> tuple[float, ...]:
        """Membership distance from the previous drop, per drop. Starts at 0.0."""
        return churn.series([drop.songs for drop in self.drops])[1]

    @property
    def rbo_drift(self) -> tuple[float, ...]:
        """Top-weighted distance from the first drop, per drop. Starts at 0.0.

        Beside :attr:`drift` rather than instead of it: the pair is the point. Movement with a
        high membership distance and a low top-weighted one is churn at the bottom of the list,
        which is the ordinary state; the reverse is the model changing its mind about rank 1.
        """
        if not self.drops:
            return ()
        first = self.drops[0].songs
        return (0.0,) + tuple(churn.rbo(first, drop.songs) for drop in self.drops[1:])

    @property
    def settled_at(self) -> str | None:
        """The earliest cutoff whose membership already matched the final drop, or None.

        The one number here that is a claim rather than a measurement, so it is derived and never
        stored: "the model knew by this date" is exactly the kind of statement that should be
        recomputable from the drops when someone disputes it.
        """
        if not self.drops:
            return None
        final = set(self.drops[-1].songs)
        for drop in self.drops:
            if set(drop.songs) == final:
                return drop.cutoff
        return None


@dataclass(frozen=True)
class DropsReport:
    """Every night walked, plus what the walk was allowed to cover.

    ``short`` counts nights that got fewer drops than ``depth`` asked for because the corpus ran
    out behind them. Reported rather than padded: a night with three drops sitting beside nights
    with twelve has a much shorter story, and a chart that does not say so reads as a model that
    settled early.
    """

    nights: tuple[NightDrops, ...]
    depth: int
    skipped: int
    short: int

    @property
    def mean_step(self) -> float:
        """Average step churn over every drop that had one behind it.

        Pooled over drops and not over nights, so a night with twelve drops weighs more than one
        with three -- it contains more movement, and averaging the per-night averages would let a
        short night with one big jump outvote a long quiet one.
        """
        values = [value for night in self.nights for value in night.step[1:]]
        return sum(values) / len(values) if values else 0.0

    @property
    def settled_early(self) -> int:
        """Nights whose shortlist had stopped moving before the last cutoff.

        The aggregate that makes :attr:`NightDrops.settled_at` worth computing. On its own that
        property reads as a date per night and invites being scanned rather than checked; the
        count answers the question it is really being asked, which is whether the model ever knows
        the answer early. Zero here is not a broken walk -- it means every additional show changed
        the top of the list, which is a statement about the band and belongs on the page.

        A night with ONE drop can never be counted, and that is the definition rather than an
        off-by-one: its only list is also its final list, so there is no earlier evidence for it
        to have already agreed with. Those nights are the same ones ``short`` counts.
        """
        return sum(1 for night in self.nights
                   if len(night.drops) > 1 and night.settled_at != night.drops[-1].cutoff)


def _cutoffs(all_nights: Sequence[tuple[str, set[str]]], index: int,
             depth: int) -> list[str]:
    """The ``depth`` gradeable show dates immediately before ``index``, oldest first.

    Taken from the gradeable list rather than from every date in the corpus, so a cutoff is always
    a night the model could have learned something from. A cutoff landing on a two-song fragment
    would produce a drop indistinguishable from the one before it and read as the model having
    settled.
    """
    start = max(0, index - depth)
    return [all_nights[position][0] for position in range(start, index)]


def predict_drop(corpus: Sequence[Mapping], spec: ReplaySpec, night: str, cutoff: str,
                 set_len: int) -> Drop:
    """One drop: the night predicted from a corpus that stops at ``cutoff``.

    The filter is ``<= cutoff`` and the cutoff is strictly before the night, so the night itself
    can never be in the filtered corpus. ``asof`` stays the night regardless -- see the module
    docstring on why both are here.

    Public because :mod:`setlistkit.picks.upcoming` predicts the same way for a night that has not
    happened, and this one line is the leak boundary. Two copies of it is two places for the
    comparison to drift to ``<`` or for the filter to be skipped entirely, and only one of them
    would have a test watching it.
    """
    visible = [show for show in corpus if show["date"] <= cutoff]
    chances = rollout_night(visible, spec.score, spec.predict, asof=night, set_len=set_len)
    return Drop(cutoff=cutoff, chances=tuple(chances))


def drops(conn, shows: Iterable[Mapping], spec: ReplaySpec, *, depth: int = DEFAULT_DEPTH,
          dry_run: bool = False, force: bool = False, progress=None) -> DropsReport:
    """Predict every night in the window from its last ``depth`` cutoffs, and grade the last.

    Writes ONE iteration per cutoff holding every night that cutoff looked ahead to, which is why
    the whole walk is computed before anything is stored: ``ledger.record`` replaces a cutoff's
    picks wholesale, so writing it once per night would leave only the last night written and
    silently discard the rest.

    ``progress`` is called with each :class:`NightDrops` as it completes.
    """
    corpus = list(shows)
    walked, by_cutoff, skipped, short = _walk(corpus, spec, depth, progress)
    graded = _store_and_grade(conn, walked, by_cutoff, spec, dry_run=dry_run, force=force)
    return DropsReport(nights=tuple(graded), depth=depth, skipped=skipped, short=short)


def _walk(corpus: Sequence[Mapping], spec: ReplaySpec, depth: int,
          progress) -> tuple[list[NightDrops], dict[str, dict[str, tuple[SongChance, ...]]],
                             int, int]:
    """Every night in the window predicted from its cutoffs, plus the picks grouped for writing.

    The grouping is the reason this returns two collections rather than one. The walk is ordered
    by NIGHT because that is the story being told, and the ledger is written by CUTOFF because
    that is what an iteration is; building the second view here, while the first is being built,
    is cheaper and clearer than re-deriving it from the nights afterwards.
    """
    all_nights = gradeable(corpus, spec.min_songs)
    by_cutoff: dict[str, dict[str, tuple[SongChance, ...]]] = {}
    walked: list[NightDrops] = []
    skipped = short = 0

    for index, (night, played) in enumerate(all_nights):
        if spec.since is not None and night < spec.since:
            continue
        if spec.until is not None and night > spec.until:
            continue
        cutoffs = _cutoffs(all_nights, index, depth)
        if not cutoffs:
            skipped += 1
            continue
        if len(cutoffs) < depth:
            short += 1
        night_drops = tuple(predict_drop(corpus, spec, night, cutoff, len(played))
                            for cutoff in cutoffs)
        for drop in night_drops:
            by_cutoff.setdefault(drop.cutoff, {})[night] = drop.chances
        walked.append(NightDrops(night=night, played=tuple(sorted(played)),
                                 drops=night_drops, outcome=None))
        if progress is not None:
            progress(walked[-1])
    return walked, by_cutoff, skipped, short


def _store_and_grade(conn, walked: Sequence[NightDrops],
                     by_cutoff: Mapping[str, Mapping[str, tuple[SongChance, ...]]],
                     spec: ReplaySpec, *, dry_run: bool,
                     force: bool) -> list[NightDrops]:
    """Write every iteration, then grade each night against what came back out.

    A dry run scores the last drop in memory instead. The two paths are the same arithmetic --
    :func:`~setlistkit.picks.ledger.grade` calls :func:`~setlistkit.picks.ledger.score_outcome`
    on rows that made a round trip through SQLite -- so a page built dry and a page built from the
    ledger report the same numbers.
    """
    if dry_run:
        return [NightDrops(night=night.night, played=night.played, drops=night.drops,
                           outcome=ledger.score_outcome(night.drops[-1].chances, night.played))
                for night in walked]

    before = ledger.table_counts(conn)
    graded: list[NightDrops] = []
    with transaction(conn):
        for cutoff in sorted(by_cutoff):
            ledger.record(conn, cutoff, by_cutoff[cutoff], spec.as_record())
        for night in walked:
            graded.append(NightDrops(night=night.night, played=night.played, drops=night.drops,
                                     outcome=ledger.grade(conn, night.night, night.played)))
        guard_no_shrink(before, ledger.table_counts(conn), force=force)
    return graded

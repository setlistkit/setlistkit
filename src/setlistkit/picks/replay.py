# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Walk the corpus one night at a time, predicting each before revealing it, and grade the lot.

The obvious way to get a prediction facility working again is to go and find a list of upcoming
shows. That reading is wrong, and following it costs a source, a schedule table, venue columns on
``shows``, and parser changes across four parsers before a single prediction can be graded.

None of it is needed, because PREDICTING IS BLINDING and the corpus can be blinded at any date.
Take a span of recent history, hide everything past a cutoff, predict the next night, reveal that
one night, grade it, advance the cutoff. The corpus supplies the "upcoming" dates *and* the ground
truth, so the whole facility gets wired end to end and shown to be healthy with no new source work
at all. A real schedule is only needed to publish a page about a show that has not happened yet,
which is a report-layer concern and a later spec.

The schema had already ruled the alternative out, in a comment on ``shows``: a record that enters
the corpus with no setlist in it is what a bad merge produces, "so this schema does not offer
somewhere to write one." An upcoming show is exactly that record.

CONFIG IS FROZEN ACROSS A REPLAY, and nothing here can change that. The pipeline this comes from
auto-tuned nightly on the most recent window and adopted whatever scored best in sample; an
out-of-sample probe caught the adopted config scoring about 1.4 sigma WORSE than the values it
replaced. :func:`setlistkit.model.backtest.holdout` already encodes the fix by RETURNING a config
instead of writing one, and nothing here writes one either.

THE ONE THING THE PREDICTOR IS TOLD ABOUT THE NIGHT IT IS PREDICTING is how many songs were
played. That is a lookahead of exactly one integer, and it is deliberate for the reason
``backtest.hit_rates`` gives: asking for more guesses than there were answers inflates recall for
free. It is worth being explicit that a LIVE prediction has no such number and would have to guess
it from a schedule or a set-length prior -- which is the open question this slice does not answer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass

from ..model.backtest import gradeable
from ..model.rollout import SongChance, rollout_night
from ..model.scores import ScoreConfig
from ..model.setlists import PredictConfig
from ..store.migrations import transaction
from . import ledger
from .ledger import Outcome

# How many songs a night needs before it is worth grading. Eight is what `backtest` defaults to,
# and the two must agree: a night that counts as a test there and not here would make the two
# reports disagree about the same corpus. It is deliberately NOT `ScoreConfig.min_show_songs`,
# which is lower and answers a different question -- "complete enough to learn from" rather than
# "complete enough to grade". A night recorded as two songs grades the taper's notes.
MIN_GRADEABLE_SONGS = 8


@dataclass(frozen=True)
class ReplaySpec:
    """Everything a replay is run UNDER, frozen for its whole duration.

    One object rather than five parameters threaded through three functions, and frozen for more
    than tidiness: it is also what gets written to ``pick_iterations.config_json``, so the record
    of what produced a prediction and the thing that produced it are the same object. A config
    assembled separately for storage is a config that can describe a run that never happened.

    ``since``/``until`` are inclusive YYYY-MM-DD bounds on the nights to GRADE. They do not bound
    training: a night at the start of the window is still predicted from the whole corpus behind
    it, which is what an honest prediction of that night would have had.
    """

    score: ScoreConfig
    predict: PredictConfig
    since: str | None = None
    until: str | None = None
    min_songs: int = MIN_GRADEABLE_SONGS

    def as_record(self) -> dict:
        """The whole spec as plain data, for the ledger's ``config_json``."""
        return {"score": asdict(self.score), "predict": asdict(self.predict),
                "min_songs": self.min_songs}


@dataclass(frozen=True)
class NightResult:
    """One night of the walk: what was predicted for it, from where, and how it did."""

    night: str
    cutoff: str
    outcome: Outcome
    chances: tuple[SongChance, ...]


@dataclass(frozen=True)
class ReplayReport:
    """The whole walk. ``skipped`` is nights in the window with no played show behind them.

    Skipped is reported rather than swallowed for the usual reason: a window whose first night is
    also the corpus's first night silently grades one fewer night than it looks like it did, and
    "90 nights" against "91 in the window" is the only thing that says so.
    """

    nights: tuple[NightResult, ...]
    skipped: int

    @property
    def mean_recall(self) -> float:
        """Recall averaged over nights, so a long night and a short one weigh the same."""
        if not self.nights:
            return 0.0
        return sum(row.outcome.recall_at_n for row in self.nights) / len(self.nights)

    @property
    def total_hits(self) -> int:
        """Songs the model called, summed over the walk."""
        return sum(row.outcome.actual_hits for row in self.nights)

    @property
    def total_expected(self) -> float:
        """Songs the model CLAIMED it would call, summed over the walk."""
        return sum(row.outcome.expected_hits for row in self.nights)

    @property
    def calibration(self) -> float:
        """Hits over expected hits, pooled across every night. Near 1.0 is the target.

        Pooled, and NOT the mean of the per-night ratios. A night that expected 0.4 hits and got
        two scores a ratio of 5.0, which would drag an average around on its own; pooling weighs
        each night by how much it was actually claiming. This design does not predict what the
        number is, and no test asserts a floor for it -- a threshold picked before the first
        measurement is a threshold fitted to hope.
        """
        return self.total_hits / self.total_expected if self.total_expected > 0 else 0.0


def _targets(all_nights: Sequence[tuple[str, set[str]]],
             spec: ReplaySpec) -> list[tuple[str, set[str], str | None]]:
    """(night, songs played, the cutoff to predict it from) for every night in the window.

    The cutoff is the played show IMMEDIATELY BEFORE the night, taken from the full gradeable list
    rather than the windowed one -- the show before the window's first night is outside the window
    and is still the honest place to predict from. ``None`` where there is no show behind it at
    all, which is the corpus's own first night and nothing else.
    """
    out = []
    for index, (night, songs) in enumerate(all_nights):
        if spec.since is not None and night < spec.since:
            continue
        if spec.until is not None and night > spec.until:
            continue
        out.append((night, songs, all_nights[index - 1][0] if index else None))
    return out


def replay(conn, shows: Iterable[Mapping], spec: ReplaySpec, *, dry_run: bool = False,
           force: bool = False, progress=None) -> ReplayReport:
    """Predict and grade every gradeable night in the window, oldest first.

    Writes the whole walk in ONE transaction, so a crash halfway leaves the previous ledger
    untouched and the no-shrink guard still has something to refuse. ``dry_run`` computes and
    grades everything in memory and writes nothing -- the metrics are identical, since
    :func:`~setlistkit.picks.ledger.grade` is the same arithmetic applied to rows that have made a
    round trip through SQLite.

    ``progress`` is called with each :class:`NightResult` as it lands. A walk over eighteen months
    is a couple of minutes of scoring, and silence for that long is indistinguishable from a hang.
    """
    corpus = list(shows)
    targets = _targets(gradeable(corpus, spec.min_songs), spec)
    before = ledger.table_counts(conn)
    results: list[NightResult] = []
    skipped = 0
    with nullcontext() if dry_run else transaction(conn):
        for night, played, cutoff in targets:
            if cutoff is None:
                skipped += 1
                continue
            result = _one_night(conn, corpus, spec, (night, played, cutoff), dry_run=dry_run)
            results.append(result)
            if progress is not None:
                progress(result)
        if not dry_run:
            # Inside the transaction on purpose: the counts are visible before the commit, so
            # raising here rolls the whole walk back rather than leaving a shrunk ledger behind.
            ledger.guard_no_shrink(before, ledger.table_counts(conn), force=force)
    return ReplayReport(nights=tuple(results), skipped=skipped)


def _one_night(conn, corpus: Sequence[Mapping], spec: ReplaySpec,
               target: tuple[str, set[str], str], *, dry_run: bool) -> NightResult:
    """Predict one night from the show before it, and grade it against what was played.

    The night's own length is the one thing the predictor is told about it -- see the module
    docstring. A dry run scores the list in memory; a real one stores it and grades what comes
    BACK OUT, which is the round trip :func:`~setlistkit.picks.ledger.grade` exists to buy.
    """
    night, played, cutoff = target
    chances = rollout_night(corpus, spec.score, spec.predict, asof=night, set_len=len(played))
    if dry_run:
        outcome = ledger.score_outcome(chances, played)
    else:
        ledger.record(conn, cutoff, {night: chances}, spec.as_record())
        outcome = ledger.grade(conn, night, played)
    return NightResult(night=night, cutoff=cutoff, outcome=outcome, chances=tuple(chances))

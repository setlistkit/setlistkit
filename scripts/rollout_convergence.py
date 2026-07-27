#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""How many rollouts does a confidence need before it stops moving?

``PredictConfig.rollouts`` defaults to 500, and that number is INHERITED, not measured here. It
comes from the pipeline this facility was ported from, whose own convergence study concluded
"rank-30 probability around 0.05 gives a standard error around 0.01" -- against a different band's
corpus, and under a config that still carried a base-rate blend this project has since dropped.
Two of the three things that number depended on have changed. So it gets re-measured rather than
trusted, and this is the thing that measures it.

Not a pytest test, deliberately. The output is a number to READ and then write down beside the
default; a test would have to assert a threshold, and a threshold picked to make today's corpus
pass is a threshold fitted to today's corpus.

WHAT IS MEASURED. A confidence is a share of draws, so it is a binomial proportion and its noise
is a function of the rollout count alone -- ``sqrt(p(1-p)/R)`` -- which is knowable without running
anything. What is NOT knowable in advance is where the interesting p values sit for a real corpus:
if rank 30 lands at p=0.30 rather than the 0.05 the inherited study assumed, the standard error at
500 draws is six times larger than that study's conclusion. So this runs the real scorer over the
real corpus, reads the p values off the ranking, and reports both the measured spread across
independent seeds and the theoretical standard error, which should agree.

Usage:

    pyenv/bin/python scripts/rollout_convergence.py --config phish-slkit.toml
    pyenv/bin/python scripts/rollout_convergence.py --config slkit.toml --set-len 18 --trials 12

Exit code is 0 unless the corpus cannot be read. Nothing here passes or fails.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# pylint: disable=wrong-import-position
from setlistkit.config import load_config                            # noqa: E402
from setlistkit.model.rollout import rollout_night                   # noqa: E402
from setlistkit.model.scores import ScoreConfig                      # noqa: E402
from setlistkit.model.settings import predict_config                 # noqa: E402
from setlistkit.store import Store                                   # noqa: E402

# The counts to compare. Spanning a factor of 32 around the inherited default, so the curve is
# visible rather than just the endpoint -- if 500 is already flat, 250 probably is too, and that
# is worth being able to see rather than infer.
LADDER = (50, 100, 250, 500, 1000, 1600)

# Ranks to report. Rank 1 is the least noisy place in the list and rank 30 the most, because a
# proportion's variance peaks at p=0.5 and falls off toward both ends -- but rank 30 is also the
# one whose p a reader has no intuition for, which is the whole reason the inherited study's
# assumption about it was worth checking.
WATCH = (1, 10, 30)


def _corpus(config_path: str):
    """Every stored show, and the corpus's last date."""
    config = load_config(config_path)
    with Store(config.data_root) as store:
        store.init()
        span = store.corpus.first_and_last()
        if span is None:
            raise SystemExit(f"the corpus in {config.data_root} is empty; run `slkit ingest`")
        return list(store.corpus.shows()), span[1], predict_config(config)


def _spread(confidences: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation of one rank's confidence across independent seeds."""
    if len(confidences) < 2:
        return (confidences[0] if confidences else 0.0), 0.0
    return statistics.fmean(confidences), statistics.stdev(confidences)


def _theoretical(p: float, rollouts: int) -> float:
    """The standard error a binomial proportion has at this p and this many draws.

    An UPPER BOUND on what is measured, not a prediction of it, and the gap is not an error. This
    is the noise on one fixed song's share; what the table measures is the share at a fixed RANK,
    which is an order statistic over thirty-odd songs. When one song's estimate wobbles down,
    whichever song is next takes its place, so the number at rank 10 moves far less than the song
    at rank 10 does. Measured spread running two to four times below this column is that effect,
    and measured spread running ABOVE it would mean something other than draw noise is moving the
    result between seeds -- which is the reason to print both.
    """
    return (p * (1.0 - p) / rollouts) ** 0.5


def _measure(shows, score_config, predicting, *, asof, set_len, rollouts, trials):
    """One rung of the ladder: run ``trials`` independent seeds and report the spread per rank."""
    runs = [rollout_night(shows, score_config, replace(predicting, rollouts=rollouts, seed=seed),
                          asof=asof, set_len=set_len)
            for seed in range(1, trials + 1)]
    out = {}
    for rank in WATCH:
        confidences = [run[rank - 1].confidence for run in runs if len(run) >= rank]
        if confidences:
            out[rank] = _spread(confidences)
    # How much the IDENTITY of the list moves, not just the numbers on it. A stable top-30 whose
    # members keep swapping is not converged, and the per-rank spread cannot see that at all.
    tops = [{chance.song for chance in run[:30]} for run in runs]
    churn = statistics.fmean([len(tops[0] ^ other) / 2 for other in tops[1:]]) if len(tops) > 1 \
        else 0.0
    return out, churn


def main(argv=None) -> int:
    """Walk the ladder and print what each rollout count buys."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to an slkit.toml")
    parser.add_argument("--asof", metavar="YYYY-MM-DD",
                        help="the night to draw (default: the day after the last stored show)")
    parser.add_argument("--set-len", type=int, default=None,
                        help="songs per draw (default: set_len from [model])")
    parser.add_argument("--trials", type=int, default=8,
                        help="independent seeds per rollout count (default: 8)")
    args = parser.parse_args(argv)

    shows, last, predicting = _corpus(args.config)
    asof = args.asof or _day_after(last)
    set_len = args.set_len or predicting.set_len

    print(f"corpus of {len(shows)} show(s) through {last}; drawing {asof}, "
          f"set of {set_len}, {args.trials} seeds per rung")
    print(f"\n{'draws':>6}  " + "  ".join(f"{'rank ' + str(r):>22}" for r in WATCH) + "   churn")
    print(f"{'':>6}  " + "  ".join(f"{'p +- measured (theory)':>22}" for _ in WATCH))
    for rollouts in LADDER:
        spreads, churn = _measure(shows, ScoreConfig(), predicting, asof=asof, set_len=set_len,
                                  rollouts=rollouts, trials=args.trials)
        cells = []
        for rank in WATCH:
            if rank not in spreads:
                cells.append(f"{'--':>22}")
                continue
            mean, sd = spreads[rank]
            cells.append(f"{mean:.3f} +-{sd:.4f} ({_theoretical(mean, rollouts):.4f})".rjust(22))
        print(f"{rollouts:>6}  " + "  ".join(cells) + f"  {churn:5.1f}")
    print("\nchurn is how many of the top 30 differ between two seeds, averaged over the trials.\n"
          "Read the rung where the rank-30 spread stops mattering for what you publish: a\n"
          "confidence printed to two decimal places is not improved by a standard error below\n"
          "0.005, and the cost of a prediction is in the scoring pass, not the draws.")
    return 0


def _day_after(date: str) -> str:
    """The earliest date the corpus does not hold. Not today -- see `catalog/window.py`."""
    import datetime as dt  # pylint: disable=import-outside-toplevel
    return (dt.date.fromisoformat(date) + dt.timedelta(days=1)).isoformat()


if __name__ == "__main__":
    sys.exit(main())

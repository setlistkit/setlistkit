# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""``slkit predict``, ``slkit replay`` and ``slkit drops``: one night, every night, every night twice.

Three verbs over the same machinery, and the differences between them are the whole design.

``predict`` answers a night the corpus does not hold, which is what a person means by predicting.
It has to guess how many songs will be played, because nothing tells it -- that is the configured
``set_len``, and it is the open question this slice does not answer.

``replay`` answers nights the corpus DOES hold, one at a time, by hiding everything from the
predictor that it should not have been able to see. It is the only one of these that can be
graded, because it is the only one where the answer is already known. It is also how the facility
gets exercised end to end without acquiring a schedule source: the corpus supplies the upcoming
dates and the ground truth both.

``drops`` answers each of those nights SEVERAL times, from successively later cutoffs, all of them
still strictly before the night. That turns a verdict into a story: not only whether the model was
right, but whether it knew early or worked it out at the last minute. Its final drop is the same
prediction ``replay`` makes, so the two agree about every graded number.

NONE OF THEM READS THE CLOCK. ``predict`` defaults its as-of date to the day after the corpus's
last show and the walks anchor their window on ``last_show``, so two runs over the same store
agree with each other whenever they are run -- the same rule ``catalog/window.py`` states at
length and refuses ``today`` to enforce.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

from ..catalog.window import resolve_explained
from ..diagnostics import ERROR, Diagnostic, DiagnosticError
from ..model.predict import predict_night
from ..model.settings import predict_config, replay_window, score_config
from ..picks.drops import DEFAULT_DEPTH, drops as walk_drops
from ..picks.replay import ReplaySpec, replay as walk
from ..store import Store
from ..store.daterange import check_date

EXIT_OK = 0

# How wide the confidence column prints. A confidence is a share of rollouts, so at 500 rollouts
# it carries three decimal places of real resolution and two of useful ones.
_CONF = "{:.2f}"


def _configs(config, args):
    """The two frozen configs this run works under, with the flags folded in.

    ``--rollouts`` and ``--set-len`` override the config file rather than editing it, because they
    are iteration flags: a lower rollout count shows the direction of a change three times faster
    and is the wrong thing to leave behind in a file that a nightly run reads.
    """
    predicting = predict_config(config)
    if getattr(args, "rollouts", None):
        predicting = replace(predicting, rollouts=args.rollouts)
    if getattr(args, "set_len", None):
        predicting = replace(predicting, set_len=args.set_len)
    # Read from the band's own config rather than constructed from the defaults. The defaults
    # were fitted on moe. and `era_full_from_year = 2023` in particular is a fact about moe.'s
    # corpus; a second band has to be able to say its own corpus measured differently. See
    # `settings.score_config` for the bar a value written there has to clear.
    return score_config(config), predicting


def _corpus(store) -> tuple[list[dict], str, str]:
    """Every stored show, plus the corpus's first and last dates. Refuses an empty store.

    Refused here rather than allowed to produce an empty prediction: a scorer with nothing to
    score returns an empty ranking, which prints as a prediction that named no songs rather than
    as a corpus nobody has ingested.
    """
    span = store.corpus.first_and_last()
    if span is None:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="nothing to predict from: the corpus is empty",
            detail="Run `slkit ingest` first. A prediction is made out of play history, and\n"
                   "there is none stored.",
        ))
    return store.corpus.shows(), span[0], span[1]


def _next_night(last: str) -> str:
    """The day after the corpus's last show: the earliest date it does not already hold.

    Not ``today``, deliberately, and for the reason ``catalog/window.py`` gives -- an as-of date
    tied to wall-clock time makes two runs over the same store disagree. This is a pure function
    of the data, and it is honest about what it is: the first night we have no record of, which is
    the nearest thing to "next show" available without a schedule.
    """
    return (dt.date.fromisoformat(last) + dt.timedelta(days=1)).isoformat()


def predict(config, args) -> int:
    """`slkit predict`: one night's ranked chances and the setlist they construct."""
    scoring, predicting = _configs(config, args)
    with Store(config.data_root) as store:
        store.init()
        shows, _first, last = _corpus(store)
        asof = check_date(args.asof, "--asof") or _next_night(last)
        prediction = predict_night(shows, scoring, predicting, asof=asof)
    print(f"predict {prediction.asof}: reading {len(shows)} show(s) through {last}")
    print(f"  {predicting.rollouts} rollouts, seed {predicting.seed}, "
          f"set of {prediction.set_len}, top {predicting.top_n} recorded")
    if asof <= last:
        # Said out loud, because it changes what the output means. Predicting a night the corpus
        # already holds is a backtest with the answer withheld -- honest, and gradeable, but not
        # the thing somebody typing `slkit predict` usually has in mind.
        print(f"  note: {prediction.asof} is inside the corpus, so this is a blinded re-prediction"
              f"\n        of a night already played. Nothing dated on or after it was read.")
    print()
    for chance in prediction.chances:
        print(f"  {chance.rank:3d}  {_CONF.format(chance.confidence)}  {chance.song}")
    print()
    print(f"  set : {_setlist_line(prediction.setlist)}")
    print(f"  enc : {_setlist_line(prediction.encore) or '(nothing the set had not already used)'}")
    return EXIT_OK


def _setlist_line(entries) -> str:
    """A constructed setlist as one line, segues marked the way a setlist is written."""
    return " ".join(entry.song + (" >" if entry.segue else ",") for entry in entries).rstrip(",")


def _window(config, store, args) -> tuple[str, str, str]:
    """The span of nights to grade, and one line saying where it came from.

    The flags win over the configured window, one endpoint at a time, so ``--since`` alone narrows
    the start and leaves the configured end alone rather than silently widening it to the anchor.
    """
    span = store.corpus.first_and_last()
    resolved = resolve_explained(replay_window(config), first=span[0], last=span[1])
    since = check_date(args.since, "--since") or resolved.since.value
    until = check_date(args.until, "--until") or resolved.until.value
    how = (f"{resolved.since.words}, anchored on {resolved.anchor}"
           if args.since is None else "given on the command line")
    return since, until, how


def replay(config, args) -> int:
    """`slkit replay`: predict and grade every gradeable night in the window, oldest first."""
    scoring, predicting = _configs(config, args)
    with Store(config.data_root) as store:
        store.init()
        shows, _first, last = _corpus(store)
        since, until, how = _window(config, store, args)
        spec = ReplaySpec(score=scoring, predict=predicting, since=since, until=until)
        print(f"replay {since} .. {until} ({how}); corpus of {len(shows)} show(s) through {last}")
        print(f"  {predicting.rollouts} rollouts, seed {predicting.seed}, "
              f"top {predicting.top_n} recorded, grading nights of "
              f"{spec.min_songs}+ songs")
        if args.dry_run:
            print("  dry run: nothing will be written to the ledger")
        print()
        report = walk(store.conn, shows, spec, dry_run=args.dry_run, force=args.force,
                      progress=_report_night)
    _report_aggregate(report)
    return EXIT_OK


def drops_cmd(config, args) -> int:
    """`slkit drops`: predict every night in the window from several cutoffs, and grade the last.

    Same window resolution as ``replay`` on purpose -- a churn number and a recall number quoted
    off the same corpus have to be talking about the same nights, and two commands that spell
    "the last six months" differently make that impossible to check.
    """
    scoring, predicting = _configs(config, args)
    depth = args.depth or DEFAULT_DEPTH
    with Store(config.data_root) as store:
        store.init()
        shows, _first, last = _corpus(store)
        since, until, how = _window(config, store, args)
        spec = ReplaySpec(score=scoring, predict=predicting, since=since, until=until)
        print(f"drops {since} .. {until} ({how}); corpus of {len(shows)} show(s) through {last}")
        print(f"  depth {depth}: each night predicted from its last {depth} cutoff(s), "
              f"{predicting.rollouts} rollouts, top {predicting.top_n} recorded")
        if args.dry_run:
            print("  dry run: nothing will be written to the ledger")
        print()
        report = walk_drops(store.conn, shows, spec, depth=depth, dry_run=args.dry_run,
                            force=args.force, progress=_report_drops)
    _report_drops_aggregate(report)
    return EXIT_OK


def _report_drops(night) -> None:
    """One night's drops, printed as the walk finishes it."""
    print(f"  {night.night}  {len(night.drops):>2} drop(s) from {night.drops[0].cutoff}  "
          f"drift {night.drift[-1]:.3f}  rbo {night.rbo_drift[-1]:.3f}  "
          f"settled {night.settled_at or 'never'}", flush=True)


def _report_drops_aggregate(report) -> None:
    """What the whole walk came to. Numbers, no verdict -- the same rule the replay follows."""
    print()
    if not report.nights:
        print("  no gradeable nights in that window; nothing was predicted")
        return
    graded = [night for night in report.nights if night.outcome is not None]
    total = sum(len(night.drops) for night in report.nights)
    print(f"  {len(report.nights)} night(s), {total} drop(s): mean step churn "
          f"{report.mean_step:.3f}")
    print(f"  {report.settled_early} night(s) had settled before their final cutoff")
    if graded:
        hits = sum(night.outcome.actual_hits for night in graded)
        expected = sum(night.outcome.expected_hits for night in graded)
        if expected > 0:
            print(f"  {len(graded)} graded: {hits} hits against {expected:.1f} expected, "
                  f"calibration {hits / expected:.3f}")
        else:
            # Not a divide-by-zero dodge: a run where every confidence came back zero claimed
            # nothing, and printing "calibration 0.000" for it would read as a bad model rather
            # than as a model that made no claim to be graded on.
            print(f"  {len(graded)} graded: {hits} hits, but nothing was claimed")
    if report.short:
        print(f"  {report.short} night(s) had less history behind them than depth {report.depth}")
    if report.skipped:
        print(f"  {report.skipped} night(s) skipped: no played show behind them to predict from")


def _report_night(result) -> None:
    """One graded night, printed as it lands.

    flush=True because a walk over eighteen months is minutes of scoring and Python buffers stdout
    whenever it is not a terminal -- redirected to a log, unflushed progress arrives after the job
    it was reporting on has finished, which is not progress.
    """
    outcome = result.outcome
    print(f"  {result.night}  from {result.cutoff}  "
          f"{outcome.actual_hits:3d}/{outcome.n_actual:<3d} played  "
          f"recall {outcome.recall_at_n:.3f}  expected {outcome.expected_hits:5.1f}  "
          f"calibration {outcome.calibration_ratio:.2f}", flush=True)


def _report_aggregate(report) -> None:
    """What the whole walk came to.

    No verdict, and no threshold. Calibration near 1.0 is the target and this prints the number
    rather than judging it: a floor picked before the first measurement is a floor fitted to hope.
    """
    print()
    if not report.nights:
        print("  no gradeable nights in that window; nothing was predicted")
        return
    print(f"  {len(report.nights)} night(s): mean recall {report.mean_recall:.3f}, "
          f"{report.total_hits} hits against {report.total_expected:.1f} expected, "
          f"calibration {report.calibration:.3f}")
    if report.skipped:
        print(f"  {report.skipped} night(s) skipped: no played show behind them to predict from")

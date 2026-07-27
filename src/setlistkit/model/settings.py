# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The `[model]` config stanza, and the only place it is read.

One module owns one section, the same way :mod:`setlistkit.catalog.window` owns
`[reports.*.window]`. What lives in config rather than in source is every number a person might
reasonably want to change without editing Python -- the set length, the rollout count, the seed,
the span a replay walks -- and what stays in source is everything whose value is a measured claim
(see :class:`~setlistkit.model.scores.ScoreConfig`, whose docstring is the measurement).

An unknown key is refused rather than ignored. The whole point of these being in config is that a
number can be changed there, and a silently skipped ``rollout = 200`` is a change somebody
believes they made.

THE REPLAY SPAN IS A WINDOW, NOT A PAIR OF DATES. `replay_since_back` is an offset from the
corpus's own last show, resolved through the same validated grammar report windows use -- so "the
last eighteen months" stays a pure function of the store, slides forward on its own as shows are
ingested, and never depends on what day the command was run. The version this comes from hardcoded
``RUN_START = "2026-06-11"`` and ``PRE_JUNE = "2026-06-01"``, which is exactly the failure mode
`[reports.*.window]` was built to end.
"""

from __future__ import annotations

from ..catalog.window import WindowError, WindowSpec, parse_offset
from ..diagnostics import ERROR, Diagnostic, DiagnosticError
from .scores import ScoreConfig
from .setlists import PredictConfig

# field -> whether a float is acceptable there. Everything else must be a whole number, CHECKED
# rather than coerced: TOML's `set_len = "10"` is a string, and `int("10")` would paper over it
# here while the same value reached the ledger's stored config as a quoted one. Same lesson as
# `min_year`, and the same fix.
_PREDICT_FIELDS = {"set_len": False, "encore_len": False, "rollouts": False, "seed": False,
                   "segue_threshold": True, "top_n": False}

# Keys the stanza may hold that are not PredictConfig fields. Kept beside the others so the
# unknown-key check sees one vocabulary rather than rejecting a key this module itself reads.
_SPAN_KEY = "replay_since_back"

# `[model.score]` arrives as a sub-table of `[model]`, so it shows up here as a key named "score".
# Named rather than "allow any dict", so a typo'd `[model.scores]` is still refused.
_SCORE_KEY = "score"

# How far back a replay walks when the config does not say. Eighteen months of recent history is
# roughly ninety gradeable nights for this corpus, which is enough for the standard error on the
# aggregate to mean something without the walk taking long enough that nobody runs it.
DEFAULT_REPLAY_SPAN = "P18M"

# `[model.score]` field -> the type TOML must hand over. `int` means a whole number and refuses a
# float; `(int, float)` accepts either and stores a float, because a half-life of 400 and 400.0
# are the same number and refusing one of them helps nobody.
_SCORE_FIELDS: dict[str, type | tuple[type, ...]] = {
    "recency_halflife_days": (int, float),
    "era_full_from_year": int,
    "era_weight_before": (int, float),
    "gap_weight": (int, float),
    "gap_cap": (int, float),
    "min_show_songs": int,
    "lookback_years": int,
}
_SCORE_NAMES = {key: ("a whole number" if wanted is int else "a number")
                for key, wanted in _SCORE_FIELDS.items()}
_SCORE_EXAMPLES = {"recency_halflife_days": "900", "era_full_from_year": "2009",
                   "era_weight_before": "0.5", "gap_weight": "0.35", "gap_cap": "3.0",
                   "min_show_songs": "5", "lookback_years": "12"}


def _refuse(config, summary: str, detail: str) -> DiagnosticError:
    """One shape for every complaint about this stanza, naming the file it is in."""
    return DiagnosticError(Diagnostic(severity=ERROR, summary=summary,
                                      path=str(config.source_path), detail=detail))


def _check_keys(config, raw) -> None:
    known = set(_PREDICT_FIELDS) | {_SPAN_KEY, _SCORE_KEY}
    extra = set(raw) - known
    if extra:
        raise _refuse(
            config, f"[model] has unknown key(s): {', '.join(sorted(extra))}",
            f"Known keys are: {', '.join(sorted(known))}.\n\n"
            "Refused rather than ignored: these live in config so a number can be changed\n"
            "without editing source, and a key that is quietly skipped is a change somebody\n"
            "believes they made.")


def predict_config(config) -> PredictConfig:
    """`[model]` as a :class:`~setlistkit.model.setlists.PredictConfig`, defaults where silent.

    Shape is checked here; whether the values make SENSE together is not, matching how
    `[reports.*.window]` is handled -- a config is only as broken as using it proves it to be, and
    a stanza nobody has asked to predict from should not stop the rest of the config loading.
    """
    raw = config.section("model")
    _check_keys(config, raw)
    values = {}
    for key, allow_float in _PREDICT_FIELDS.items():
        if key not in raw:
            continue
        value = raw[key]
        wanted = (int, float) if allow_float else int
        if not isinstance(value, wanted) or isinstance(value, bool):
            example = "0.45" if allow_float else "10"
            raise _refuse(
                config,
                f"[model] {key} must be {'a number' if allow_float else 'a whole number'}, "
                f"got {value!r}",
                f"Write it unquoted:\n\n    [model]\n    {key} = {example}")
        values[key] = float(value) if allow_float else int(value)
    return PredictConfig(**values)


def score_config(config) -> ScoreConfig:
    """`[model.score]` as a :class:`~setlistkit.model.scores.ScoreConfig`, defaults where silent.

    THIS STANZA IS AN EXCEPTION TO THE RULE AT THE TOP OF THIS MODULE, and it is worth saying why
    rather than letting the exception quietly become the rule.

    The rule is that a measured claim lives in source. It was written when there was one band, and
    it was right: ``ScoreConfig``'s docstring IS the measurement, taken over moe.'s corpus, and a
    config key would have let somebody move a number without moving the evidence.

    A second band breaks it, and not at the margin. ``era_full_from_year = 2023`` is a fact about
    moe. wearing the costume of a constant. ``recency_halflife_days = 200`` was fitted against a
    band whose rotation is dominated by the last few years; against one that returns to material
    it has not played in half a decade, the same number does not mean the same thing -- it means
    the deep catalog is weighted to nothing, which the source docstring already reports as "87% of
    the corpus cannot reach this scorer at all".

    So the defaults stay exactly where they are and stay the measured claim. What this adds is the
    ability for a band to say, in its own config, that its own corpus measured differently. THE
    BAR FOR WRITING A KEY HERE IS THE SAME BAR THE DEFAULTS MET: a walk-forward sweep on a window
    that is not the window the result is reported from. A value typed in because it looked better
    in sample is the exact failure the scores docstring records this scorer's ancestor making.

    ``cooldown`` is deliberately NOT readable here. It is a tuple whose index means shows-since-
    last-play, it is the largest single lever in the scorer, and a TOML array offers no way to say
    that -- a four-element list typed in the wrong order is a silent, plausible catastrophe. It
    stays in source until there is a reason and a shape for moving it.
    """
    # The parent stanza is checked too, so a typo'd `[model.scores]` is caught by the command that
    # wanted the scores rather than silently read as defaults by the one command that would notice.
    _check_keys(config, config.section("model"))
    raw = config.section("model", "score")
    extra = set(raw) - set(_SCORE_FIELDS)
    if extra:
        raise _refuse(
            config, f"[model.score] has unknown key(s): {', '.join(sorted(extra))}",
            f"Known keys are: {', '.join(sorted(_SCORE_FIELDS))}.\n\n"
            "`cooldown` is deliberately absent: it is a tuple indexed by shows-since-last-play,\n"
            "and a TOML array gives no way to say so.")
    values = {}
    for key, wanted in _SCORE_FIELDS.items():
        if key not in raw:
            continue
        value = raw[key]
        if key == "lookback_years" and value is None:
            continue                     # explicit null means "the whole corpus", the default
        if not isinstance(value, wanted) or isinstance(value, bool):
            raise _refuse(
                config, f"[model.score] {key} must be {_SCORE_NAMES[key]}, got {value!r}",
                f"Write it unquoted:\n\n    [model.score]\n    {key} = {_SCORE_EXAMPLES[key]}")
        values[key] = float(value) if wanted is not int else int(value)
    return ScoreConfig(**values)


def replay_window(config) -> WindowSpec:
    """The span a replay grades, as a window anchored on the corpus's last show.

    Validated here rather than at resolve time so a typo is reported by the command that read the
    config, not by the one that eventually resolved it. `PT18M` is the trap the grammar exists to
    close -- eighteen MINUTES, and a perfectly valid ISO 8601 duration.

    The `WindowError` is re-raised as a diagnostic naming the file, because `slkit` renders those
    and lets anything else out as a traceback. A config error a reader cannot locate is half an
    error, and the grammar's own message does not say which file it read.
    """
    raw = config.section("model")
    _check_keys(config, raw)
    offset = str(raw.get(_SPAN_KEY) or DEFAULT_REPLAY_SPAN)
    try:
        parse_offset(offset)
    except WindowError as exc:
        raise _refuse(
            config, f"[model] {_SPAN_KEY} is not a valid window offset: {offset!r}",
            f"{exc}\n\nHow far back `slkit replay` walks, as an offset from the last stored\n"
            f"show:\n\n    [model]\n    {_SPAN_KEY} = \"{DEFAULT_REPLAY_SPAN}\"") from exc
    return WindowSpec(anchor="last_show", since_back=offset)

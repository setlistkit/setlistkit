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

# How far back a replay walks when the config does not say. Eighteen months of recent history is
# roughly ninety gradeable nights for this corpus, which is enough for the standard error on the
# aggregate to mean something without the walk taking long enough that nobody runs it.
DEFAULT_REPLAY_SPAN = "P18M"


def _refuse(config, summary: str, detail: str) -> DiagnosticError:
    """One shape for every complaint about this stanza, naming the file it is in."""
    return DiagnosticError(Diagnostic(severity=ERROR, summary=summary,
                                      path=str(config.source_path), detail=detail))


def _check_keys(config, raw) -> None:
    known = set(_PREDICT_FIELDS) | {_SPAN_KEY}
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

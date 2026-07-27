# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for reading the `[model]` stanza.

The stanza exists so a number can be changed without editing Python. Everything below follows from
taking that seriously: an unknown key is a change somebody believes they made, and a quoted number
is a value that would reach the ledger's stored config still quoted. Both are refused rather than
smoothed over, and the tests are here because both failure modes are silent.
"""

import pytest

from setlistkit.config import load_config
from setlistkit.diagnostics import DiagnosticError
from setlistkit.model.settings import DEFAULT_REPLAY_SPAN, predict_config, replay_window

BASE = ('data_root = "state"\n'
        'user_agent = "famoe.ly nightly (you@example.com)"\n')


def _config(tmp_path, stanza=""):
    path = tmp_path / "slkit.toml"
    path.write_text(BASE + stanza, encoding="utf-8")
    return load_config(str(path), env={}, cwd=tmp_path)


def test_a_config_with_no_model_stanza_gets_every_default(tmp_path):
    assert predict_config(_config(tmp_path)).rollouts == 500


def test_a_stated_value_wins_over_the_default(tmp_path):
    predicting = predict_config(_config(tmp_path, "[model]\nrollouts = 200\nseed = 7\n"))
    assert predicting.rollouts == 200 and predicting.seed == 7
    assert predicting.top_n == 30


def test_the_segue_threshold_may_be_a_fraction(tmp_path):
    assert predict_config(_config(tmp_path, "[model]\nsegue_threshold = 0.6\n")) \
        .segue_threshold == 0.6


def test_a_whole_number_is_acceptable_for_the_one_field_that_takes_a_fraction(tmp_path):
    """`segue_threshold = 1` is a band that has to have segued every time. Unusual, not wrong."""
    assert predict_config(_config(tmp_path, "[model]\nsegue_threshold = 1\n")) \
        .segue_threshold == 1.0


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path):
    """`rollout = 200` is the real case: singular, plausible, and does nothing."""
    with pytest.raises(DiagnosticError) as caught:
        predict_config(_config(tmp_path, "[model]\nrollout = 200\n"))
    assert "unknown key" in caught.value.diagnostic.summary
    assert "rollout" in caught.value.diagnostic.summary
    # The complaint names the file, because a config error a reader cannot locate is half an error.
    assert "slkit.toml" in caught.value.diagnostic.path


def test_a_quoted_number_is_refused_rather_than_coerced(tmp_path):
    """`int("10")` would work here and leave the quoted value in the ledger's stored config."""
    with pytest.raises(DiagnosticError) as caught:
        predict_config(_config(tmp_path, '[model]\nset_len = "10"\n'))
    assert "must be a whole number" in caught.value.diagnostic.summary


def test_a_fraction_is_refused_where_a_whole_number_is_required(tmp_path):
    with pytest.raises(DiagnosticError):
        predict_config(_config(tmp_path, "[model]\nrollouts = 200.5\n"))


def test_a_boolean_is_not_a_number(tmp_path):
    """Python says `isinstance(True, int)`. TOML users do not."""
    with pytest.raises(DiagnosticError):
        predict_config(_config(tmp_path, "[model]\nset_len = true\n"))


def test_the_replay_span_defaults_to_eighteen_months(tmp_path):
    window = replay_window(_config(tmp_path))
    assert window.anchor == "last_show"
    assert window.since_back == DEFAULT_REPLAY_SPAN == "P18M"


def test_the_replay_span_is_anchored_on_the_corpus_rather_than_on_today(tmp_path):
    """The failure the window grammar exists to end: the version this comes from hardcoded
    `RUN_START = "2026-06-11"`, so its window meant something different every week."""
    assert replay_window(_config(tmp_path, '[model]\nreplay_since_back = "P3Y"\n')).anchor \
        == "last_show"


def test_a_stated_span_wins(tmp_path):
    assert replay_window(_config(tmp_path, '[model]\nreplay_since_back = "P2Y6M"\n')) \
        .since_back == "P2Y6M"


def test_eighteen_minutes_is_refused_where_eighteen_months_was_meant(tmp_path):
    """`PT18M` is a perfectly valid ISO 8601 duration and is the trap the grammar closes.

    Validated when the config is READ, so the typo is reported by the command that loaded it
    rather than by whatever eventually tried to resolve the window.
    """
    with pytest.raises(DiagnosticError):
        replay_window(_config(tmp_path, '[model]\nreplay_since_back = "PT18M"\n'))


def test_the_span_key_is_not_mistaken_for_an_unknown_one(tmp_path):
    """Both functions check the same vocabulary, so neither may reject what the other reads."""
    stanza = '[model]\nreplay_since_back = "P1Y"\nrollouts = 100\n'
    assert predict_config(_config(tmp_path, stanza)).rollouts == 100
    assert replay_window(_config(tmp_path, stanza)).since_back == "P1Y"

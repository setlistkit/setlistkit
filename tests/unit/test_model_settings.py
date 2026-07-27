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
from setlistkit.model.settings import (
    DEFAULT_REPLAY_SPAN,
    predict_config,
    replay_window,
    score_config,
)

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


def test_a_band_that_says_nothing_about_scoring_gets_the_measured_defaults(tmp_path):
    """The defaults are a measurement, not a placeholder. Silence must not move them."""
    scoring = score_config(_config(tmp_path))
    assert scoring.recency_halflife_days == 200.0
    assert scoring.era_full_from_year == 2023
    assert scoring.cooldown == (0.12, 0.55, 0.82, 0.93)


def test_a_band_may_state_its_own_half_life_without_moving_anybody_elses(tmp_path):
    """The whole reason this stanza exists: two bands, one scorer, two corpora.

    `era_full_from_year = 2023` is a fact about moe. wearing the costume of a constant, and a
    band whose corpus starts in 2009 has to be able to say so in its own config.
    """
    scoring = score_config(_config(
        tmp_path, "[model.score]\nrecency_halflife_days = 900\nera_full_from_year = 2009\n"))
    assert scoring.recency_halflife_days == 900.0
    assert scoring.era_full_from_year == 2009
    # Untouched fields keep the measured defaults rather than being reset alongside.
    assert scoring.gap_weight == 0.35
    assert score_config(_config(tmp_path)).recency_halflife_days == 200.0


def test_a_whole_number_is_fine_where_a_fraction_is_allowed(tmp_path):
    """900 and 900.0 are the same half-life, and refusing one of them helps nobody."""
    assert score_config(_config(
        tmp_path, "[model.score]\nrecency_halflife_days = 900\n")).recency_halflife_days == 900.0


def test_a_fraction_is_refused_where_only_a_whole_number_makes_sense(tmp_path):
    """A year is not 2009.5, and coercing it would hide the fact that somebody meant something."""
    with pytest.raises(DiagnosticError):
        score_config(_config(tmp_path, "[model.score]\nera_full_from_year = 2009.5\n"))


def test_a_quoted_half_life_is_refused_rather_than_coerced(tmp_path):
    """Same lesson as `min_year`: `int("900")` papers over it here and not downstream.

    Named for its stanza rather than for its lesson, because the `[model]` test above teaches the
    same lesson and shared the same name -- which meant the later definition shadowed the earlier
    and one of the two silently stopped running.
    """
    with pytest.raises(DiagnosticError):
        score_config(_config(tmp_path, '[model.score]\nrecency_halflife_days = "900"\n'))


def test_an_unknown_scoring_key_is_refused_because_it_is_a_change_nobody_made(tmp_path):
    with pytest.raises(DiagnosticError) as caught:
        score_config(_config(tmp_path, "[model.score]\nhalflife = 900\n"))
    assert "halflife" in str(caught.value)


def test_the_cooldown_cannot_be_set_from_config_at_all(tmp_path):
    """Deliberately unreachable, and the refusal says so.

    It is a tuple indexed by shows-since-last-play and the largest single lever in the scorer.
    A four-element TOML array typed in the wrong order is a silent, plausible catastrophe.
    """
    with pytest.raises(DiagnosticError) as caught:
        score_config(_config(tmp_path, "[model.score]\ncooldown = [0.12, 0.55, 0.82, 0.93]\n"))
    assert "cooldown" in str(caught.value)


def test_a_typo_in_the_stanza_name_is_refused_by_the_command_that_wanted_it(tmp_path):
    """`[model.scores]` would otherwise read as an empty table and silently yield defaults."""
    with pytest.raises(DiagnosticError) as caught:
        score_config(_config(tmp_path, "[model.scores]\nrecency_halflife_days = 900\n"))
    assert "scores" in str(caught.value)


def test_the_scoring_sub_table_does_not_make_the_parent_stanza_complain(tmp_path):
    """`[model.score]` arrives as a key named "score" inside `[model]`, and must be known there."""
    config = _config(tmp_path, "[model]\nrollouts = 200\n\n[model.score]\ngap_cap = 4.0\n")
    assert predict_config(config).rollouts == 200
    assert replay_window(config).since_back == DEFAULT_REPLAY_SPAN
    assert score_config(config).gap_cap == 4.0

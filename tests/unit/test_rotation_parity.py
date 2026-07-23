# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Cross-language parity: pin setlistkit.model.scores against the Songbook's browser JS.

See tests/unit/fixtures/rotation_formula.js for why this runs a maintained COPY of
aggregateWindow()'s rotation arithmetic (pinned to dueRatio()) rather than famoe.ly's real
logic.js -- setlistkit and famoe.ly are separate repositories and neither checks out the
other. The formula both sides
are checked against is published in src/setlistkit/catalog/songbook.py's module docstring.

node is an optional runtime dependency of this one test, not a declared project dependency --
see the skip in _js_side() below.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from setlistkit.model.scores import OVERDUE_FALLBACK_RATIO, overdue_ratio

JS_FORMULA = Path(__file__).resolve().parent / "fixtures" / "rotation_formula.js"

# Each case is (label, n, lastIdx, gaps) in show-index space, chosen to exercise every branch
# of the formula published in songbook.py: an ordinary song with repeat gaps, a song played in
# the newest show (gap == 0), a single-play song at three positions in the window (the
# "can never be overdue" property, pinned again here against the JS side), and the degenerate
# n == 0 case -- the only way the ratio's own fallback can fire, since a real window always
# has at least one show for a song to have a base rate at all.
CASES = [
    ("repeat gaps",            10, 7, [3, 3]),
    ("played in newest show",  10, 9, [2, 4]),
    ("single play, early",      8, 0, []),
    ("single play, middle",     8, 4, []),
    ("single play, newest",     8, 7, []),
    ("empty window",            0, 0, []),   # meanGap falls back to n == 0 -> the fallback
]


def _python_side(n, last_idx, gaps):
    """The same four lines rotation_formula.js implements, computed in Python.

    Deliberately not calling rotation() here: that function's signature takes a list of
    (date, songs) shows, and reconstructing a synthetic show list just to reach the n == 0
    degenerate case would obscure the number actually under test. overdue_ratio() is the one
    piece of the formula both languages truly duplicate, so it is what gets called directly.
    """
    gap = n - 1 - last_idx
    mean_gap = (sum(gaps) / len(gaps)) if gaps else float(n)
    return {"gap": gap, "meanGap": mean_gap, "ratio": overdue_ratio(gap, mean_gap)}


def _js_side(cases):
    if shutil.which("node") is None:
        pytest.skip("node not on PATH -- see tests/unit/fixtures/rotation_formula.js for why "
                    "this check needs it, and why it is a skip rather than a hard failure")
    payload = [{"n": n, "lastIdx": last_idx, "gaps": gaps, "fallback": OVERDUE_FALLBACK_RATIO}
               for _, n, last_idx, gaps in cases]
    result = subprocess.run(["node", str(JS_FORMULA)], input=json.dumps(payload),
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_python_and_js_agree_on_every_case():
    """The actual gate. Python's overdue_ratio() and the JS copy of aggregateWindow()'s
    rotation arithmetic must produce identical gap/meanGap/ratio for identical inputs -- both are plain
    float64 division performed in the same order, so exact equality is the right assertion,
    not pytest.approx quietly forgiving a real disagreement."""
    js_results = _js_side(CASES)
    for (label, n, last_idx, gaps), js in zip(CASES, js_results):
        py = _python_side(n, last_idx, gaps)
        assert py == js, f"{label}: python={py} js={js}"


def test_a_single_play_song_is_never_overdue_in_the_js_side_either():
    """The structural property pinned in test_scores.py, checked against the JS copy too --
    it is a property of the ARITHMETIC, not of one language's implementation of it."""
    single_play_cases = [c for c in CASES if c[0].startswith("single play")]
    for result in _js_side(single_play_cases):
        assert result["ratio"] < 1.0

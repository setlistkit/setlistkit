# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The churn metrics, pinned by hand.

Both functions are pure and take two lists, so every case here is checkable on paper. That is the
point of testing them apart from the walk that uses them: a churn number that is wrong inside a
432-drop walk looks exactly like a band that changes its mind a lot.
"""

import pytest

from setlistkit.picks import churn


def test_two_identical_lists_have_not_moved():
    """Both metrics read zero for the same list, and RBO's extrapolation is why.

    A truncated equal-length RBO sum converges to noticeably UNDER 1.0 similarity, which would
    report an unchanged list as having moved. The tail term closes that gap -- to float precision,
    not exactly, which is why this uses a tolerance. Measured residual is about 1e-16, so the
    bound here is far tighter than anything a page would ever display and would still catch a
    missing or mis-weighted tail term.
    """
    same = ["a", "b", "c", "d"]
    assert churn.turnover(same, same) == 0.0
    assert churn.rbo(same, same) == pytest.approx(0.0, abs=1e-12)


def test_two_lists_sharing_nothing_are_a_full_distance_apart():
    assert churn.turnover(["a", "b"], ["c", "d"]) == 1.0
    assert churn.rbo(["a", "b"], ["c", "d"]) == 1.0


def test_turnover_is_the_jaccard_distance_and_counts_both_sides():
    """Three shared out of five in the union: 1 - 3/5.

    Written with an unequal split on purpose. A metric that counted only departures would read
    1/4 here and a metric that counted only arrivals 1/3; the union is what makes one number
    describe both.
    """
    assert churn.turnover(["a", "b", "c", "d"], ["b", "c", "d", "e"]) == pytest.approx(1 - 3 / 5)


def test_turnover_ignores_order_and_rbo_does_not():
    """The case that justifies shipping two numbers instead of one.

    Same songs, exactly reversed. Membership says nothing changed; a reader looking at the top of
    the list would say everything did.

    Twelve items rather than four, deliberately. On a short list RBO's tail term extrapolates the
    agreement found at the deepest common depth outward and swamps the ordering signal -- a
    reversed four-item list scores only ~0.22. A shortlist is thirty items, so twelve is the
    honest end of the range to assert against.
    """
    forward = list("abcdefghijkl")
    assert churn.turnover(forward, list(reversed(forward))) == 0.0
    assert churn.rbo(forward, list(reversed(forward))) > 0.4


def test_rbo_weights_the_top_of_the_list_far_more_heavily():
    """One swap at ranks 1-2 must cost more than the same swap at ranks 9-10.

    This is the whole reason RBO is here rather than a plain positional diff. A song entering at
    rank 2 is the model changing its mind; the same song entering at rank 10 is the tail moving.
    """
    base = list("abcdefghij")
    top_swapped = ["b", "a"] + base[2:]
    tail_swapped = base[:8] + ["j", "i"]
    assert churn.rbo(base, top_swapped) > churn.rbo(base, tail_swapped)


def test_both_metrics_are_symmetric():
    """Distance, not direction. A chart that flipped sign on argument order would be unreadable."""
    first, second = ["a", "b", "c"], ["c", "a", "d"]
    assert churn.turnover(first, second) == churn.turnover(second, first)
    assert churn.rbo(first, second) == pytest.approx(churn.rbo(second, first))


def test_two_empty_lists_agree_but_an_empty_against_a_full_one_does_not():
    """An empty drop is not a drop that agreed, and the two cases must not collapse."""
    assert churn.turnover([], []) == 0.0
    assert churn.rbo([], []) == 0.0
    assert churn.turnover([], ["a"]) == 1.0
    assert churn.rbo([], ["a"]) == 1.0


def test_series_returns_drift_and_step_indexed_with_the_drops():
    """Both series start at 0.0 and are as long as the drop list.

    The leading zero carries weight: it is what lets a chart plot drops, drift and step against
    one x-axis without offsetting one of them, which is where an off-by-one would live.
    """
    drops = [["a", "b"], ["a", "b"], ["a", "c"]]
    drift, step = churn.series(drops)
    assert len(drift) == len(step) == len(drops)
    assert drift[0] == step[0] == 0.0
    # Drop 2 is unchanged from both the first drop and the previous one.
    assert drift[1] == 0.0 and step[1] == 0.0
    # Drop 3 swaps one of two songs: shared 1, union 3.
    assert drift[2] == pytest.approx(1 - 1 / 3)
    assert step[2] == pytest.approx(1 - 1 / 3)


def test_series_of_no_drops_is_two_empty_series():
    assert churn.series([]) == ((), ())


def test_step_and_drift_diverge_when_a_list_wanders_and_comes_back():
    """Drift is measured from the first drop, step from the previous one, and they must differ.

    A list that leaves and returns has zero drift and non-zero step. If these were the same
    walk with a different starting index, this case would read identically on both series and
    the page would be showing one number twice.
    """
    drift, step = churn.series([["a", "b"], ["c", "d"], ["a", "b"]])
    assert drift[2] == 0.0
    assert step[2] == 1.0

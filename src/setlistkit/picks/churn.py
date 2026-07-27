# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""How much did a shortlist move between two drops, measured two ways that disagree usefully.

A night predicted from several successive cutoffs produces several shortlists, and the question
the Scorecard asks is how much they differ. There is no single right answer to that, because two
lists can differ in two unrelated ways:

* **Membership.** Which songs are on the list at all. A song leaving the top 30 entirely is a
  real change of mind and has nothing to do with where it sat.
* **Position.** A list can keep every song and reverse them. Membership calls that identical;
  a reader looking at rank 1 would not.

So both are here, and neither is presented as the number. :func:`turnover` answers the first,
:func:`rbo` the second, and where they disagree is itself informative: high turnover with high
RBO means the churn is all at the bottom of the list, which is the ordinary state and not news.
The reverse -- low turnover, low RBO -- means the same songs reshuffled at the top, which is the
model changing its mind about what matters.

BOTH ARE DISTANCES IN THE SAME DIRECTION. 0.0 means the two lists agree, 1.0 means they share
nothing. RBO is usually quoted as a similarity, and it is inverted here rather than shipped with
the opposite polarity of its neighbor: two numbers on a page that both describe "how much moved"
must not run in opposite directions, because nobody rereads the axis label on the second chart.
"""

from __future__ import annotations

from collections.abc import Sequence

# The top-weighting constant for RBO. At 0.9 roughly 86% of the weight lands in the first ten
# ranks, which is the shape wanted here: a song entering at rank 4 is news and one entering at
# rank 28 is not. Inherited from the pipeline this reproduces, and left at its value so the two
# pages' numbers can be compared to each other at all.
RBO_P = 0.9


def turnover(before: Sequence[str], after: Sequence[str]) -> float:
    """Share of the two lists' membership that is not shared. Order ignored.

    Jaccard distance over the sets: one minus the intersection over the union. Two empty lists
    are identical rather than undefined -- nothing changed, because there was nothing to change.

    Not "how many songs left", which sounds equivalent and is not: a list that loses five songs
    also gains five, so counting one side reports half the movement and counting both double
    counts it. The union normalizes that without a choice about which side to privilege.
    """
    first, second = set(before), set(after)
    union = first | second
    if not union:
        return 0.0
    return 1.0 - len(first & second) / len(union)


def rbo(before: Sequence[str], after: Sequence[str], p: float = RBO_P) -> float:
    """Rank-biased overlap between two ranked lists, as a DISTANCE in [0, 1].

    Walks both lists depth by depth, accumulating the share of each prefix the two share, with
    each depth weighted ``p**d`` so disagreement near the top dominates. The tail term extrapolates
    the deepest common prefix onward, which is what brings two identical lists to 0.0 rather than
    to a number visibly short of it -- an equal-length truncated sum converges to noticeably under
    1.0 similarity and would report a list as having moved when it had not. The extrapolation
    closes that gap to float precision and not further: identical lists land within about 1e-16 of
    zero, so a caller comparing to 0.0 exactly should round or use a tolerance.

    ONE CONSEQUENCE OF THE EXTRAPOLATION IS WORTH KNOWING before reading a number off a short
    list. It assumes the agreement seen at the deepest common depth continues forever, so a fully
    reversed four-item list scores only about 0.22 -- the two lists share every member by depth 4,
    and the tail term treats that as agreement stretching out past the end. On a 30-item shortlist
    the weighting has room to work and the effect is small; on a handful of items RBO is measuring
    membership as much as order.

    Two empty lists agree (0.0). One empty list against a populated one shares nothing (1.0), and
    that is the honest reading: a drop that recorded no picks is not a drop that agreed.
    """
    depth = min(len(before), len(after))
    if depth == 0:
        return 0.0 if len(before) == len(after) else 1.0

    seen_before: set[str] = set()
    seen_after: set[str] = set()
    shared = 0
    total = 0.0
    for index in range(depth):
        left, right = before[index], after[index]
        # Counted from the SEEN sets rather than by intersecting the two prefixes each time. Same
        # answer, and it keeps the walk linear instead of quadratic in the list length.
        if left == right:
            shared += 1
        else:
            if left in seen_after:
                shared += 1
            if right in seen_before:
                shared += 1
        seen_before.add(left)
        seen_after.add(right)
        total += (shared / (index + 1)) * (p ** (index + 1))

    similarity = (1 - p) / p * total + (shared / depth) * (p ** depth)
    # Clamped because the extrapolation is an estimate, not an identity: floating point can put
    # an identical pair a whisker over 1.0, and a negative distance on a chart axis is a bug
    # someone has to go and find.
    return max(0.0, min(1.0, 1.0 - similarity))


def series(drops: Sequence[Sequence[str]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(drift from the first drop, step change from the previous drop) across a night's drops.

    Both series are the same length as ``drops`` and both start at 0.0, because the first drop has
    nothing behind it to have moved from. That leading zero is kept rather than trimmed so the two
    series and the drop list all index together -- a chart that has to offset one series by one
    point is a chart that will eventually be drawn off by one.

    Turnover only. RBO is the same walk with a different function and is left to the caller, which
    is what stops this from having to grow a "which metric" parameter.
    """
    if not drops:
        return (), ()
    drift = [0.0]
    step = [0.0]
    for index in range(1, len(drops)):
        drift.append(turnover(drops[0], drops[index]))
        step.append(turnover(drops[index - 1], drops[index]))
    return tuple(drift), tuple(step)

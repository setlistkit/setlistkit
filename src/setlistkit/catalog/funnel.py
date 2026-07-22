# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""How many items took each branch: one counter per EDGE of the ingest funnel.

Not :class:`~setlistkit.catalog.parse.Census`, and deliberately beside it rather than inside it.
The census counts TOKENS so ``slkit pack lint`` can rank which title to fix first; this counts
DECISIONS so a person can see which rule the corpus actually runs into. A census keyed by token
cannot answer "how many items did the band filter refuse", and a funnel keyed by edge cannot
answer "which spelling should we add" -- two questions, two shapes, and collapsing them would
serve neither.

THE EDGE IDS ARE DECLARED ONCE, HERE. The parser increments them and the diagram generator
renders them, and if those two ever spelled an id differently the label would simply not appear
-- a diagram that looks finished and is missing a number. :data:`EDGES` is the single list both
sides read, and a test asserts the counter and the renderer agree about it.

Always collected, never behind a flag, for the reason the census gives in its own docstring: a
count gathered only in profiling mode is a count that is empty exactly when someone forgot to ask
for it. The flag on ``slkit ingest`` decides whether the numbers are WRITTEN, not whether they are
taken.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# stage -> node -> the edges leaving it. Every branch of a decision appears here, including the
# ones that continue down the ladder, because "how many got through" is exactly the number an
# unlabelled diagram leaves you guessing at.
#
# The three-tuple is (edge id, edge label, destination). The label is what the diagram prints
# before the count, so it matches the prose on the page rather than the identifier in the code.
EDGES: dict[str, dict[str, tuple[tuple[str, str, str], ...]]] = {
    "tapes": {
        "start": (("s1.start", "cached items", "billed_as_other"),),
        "billed_as_other": (
            ("s1.billing.fail", "matched", "Skipped OTHER_BILLING"),
            ("s1.billing.pass", "no", "band_filter"),
        ),
        "band_filter": (
            ("s1.band.fail", "no", "Skipped NOT_THIS_BAND"),
            ("s1.band.pass", "yes", "_show_date"),
        ),
        "_show_date": (
            ("s1.date.fail", "unreadable", "Skipped NO_DATE"),
            ("s1.date.pass", "date", "night billing"),
        ),
        "night billing": (
            ("s1.night.fail", "yes", "Skipped OTHER_BILLING (night)"),
            ("s1.night.pass", "no", "drop_dates"),
        ),
        "drop_dates": (
            ("s1.drop.fail", "listed", "Skipped DROPPED_DATE"),
            ("s1.drop.pass", "no", "admitted tape"),
        ),
        # Which parse produced the setlist. Not a refusal -- both outcomes are an admitted tape --
        # but it is the branch that says how often a taper's filenames rescued a description the
        # parser could not read.
        "admitted tape": (
            ("s1.parse.description", "description won", "setlist"),
            ("s1.parse.tracks", "tracks won", "setlist"),
        ),
    },
    "tokens": {
        "seen": (("s2.seen", "tokens", "_claimed"),),
        "_claimed": (
            ("s2.claimed.pass", "yes — skip every drop rule", "canonicalize"),
            ("s2.claimed.fail", "no", "_looks_songlike"),
        ),
        "_looks_songlike": (
            ("s2.shape.fail", "fails", "dropped"),
            ("s2.shape.pass", "passes", "CREDIT / CREW_ROLE"),
        ),
        "CREDIT / CREW_ROLE": (
            ("s2.credit.fail", "matches", "dropped"),
            ("s2.credit.pass", "no", "junk / URLISH"),
        ),
        "junk / URLISH": (
            ("s2.junk.fail", "matches", "dropped"),
            ("s2.junk.pass", "no", "canonicalize"),
        ),
        "canonicalize": (
            ("s2.canon.fail", "empty", "dropped"),
            ("s2.canon.pass", "named", "is_non_song"),
        ),
        "is_non_song": (
            ("s2.nonsong.fail", "yes", "tagged"),
            ("s2.nonsong.pass", "no", "unknown-title gate"),
        ),
        "unknown-title gate": (
            ("s2.unknown.fail", "yes", "dropped"),
            ("s2.unknown.pass", "no", "emit"),
        ),
    },
    # Which rung of _canonicalize resolved the name. Every call leaves by exactly one of these.
    "naming": {
        "canonicalize": (
            ("s3.alias", "alias hit", "canonical name"),
            ("s3.exact", "exact vocab hit", "canonical name"),
            ("s3.fuzzy", "difflib ≥ 0.90", "canonical name"),
            ("s3.band", "0.80 ≤ r < 0.90 (proposed)", "review list"),
            ("s3.fallthrough", "no match", "mints a NEW song"),
        ),
    },
}

# Every id, flattened. The generator imports this; a test asserts it covers what the parser
# increments, so an edge added to one side without the other fails loudly instead of silently
# rendering an unlabelled arrow.
EDGE_IDS: tuple[str, ...] = tuple(
    edge_id
    for stage in EDGES.values()
    for edges in stage.values()
    for edge_id, _label, _to in edges
)


@dataclass
class Funnel:
    """One counter per edge id. Mutable on purpose -- it is written to as the parse runs."""

    counts: Counter = field(default_factory=Counter)

    def hit(self, edge_id: str, n: int = 1) -> None:
        """Record ``n`` items taking ``edge_id``."""
        self.counts[edge_id] += n

    def branch(self, node: str, passed: bool) -> bool:
        """Record which way a two-way gate went, and hand the answer straight back.

        This is why the ids are mechanical. Every gate in the parse is the same shape -- a
        boolean, a refusing side and a continuing side -- so one helper drives all of them and a
        call site reads as the test it already was:

            if not rules.funnel.branch("s2.shape", _looks_songlike(piece, rules)):

        rather than a hit, a test, and a second hit on the other side of the ``continue``. The
        pair cannot then go out of sync, which the hand-written version could: a gate whose pass
        side was counted and whose fail side was not looks perfectly healthy until the node fails
        to reconcile.

        A decorator was the obvious thing to reach for and does not fit. These are BRANCHES inside
        one function, not calls -- ``CREDIT.search(piece) or CREW_ROLE.search(piece)`` is an
        inline regex with nothing to wrap -- and naming edges from the call stack would tie the
        labels on a published diagram to internal function names, so a rename would silently move
        a number on a live page. The human label lives in :data:`EDGES`; only the id is mechanical.
        """
        self.hit(f"{node}.pass" if passed else f"{node}.fail")
        return passed

    def merge(self, other: "Funnel") -> None:
        """Fold a scratch funnel into this one, for a parse attempt that WON its item.

        Mirrors :func:`~setlistkit.catalog.parse._merge_census` and exists for the same reason:
        a weak description parse is retried against the tracklist, both attempts walk the same
        gates, and counting both would report every token of every retried item twice.
        """
        self.counts.update(other.counts)

    def as_dict(self) -> dict[str, int]:
        """Every declared edge, zeros included.

        Zeros are emitted rather than omitted because a rule that never fired is a finding -- it
        is either dead or defending something -- and a missing key reads as "not measured", which
        is a different and much weaker statement.
        """
        return {edge_id: self.counts.get(edge_id, 0) for edge_id in EDGE_IDS}


def imbalances(funnel: Funnel) -> list[tuple[str, int, int]]:
    """Nodes where the traffic in does not equal the traffic out.

    THE POINT OF THE WHOLE MODULE. A funnel whose arithmetic does not close has an uncounted
    path, and an uncounted path is exactly the thing a diagram of counts would hide: every arrow
    carries a number, the picture looks complete, and some fraction of the corpus went somewhere
    nobody drew. Reported rather than asserted, so a run still finishes and still tells you.

    Returns ``(node, in, out)`` for each node that does not reconcile. An empty list is the
    result to want.
    """
    counts = funnel.counts
    out: list[tuple[str, int, int]] = []
    for stage in EDGES.values():
        # node -> what arrived, summed from every edge naming it as a destination
        arrived: Counter = Counter()
        for edges in stage.values():
            for edge_id, _label, destination in edges:
                arrived[destination] += counts.get(edge_id, 0)
        for node, edges in stage.items():
            leaving = sum(counts.get(edge_id, 0) for edge_id, _l, _t in edges)
            entering = arrived.get(node, 0)
            # The first node of a stage has no inbound edge; its own single "start" edge IS the
            # count that entered, so there is nothing to reconcile it against.
            if node not in arrived:
                continue
            if entering != leaving:
                out.append((node, entering, leaving))
    return out

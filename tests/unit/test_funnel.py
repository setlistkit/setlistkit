# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for the ingest funnel: one counter per edge, and the arithmetic that has to close.

Two things are pinned here, both quoted from the module's own docstring. First, the
RECONCILIATION RULE: at every node, what arrived equals what left, over every stage. It is
the test that already caught a real bug once while the profiler was being built -- the
unknown-title gate only examined titles the vocabulary did NOT know, and counting only the
examined ones left the node short by every known song in the corpus. Second, EDGE COVERAGE:
:data:`~setlistkit.catalog.funnel.EDGE_IDS` is the one registry both the parser and the diagram
generator read, and a parser that increments an id the registry never declared would silently
vanish from every rendered count -- `as_dict` only ever reads the declared ids, so a typo'd edge
is not an error, it is a missing arrow nobody notices. The integration tests below run a real
parse and check every id it actually touched against the registry, which is the only way to
catch that class of drift; full edge enumeration is not the goal, and neither scenario tries to
walk every one of the twenty-odd declared edges.
"""

import re

from setlistkit.catalog import ArchivePolicy, Normalizer, parse_archive_items, title_band_filter
from setlistkit.catalog.funnel import EDGE_IDS, EDGES, Funnel, imbalances


# --- Funnel: the counter itself -------------------------------------------------------------

def test_hit_accumulates_and_defaults_to_one():
    funnel = Funnel()
    funnel.hit("s1.start")
    funnel.hit("s1.start", 3)
    assert funnel.counts["s1.start"] == 4


def test_branch_records_the_side_it_took_and_returns_the_answer_unchanged():
    funnel = Funnel()
    assert funnel.branch("s2.claimed", True) is True
    assert funnel.branch("s2.claimed", False) is False
    assert funnel.counts["s2.claimed.pass"] == 1
    assert funnel.counts["s2.claimed.fail"] == 1


def test_merge_folds_a_scratch_funnels_counts_into_the_running_one():
    running = Funnel()
    running.hit("s1.start", 2)
    scratch = Funnel()
    scratch.hit("s1.start", 5)
    scratch.hit("s2.seen", 1)
    running.merge(scratch)
    assert running.counts["s1.start"] == 7
    assert running.counts["s2.seen"] == 1


def test_as_dict_includes_every_declared_edge_with_zeros_for_the_unhit_ones():
    funnel = Funnel()
    funnel.hit("s1.start", 9)
    out = funnel.as_dict()
    assert set(out) == set(EDGE_IDS)
    assert out["s1.start"] == 9
    assert out["s1.billing.fail"] == 0        # never hit, present anyway


# --- imbalances: the reconciliation check itself -------------------------------------------

def test_imbalances_is_empty_when_a_stage_reconciles():
    """Every downstream edge of the "tapes" stage is walked, not just the first one or two --
    an untouched tail node's own outgoing edges would otherwise sit at zero while a nonzero
    count arrived at it, which is an imbalance in its own right and not the thing this test is
    supposed to be about."""
    funnel = Funnel()
    funnel.hit("s1.start", 10)
    funnel.hit("s1.billing.fail", 3)
    funnel.hit("s1.billing.pass", 7)
    funnel.hit("s1.band.fail", 2)
    funnel.hit("s1.band.pass", 5)
    funnel.hit("s1.date.fail", 1)
    funnel.hit("s1.date.pass", 4)
    funnel.hit("s1.night.fail", 1)
    funnel.hit("s1.night.pass", 3)
    funnel.hit("s1.drop.fail", 1)
    funnel.hit("s1.drop.pass", 2)
    funnel.hit("s1.parse.description", 1)
    funnel.hit("s1.parse.tracks", 1)
    assert imbalances(funnel) == []


def test_imbalances_reports_a_node_whose_in_and_out_disagree():
    """Nothing past `billed_as_other` is hit, so the one-short count at the top cascades: the 6
    that leave `billed_as_other` arrive at `band_filter`, which has no outgoing edges of its own
    hit at all, so it disagrees too. `imbalances` reports every node that fails to reconcile,
    not just the first one -- both belong in the expected list."""
    funnel = Funnel()
    funnel.hit("s1.start", 10)
    funnel.hit("s1.billing.fail", 3)
    funnel.hit("s1.billing.pass", 6)          # one short: 9 left, 10 arrived
    assert imbalances(funnel) == [
        ("billed_as_other", 10, 9),
        ("band_filter", 6, 0),
    ]


def test_the_first_node_of_a_stage_has_nothing_to_reconcile_against():
    """"start", "seen" and "canonicalize" are each the first node of their stage: nothing in
    the registry names them as a destination, so there is no inbound count to hold them to. The
    rest of the stage is walked straight through -- every "pass" edge hit, no "fail" edge --
    so the only thing under test is "start" itself, not an unpopulated tail."""
    funnel = Funnel()
    funnel.hit("s1.start", 5)                 # no edges lead TO "start"
    funnel.hit("s1.billing.pass", 5)
    funnel.hit("s1.band.pass", 5)
    funnel.hit("s1.date.pass", 5)
    funnel.hit("s1.night.pass", 5)
    funnel.hit("s1.drop.pass", 5)
    funnel.hit("s1.parse.description", 5)
    assert imbalances(funnel) == []


def test_edge_ids_flattens_every_edge_in_the_registry_with_no_duplicates():
    flattened = [edge_id for stage in EDGES.values() for edges in stage.values()
                 for edge_id, _label, _to in edges]
    assert list(EDGE_IDS) == flattened
    assert len(set(EDGE_IDS)) == len(EDGE_IDS)


# --- a real parse: the rule held against production code ------------------------------

_VOCAB = ["Rebubula", "Meat", "Ophelia", "Aurora"]


class _StubNormalizer(Normalizer):
    """Enough policy to walk most of the funnel; none of a real band's data."""

    def __init__(self):
        super().__init__(_VOCAB)

    def aliases(self):
        return {"reb": "Rebubula"}

    def non_song_patterns(self):
        return [re.compile(r"^tuning$")]

    def protected_titles(self):
        return set()


def _policy(**overrides):
    defaults = dict(
        drop_dates=frozenset({"2026-02-01"}),
        band_filter=title_band_filter("Example"),
        side_projects=(re.compile(r"acoustic\s+duo", re.I),),
        junk_patterns=(r"stagehand",),
    )
    defaults.update(overrides)
    return ArchivePolicy(**defaults)


def _item(identifier, date, description, title="Example. Live at Northlands on {date}"):
    return {"identifier": identifier, "date": date,
            "title": title.format(date=date), "description": description}


def test_a_realistic_parse_reconciles_at_every_node():
    """One run touching most of the ladder: a refused band, a refused date, a dropped date, a
    credit-line drop, a junk drop, a tagged non-song, an alias hit, an exact hit, a fuzzy hit, a
    near-miss (0.80-0.90) fall-through and both outcomes of the unknown-title gate.
    `imbalances` must come back empty regardless of which paths a given run exercises.
    """
    items = [
        _item("t1", "2026-01-31",
              "Set 1:\n01. REB\n02. Meat\n03. Tuning\n04. Al Schnier - guitar, vocals\n"
              "05. stagehand notes\n06. Opheliaa\n07. Wibbly Wobble Nonsense\n"
              "08. Some Really Quite Long Unnamed Fragment 7\n09. Arorra\n"),
        _item("t2", "2026-02-01", "Set 1:\n01. Meat\n02. Aurora\n"),   # a date the pack refuses
        _item("t3", "2026-03-01", "Set 1:\n01. Meat\n",
              title="Some Other Band. Live at The Fillmore on {date}"),   # not this band
        _item("t4", "", "Set 1:\n01. Meat\n"),                            # no believable date
    ]
    result = parse_archive_items(items, normalizer=_StubNormalizer(), policy=_policy())
    assert imbalances(result.funnel) == []
    # every id the parser actually touched is one the registry declares -- a typo'd edge id
    # would otherwise vanish from as_dict() silently instead of failing this assertion.
    assert set(result.funnel.counts) <= set(EDGE_IDS)
    # and the run actually exercised the paths the docstring above claims it did
    touched = set(result.funnel.counts)
    for edge_id in ("s1.band.fail", "s1.date.fail", "s1.drop.fail", "s2.credit.fail",
                    "s2.junk.fail", "s2.nonsong.fail", "s2.nonsong.pass", "s2.unknown.fail",
                    "s2.unknown.pass", "s3.alias", "s3.exact", "s3.fuzzy", "s3.band",
                    "s3.fallthrough"):
        assert edge_id in touched, edge_id


def test_a_weak_description_retry_still_reconciles():
    """A description too thin to count as a setlist falls back to the tracklist, and the
    scratch funnel from that attempt is folded in only if it wins -- exactly the merge path
    :meth:`Funnel.merge` exists for. The rule has to survive it.
    """
    item = dict(_item("w1", "2026-01-31", "a lovely night, thanks all"),
                tracks=[{"title": "Meat"}, {"title": "Aurora"}, {"title": "Rebubula"},
                        {"title": "Ophelia"}])
    result = parse_archive_items([item], normalizer=_StubNormalizer(), policy=_policy())
    assert imbalances(result.funnel) == []
    assert result.shows[0]["source"] == "tracks"

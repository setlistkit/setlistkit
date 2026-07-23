"""Reconciling the several tapes of one night into one duration per performance.

Like the reading tests next door, nearly every case here is a real one: a coarse tape that lumped
a segue into one file, a taper who posted four mic feeds and outvoted the rest of the night, a
17-track reel whose extra "tracks" were two 18-second false starts. They are written as the shape
that broke, so a regression names the failure rather than a line number.
"""
import re

from setlistkit.catalog import durations as D
from setlistkit.catalog import lengths as L
from setlistkit.catalog.normalizer import Normalizer


class _Pack(Normalizer):
    """A normalizer with just enough pack policy to mark a track as not-music."""

    def __init__(self, non_songs=()):
        super().__init__([])
        self._non_songs = [re.compile(p) for p in non_songs]

    def non_song_patterns(self):
        return list(self._non_songs)


DATE = "2023-01-19"


def _obs(song, seconds, identifier="a", *, set_label="1", position=1, date=DATE,
         show_type=L.ELECTRIC, combined_with=()):
    return L.Observation(slot=L.Slot(date, set_label, position, song), identifier=identifier,
                         seconds=seconds, show_type=show_type, combined_with=combined_with)


def _entry(song, segue=False):
    return {"song": song, "segue": segue, "non_song": False}


def _show(*sets, encore=()):
    return {"sets": [list(one) for one in sets], "encore": list(encore)}


def _reconcile(observations, shows=None, *, uploaders=None, splits=None, exclusions=None):
    return L.reconcile(observations, shows or {DATE: _show([_entry("Moth")])},
                       uploaders=uploaders or {}, splits=splits or {},
                       exclusions=exclusions)


def _only(observations, **kwargs):
    """The single performance a set of observations reconciles to."""
    performances, _ = _reconcile(observations, **kwargs)
    assert len(performances) == 1
    return performances[0]


# ---- the identity of a performance -----------------------------------------------------------

def test_slots_sort_into_play_order_with_the_encore_last():
    """Ordered on (date, set, position) before song, so a pile of these is the night in order.
    "E" sorts after the digits, which is the encore landing where it was played."""
    slots = [L.Slot(DATE, "2", 1, "Meat"), L.Slot(DATE, "E", 1, "Gone"),
             L.Slot(DATE, "1", 2, "Moth"), L.Slot(DATE, "1", 1, "Buster")]
    assert [slot.song for slot in sorted(slots)] == ["Buster", "Moth", "Meat", "Gone"]


# ---- how fine a tape is ----------------------------------------------------------------------

def test_the_finest_tape_is_counted_in_real_boundaries_not_in_files():
    """2024-07-29: a 17-track reel opening with two 18-second false starts outranked three good
    tapes and booked Big World at 0:18 against their 5:14, 5:32 and 6:14. A stub is not a
    boundary drawn inside the music, and counting it promotes the sloppiest tape of the night to
    arbiter of the whole night."""
    sloppy = {"identifier": "sloppy",
              "tracks": [{"seconds": 18.0}, {"seconds": 18.0}] + [{"seconds": 300.0}] * 15}
    clean = {"identifier": "clean", "tracks": [{"seconds": 300.0}] * 16}
    splits = L.track_splits([sloppy, clean])
    assert splits == {"sloppy": 15, "clean": 16}


def test_a_track_with_no_parsed_duration_is_not_a_boundary():
    """seconds is NULL where the raw length could not be parsed. Counting it as a split would
    hand rank to the tape we understand least."""
    assert L.track_splits([{"identifier": "a", "tracks": [{"seconds": None},
                                                          {"seconds": 300.0}]}]) == {"a": 1}


# ---- one tape's reading, as votes ------------------------------------------------------------

def test_a_track_that_is_not_music_consumes_its_track_but_casts_no_vote():
    """The MC's introduction has to stay consumed -- dropping it earlier would slide every song
    after it up one slot -- but a repertoire where "Intro" runs 40 seconds every night is a
    repertoire with a fictional song in it."""
    pack = _Pack(non_songs=[r"^intro$"])
    tape = D.Tape("moe2023-01-19", DATE)
    reading = D.Reading(rows=(D.Row("1", 1, "Intro", 40.0, "01 Intro.flac"),
                              D.Row("1", 2, "Moth", 600.0, "02 Moth.flac")))
    votes, edges = L.observations_of(tape, reading, pack)
    assert [v.slot.song for v in votes] == ["Moth"]
    assert [e.kind for e in edges] == ["non_song_excluded"]
    assert edges[0].detail["seconds"] == 40.0


def test_an_observation_records_which_source_named_it():
    """Nothing downstream reads this, which is exactly why it is easy to delete -- and deleting it
    is how the disagreement tolerances become magic numbers. They are justified by how closely two
    independent sources agree, and that cannot be re-measured without knowing which was which."""
    tape = D.Tape("moe2023-01-19", DATE)
    reading = D.Reading(rows=(D.Row("1", 1, "Moth", 600.0, "01 Moth.flac"),))
    votes, _ = L.observations_of(tape, reading, _Pack(), L.ELECTRIC, D.DESCRIPTION)
    assert votes[0].named_by == D.DESCRIPTION


def test_songs_nobody_claimed_on_a_believed_tape_are_reported_not_swallowed():
    """This is where the next alias comes from: either the taper spelled it in a way we do not
    recognize, or it was folded into a neighbor's file."""
    pack = _Pack(non_songs=[r"^intro$"])
    night = D.Night.of(_show([_entry("Intro"), _entry("Moth"), _entry("Meat")]), pack)
    reading = D.Reading(rows=(D.Row("1", 2, "Moth", 600.0, "02 Moth.flac"),),
                        claimed=frozenset({1}))
    edges = L.unclaimed_songs(D.Tape("moe2023-01-19", DATE), reading, night, pack)
    # Meat is missing and is worth a look. Intro is missing and is not music.
    assert [e.song for e in edges] == ["Meat"]


# ---- one taper, one vote ---------------------------------------------------------------------

def test_one_uploader_posting_several_feeds_is_one_ballot():
    """2023-04-27: one taper's two mic feeds and the matrix built from them are three archive.org
    items with a single set of track splits between them."""
    uploaders = {"ck61": "chris", "ck63": "chris", "matrix": "chris", "dave1": "dave"}
    grouped = L.ballots([_obs("Moth", 600.0, "ck61"), _obs("Moth", 601.0, "ck63"),
                         _obs("Moth", 600.5, "matrix"), _obs("Moth", 300.0, "dave1")], uploaders)
    assert sorted(grouped) == ["chris", "dave"]
    assert len(grouped["chris"]) == 3


def test_a_tape_with_no_known_uploader_votes_as_itself():
    """Weaker consolidation, not a wrong one. The caller is expected to say how often it happens
    rather than let it degrade quietly -- see the design doc on the 425 uncredited tapes."""
    grouped = L.ballots([_obs("Moth", 600.0, "a"), _obs("Moth", 601.0, "b")], {})
    assert sorted(grouped) == ["a", "b"]


def test_one_taper_cannot_outvote_the_rest_of_the_night_with_four_uploads():
    """The whole reason ballots exist. Three items at 10:00 from one person against two other
    tapers at 5:00 -- by tapes it is 3-2, by tapers it is 1-2, and the tapers are right."""
    uploaders = {"ck61": "chris", "ck63": "chris", "matrix": "chris",
                 "dave1": "dave", "eve1": "eve"}
    performance = _only([_obs("Moth", 600.0, "ck61"), _obs("Moth", 600.0, "ck63"),
                         _obs("Moth", 600.0, "matrix"),
                         _obs("Moth", 300.0, "dave1"), _obs("Moth", 305.0, "eve1")],
                        uploaders=uploaders)
    assert performance.seconds == 302.5
    assert performance.consensus.n_ballots == 3
    assert performance.consensus.resolved_by == L.OUTLIER_DROPPED
    assert performance.consensus.suspect is False


# ---- who is allowed to speak for a performance -----------------------------------------------

def test_a_single_ballot_cannot_outvote_itself():
    assert L.largest_cluster([_obs("Moth", 600.0, "a"), _obs("Moth", 900.0, "b")],
                             45, {"a": "chris", "b": "chris"}) is None


def test_ballots_that_already_agree_have_no_outlier_to_drop():
    assert L.largest_cluster([_obs("Moth", 600.0, "a"), _obs("Moth", 610.0, "b")], 45, {}) is None


def test_a_tie_is_not_a_consensus():
    """Two against two means we really cannot say which pair is right. Deciding it by whichever
    tape happens to sort first is a coin flip published as a measurement."""
    tapes = [_obs("Moth", 300.0, "a"), _obs("Moth", 305.0, "b"),
             _obs("Moth", 600.0, "c"), _obs("Moth", 605.0, "d")]
    assert L.largest_cluster(tapes, 45, {}) is None
    assert _only(tapes).consensus.suspect is True


def test_a_resolved_dispute_is_not_reported_as_an_uncorroborated_measurement():
    """n_tapes is what we KEPT; n_tapes_seen is what the night offered. Reporting only the first
    turns "we discarded one tape" into "only two tapes exist"."""
    performance = _only([_obs("Moth", 300.0, "a"), _obs("Moth", 305.0, "b"),
                         _obs("Moth", 900.0, "c")])
    assert (performance.consensus.n_tapes, performance.consensus.n_tapes_seen) == (2, 3)
    assert performance.consensus.spread_seconds == 5.0
    assert performance.consensus.spread_all_tapes == 600.0


# ---- when two tapes of one night disagree by minutes -----------------------------------------

def test_tapes_that_agree_are_a_measurement():
    """Two tapers who both caught a standalone song put it a median of 12.4 seconds apart across
    this corpus. This is what the ordinary case looks like."""
    performance = _only([_obs("Moth", 300.0, "a"), _obs("Moth", 302.0, "b"),
                         _obs("Moth", 305.0, "c")])
    assert performance.seconds == 302.0
    assert performance.consensus.suspect is False
    assert performance.consensus.resolved_by is None


def test_a_coarse_tape_that_lumped_a_segue_loses_to_the_one_that_drew_the_boundary():
    """11-track tape: "Mar-De-Ma" 25:19. 17-track tape: "Mar-De-Ma" 6:10 + "George" 19:11 = 25:21.
    25:19 is not the length of Mar-De-Ma, it is the length of Mar-De-Ma AND George, and no amount
    of averaging makes it otherwise."""
    performance = _only([_obs("Mar-De-Ma", 1519.0, "coarse"), _obs("Mar-De-Ma", 370.0, "fine")],
                        splits={"coarse": 11, "fine": 17})
    assert performance.seconds == 370.0
    assert performance.consensus.resolved_by == L.FINEST_TAPE
    assert performance.consensus.n_tapes == 1
    assert performance.consensus.suspect is False


def test_a_broken_track_does_not_win_on_track_count():
    """The coarse-tape premise predicts a SHORTER but still musical value. 18 seconds is not a
    finer reading of Big World, it is a false start, and no track count earns the right to
    overrule every other tape of the night with one."""
    performance = _only([_obs("Big World", 18.0, "sloppy"), _obs("Big World", 314.0, "clean")],
                        splits={"sloppy": 17, "clean": 13})
    assert performance.consensus.suspect is True
    assert performance.consensus.resolved_by is None


def test_equally_fine_tapes_stay_disputed():
    """Equal granularity means the tapers themselves could not tell us which boundary is right."""
    performance = _only([_obs("Moth", 300.0, "a"), _obs("Moth", 900.0, "b")],
                        splits={"a": 15, "b": 15})
    assert performance.consensus.suspect is True


def test_dropping_an_outlier_can_narrow_a_dispute_without_settling_it():
    """Partial progress is reported as partial.

    The cluster is built by breaking the chain at gaps wider than the tolerance, so its members
    are each within 45s of their NEIGHBOR while the run as a whole drifts 120s end to end. One
    reel plainly failed and is dropped; the four survivors still do not agree, and there is
    nothing to choose between them. Setting resolved_by while leaving suspect set is the honest
    description of that, and it is why the two are separate fields.
    """
    equal = {name: 15 for name in "abcde"}
    performance = _only([_obs("Moth", 300.0, "a"), _obs("Moth", 340.0, "b"),
                         _obs("Moth", 380.0, "c"), _obs("Moth", 420.0, "d"),
                         _obs("Moth", 3000.0, "e")], splits=equal)
    assert performance.consensus.resolved_by == L.OUTLIER_DROPPED
    assert performance.consensus.n_tapes == 4
    assert performance.consensus.spread_seconds == 120.0
    assert performance.consensus.suspect is True


def test_a_disputed_performance_says_so_where_a_human_will_see_it():
    _, edges = _reconcile([_obs("Moth", 300.0, "a"), _obs("Moth", 900.0, "b")])
    assert [e.kind for e in edges] == ["tapes_disagree"]
    assert edges[0].detail["values"] == [300.0, 900.0]


def test_a_segued_song_is_allowed_to_wander_further():
    """There is no objective boundary inside a segue: where the Bring You Down jam stops being
    Bring You Down is the taper's aesthetic call, and two tapers routinely put it half a minute
    apart while agreeing almost exactly on where the PAIR begins and ends."""
    tapes = [_obs("Buster", 300.0, "a"), _obs("Buster", 380.0, "b")]
    standalone = _show([_entry("Buster"), _entry("Moth")])
    segued = _show([_entry("Buster", segue=True), _entry("Moth")])
    assert _only(tapes, shows={DATE: standalone}).consensus.suspect is True
    assert _only(tapes, shows={DATE: segued}).consensus.suspect is False


def test_a_segue_pair_sharing_one_file_times_the_run_and_never_a_song():
    """The time is real but it belongs to the run, and there is no rule for dividing it between
    the two songs that is not an invention."""
    performances, _ = _reconcile([_obs("Buster", 900.0, "a", combined_with=("Moth",)),
                                  _obs("Moth", 600.0, "a", position=2)])
    assert [p.slot.song for p in performances] == ["Moth"]


# ---- a song played twice in one night --------------------------------------------------------

def test_the_short_half_of_a_sandwich_is_set_aside_and_the_total_kept():
    """Moth > [Water > Yellow Tigers] > Moth. Neither half is a performance of the song, any more
    than the first half of a sentence is a short sentence, and the short halves were dragging
    every jam vehicle's median down."""
    performances, edges = _reconcile([_obs("Moth", 600.0, "a"),
                                      _obs("Moth", 137.0, "a", position=5)])
    long_half, short_half = sorted(performances, key=lambda p: -p.seconds)
    assert long_half.sandwich == L.Sandwich(parts=2, total_seconds=737.0, is_longest_part=True)
    assert short_half.sandwich.is_longest_part is False
    assert long_half.withheld is None
    assert short_half.withheld == L.SANDWICH_SHORT_HALF
    assert [e.kind for e in edges] == ["song_played_twice"]


def test_a_song_played_once_carries_no_sandwich_at_all():
    """None rather than four columns reading False/None/None/True on the ninety-eight percent of
    rows that are not sandwiches."""
    assert _only([_obs("Moth", 600.0, "a")]).sandwich is None


# ---- who votes for a song's nominal length ---------------------------------------------------

def test_an_acoustic_performance_is_measured_and_still_does_not_vote():
    """An acoustic Lazarus is a real six-minute performance and it stays in the table, tagged. It
    is not evidence about how long the ELECTRIC band plays Lazarus, and averaging the two produces
    a number describing neither."""
    performance = _only([_obs("Lazarus", 360.0, "a", show_type="acoustic")])
    assert performance.seconds == 360.0
    assert performance.withheld == "acoustic"
    assert L.song_stats([performance]) == []


def test_a_performance_a_human_ruled_out_is_kept_tagged_and_silent():
    """Neither a tape that cut off mid-song nor a two-minute reprise is detectable from metadata,
    because both look exactly like a really unusual performance. Nothing is deleted."""
    performance = _only([_obs("Moth", 247.0, "a")],
                        exclusions={(DATE, "1", 1): {"song": "Moth", "reason": "truncated"}})
    assert performance.seconds == 247.0
    assert performance.excluded == "truncated"
    assert L.song_stats([performance]) == []


def test_a_disputed_performance_is_counted_as_withheld_rather_than_quietly_dropped():
    """The previous implementation dropped suspect performances at the point of aggregation
    without counting them anywhere, so the exclusion tally was missing its largest category."""
    performance = _only([_obs("Moth", 300.0, "a"), _obs("Moth", 900.0, "b")])
    assert performance.withheld == L.TAPES_DISAGREE
    assert L.withheld_counts([performance]) == {L.TAPES_DISAGREE: 1}


def test_withheld_counts_only_names_the_performances_actually_held_back():
    kept = _only([_obs("Moth", 600.0, "a")])
    assert L.withheld_counts([kept]) == {}


# ---- the per-song pool -----------------------------------------------------------------------

def _perf(song, seconds, date, set_label="1", position=1, **kwargs):
    consensus = L.Consensus(n_tapes=1, n_tapes_seen=1, n_ballots=1, spread_seconds=0.0,
                            spread_all_tapes=0.0, suspect=False)
    return L.Performance(slot=L.Slot(date, set_label, position, song), seconds=seconds,
                         consensus=consensus, segued=False, **kwargs)


def test_the_pool_reports_where_the_longest_one_was():
    rows = [_perf("Moth", 300.0, "2023-01-19"), _perf("Moth", 900.0, "2023-03-11"),
            _perf("Moth", 600.0, "2023-05-02")]
    stat = L.song_stats(rows)[0]
    assert (stat.n, stat.median_seconds, stat.mean_seconds) == (3, 600.0, 600.0)
    assert (stat.min_seconds, stat.max_seconds) == (300.0, 900.0)
    assert stat.longest_date == "2023-03-11"


def test_a_song_played_once_is_its_own_every_percentile():
    """n=1 has no distribution. Clamping rather than indexing off the end is what keeps it from
    being an error case that has to be special-pleaded at every consumer."""
    stat = L.song_stats([_perf("Timmy Tucker", 421.0, DATE)])[0]
    assert (stat.p10_seconds, stat.median_seconds, stat.p90_seconds) == (421.0, 421.0, 421.0)
    assert stat.stdev_seconds == 0.0


def test_the_pool_leads_with_the_longest_songs():
    stats = L.song_stats([_perf("Buster", 300.0, DATE), _perf("Moth", 900.0, DATE, position=2)])
    assert [s.song for s in stats] == ["Moth", "Buster"]


def test_percentiles_are_nearest_rank_on_the_sorted_values():
    """p10 and p90 are what the page draws its whiskers from, so they are real observed lengths
    rather than an interpolation between two nights that never happened."""
    rows = [_perf("Moth", secs, DATE, position=i)
            for i, secs in enumerate([10.0, 20.0, 30.0, 40.0, 50.0,
                                      60.0, 70.0, 80.0, 90.0, 100.0], start=1)]
    stat = L.song_stats(rows)[0]
    assert (stat.p10_seconds, stat.median_seconds, stat.p90_seconds) == (10.0, 55.0, 90.0)


def test_a_song_timed_twice_puts_its_p90_at_the_longer_of_the_two():
    """the sample size that tells nearest-rank apart from floor-rank, which n=10 above cannot.

    `int(fraction * (n - 1))` agrees with the ceiling at n=10 and disagrees everywhere small: at
    n=2 it puts p90 at index 0 and returns the SHORTER take, at n=3 it returns the median exactly.
    That shipped -- 106 of the 389 real songs with more than one timing had p90 <= median, and
    Vocal Jam published a 19.6-minute median beside a 7.4-minute p90."""
    rows = [_perf("Vocal Jam", 445.8, DATE), _perf("Vocal Jam", 1901.8, "2023-03-11")]
    stat = L.song_stats(rows)[0]
    assert stat.p90_seconds == 1901.8
    assert stat.p10_seconds == 445.8
    assert stat.p10_seconds <= stat.median_seconds <= stat.p90_seconds


def test_a_percentile_is_never_on_the_wrong_side_of_the_median():
    """the property the off-by-one broke, at every sample size rather than at a chosen one.

    Lengths carry hundredths, because that is how archive.org states them, and the rounding is
    what makes this a property worth asserting rather than arithmetic: rounding the median while
    leaving the percentiles raw put p90 0.04s BELOW the median for every song whose timings were
    identical, and a bar drawn from median to p90 has a negative width."""
    for count in range(1, 12):
        rows = [_perf("Moth", 100.0 * n + 0.56, DATE, position=n) for n in range(1, count + 1)]
        stat = L.song_stats(rows)[0]
        assert stat.min_seconds <= stat.p10_seconds <= stat.median_seconds, f"n={count}"
        assert stat.median_seconds <= stat.p90_seconds <= stat.max_seconds, f"n={count}"


def test_every_published_length_shares_one_precision():
    """Six numbers printed side by side get compared to each other, so they round alike."""
    rows = [_perf("Moth", secs, DATE, position=i)
            for i, secs in enumerate([492.681, 492.684, 501.999], start=1)]
    stat = L.song_stats(rows)[0]
    for value in (stat.median_seconds, stat.mean_seconds, stat.min_seconds, stat.max_seconds,
                  stat.p10_seconds, stat.p90_seconds, stat.stdev_seconds):
        assert value == round(value, 1)


# ---- the exclusions ledger ---------------------------------------------------------------------

def _excl(song, reason="truncated"):
    return {"song": song, "reason": reason}


def test_an_exclusion_stops_a_performance_voting_without_deleting_it():
    """A ruled-out performance is still measured, still stored and still published -- tagged.
    Deleting it would hide the judgement instead of recording it."""
    perf = _only([_obs("Moth", 247.0)],
                 exclusions={(DATE, "1", 1): _excl("Moth")})
    assert perf.seconds == 247.0
    assert perf.excluded == "truncated"
    assert perf.withheld == "truncated"


def test_an_exclusion_that_lands_on_a_different_song_is_refused():
    """THE ONE THAT BIT US. Position is a number somebody counted off a page, so it is exactly
    the key that can be one out and still hit a real row -- and the seeded ledger's "2:17 Moth
    reprise" landed on an eighteen-minute Pit. A good measurement stopped voting and the bad one
    kept voting, and the run reported a tidy tally of exclusions applied."""
    perf = _only([_obs("The Pit", 1109.0)],
                 exclusions={(DATE, "1", 1): _excl("Moth", "reprise")})
    assert perf.excluded is None
    assert perf.withheld is None


def test_a_refused_exclusion_is_reported_with_what_it_hit_instead():
    """Silently skipping it is the same failure one step later: somebody listened to that
    performance, and their judgement has stopped being honored."""
    performances, _ = _reconcile([_obs("The Pit", 1109.0)],
                                 exclusions={(DATE, "1", 1): _excl("Moth", "reprise")})
    unmatched = L.unmatched_exclusions(performances, {(DATE, "1", 1): _excl("Moth", "reprise")})
    assert unmatched == [{"date": DATE, "set": "1", "position": 1, "song": "Moth",
                          "reason": "reprise", "found": "The Pit"}]


def test_an_exclusion_for_a_slot_nobody_played_is_reported_rather_than_ignored():
    performances, _ = _reconcile([_obs("Moth", 600.0)])
    unmatched = L.unmatched_exclusions(performances, {("1999-01-01", "2", 9): _excl("Moth")})
    assert unmatched[0]["found"] is None


def test_an_excluded_performance_does_not_reach_its_song_pool():
    rows = [_perf("Moth", 600.0, DATE), _perf("Moth", 4.0, "2023-03-11")]
    assert L.song_stats(rows)[0].n == 2
    ruled_out = [r for r in rows if r.slot.date != "2023-03-11"]
    assert L.song_stats(ruled_out)[0].n == 1


def test_a_zero_length_track_is_not_a_performance_of_zero_seconds():
    """archive.org states some tracks as 0:00 -- a placeholder or an unfinished derivative.
    As a vote it is the smallest number there is, so it lands as the song's minimum: Moth
    published a floor of 0:00 across 350 performances before this."""
    tape = D.Tape("moe2023-01-19", DATE)
    reading = D.Reading(rows=(D.Row("1", 1, "Moth", 0.0, "01 Moth.flac"),
                              D.Row("1", 2, "Meat", 600.0, "02 Meat.flac")))
    votes, edges = L.observations_of(tape, reading, _Pack())
    assert [v.slot.song for v in votes] == ["Meat"]
    assert [e.kind for e in edges] == ["zero_length_track"]


# --- as_row / from_row -------------------------------------------------------------------------


def _performance(**kw):
    """A performance with every optional part populated, so a round trip has something to lose."""
    base = dict(
        slot=L.Slot(date="2023-01-19", set_label="2", position=4, song="Rebubula"),
        seconds=612.5,
        consensus=L.Consensus(n_tapes=3, n_tapes_seen=5, n_ballots=7, spread_seconds=4.5,
                              spread_all_tapes=91.0, suspect=False, resolved_by="finest_tape"),
        segued=True, show_type=L.ELECTRIC, excluded=None,
        sandwich=L.Sandwich(parts=2, total_seconds=980.0, is_longest_part=True),
    )
    return L.Performance(**{**base, **kw})


def test_from_row_is_the_exact_inverse_of_as_row():
    """The pair is only correct together, so it is tested together."""
    for performance in (_performance(),
                        _performance(sandwich=None),
                        _performance(excluded="hand_excluded"),
                        _performance(show_type="acoustic"),
                        _performance(consensus=L.Consensus(1, 1, 1, 0.0, 0.0, True))):
        assert L.from_row(L.as_row(performance)) == performance


def test_a_rehydrated_performance_recomputes_the_withheld_it_was_stored_with():
    """``withheld`` is a property over four other fields, and ``from_row`` deliberately does not
    read the stored column. That is only safe if the two always agree -- if they can diverge, a
    ranged export would hold back a different set of performances than the stored tally explains.
    """
    for performance in (_performance(),
                        _performance(show_type="acoustic"),
                        _performance(excluded="hand_excluded"),
                        _performance(sandwich=L.Sandwich(2, 980.0, is_longest_part=False)),
                        _performance(consensus=L.Consensus(1, 4, 4, 200.0, 200.0, True))):
        row = L.as_row(performance)
        assert L.from_row(row).withheld == row["withheld"]


def test_song_stats_over_rehydrated_rows_matches_stats_over_the_originals():
    """What the ranged export relies on: a performance that has been through the store and back
    votes exactly as it did before, so a window changes the population and never the method."""
    performances = [
        _performance(slot=L.Slot("2023-01-19", "1", n, "Rebubula"), seconds=600.0 + n)
        for n in range(1, 6)
    ] + [_performance(slot=L.Slot("2023-01-20", "1", 1, "Plane Crash"), seconds=1200.0)]
    direct = L.song_stats(performances)
    round_tripped = L.song_stats([L.from_row(L.as_row(p)) for p in performances])
    assert [vars(s) for s in direct] == [vars(s) for s in round_tripped]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""tests for the band-agnostic Normalizer mechanism.

These lock the traps the old songnorm.py learned the hard way: the case-sensitive (UM)
attribution rule that must run before lowercasing, the "with" that hides inside a real
song title, the parenthesised titles that are NOT attributions, the segue tokens, and the
protected-title guard that keeps a shape rule from deleting a real song.

No pack is loaded here. The base class works with an injected vocabulary and empty policy
hooks; a synthetic subclass supplies aliases / non-song patterns / protected titles so the
policy-reading methods can be exercised without the moe. data.
"""

import re

from setlistkit.catalog import Normalizer


# A synthetic pack: enough policy to exercise the hooks, none of the real moe. data.
class _StubNormalizer(Normalizer):
    def aliases(self):
        return {"rec chem": "Recreational Chemistry", "zoz": "Z0Z (Zed Nought Z)"}

    def non_song_patterns(self):
        return [re.compile(r"^setbreak$"), re.compile(r"nounc"), re.compile(r"intro")]

    def protected_titles(self):
        return {"ATL", "TLH", "NYC"}


# --- squash -------------------------------------------------------------------------------

def test_squash_reduces_to_bare_alnum():
    """every spacing/punctuation/case variant collapses to the same key."""
    norm = Normalizer()
    assert norm.squash("Worm Wood") == "wormwood"
    assert norm.squash("worm-wood") == "wormwood"
    assert norm.squash("Wormwood") == "wormwood"


def test_squash_ampersand_is_configurable_both_ways():
    """'&' has no single right answer, so the caller picks which spelling to emit."""
    norm = Normalizer()
    assert norm.squash("Ups & Downs") == "upsanddowns"          # default amp="and"
    assert norm.squash("Ups & Downs", amp="") == "upsdowns"     # deletion form


# --- normalize ----------------------------------------------------------------------------

def test_normalize_strips_the_and_lowercases():
    norm = Normalizer()
    assert norm.normalize("The Low Spark") == "low spark"


def test_normalize_ampersand_becomes_and():
    norm = Normalizer()
    assert norm.normalize("Hi & Lo") == "hi and lo"


def test_normalize_collapses_punctuation_and_whitespace():
    norm = Normalizer()
    assert norm.normalize("Mar-De-Ma") == "mar de ma"
    assert norm.normalize("St.  Augustine") == "st augustine"


# --- strip_segue --------------------------------------------------------------------------

def test_strip_segue_flags_and_removes_all_forms():
    norm = Normalizer()
    for token in ("->", "~>", ">", "→"):
        stripped, seg = norm.strip_segue(f"Brent Black{token}")
        assert stripped == "Brent Black"
        assert seg is True


def test_strip_segue_bare_tilde_is_cleaned_but_not_flagged():
    """a lone trailing '~' is noise, not a segue: it is removed without setting the flag.

    Verbatim port behavior — the original's bare-'~' branch sets `changed` but never `seg`.
    Only the '~>' pair marks a segue."""
    norm = Normalizer()
    assert norm.strip_segue("Brent Black~") == ("Brent Black", False)


def test_strip_segue_no_marker_is_not_a_segue():
    norm = Normalizer()
    stripped, seg = norm.strip_segue("Rec Chem")
    assert (stripped, seg) == ("Rec Chem", False)


# --- strip_attribution --------------------------------------------------------------------

def test_strip_attribution_drops_cover_credits():
    norm = Normalizer()
    assert norm.strip_attribution("Ophelia [The Band]") == "Ophelia"
    assert norm.strip_attribution("No Rain (Blind Melon cover)") == "No Rain"
    assert norm.strip_attribution("Corduroy (UM)") == "Corduroy"


def test_strip_attribution_keeps_real_parenthesised_titles():
    """parenthesised phrases that are part of the name survive; only attributions go."""
    norm = Normalizer()
    assert norm.strip_attribution("Z0Z (Zed Nought Z)") == "Z0Z (Zed Nought Z)"
    assert norm.strip_attribution("Breathe (In the Air)") == "Breathe (In the Air)"


def test_strip_attribution_removes_encore_mark_and_timecode():
    norm = Normalizer()
    assert norm.strip_attribution("e. Ophelia") == "Ophelia"
    assert norm.strip_attribution("(09:39) Tamborine") == "Tamborine"


def test_normalize_um_attribution_is_case_sensitive_and_runs_before_lower():
    """the (UM) rule is uppercase-only and must fire before lowercasing.

    End-to-end through normalize(), not strip_attribution() in isolation: this pins the
    ordering (strip before lower) AND the case-sensitivity. Reordering .lower() ahead of
    strip_attribution would break the first assertion; matching lowercase initialisms would
    break the second (a bare '(um)' is a taper's word, not a cover credit)."""
    norm = Normalizer()
    assert norm.normalize("Corduroy (UM)") == "corduroy"
    assert norm.normalize("Corduroy (um)") == "corduroy um"


def test_strip_attribution_guest_needs_a_separator():
    """a bare 'with' stays (Don't Fuck With Flo); a dashed or slashed credit is cut."""
    norm = Normalizer()
    assert norm.strip_attribution("Don't Fuck With Flo") == "Don't Fuck With Flo"
    assert norm.strip_attribution("St. Augustine -with emma d") == "St. Augustine"
    assert norm.strip_attribution("Meat w/ Haley Jane") == "Meat"


# --- build_vocab / canonicalize -----------------------------------------------------------

def test_build_vocab_is_deterministic_on_normalized_collisions():
    """when two display names share a normalized key the sorted winner is stable."""
    norm = Normalizer(vocabulary=["The Faker", "Faker"])
    canon, n2c = norm.build_vocab()
    assert canon == ["Faker", "The Faker"]
    assert n2c["faker"] == "Faker"        # sorted() picks "Faker" before "The Faker"


def test_canonicalize_exact_normalized_match():
    norm = Normalizer(vocabulary=["Recreational Chemistry"])
    assert norm.canonicalize("recreational chemistry") == ("Recreational Chemistry", False)


def test_canonicalize_fuzzy_within_cutoff():
    norm = Normalizer(vocabulary=["Recreational Chemistry"])
    canon, seg = norm.canonicalize("Recreational Chemistr")   # dropped trailing y, inside 0.9
    assert canon == "Recreational Chemistry"


def test_canonicalize_unknown_falls_through_to_cleaned_display():
    norm = Normalizer(vocabulary=["Recreational Chemistry"])
    canon, seg = norm.canonicalize("Bat Country [Avenged Sevenfold]>")
    assert canon == "Bat Country"       # attribution stripped, segue flagged
    assert seg is True


def test_canonicalize_empty_after_normalize_is_none():
    norm = Normalizer(vocabulary=["Anything"])
    canon, seg = norm.canonicalize("(cover)")   # a pure attribution normalizes to nothing
    assert canon is None


def test_canonicalize_uses_pack_aliases():
    norm = _StubNormalizer(vocabulary=["Recreational Chemistry"])
    assert norm.canonicalize("Rec Chem")[0] == "Recreational Chemistry"
    assert norm.canonicalize("ZOZ")[0] == "Z0Z (Zed Nought Z)"


def test_alias_targets_fold_into_the_vocabulary():
    """an alias value is a canonical name even if the caller never listed it."""
    norm = _StubNormalizer(vocabulary=[])
    canon, _ = norm.build_vocab()
    assert "Recreational Chemistry" in canon
    assert "Z0Z (Zed Nought Z)" in canon


# --- synonym_map --------------------------------------------------------------------------

def test_synonym_map_collapses_spelling_variants():
    """three spellings of one song map onto the single canonical form."""
    norm = _StubNormalizer(vocabulary=[])
    names = ["Recreational Chemistry", "Rec Chem", "recreational chemistry"]
    mapped = norm.synonym_map(names)
    assert set(mapped.values()) == {"Recreational Chemistry"}


# --- is_non_song + protected guard --------------------------------------------------------

def test_is_non_song_matches_injected_patterns():
    norm = _StubNormalizer()
    assert norm.is_non_song("Setbreak") is True
    assert norm.is_non_song("Al.nouncements") is True
    assert norm.is_non_song("Recreational Chemistry") is False


def test_is_non_song_catches_guest_and_bare_notes():
    norm = _StubNormalizer()
    assert norm.is_non_song("w/Andy Frasco") is True   # slashed credit, no space
    assert norm.is_non_song("with Haley Jane") is True  # spelled-out form
    assert norm.is_non_song("(UM)") is True             # bare parenthesised note


def test_guest_note_misses_space_after_slash_KNOWN_BUG():
    """LOCKS a latent bug ported verbatim from songnorm.py, to be fixed as a measured change.

    GUEST_NOTE is `^\\(?\\s*=?\\s*w(?:ith|/)\\b`. The `\\b` after "/" cannot fire before a
    space, so the regex's OWN documented targets -- "w/ Andy Frasco", "(w/ BRONCO)" -- slip
    through and are treated as songs. Only "w/word" (no space) and the "with " form are
    caught. When the pack/normalizer fix lands (its own commit), these flip to True."""
    norm = _StubNormalizer()
    assert norm.is_non_song("w/ Andy Frasco") is False
    assert norm.is_non_song("(w/ BRONCO)") is False


def test_protected_title_is_always_a_song():
    """ATL/TLH/NYC survive even when a pattern DOES match them; formatting is ignored."""
    class _Trap(Normalizer):
        def non_song_patterns(self):
            # each of these would delete a protected moe. song without the guard
            return [re.compile(r"atl"), re.compile(r"tlh"), re.compile(r"nyc")]

        def protected_titles(self):
            return {"ATL", "TLH", "NYC"}

    trap = _Trap()
    for title in ("ATL", "TLH", "NYC"):
        assert trap.is_non_song(title) is False
    # squash-both-sides: punctuation in the entry does not sneak a protected title past the guard
    assert trap.is_non_song("A.T.L") is False


def test_protected_guard_beats_a_matching_pattern():
    class _Trap(Normalizer):
        def non_song_patterns(self):
            return [re.compile(r"nyc")]      # would delete NYC without the guard

        def protected_titles(self):
            return {"NYC"}

    assert _Trap().is_non_song("NYC") is False
    # a non-protected entry the same pattern hits is still dropped
    assert _Trap().is_non_song("nyc jam") is True


# --- base class stands alone --------------------------------------------------------------

def test_base_normalizer_has_empty_policy_hooks():
    norm = Normalizer()
    assert norm.aliases() == {}
    assert norm.non_song_patterns() == []
    assert norm.protected_titles() == set()
    # with no patterns nothing is a non-song except the built-in shape rules
    assert norm.is_non_song("Anything At All") is False
    assert norm.is_non_song("(UM)") is True

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
from setlistkit.catalog.normalizer import clean_song


# A synthetic pack: enough policy to exercise the hooks, none of the real moe. data.
class _StubNormalizer(Normalizer):
    def aliases(self):
        return {"rec chem": "Recreational Chemistry", "zoz": "Z0Z (Zed Nought Z)"}

    def non_song_patterns(self):
        return [re.compile(r"^setbreak$"), re.compile(r"nounc"), re.compile(r"intro")]

    def protected_titles(self):
        return {"ATL", "NYC"}


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


def test_guest_note_catches_a_space_after_the_slash():
    """the bug ported from songnorm.py, now fixed: these are the regex's OWN examples.

    `w(?:ith|/)\\b` could not match them, because a `\\b` cannot fire between "/" and a space,
    so the one thing GUEST_NOTE was written to catch went into the vocabulary as a song. The
    boundary now sits on "with" alone."""
    norm = _StubNormalizer()
    assert norm.is_non_song("w/ Andy Frasco") is True
    assert norm.is_non_song("(w/ BRONCO)") is True


def test_a_song_whose_name_starts_with_with_survives():
    """why the boundary has to stay on the spelled-out form: "with" is a word, "w/" is not."""
    norm = _StubNormalizer()
    assert norm.is_non_song("Within Your Reach") is False


def test_a_guest_annotation_is_not_a_different_song():
    """"Moth (w/ Daniel Donato)" is a Moth. Left attached it files a real performance under its
    own name with n=1, which is how one song appears three times with a sample size of one."""
    assert clean_song("Moth (w/ Daniel Donato)") == "Moth"
    assert clean_song("Moth (with the horns)") == "Moth"
    assert clean_song('Moth"') == "Moth"
    # A parenthetical that is not a guest credit stays put -- it may be part of the title.
    assert clean_song("Rebubula (reprise)") == "Rebubula (reprise)"


def test_cleaning_a_song_name_does_not_eat_its_apostrophe():
    """a curly apostrophe is in the same character class as a curly quote, and an earlier form of
    this deleted every one of them wherever it stood.

    "Hey, It's Christmas" was published as "Hey, Its Christmas" -- a real song under a name it
    does not have, which then matched nothing else spelled correctly. A quote character between
    two letters is an apostrophe."""
    assert clean_song("Hey, It’s Christmas") == "Hey, It’s Christmas"
    assert clean_song("Rob’s Speech") == "Rob’s Speech"
    assert clean_song("Don't Fuck With Flo") == "Don't Fuck With Flo"


def test_cleaning_leaves_a_spoken_moment_its_quotes():
    """the quotes ARE the classification, so cleaning them off destroys it.

    Everything downstream asks is_non_song about the CLEANED entry. Strip the quotes here and the
    classifier is handed 'Penguin Joke' with nothing to recognize -- which is how a knock-knock
    joke came to be published with a median length of 4m45s."""
    assert clean_song('"Penguin Joke"') == '"Penguin Joke"'
    assert clean_song('  "thank you very much everybody..."  ') == '"thank you very much everybody..."'
    # A stray quote on one end only is still parse debris, not a classification.
    assert clean_song('Moth"') == "Moth"


def test_the_two_halves_of_an_export_file_a_song_under_one_name():
    """the join this function exists to keep honest. features keyed songs by the raw setlist
    string while the length chain cleaned them, so a song's lengths and its structural profile
    were two records nothing could put back together."""
    from setlistkit.catalog.features import song_features

    shows = [{"date": "2024-01-01",
              "sets": [[{"song": "Hey, It’s Christmas"}, {"song": "Moth (w/ Daniel Donato)"}]]}]
    assert sorted(f.song for f in song_features(shows)) == ["Hey, It’s Christmas", "Moth"]


def test_an_entry_wrapped_in_quotes_is_a_spoken_moment_not_a_song():
    """how a setlist writes down banter. Left as songs these get TIMED: the corpus published
    "thank you very much everybody..." with a median length of 14 seconds, and 25 more like it."""
    norm = _StubNormalizer()
    assert norm.is_non_song('"thank you very much everybody..."') is True
    assert norm.is_non_song('"Penguin Joke"') is True
    assert norm.is_non_song("“Band Interview”") is True     # curly quotes are the same convention


def test_a_song_carrying_a_quoted_note_keeps_its_slot():
    """the trap in the quoted rule: the quotes have to wrap the WHOLE entry.

    'Wind it Up "False Start"' is a Wind it Up that went wrong, not a spoken moment, and a rule
    that matched a quote anywhere would delete the song along with the note."""
    norm = _StubNormalizer()
    assert norm.is_non_song('Wind it Up "False Start"') is False
    assert norm.is_non_song('end of "soundcheck" etc') is False


def test_a_protected_title_survives_being_quoted():
    """the escape hatch the quoted rule leans on: it is a shape rule with no vocabulary behind
    it, so the day a real title arrives in quotes, the pack is what saves it."""
    class _Quoted(Normalizer):
        def protected_titles(self):
            return {"SuperJam"}

    assert _Quoted().is_non_song('"SuperJam"') is False


def test_protected_title_is_always_a_song():
    """ATL and NYC survive even when a pattern DOES match them; formatting is ignored."""
    class _Trap(Normalizer):
        def non_song_patterns(self):
            # each of these would delete a protected moe. song without the guard
            return [re.compile(r"atl"), re.compile(r"nyc")]

        def protected_titles(self):
            return {"ATL", "NYC"}

    trap = _Trap()
    for title in ("ATL", "NYC"):
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

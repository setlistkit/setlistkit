# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Tests for `slkit pack suggest-vocab` -- the pack-bootstrap worklist.

These lock the properties the objective turns on: nothing is silently dropped (banter is segregated,
never deleted), and the aggressive number-stripping that FEEDS the clustering never reaches the
LABEL a human reads -- so "25 or 6 to 4" is never shown as "or 6 to 4", and a track index never
appears as a candidate name.
"""

import json

from setlistkit.catalog.suggest import (
    preclean_display,
    preclean_key,
    render_markdown,
    suggest,
)
from setlistkit.cli.main import EXIT_DIAGNOSTIC, EXIT_OK, main


# --- preclean: the archive.org noise the normalizer never sees ----------------------------

def test_sure_track_numbers_are_stripped_from_display():
    """A leading zero or a dotted index is unambiguously a track number, gone even from display."""
    assert preclean_display("01 Cut The Cable") == "Cut The Cable"
    assert preclean_display("02. Domino Theory") == "Domino Theory"
    assert preclean_display("08 Den") == "Den"


def test_bare_number_survives_in_display_but_not_in_key():
    """A bare 1-2 digit lead is ambiguous: kept for the label, dropped for the clustering key."""
    assert preclean_display("10 All In Time") == "10 All In Time"      # label keeps it
    assert preclean_key("10 All In Time") == "All In Time"            # key drops the number


def test_number_led_real_title_is_never_mangled():
    """The whole point of the bare/sure split: a song that IS a number keeps its name."""
    assert preclean_display("25 or 6 to 4") == "25 or 6 to 4"
    assert preclean_display("32 Things") == "32 Things"
    # Four-digit titles are never touched at all, by either pass.
    assert preclean_display("1348") == "1348"
    assert preclean_key("1999") == "1999"


def test_sigils_and_encore_flag_are_stripped():
    assert preclean_display("#@+$Ophelia") == "Ophelia"
    assert preclean_display("*Der Bluten Kat >") == "Der Bluten Kat"
    assert preclean_display("(E) Helplessly Hoping") == "Helplessly Hoping"
    assert preclean_display("Wappy Sprayberry#") == "Wappy Sprayberry"


# --- non-song segregation: keep-and-flag, never drop --------------------------------------

def test_parenthesised_banter_is_segregated_not_dropped():
    titles = ["Ophelia", "(Andy Farag Intro)", "(Outro)", "Bridgeless"]
    result = suggest(titles)
    songs = {c.canonical for c in result.songs}
    assert "Ophelia" in songs
    assert "(Andy Farag Intro)" in result.non_songs
    assert "(Outro)" in result.non_songs
    # nothing vanished: every input is accounted for somewhere
    accounted = {m for c in result.songs for m in c.members} | set(result.non_songs)
    assert accounted == set(titles)


def test_blank_titles_are_skipped():
    """Empty / whitespace-only entries are neither songs nor flagged non-songs -- just dropped."""
    result = suggest(["Ophelia", "", "   "])
    assert [c.canonical for c in result.songs] == ["Ophelia"]
    assert result.non_songs == ()


def test_all_noise_title_is_segregated_not_an_empty_song():
    """A title that precleans to nothing must not mint a blank-keyed 'song'."""
    result = suggest(["#@+$", "Ophelia"])
    assert "#@+$" in result.non_songs
    assert all(c.canonical for c in result.songs)


# --- clustering: fold the near-duplicates ------------------------------------------------

def test_track_numbered_variants_fold_into_one_song():
    titles = ["01 1348", "08 1348", "1348", "1348 >", "16 1348"]
    result = suggest(titles)
    assert len(result.songs) == 1
    cluster = result.songs[0]
    assert cluster.canonical == "1348"           # the de-numbered label, not "16 1348"
    assert cluster.size == 5                       # every raw variant retained


def test_fuzzy_spelling_variants_merge_and_pick_the_common_spelling():
    """The medoid label: the majority spelling wins over a lone typo."""
    titles = ["Bridgeless", "Bridgeless >", "09 Bridgeless", "Bridgelesss"]
    result = suggest(titles)
    assert len(result.songs) == 1
    assert result.songs[0].canonical == "Bridgeless"   # not the doubled-s typo


def test_distinct_songs_do_not_merge():
    """A conservative cutoff keeps unrelated short titles apart."""
    result = suggest(["Glory", "Prowler", "Wingtips"], cutoff=0.88)
    assert len(result.songs) == 3


# --- output artifacts ---------------------------------------------------------------------

def test_aliases_map_variant_key_to_canonical_only_for_real_folds():
    result = suggest(["Believe The Lie", "Belive The Lie", "13 Believe The Lie"])
    aliases = result.aliases()
    assert aliases.get("belive the lie") == "Believe The Lie"
    # the canonical's own key never aliases to itself
    assert "believe the lie" not in aliases


def test_vocabulary_is_sorted_and_deduped():
    result = suggest(["Prowler", "Glory", "Glory >"])
    assert result.vocabulary() == ["Glory", "Prowler"]


def test_markdown_shows_every_member_and_segregates_non_songs():
    md = render_markdown(suggest(["01 1348", "1348", "(Outro)"]))
    assert "## Song candidates" in md
    assert "## Probable non-songs" in md
    assert "(Outro)" in md
    assert "01 1348" in md          # the folded variant is shown for the human to audit


# --- CLI end to end -----------------------------------------------------------------------

def _titles_file(tmp_path, titles):
    path = tmp_path / "titles.json"
    path.write_text(json.dumps(titles), encoding="utf-8")
    return path


def test_cli_writes_three_artifacts(tmp_path, capsys):
    src = _titles_file(tmp_path, ["01 1348", "1348", "Belive The Lie", "Believe The Lie", "(Outro)"])
    out = tmp_path / "pack"
    assert main(["pack", "suggest-vocab", "--from", str(src), "--out", str(out)]) == EXIT_OK
    assert (out / "vocabulary.json").is_file()
    assert (out / "aliases.json").is_file()
    assert (out / "SUGGESTIONS.md").is_file()
    vocab = json.loads((out / "vocabulary.json").read_text())
    assert "1348" in vocab
    assert "probable non-songs" in capsys.readouterr().out


def test_cli_refuses_to_clobber_without_force(tmp_path, capsys):
    src = _titles_file(tmp_path, ["Ophelia"])
    out = tmp_path / "pack"
    out.mkdir()
    (out / "vocabulary.json").write_text('["Hand Made Real Song"]', encoding="utf-8")
    assert main(["pack", "suggest-vocab", "--from", str(src), "--out", str(out)]) == EXIT_DIAGNOSTIC
    assert "refusing to overwrite" in capsys.readouterr().err
    # the human's file is untouched
    assert "Hand Made Real Song" in (out / "vocabulary.json").read_text()


def test_cli_force_overwrites(tmp_path):
    src = _titles_file(tmp_path, ["Ophelia"])
    out = tmp_path / "pack"
    out.mkdir()
    (out / "vocabulary.json").write_text('["Stale"]', encoding="utf-8")
    assert main(
        ["pack", "suggest-vocab", "--from", str(src), "--out", str(out), "--force"]) == EXIT_OK
    assert json.loads((out / "vocabulary.json").read_text()) == ["Ophelia"]


def test_cli_empty_placeholder_files_do_not_need_force(tmp_path):
    """A `git init`ed pack with empty [] / {} placeholders is safe to fill without --force."""
    src = _titles_file(tmp_path, ["Ophelia"])
    out = tmp_path / "pack"
    out.mkdir()
    (out / "vocabulary.json").write_text("[]", encoding="utf-8")
    (out / "aliases.json").write_text("{}", encoding="utf-8")
    assert main(["pack", "suggest-vocab", "--from", str(src), "--out", str(out)]) == EXIT_OK


def test_cli_bad_input_is_a_diagnostic(tmp_path, capsys):
    bad = tmp_path / "titles.json"
    bad.write_text('{"not": "an array"}', encoding="utf-8")
    out = tmp_path / "pack"
    assert main(["pack", "suggest-vocab", "--from", str(bad), "--out", str(out)]) == EXIT_DIAGNOSTIC
    assert "not a JSON array of strings" in capsys.readouterr().err


def test_cli_missing_input_is_a_diagnostic(tmp_path, capsys):
    out = tmp_path / "pack"
    ghost = tmp_path / "ghost.json"
    assert main(["pack", "suggest-vocab", "--from", str(ghost), "--out", str(out)]) == EXIT_DIAGNOSTIC
    assert "could not read titles" in capsys.readouterr().err


def test_cli_pack_lint_still_defaults(tmp_path):
    """The pack dispatcher must not have broken the bare `pack lint` path: an empty directory is
    not a valid pack, so lint routes through and reports a diagnostic rather than crashing."""
    assert main(["pack", "lint", "--pack", str(tmp_path)]) == EXIT_DIAGNOSTIC

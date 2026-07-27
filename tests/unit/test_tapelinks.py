# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Where a reader goes to hear a tape, per source.

The bug these guard against does not look like a bug. Every URL this module can build is
well-formed, so nothing downstream -- no parser, no build step, no link checker that only reads
shape -- can tell a correct one from a confident 404. The only place the difference is knowable is
here, against the source that supplied the row.
"""

from setlistkit.catalog import tapelinks


def test_an_archive_tape_links_to_its_own_item_page():
    assert tapelinks.item_url("archive_org", "moe2024-06-01.flac16") == \
        "https://archive.org/details/moe2024-06-01.flac16"


def test_a_phish_in_tape_has_no_per_recording_page_and_says_so():
    """The tempting answer is the night's URL, and it is wrong for this question.

    phish.in addresses recordings by date -- one official transfer a night, no per-transfer page.
    A caller asking "link me to THIS tape" and silently receiving a page about the night gets an
    answer to a question it did not ask, and cannot detect the substitution.
    """
    assert tapelinks.item_url("phish_in", "phish_in-2042") is None


def test_an_archive_url_is_never_built_for_a_source_that_is_not_archive_org():
    """The regression this exists for: `details/phish_in-2042` is well-formed and 404s.

    The old code built the URL from a literal prefix and an identifier, which was indistinguishable
    from a rule for as long as there was one supplier.
    """
    for source in ("phish_in", "phish_net", "setlist_fm", "", "made_up"):
        assert tapelinks.item_url(source, "phish_in-2042") != \
            "https://archive.org/details/phish_in-2042"


def test_a_night_on_phish_in_is_the_date_itself():
    assert tapelinks.night_url("phish_in", "2026-07-24") == "https://phish.in/2026-07-24"


def test_a_night_on_archive_org_is_a_search_scoped_to_the_collection_and_the_date():
    """A SEARCH, deliberately, and not a tape we hold.

    The count beside this link is what we last saw. Tapes land weeks after the show, so the reader
    is being sent to look at what is there now -- linking to one of our own rows would freeze the
    page's answer at the moment of the last pull.
    """
    assert tapelinks.night_url("archive_org", "2026-06-17", collection="moe") == (
        "https://archive.org/search?"
        "query=collection%3Amoe%20AND%20date%3A%5B2026-06-17%20TO%202026-06-17%5D")


def test_archive_org_without_a_collection_declines_rather_than_searching_the_world():
    """A date alone matches every band who played that night, which is not the question asked."""
    assert tapelinks.night_url("archive_org", "2026-06-17") is None


def test_an_unknown_source_gets_no_link_at_all():
    """Nothing is rendered rather than something guessed: a dead link looks like a fact."""
    assert tapelinks.night_url("", "2026-06-17", collection="moe") is None
    assert tapelinks.night_url("phish_net", "2026-06-17") is None
    assert tapelinks.item_url("", "whatever") is None


def test_a_missing_date_produces_no_night_link():
    assert tapelinks.night_url("phish_in", "") is None
    assert tapelinks.night_url("archive_org", "", collection="moe") is None


def test_a_missing_identifier_produces_no_item_link():
    assert tapelinks.item_url("archive_org", "") is None


def test_identifiers_and_collections_that_need_escaping_are_escaped():
    """Both fields are somebody else's strings: an item name and a config value.

    A collection with a space in it building a raw URL is the kind of thing that works on every
    band whose name is one word.
    """
    assert tapelinks.item_url("archive_org", "moe 2024/06") == \
        "https://archive.org/details/moe%202024%2F06"
    assert "collection%3Athe%20band" in tapelinks.night_url(
        "archive_org", "2026-06-17", collection="the band")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Catalog: songs, vocabulary, segues, families, and durations.

This layer is useful standalone to someone who never touches prediction: a jam-band song
graph and normalizer that stands on its own. It therefore imports nothing from ``model``,
``picks``, or ``report`` — a rule enforced by a test. (Populated in a later phase.)
"""

from .features import SongFeature, song_features
from .lint import lint
from .merge import (COMPLETE_FRAC, DEFAULT_RANKS, MergePolicy, MergeResult, apply_overrides,
                    merge_shows, override_disagreements, overrides_from_mapping, pick_show)
from .normalizer import Normalizer
from .pack import CorpusPolicy, Pack, load_pack
from .parse import (ArchivePolicy, count_songs, parse_archive_item, parse_archive_items,
                    title_band_filter)

__all__ = ["COMPLETE_FRAC", "DEFAULT_RANKS", "ArchivePolicy", "CorpusPolicy", "MergePolicy",
           "MergeResult", "Normalizer", "Pack", "SongFeature", "apply_overrides", "count_songs",
           "lint", "load_pack", "merge_shows", "override_disagreements",
           "overrides_from_mapping", "parse_archive_item", "parse_archive_items", "pick_show",
           "song_features", "title_band_filter"]

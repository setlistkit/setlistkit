# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Catalog: songs, vocabulary, segues, families, and durations.

This layer is useful standalone to someone who never touches prediction: a jam-band song
graph and normalizer that stands on its own. It therefore imports nothing from ``model``,
``picks``, or ``report`` — a rule enforced by a test. (Populated in a later phase.)
"""

from .lint import lint
from .normalizer import Normalizer
from .pack import Pack, load_pack

__all__ = ["Normalizer", "Pack", "load_pack", "lint"]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The archive.org importer: the mirror source.

archive.org carries the audio -- tapes, track durations, show types -- so it is the source
``ingest`` reads explicitly to build the mirror, not a fold-in. Hence ``candidates=None``: its
merge records come out of the same ``parse_archive_items`` pass the mirror needs, which ``ingest``
already runs directly, and routing them back through a generic candidate call would parse twice.
"""

from __future__ import annotations

from ..cli.common import required_setting
from ..sources.archive_org import NAMESPACE, ArchiveOrgClient
from .base import Importer


def _identity(config) -> str:
    """The archive.org collection that holds the band's tapes, or a diagnostic naming the table.

    Uses the shared ``required_setting`` so the "not set" message is the one every other missing
    config value gives, rather than a second dialect of the same error.
    """
    return required_setting(
        config, ("sources", NAMESPACE), "collection",
        "A pull needs to know which archive.org collection holds this band's tapes.")


IMPORTER = Importer(
    id=NAMESPACE,
    feeds_mirror=True,
    make_client=ArchiveOrgClient,
    identity=_identity,
    candidates=None,          # read explicitly by ingest for the mirror; see the module docstring
)

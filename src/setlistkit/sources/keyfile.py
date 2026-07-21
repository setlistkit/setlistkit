# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""API key files: read one, but refuse it if the world can read it.

A key never lives inline in the config and never in an env var a shell history can leak. It
lives in its own file, named by path, and before it is ever used this module stats that
file. A key readable by ``other`` is a hard failure -- the whole point of a key file is that
it is not sitting where anyone with a shell on the box can copy it. Owner and group bits are
the operator's business; the world bit is not negotiable, and 0600 is the recommendation.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from ..diagnostics import ERROR, Diagnostic, DiagnosticError


def read_api_key(path) -> str:
    """Return the key held in the file at ``path``, or raise a diagnostic that says how to fix it.

    Fails -- never warns -- when the file is missing, empty, or readable by ``other``: a key the
    world can read is not a secret, and the caller asked for a secret. ``~`` is expanded so a
    config can point at ``~/.config/setlistkit/setlistfm.api_key``.
    """
    key_path = Path(os.path.expanduser(str(path)))
    if not key_path.is_file():
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="api key file not found",
            path=str(key_path),
            detail="A source is configured with an api_key_file that does not exist. Create it\n"
                   "(containing only the key), or fix the path in your config.",
        ))
    # Open once, then check the mode of THIS descriptor and read from it. A separate stat()
    # then read() leaves a window where a 0600 file is swapped for a world-readable one between
    # the check and the read; checking the fd we read from means both see the same inode.
    try:
        fd = os.open(key_path, os.O_RDONLY)
    except OSError as err:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="api key file could not be opened",
            path=str(key_path),
            detail=f"opening the api_key_file failed: {err}",
        )) from err
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        if os.fstat(handle.fileno()).st_mode & stat.S_IROTH:
            raise DiagnosticError(Diagnostic(
                severity=ERROR,
                summary="api key file is world-readable",
                path=str(key_path),
                detail="Anyone on this machine can read the key. Restrict it:\n\n"
                       f"    chmod 600 {key_path}\n\n"
                       "Owner and group access are your call; world-readable is refused.",
            ))
        key = handle.read().strip()
    if not key:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="api key file is empty",
            path=str(key_path),
            detail="The api_key_file exists but holds no key. Put the API key in it.",
        ))
    return key

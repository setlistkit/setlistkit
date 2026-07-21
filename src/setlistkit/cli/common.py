# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Config resolution shared by more than one subcommand.

Here rather than in ``main`` so ``ingest`` can reach it without importing the dispatcher that
dispatches to it. Every function turns a missing or malformed setting into a
:class:`~setlistkit.diagnostics.Diagnostic` naming the file and the table it should have been in,
because "which config, which key" is the entire content of the answer a user needs.

Relative paths resolve against the CONFIG FILE'S directory, never the process cwd. A downstream
deployment commits its config next to its pack, and it has to mean the same thing whether ``slkit``
runs from that directory, from a systemd unit, or from cron.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import load_config
from ..diagnostics import ERROR, Diagnostic, DiagnosticError


def required_setting(config, section: tuple[str, ...], key: str, what: str) -> str:
    """A non-empty string setting, or a diagnostic naming the table it was missing from."""
    value = str(config.section(*section).get(key) or "").strip()
    if not value:
        table = ".".join(section)
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary=f"[{table}] {key} is not set",
            path=str(config.source_path),
            detail=f"{what}\n\nAdd it to your config:\n\n    [{table}]\n    {key} = \"...\"",
        ))
    return value


def min_year(config, source: str, flag: int | None) -> int | None:
    """The play-year floor: the ``--min-year`` flag, else ``min_year`` in the source's table.

    A floor keeps a late-uploaded decades-old recording from drifting into the vocabulary. It is
    optional, and absent means no floor at all rather than a default year -- guessing one would
    silently discard shows nobody asked us to discard.
    """
    if flag is not None:
        return flag
    configured = config.section("sources", source).get("min_year")
    if configured is None:
        return None
    # A quoted year in TOML is the easy typo, and int("2020") would paper over it here while the
    # same value went into a cache key as a string somewhere else. One shape, checked once.
    if not isinstance(configured, int) or isinstance(configured, bool):
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary=f"[sources.{source}] min_year must be a number, got {configured!r}",
            path=str(config.source_path),
            detail=f"Write it unquoted:\n\n    [sources.{source}]\n    min_year = 2020",
        ))
    return configured


def resolve_pack_dir(pack_arg, config_arg) -> Path:
    """The pack directory to work on: the ``--pack`` flag, else ``[catalog] pack`` in config."""
    if pack_arg:
        return Path(pack_arg).expanduser().resolve()
    config = load_config(config_arg)
    configured = config.section("catalog").get("pack")
    if not configured:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="no pack configured",
            path=str(config.source_path),
            detail="Pass --pack PATH, or set [catalog] pack in your config to a pack directory.",
        ))
    path = Path(os.path.expanduser(str(configured)))
    if not path.is_absolute():
        path = config.source_path.parent / path
    return path.resolve()

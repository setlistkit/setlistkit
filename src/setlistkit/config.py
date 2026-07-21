# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The config contract: locate a TOML file, parse it, and resolve ``data_root``.

Resolution order for the file itself, first match wins:

1. an explicit path (the ``--config`` flag)
2. the ``SLKIT_CONFIG`` environment variable
3. ``./slkit.toml`` in the current directory
4. ``$XDG_CONFIG_HOME/setlistkit/config.toml`` (``$XDG_CONFIG_HOME`` defaults to ``~/.config``)

The file never holds a credential and is safe to commit in a downstream deployment repo:
the setlist.fm key lives in its own file, referenced by path. ``data_root`` is where all
mutable state lives, deliberately outside the repository and behind this pointer.

``user_agent`` identifies the deployment to every upstream API. It ships as a sentinel that
the program refuses to run against the network until it is changed, so nobody accidentally
sends anonymous or impersonating traffic. That refusal is enforced here, at
:func:`require_network_identity`, which the source clients call before any request.

It is a PREFIX, not the whole header. During a bulk run -- a pull of a whole collection, which
is thousands of requests -- the client appends a second comment naming the run and its position
in it, so the host can tell one job from twenty and see that it is finite::

    user_agent = "famoe.ly/0.1 (+mailto:you@example.com; AI agent)"
    on the wire = "famoe.ly/0.1 (+mailto:you@example.com; AI agent) (batch 3f7a9c21; item 100/4514)"

Documented here rather than left as a surprise, because the configured string belongs to the
operator and finding bytes in it you did not write is not a thing a tool should do quietly. Set
``user_agent_batch_progress = false`` to send the configured string unchanged; the default is
true, because being legible to the host you are spending bandwidth on is the better default and
the cost is nothing.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .diagnostics import ERROR, Diagnostic, DiagnosticError

# The shipped stub value. Any user_agent still bearing the CHANGE-ME marker means the
# operator has not identified themselves, and no network call may proceed.
SENTINEL_USER_AGENT = "CHANGE-ME (setlistkit; you@example.com)"
_SENTINEL_MARKER = "CHANGE-ME"

# Filenames used during resolution.
LOCAL_CONFIG_NAME = "slkit.toml"
XDG_CONFIG_RELPATH = Path("setlistkit") / "config.toml"


@dataclass(frozen=True)
class Config:
    """A resolved configuration.

    ``data_root`` is absolute. ``raw`` is the full parsed TOML so sections not yet modelled
    (``[sources.*]``, ``[report]``) remain reachable by later layers without this type
    having to grow a field per phase. ``source_path`` records which file was loaded, which
    matters for error messages and for resolving relative paths within the config.
    """

    data_root: Path
    user_agent: str
    source_path: Path
    # Whether a bulk run appends its batch id and progress to user_agent. True by default: the
    # host whose bandwidth we are spending gets to see what the job is and how big it is. An
    # operator whose upstream keys on an exact registered User-Agent turns it off.
    user_agent_batch_progress: bool = True
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    def section(self, *names: str) -> dict:
        """Return a copy of a nested table (e.g. ``section("sources", "setlistfm")``).

        A shallow copy, so a caller cannot mutate this frozen config's state through the
        returned dict. Returns ``{}`` for any missing path.
        """
        node: object = self.raw
        for name in names:
            if not isinstance(node, dict):
                return {}
            node = node.get(name, {})
        return dict(node) if isinstance(node, dict) else {}

    @property
    def user_agent_is_sentinel(self) -> bool:
        """True while user_agent still bears the shipped CHANGE-ME marker."""
        return _SENTINEL_MARKER in self.user_agent


def _search_locations(explicit_path, env, cwd) -> list[Path]:
    """The ordered candidate paths, filtered to those actually configured/present."""
    locations: list[Path] = []
    if explicit_path is not None:
        locations.append(Path(explicit_path).expanduser())
    env_path = env.get("SLKIT_CONFIG")
    if env_path:
        locations.append(Path(env_path).expanduser())
    locations.append(cwd / LOCAL_CONFIG_NAME)
    xdg = env.get("XDG_CONFIG_HOME")
    xdg_base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    locations.append(xdg_base / XDG_CONFIG_RELPATH)
    return locations


def resolve_config_path(explicit_path=None, *, env=None, cwd=None) -> Path:
    """Find the config file. Raise a :class:`DiagnosticError` if none exists.

    An explicit ``--config`` path that does not exist is its own, more specific error: the
    user named a file and it is not there, which is different from finding no config at all.
    """
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else Path(cwd)

    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="config file not found",
            path=str(candidate),
            detail="This path was given explicitly with --config but does not exist. Check\n"
                   "the path, or drop the flag to fall back to ./slkit.toml and\n"
                   "$XDG_CONFIG_HOME/setlistkit/config.toml.",
        ))

    for candidate in _search_locations(None, env, cwd):
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(f"  - {p}" for p in _search_locations(None, env, cwd))
    raise DiagnosticError(Diagnostic(
        severity=ERROR,
        summary="no configuration file found",
        detail="Searched, in order:\n"
               f"{searched}\n\n"
               "Create one of these, or point at a file with --config or SLKIT_CONFIG.\n"
               "A starter config lives at slkit.example.toml.",
    ))


def _resolve_data_root(raw_value: str, source_path: Path) -> Path:
    """Expand ``~`` and make ``data_root`` absolute, relative paths anchored at the config.

    Anchoring a relative ``data_root`` at the config file's directory (not the process cwd)
    means a committed downstream config behaves the same no matter where ``slkit`` is run.
    """
    expanded = os.path.expanduser(raw_value)
    path = Path(expanded)
    if not path.is_absolute():
        path = source_path.parent / path
    return path.resolve()


def load_config(explicit_path=None, *, env=None, cwd=None) -> Config:
    """Locate, parse, and validate a config file into a :class:`Config`.

    Raises :class:`DiagnosticError` for a missing file, malformed TOML, or a missing
    required key. All three render through the one shared diagnostic format.
    """
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else Path(cwd)

    source_path = resolve_config_path(explicit_path, env=env, cwd=cwd)

    text = source_path.read_text(encoding="utf-8")
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="config file is not valid TOML",
            path=str(source_path),
            detail=str(exc),
        )) from exc

    missing = [key for key in ("data_root", "user_agent") if key not in raw]
    if missing:
        keys = ", ".join(missing)
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary=f"config is missing required key(s): {keys}",
            path=str(source_path),
            detail="Every config needs a data_root (where mutable state lives) and a\n"
                   "user_agent (how this deployment identifies itself to upstream APIs).\n"
                   "See slkit.example.toml for a starting point.",
        ))

    if not str(raw["data_root"]).strip():
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="data_root is empty",
            path=str(source_path),
            detail="data_root points at where mutable state lives and must name a real\n"
                   "directory. An empty value would resolve to the config's own directory,\n"
                   "writing state next to a committable file. Set an explicit path.",
        ))

    # Checked rather than coerced: bool("false") is True, so a quoted TOML boolean would turn
    # the setting into its opposite and say nothing. Same lesson as min_year.
    batch_progress = raw.get("user_agent_batch_progress", True)
    if not isinstance(batch_progress, bool):
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary=f"user_agent_batch_progress must be true or false, got {batch_progress!r}",
            path=str(source_path),
            detail="Write it unquoted:\n\n    user_agent_batch_progress = false\n\n"
                   "It controls whether a bulk run appends its batch id and progress to\n"
                   "user_agent. Leave it out to keep the default, which is true.",
        ))

    data_root = _resolve_data_root(str(raw["data_root"]), source_path)
    return Config(
        data_root=data_root,
        user_agent=str(raw["user_agent"]),
        source_path=source_path,
        user_agent_batch_progress=batch_progress,
        raw=raw,
    )


def require_network_identity(config: Config) -> None:
    """Abort before any network call if ``user_agent`` is still the shipped sentinel.

    Source clients call this before their first request. Identifying yourself is mandatory:
    the identity belongs to whoever runs the deployment, so an operator is accountable for
    their own traffic and no request is ever disguised as a browser or as the project.
    """
    if config.user_agent_is_sentinel or not config.user_agent.strip():
        summary = (
            "user_agent is still the placeholder; refusing to touch the network"
            if config.user_agent_is_sentinel
            else "user_agent is empty; refusing to touch the network"
        )
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary=summary,
            path=str(config.source_path),
            detail="Every outbound request must identify this deployment and a contact, so\n"
                   "an operator is accountable for the traffic. Set user_agent in your config\n"
                   "to something like:\n\n"
                   '    user_agent = "famoe.ly nightly (you@example.com)"\n\n'
                   "Do not impersonate a browser, and do not claim to be setlistkit itself.",
        ))

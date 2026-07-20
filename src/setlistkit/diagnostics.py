# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""One Diagnostic type, one renderer, used everywhere.

The design spec makes diagnostics a first-class requirement, not a polish item: every
failure says what broke, *where*, and what to do about it. The same type and renderer
serve config errors, pack schema errors, and lint findings, so a user learns one output
format and CI parses one shape.

The renderer produces a code frame with a caret when a source position is known::

    error: free-floating pattern must justify itself

      packs/moe/classifiers.json:148:14
      147 |   { "pattern": "reprise",
      148 |     "why": "",
          |            ^^ empty
      149 |     "must_not_match": [] }

      "reprise" matches anywhere in a title, so it can delete a real song.
      Either anchor it (^reprise$) or give a reason and a counter-example.

and degrades gracefully: with only a path it prints the location line; with neither it
prints just the headline and detail. Nothing here touches the network or the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Severities. "error" is fatal (non-zero exit); "warning" and "note" are advisory. Kept as
# bare strings rather than an enum so a Diagnostic serialises to JSON (`--format json`,
# honest exit codes for CI) without a custom encoder.
ERROR = "error"
WARNING = "warning"
NOTE = "note"

_SEVERITIES = (ERROR, WARNING, NOTE)


@dataclass(frozen=True)
class Diagnostic:
    """A single diagnosable problem.

    Only ``severity`` and ``summary`` are required. Supply ``path``/``line``/``col`` to
    anchor it to a location, and ``source`` (the full text of that file) to render a code
    frame with a caret. ``detail`` is the explanatory paragraph shown below the frame.

    ``source`` is excluded from equality and repr: two diagnostics describing the same
    problem are equal regardless of how much surrounding text was attached for rendering,
    which keeps tests asserting on identity rather than on incidental file contents.
    """

    severity: str
    summary: str
    path: str | None = None
    line: int | None = None          # 1-based
    col: int | None = None           # 1-based
    length: int = 1                  # caret width, in characters
    caret_caption: str = ""          # inline label after the caret, e.g. "empty"
    detail: str | None = None
    source: str | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITIES:
            raise ValueError(
                f"unknown severity {self.severity!r}; expected one of {_SEVERITIES}"
            )

    @property
    def is_error(self) -> bool:
        """True for fatal diagnostics (those that should drive a non-zero exit)."""
        return self.severity == ERROR

    def to_dict(self) -> dict:
        """JSON-friendly view for ``--format json``. Omits ``source`` (a rendering aid)."""
        out: dict = {"severity": self.severity, "summary": self.summary}
        if self.path is not None:
            out["path"] = self.path
        if self.line is not None:
            out["line"] = self.line
        if self.col is not None:
            out["col"] = self.col
        if self.caret_caption:
            out["caret_caption"] = self.caret_caption
        if self.detail is not None:
            out["detail"] = self.detail
        return out


class DiagnosticError(Exception):
    """A fatal condition carrying a :class:`Diagnostic`.

    Raised where a problem is detected; caught once at the CLI boundary, which renders the
    diagnostic to stderr and exits non-zero. This keeps every layer free to ``raise`` a
    well-formed diagnostic without knowing how it will be displayed.
    """

    def __init__(self, diagnostic: Diagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(diagnostic.summary)


def _location_line(diag: Diagnostic) -> str | None:
    if diag.path is None:
        return None
    loc = diag.path
    if diag.line is not None:
        loc += f":{diag.line}"
        if diag.col is not None:
            loc += f":{diag.col}"
    return f"  {loc}"


def _code_frame(diag: Diagnostic, context: int = 1) -> list[str]:
    """Render the numbered source lines around the error plus a caret line.

    Returns an empty list when there is nothing to frame (no source text, or no line
    number to anchor to). ``context`` is how many lines of surrounding source to show on
    each side.
    """
    if diag.source is None or diag.line is None:
        return []

    lines = diag.source.splitlines()
    if diag.line < 1 or diag.line > len(lines):
        return []

    first = max(1, diag.line - context)
    last = min(len(lines), diag.line + context)
    gutter = len(str(last))  # width of the widest line number shown

    out: list[str] = []
    for num in range(first, last + 1):
        text = lines[num - 1]
        out.append(f"  {num:>{gutter}} | {text}")
        if num == diag.line and diag.col is not None:
            pad = " " * (diag.col - 1)
            carets = "^" * max(1, diag.length)
            caption = f" {diag.caret_caption}" if diag.caret_caption else ""
            out.append(f"  {'':>{gutter}} | {pad}{carets}{caption}")
    return out


def render(diag: Diagnostic) -> str:
    """Render a diagnostic to the multi-line human form shown in the module docstring."""
    parts: list[str] = [f"{diag.severity}: {diag.summary}"]

    location = _location_line(diag)
    frame = _code_frame(diag)
    if location is not None:
        parts.append("")
        parts.append(location)
        parts.extend(frame)

    if diag.detail:
        parts.append("")
        parts.extend(f"  {ln}" if ln else "" for ln in diag.detail.splitlines())

    return "\n".join(parts)

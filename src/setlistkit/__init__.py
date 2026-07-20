"""setlistkit: a generic, open-source setlist prediction toolkit.

The package is layered, and the layering is enforced by a test rather than by convention:
``catalog`` stands alone (a jam-band song graph with no prediction), ``model`` builds on
``catalog``, ``picks`` builds on both, and ``report`` sits on top of everything while
nothing imports it. Mutable state lives outside the repository behind ``data_root``; band
-specific knowledge lives in data (a pack) rather than in code.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

__all__ = ["__version__"]


def _read_version() -> str:
    """Read the single-source-of-truth VERSION file that also feeds packaging.

    Falls back to the installed distribution metadata when the source tree's VERSION file
    is not present (e.g. an installed wheel run from an unrelated cwd).
    """
    version_file = Path(__file__).resolve().parent.parent.parent / "VERSION"
    try:
        return version_file.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    try:
        return version("setlistkit")
    except PackageNotFoundError:  # pragma: no cover - metadata absent in odd installs
        return "0+unknown"


__version__ = _read_version()

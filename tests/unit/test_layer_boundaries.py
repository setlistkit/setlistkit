# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The dependency rule, enforced by a test rather than by convention.

Goal 4 (the catalog is useful standalone) lives or dies on the first rule below: someone
who wants a song graph and no prediction must be able to install setlistkit and use
``catalog`` alone. These checks parse every module in a layer and reject an import that
reaches a forbidden sibling, whether written absolute (``import setlistkit.model``) or
relative (``from ..model import x``).
"""

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

# layer -> the sibling layers it must not import.
FORBIDDEN = {
    "catalog": {"model", "picks", "report"},
    "model": {"picks", "report"},
    "picks": {"report"},
    "report": set(),
}


def _imports_in_source(source: str, package_parts: list[str]) -> set[str]:
    """Absolute dotted module names a source file could reach, resolving relative imports.

    Every imported *name* is also expanded against its base package, because
    ``from .. import model`` and ``from setlistkit import model`` reach a sibling layer as a
    name, not as a dotted module path. Without that expansion the guardrail has a hole: the
    two idiomatic sibling-import forms would slip past it.
    """
    tree = ast.parse(source)
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                head = package_parts[: len(package_parts) - (node.level - 1)]
                tail = [node.module] if node.module else []
                base = ".".join(head + tail)
            if base:
                mods.add(base)
                for alias in node.names:
                    mods.add(f"{base}.{alias.name}")
    return mods


def _imported_modules(py_path: Path) -> set[str]:
    """Resolve a file's imports; its package is its directory relative to ``src/``."""
    parts = py_path.relative_to(SRC_ROOT).with_suffix("").parts
    package_parts = list(parts[:-1])  # drop the module name (or "__init__")
    return _imports_in_source(py_path.read_text(encoding="utf-8"), package_parts)


def _layer_files(layer: str) -> list[Path]:
    return sorted((SRC_ROOT / "setlistkit" / layer).rglob("*.py"))


def _reaches_model(source: str) -> bool:
    """Would this source, sitting in setlistkit.catalog, reach the model layer?"""
    mods = _imports_in_source(source, ["setlistkit", "catalog"])
    return any(m == "setlistkit.model" or m.startswith("setlistkit.model.") for m in mods)


@pytest.mark.parametrize("source", [
    "from ..model import predict",       # relative, module path
    "from .. import model",              # relative, sibling as a name
    "from setlistkit import model",      # absolute, sibling as a name
    "import setlistkit.model",           # absolute module path
    "from setlistkit.model import x",    # absolute, from a module path
])
def test_resolver_catches_every_sibling_import_form(source):
    assert _reaches_model(source) is True


@pytest.mark.parametrize("source", [
    "from . import songnorm",            # same-layer name
    "from ..diagnostics import render",  # a non-layer sibling module
    "import os",
    "from setlistkit.catalog import vocabulary",
])
def test_resolver_allows_legitimate_imports(source):
    assert _reaches_model(source) is False


@pytest.mark.parametrize("layer,forbidden", sorted(FORBIDDEN.items()))
def test_layer_does_not_import_forbidden_siblings(layer, forbidden):
    violations = []
    for py_path in _layer_files(layer):
        for mod in _imported_modules(py_path):
            for sibling in forbidden:
                target = f"setlistkit.{sibling}"
                if mod == target or mod.startswith(target + "."):
                    violations.append(f"{py_path.name} imports {mod}")
    assert not violations, (
        f"{layer} must not import {sorted(forbidden)}: " + "; ".join(violations)
    )

"""Wiring guard: every repo-local name imported by pages/app.py must exist.

Catches ImportError-class page crashes without a live Streamlit session by
statically collecting imports (ast) and resolving them (importlib) — page
modules themselves are never executed.
"""

import ast
import importlib
from pathlib import Path

import pytest

LOCAL_TOP_LEVELS = {
    "analytics",
    "churn",
    "components",
    "deep_learning",
    "explainability",
    "feature_engineering",
    "retention_engine",
    "segmentation",
    "services",
    "utils",
    "visualization",
}

TARGETS = [Path("app.py"), *sorted(Path("pages").glob("*.py"))]


def _collect_imports(path: Path) -> list[tuple[str, list[str]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in LOCAL_TOP_LEVELS:
                found.append((node.module, [a.asname or a.name for a in node.names]))
    return found


@pytest.mark.parametrize("path", TARGETS, ids=[p.name for p in TARGETS])
def test_page_imports_resolve(path: Path) -> None:
    """Every from-import in the page resolves to a real module attribute."""
    assert path.exists()
    for module_name, names in _collect_imports(path):
        module = importlib.import_module(module_name)
        for name in names:
            if hasattr(module, name):
                continue
            # Mirror from-import semantics: fall back to submodule resolution.
            try:
                importlib.import_module(f"{module_name}.{name}")
            except ImportError:
                raise AssertionError(f"{path.name}: {module_name} has no {name!r}") from None

"""Element-key guard: no view repeats one key expression.

Duplicate widget keys crash the page at runtime (StreamlitDuplicateElementKey).
Two textually identical ``key=`` expressions in one view file resolve to the
same key — the RFM matrix/relationships clash is what motivated this.
"""

from __future__ import annotations

import ast
from pathlib import Path

PAGES = sorted(Path("pages").glob("*.py"))
TARGETS = [Path("app.py"), *PAGES]


def _key_sources(path: Path) -> list[str]:
    """Raw source text of every key= expression in the file.

    Identical source text in one view means identical keys at runtime (same
    variables, same scope shape) — the RFM matrix/relationships clash was
    exactly two copies of ``key=f"{key}_matrix"``. Distinct templates such as
    ``f"f_{level}"`` vs ``f"f_{key}"`` draw from different loop variables and
    are left alone.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    expressions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "key" or keyword.value is None:
                continue
            segment = ast.get_source_segment(source, keyword.value)
            if segment:
                expressions.append(" ".join(segment.split()))
    return expressions


def test_element_keys_unique_per_view() -> None:
    """Duplicate key expressions within one view file fail loudly here first."""
    offenders = []
    for path in TARGETS:
        seen: dict[str, int] = {}
        for expression in _key_sources(path):
            seen[expression] = seen.get(expression, 0) + 1
        dupes = sorted(expression for expression, count in seen.items() if count > 1)
        if dupes:
            offenders.append(f"{path}: {dupes}")
    assert offenders == []

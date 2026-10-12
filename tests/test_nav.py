"""Nav chrome guard: every view renders the shared tab bar; sidebar stays dead."""

import ast
from pathlib import Path

from components.nav_tabs import NAV_PAGES

PAGES = sorted(Path("pages").glob("*.py"))
TARGETS = [Path("app.py"), *PAGES]


def _has_nav_call(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.Call)
        and (
            getattr(node.func, "attr", "") == "render_nav_tabs"
            or getattr(node.func, "id", "") == "render_nav_tabs"
        )
        for node in ast.walk(tree)
    )


def test_nav_order_matches_pages_directory() -> None:
    """Tab order follows the numeric page scheme, landing first."""
    page_paths = [p for _, p in NAV_PAGES[1:]]
    assert page_paths == [f"pages/{p.name}" for p in PAGES]
    assert NAV_PAGES[0] == ("Home", "app.py")


def test_every_view_renders_nav_tabs() -> None:
    """app.py + all 9 pages call render_nav_tabs (single nav source)."""
    missing = [str(p) for p in TARGETS if not _has_nav_call(p)]
    assert missing == []


def test_no_sidebar_usage() -> None:
    """Sidebar is hidden by CSS; no view may render into it."""
    offenders = [
        str(p) for p in TARGETS if "st.sidebar" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []

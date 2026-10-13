"""Nav guards: the sidebar is the single navigation home for every view."""

import ast
from pathlib import Path

from components.nav_tabs import NAV_PAGES

PAGES = sorted(Path("pages").glob("*.py"))
TARGETS = [Path("app.py"), *PAGES]

_NAV_CALLS = ("render_nav_tabs", "render_sidebar")


def _nav_call_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            getattr(node.func, "attr", "") in _NAV_CALLS
            or getattr(node.func, "id", "") in _NAV_CALLS
        )
    )


def test_nav_order_matches_pages_directory() -> None:
    """Sidebar order follows the numeric page scheme, landing first."""
    page_paths = [p for _, p in NAV_PAGES[1:]]
    assert page_paths == [f"pages/{p.name}" for p in PAGES]
    assert NAV_PAGES[0] == ("Home", "app.py")


def test_nav_targets_exist() -> None:
    """Every nav entry points at a file that exists."""
    missing = [path for _, path in NAV_PAGES if not Path(path).exists()]
    assert missing == []


def test_every_view_renders_sidebar_nav() -> None:
    """app.py + all 9 pages reach the sidebar nav (directly or via the shim)."""
    missing = [str(p) for p in TARGETS if _nav_call_count(p) == 0]
    assert missing == []


def test_nav_rendered_exactly_once() -> None:
    """One nav call per view; duplicates would repeat sidebar blocks."""
    doubled = [f"{p}:{_nav_call_count(p)}" for p in TARGETS if _nav_call_count(p) != 1]
    assert doubled == []


def test_sidebar_hosts_the_navigation() -> None:
    """The sidebar module owns nav rendering against NAV_PAGES."""
    source = Path("components/sidebar.py").read_text(encoding="utf-8")
    assert "st.sidebar" in source
    assert "NAV_PAGES" in source
    assert "page_link" in source


def test_top_tab_grid_retired() -> None:
    """The button-grid tab bar is gone: no column grid, no switch_page nav."""
    source = Path("components/nav_tabs.py").read_text(encoding="utf-8")
    assert "st.columns" not in source
    assert "st.button" not in source
    assert "switch_page" not in source

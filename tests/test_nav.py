"""Nav guards: the sidebar is the single navigation home for every view."""

import ast
from pathlib import Path

from components.nav_tabs import NAV_PAGES

PAGES = sorted(Path("pages").glob("*.py"))
TARGETS = [Path("app.py"), *PAGES]

_NAV_CALL = "render_sidebar_nav"
_FOOTER_CALL = "render_sidebar_footer"

#: Views whose slicers live in the rail; every other view keeps its
#: compute-driving configuration in-page. Changing this set is a design
#: decision, not a refactor accident — update deliberately.
SIDEBAR_FILTER_VIEWS = {
    "pages/04_Demographic_Profiling.py",
    "pages/06_Segmentation_Comparison.py",
    "pages/08_Retention_Strategies.py",
    "pages/09_Executive_Dashboard.py",
}


def _call_count(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "attr", "") == name or getattr(node.func, "id", "") == name)
    )


def test_nav_order_matches_pages_directory() -> None:
    """The rail leads with the dashboard; every view is listed exactly once."""
    assert NAV_PAGES[0] == ("Dashboard", "pages/09_Executive_Dashboard.py")
    assert sorted(p for _, p in NAV_PAGES) == sorted(
        ["app.py", *[f"pages/{p.name}" for p in PAGES]]
    )
    assert len({p for _, p in NAV_PAGES}) == len(NAV_PAGES)


def test_nav_targets_exist() -> None:
    """Every nav entry points at a file that exists."""
    missing = [path for _, path in NAV_PAGES if not Path(path).exists()]
    assert missing == []


def test_every_view_renders_sidebar_nav() -> None:
    """app.py + all 9 pages open the rail with render_sidebar_nav."""
    missing = [str(p) for p in TARGETS if _call_count(p, _NAV_CALL) == 0]
    assert missing == []


def test_nav_rendered_exactly_once() -> None:
    """One rail head per view; duplicates would repeat nav blocks."""
    doubled = [
        f"{p}:{_call_count(p, _NAV_CALL)}" for p in TARGETS if _call_count(p, _NAV_CALL) != 1
    ]
    assert doubled == []


def test_every_view_renders_sidebar_footer() -> None:
    """app.py + all 9 pages close the rail with render_sidebar_footer."""
    missing = [str(p) for p in TARGETS if _call_count(p, _FOOTER_CALL) == 0]
    assert missing == []
    doubled = [
        f"{p}:{_call_count(p, _FOOTER_CALL)}" for p in TARGETS if _call_count(p, _FOOTER_CALL) != 1
    ]
    assert doubled == []


def test_slicers_live_in_the_rail_where_declared() -> None:
    """Only the declared views put slicers in the sidebar, via sidebar_filters."""
    for path in TARGETS:
        uses = "sidebar_filters" in path.read_text(encoding="utf-8")
        expected = str(path).replace("\\", "/") in SIDEBAR_FILTER_VIEWS
        assert uses == expected, f"{path}: sidebar_filters mismatch"


def test_sidebar_hosts_the_navigation() -> None:
    """The sidebar module owns nav rendering against NAV_PAGES."""
    source = Path("components/sidebar.py").read_text(encoding="utf-8")
    assert "st.sidebar" in source
    assert "NAV_PAGES" in source
    assert "page_link" in source


def test_pages_never_touch_the_sidebar_directly() -> None:
    """Views compose the rail through sidebar helpers only (nav/filters/footer)."""
    offenders = [str(p) for p in TARGETS if "st.sidebar" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_top_tab_grid_retired() -> None:
    """The button-grid tab bar is gone: no column grid, no switch_page nav."""
    source = Path("components/nav_tabs.py").read_text(encoding="utf-8")
    assert "st.columns" not in source
    assert "st.button" not in source
    assert "switch_page" not in source

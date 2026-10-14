"""Theme tests: the light design system, the one chart styler, and the dark skin."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import pytest

from visualization import theme, tokens
from visualization.theme import (
    CANVAS,
    CARD,
    CHART_HEIGHT,
    DARK_CSS,
    GLASS_BG,
    GLASS_BLUR,
    GLASS_BORDER,
    GLASS_RADIUS,
    GLASS_SHADOW,
    LIGHT_CSS,
    MUTED,
    PRIMARY,
    TERMINAL_CSS,
    apply_chart_theme,
    build_css,
    inject_terminal_css,
    style_plotly_fig,
    styled,
)

HEX = re.compile(r"#[0-9A-Fa-f]{6}")


@pytest.fixture(autouse=True)
def _restore_mode():
    """Chart theming is module-level state; never leak a mode between tests."""
    original = tokens.active_mode()
    yield
    tokens.set_active_mode(original)


def _sample() -> px.Figure:
    return px.bar(pd.DataFrame({"a": ["x", "y"], "b": [1, 2]}), x="a", y="b")


# --------------------------------------------------------------------------- #
# Chart theming.
# --------------------------------------------------------------------------- #


def test_apply_chart_theme_uses_light_surface() -> None:
    """Default charts render on the light surface with muted Inter type."""
    fig = apply_chart_theme(_sample())
    layout = fig.layout
    assert layout.template is not None
    assert layout.paper_bgcolor == tokens.LIGHT["surface"]
    assert layout.plot_bgcolor == tokens.LIGHT["surface"]
    assert layout.font.color == tokens.LIGHT["muted"]
    assert layout.height == CHART_HEIGHT
    assert tokens.DEFAULT_MODE == tokens.MODE_LIGHT


def test_apply_chart_theme_anchors_title_top_left() -> None:
    """A factory title is re-anchored to the top-left, matching the page hierarchy."""
    fig = apply_chart_theme(
        px.bar(pd.DataFrame({"a": ["x"], "b": [1]}), x="a", y="b", title="Revenue")
    )
    title = fig.layout.title
    assert title.text == "Revenue"
    assert title.x == 0.01 and title.xanchor == "left"


def test_apply_chart_theme_accepts_explicit_title_override() -> None:
    """An explicit title argument wins over whatever the factory set."""
    fig = apply_chart_theme(_sample(), title="Override")
    assert fig.layout.title.text == "Override"


def test_apply_chart_theme_leaves_single_series_legend_alone() -> None:
    """One trace keeps Plotly's no-legend default instead of a one-item legend."""
    fig = apply_chart_theme(_sample())
    assert not fig.layout.showlegend
    assert apply_chart_theme(_sample(), show_legend=True).layout.showlegend is True


def test_apply_chart_theme_positions_multi_series_legend() -> None:
    """Multi-series figures get a horizontal legend above the plot area."""
    frame = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": [1, 2, 3, 4], "s": ["p", "q", "p", "q"]})
    fig = apply_chart_theme(px.bar(frame, x="a", y="b", color="s"))
    assert fig.layout.legend.orientation == "h"
    assert fig.layout.legend.y == 1.02 and fig.layout.legend.x == 1


def test_apply_chart_theme_remaps_literals_onto_the_dark_palette() -> None:
    """Dark mode re-maps the factories' light literals onto the dark counterparts.

    Factories are authored once against the default palette; switching mode must
    not require editing (or even knowing about) them. The figure below carries
    an explicit palette literal, so this exercises the re-map rather than the
    separate default-adoption path.
    """
    import plotly.graph_objects as go

    literal = go.Figure(go.Bar(x=["x"], y=[1], marker_color=tokens.LIGHT["primary"]))
    light_fig = apply_chart_theme(literal)
    assert light_fig.data[0].marker.color.upper() == tokens.LIGHT["primary"].upper()

    tokens.set_active_mode(tokens.MODE_DARK)
    dark_fig = apply_chart_theme(literal)
    assert dark_fig.data[0].marker.color.upper() == tokens.DARK["primary"].upper()
    assert dark_fig.layout.paper_bgcolor == tokens.DARK["surface"]


def test_style_plotly_fig_is_the_legacy_entry_point() -> None:
    """The old helper still styles a figure and still honours an explicit height."""
    fig = style_plotly_fig(_sample())
    assert fig.layout.paper_bgcolor == tokens.LIGHT["surface"]
    assert style_plotly_fig(_sample(), height=200).layout.height == 200


def test_styled_decorator_preserves_behavior() -> None:
    """The decorator styles output without changing the wrapped function."""

    @styled
    def make():
        return _sample()

    assert make.__name__ == "make"
    assert make().layout.paper_bgcolor == tokens.LIGHT["surface"]

    @styled(height=200)
    def small():
        return _sample()

    assert small().layout.height == 200


def test_normalize_to_mode_is_reversible() -> None:
    """The colour re-map round-trips, so no mode can corrupt the other's hues."""
    fig = _sample()
    tokens.set_active_mode(tokens.MODE_DARK)
    dark = theme.normalize_to_mode(fig, tokens.MODE_DARK)
    back = theme.normalize_to_mode(dark, tokens.MODE_LIGHT)
    assert back.data[0].marker.color.upper() == fig.data[0].marker.color.upper()


def test_empty_figure_is_a_titled_placeholder() -> None:
    """Empty inputs keep the card footprint with a titled message, not a blank."""
    fig = theme.empty_figure("Revenue trend")
    assert fig.layout.title.text == "Revenue trend"
    assert fig.layout.height == theme.CHART_HEIGHT
    assert fig.layout.annotations[0].text == "No data for the current selection."
    assert theme.empty_figure("T", "Custom.").layout.annotations[0].text == "Custom."


def test_compact_preset_is_dense_but_complete() -> None:
    """Compact charts render at 300px with tighter margins and smaller titles."""
    fig = theme.apply_chart_theme(_sample(), title="Titled", compact=True)
    assert fig.layout.height == tokens.COMPACT_HEIGHT == 300
    assert fig.layout.margin.t < 64 and fig.layout.margin.b < 48
    assert fig.layout.title.font.size == 13

    explicit = theme.apply_chart_theme(_sample(), height=500, compact=True)
    assert explicit.layout.height == 500

    regular = theme.apply_chart_theme(_sample(), title="Titled")
    assert regular.layout.height == theme.CHART_HEIGHT
    assert regular.layout.title.font.size == 15


def test_without_title_keeps_chrome_reclaims_margin() -> None:
    """Section-framed charts drop the in-figure title but keep everything else."""
    fig = theme.apply_chart_theme(_sample(), title="Titled")
    assert fig.layout.title.text == "Titled"
    bare = theme.without_title(fig)
    assert not bare.layout.title.text
    assert bare.layout.height == theme.CHART_HEIGHT
    assert bare.layout.margin.t == 40
    assert len(bare.data) == len(fig.data)

    compact_bare = theme.without_title(
        theme.apply_chart_theme(_sample(), compact=True), compact=True
    )
    assert compact_bare.layout.height == tokens.COMPACT_HEIGHT
    assert compact_bare.layout.margin.t == 32


def test_adopt_colorway_removes_leaked_defaults() -> None:
    """A factory that passes no colour still renders in the design system.

    Bare ``px`` figures carry whatever the process-global default template
    assigned — Plotly's sequence before Streamlit is imported, Streamlit's
    ``#00000N`` placeholders after — so this asserts the post-condition rather
    than assuming which leak is present.
    """
    leaked = _sample().data[0].marker.color
    assert leaked.upper() not in {c.upper() for c in tokens.colorway(tokens.MODE_LIGHT)}

    light_marker = apply_chart_theme(_sample()).data[0].marker.color
    assert light_marker.upper() == tokens.LIGHT["primary"].upper()
    tokens.set_active_mode(tokens.MODE_DARK)
    dark_marker = apply_chart_theme(_sample()).data[0].marker.color
    assert dark_marker.upper() == tokens.DARK["primary"].upper()


# --------------------------------------------------------------------------- #
# Stylesheets.
# --------------------------------------------------------------------------- #


def test_light_stylesheet_is_the_shipped_default() -> None:
    """The light sheet defines the surface, caps tables, and stays light."""
    assert "color-scheme: light" in LIGHT_CSS
    assert tokens.LIGHT["surface"] in LIGHT_CSS
    assert tokens.LIGHT["background"] in LIGHT_CSS
    assert ".block-container" in LIGHT_CSS
    assert ".dashboard-card" in LIGHT_CSS
    assert "max-height: 420px" in LIGHT_CSS
    assert "tabular-nums" in LIGHT_CSS
    assert '[data-testid="stPlotlyChart"]' in LIGHT_CSS


def test_dark_stylesheet_retains_the_glass_console() -> None:
    """The selectable dark skin keeps its frosted cards and glow canvas."""
    assert "color-scheme: dark" in DARK_CSS
    for token in (GLASS_BG, GLASS_BORDER, GLASS_RADIUS, GLASS_SHADOW):
        assert token in DARK_CSS
    assert f"backdrop-filter: {GLASS_BLUR}" in DARK_CSS
    assert GLASS_BLUR == "blur(12px)"
    assert tokens.DARK["background"] in DARK_CSS


def test_sheets_carry_no_colour_outside_tokens() -> None:
    """Every hex in either stylesheet is a palette value (no drifting literals)."""
    palette_values = {v.upper() for v in (*tokens.LIGHT.values(), *tokens.DARK.values())}
    for mode in tokens.MODES:
        sheet = build_css(mode)
        for found in HEX.findall(sheet):
            assert found.upper() in palette_values, f"{mode} CSS carries stray colour {found}"


def test_sidebar_visible_and_auto_nav_hidden() -> None:
    """The rail is server-suppressed for auto-nav and never CSS-hidden itself."""
    import re
    import tomllib

    with open(".streamlit/config.toml", "rb") as handle:
        client = tomllib.load(handle)["client"]
    assert client["showSidebarNavigation"] is False

    for sheet in (LIGHT_CSS, DARK_CSS):
        assert (
            re.search(r'\[data-testid="stSidebar"\][^{]*\{[^}]*display:\s*none', sheet) is None
        ), "stylesheet hides the sidebar"


def test_sidebar_width_follows_token() -> None:
    """The rail width ships in both sheets from the single token value."""
    for sheet in (LIGHT_CSS, DARK_CSS):
        assert tokens.SIDEBAR_WIDTH in sheet
        assert 'body [data-testid="stSidebar"]' in sheet
    assert tokens.SIDEBAR_WIDTH == "15rem"


def test_legacy_names_remain_importable() -> None:
    """Migration shims keep the previous theme's public names resolvable."""
    assert TERMINAL_CSS == DARK_CSS
    assert (CANVAS, CARD, PRIMARY, MUTED) == (
        tokens.LIGHT["background"],
        tokens.LIGHT["surface"],
        tokens.LIGHT["primary"],
        tokens.LIGHT["muted"],
    )
    assert callable(inject_terminal_css)
    assert theme.PRIMARY_SCALE == tokens.sequential_scale(tokens.MODE_LIGHT)
    assert theme.PRIMARY_DISCRETE == tokens.discrete_colors(tokens.MODE_LIGHT)


def test_css_is_rebuilt_from_tokens() -> None:
    """The cache-clear helper rebuilds identical sheets from the token tables."""
    before = build_css(tokens.MODE_LIGHT)
    theme.clear_theme_cache()
    assert build_css(tokens.MODE_LIGHT) == before
    assert theme.TERMINAL_CSS == theme.DARK_CSS


# --------------------------------------------------------------------------- #
# Retired palette.
# --------------------------------------------------------------------------- #


def test_no_retired_volt_colour_hides_in_any_form() -> None:
    """The volt palette is gone from factories, pages, latent viz, and config.

    Includes the rgba() form, which is how the last trace of it survived a
    hex-only scan.
    """
    offenders: list[str] = []
    targets = [
        *Path("visualization").glob("*.py"),
        *Path("pages").glob("*.py"),
        Path("deep_learning/latent_space_visualization.py"),
        Path(".streamlit/config.toml"),
    ]
    for path in targets:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "#CCFF00" in line or "#9DB800" in line or "VOLT" in line or "204,255,0" in line:
                offenders.append(f"{path}:{i}")
    assert offenders == []


def test_chart_layer_keeps_colours_in_the_token_table() -> None:
    """No figure factory hard-codes a hex; every colour traces back to tokens."""
    offenders: list[str] = []
    exempt = {"tokens.py", "theme.py"}
    for path in Path("visualization").glob("*.py"):
        if path.name in exempt:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for found in HEX.findall(line):
                offenders.append(f"{path}:{i}: {found}")
    assert offenders == [], f"hard-coded colours in chart factories: {offenders}"

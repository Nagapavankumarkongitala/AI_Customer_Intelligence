"""Streamlit + Plotly theming: one CSS loader and one figure styler.

``load_theme(mode)`` injects the stylesheet for the active mode and hands the
mode to :mod:`visualization.tokens`. ``apply_chart_theme(fig)`` is the single
function every figure factory routes through (via the ``@styled`` decorator), so
chart chrome — title placement, axes, grid, hover, legend, colourway — is defined
once instead of per chart.

Both are mode-aware. Figure factories hold default-mode (light) literals in their
authoring code; in dark mode ``apply_chart_theme`` re-maps every inactive-palette
colour onto its active counterpart, so a factory never needs to know which mode
is live and never imports Streamlit (spec §17).

Legacy names from the terminal/glassmorphism theme are re-exported below so
existing imports keep working during the migration to the component library.
"""

from __future__ import annotations

import functools
import json
from string import Template
from typing import Any, Final

from visualization import tokens

# --------------------------------------------------------------------------- #
# Legacy constants (default-mode literals, kept importable).
# Prefer tokens.palette() / tokens.segment_color() in new code.
# --------------------------------------------------------------------------- #

CANVAS = tokens.LIGHT["background"]
CARD = tokens.LIGHT["surface"]
CARD_BORDER = tokens.LIGHT["border"]
PRIMARY = tokens.LIGHT["primary"]
PRIMARY_DIM = tokens.LIGHT["secondary_dim"]
PRIMARY_PALE = tokens.LIGHT["primary_100"]
TEXT = tokens.LIGHT["text"]
MUTED = tokens.LIGHT["muted"]
RISK_RED = tokens.LIGHT["danger"]
RISK_GREEN = tokens.LIGHT["positive"]
RISK_AMBER = tokens.LIGHT["warning"]

PRIMARY_SCALE = tokens.sequential_scale(tokens.MODE_LIGHT)
PRIMARY_DISCRETE = tokens.discrete_colors(tokens.MODE_LIGHT)

# Frosted-glass tokens: the dark console's card recipe, retained for the DARK skin.
GLASS_BG = "rgba(13,25,48,0.55)"
GLASS_BORDER = "rgba(255,255,255,0.12)"
GLASS_BLUR = "blur(12px)"
GLASS_RADIUS = tokens.RADIUS["lg"]
GLASS_SHADOW = "0 8px 32px rgba(0,0,0,0.45), 0 0 24px rgba(34,211,238,0.07)"

CHART_HEIGHT = tokens.CHART_HEIGHT

# --------------------------------------------------------------------------- #
# Stylesheets. Tokenised with string.Template so CSS and tokens cannot drift:
# every colour below resolves to a key in visualization.tokens.
# --------------------------------------------------------------------------- #

_BASE_CSS = Template("""
<style>
:root {
    color-scheme: $scheme;
    --ci-bg: $background;
    --ci-surface: $surface;
    --ci-surface-alt: $surface_alt;
    --ci-border: $border;
    --ci-grid: $grid;
    --ci-text: $text;
    --ci-muted: $muted;
    --ci-muted-soft: $muted_soft;
    --ci-primary: $primary;
    --ci-primary-soft: $primary_100;
    --ci-positive: $positive;
    --ci-warning: $warning;
    --ci-danger: $danger;
    --ci-critical: $critical;
    --ci-purple: $purple;
    --ci-radius: $radius;
    --ci-shadow-card: $shadow_card;
    --ci-shadow-raised: $shadow_raised;
    --ci-shadow-hover: $shadow_hover;
}
html, body { touch-action: manipulation; }
[data-testid="stAppViewContainer"] { background: $background; }
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: $content_width;
}
h1, h2, h3 { color: $text; letter-spacing: -0.02em; }
a { color: $primary; }
/* Streamlit's auto-generated page list stays off: the sidebar renders NAV_PAGES. */
[data-testid="stSidebarNav"] {
    display: none;
}
/* Surface cards */
.dashboard-card {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: $radius;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: $shadow_card;
}
[data-testid="stMetric"] {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: $radius;
    padding: 1rem 1.25rem;
    box-shadow: $shadow_card;
}
[data-testid="stMetricValue"] {
    color: $text;
    font-variant-numeric: tabular-nums;
}
[data-testid="stMetricLabel"] { color: $muted; }
/* Native selects need explicit colours on Windows */
[data-testid="stSidebar"] select, div[data-baseweb="select"] > div {
    background-color: $surface;
    color: $text;
}
[data-testid="stDataFrame"] {
    max-height: 420px;
    overflow-y: auto;
    font-variant-numeric: tabular-nums;
    border: 1px solid $border;
    border-radius: 14px;
}
div[data-testid="stPlotlyChart"] {
    background: $surface;
    border: 1px solid $border;
    border-radius: $radius;
    padding: 8px;
    box-shadow: $shadow_card;
}
/* Insight boxes */
.insight-box {
    border-radius: 12px;
    border-left: 4px solid currentColor;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
}
.insight-box strong { display: block; margin-bottom: 0.2rem; }
.insight-box p { margin: 0; font-size: 0.9rem; }
/* Status badges */
.status-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}
.page-badge {
    background: $primary_100;
    color: $primary_700;
    border-radius: 999px;
    padding: 8px 14px;
    text-align: center;
    font-size: 12px;
    font-weight: 700;
}
@media (prefers-reduced-motion: reduce) {
    * { animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; }
}
</style>
""")

_DARK_CSS = Template("""
<style>
/* Dark console: frosted cards over a layered volt-glow canvas. */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1200px 600px at 15% -5%, rgba(34,211,238,0.07), transparent),
        radial-gradient(1000px 700px at 90% 110%, rgba(61,220,132,0.05), transparent),
        $background;
}
.dashboard-card, [data-testid="stMetric"] {
    background-color: $glass_bg;
    backdrop-filter: $glass_blur;
    -webkit-backdrop-filter: $glass_blur;
    border: 1px solid $glass_border;
    box-shadow: $glass_shadow;
}
div[data-testid="stPlotlyChart"] { background: transparent; box-shadow: none; }
[data-testid="stSidebar"] { background-color: $background; }
[data-testid="stSidebar"] select, div[data-baseweb="select"] > div {
    background-color: $surface;
    color: $text;
}
</style>
""")


def build_css(mode: str | None = None) -> str:
    """Stylesheet for ``mode``, with every colour resolved from tokens."""
    resolved = mode if mode in tokens.MODES else tokens.DEFAULT_MODE
    base = palette_css_values(resolved)
    css = _BASE_CSS.substitute(**base)
    if resolved == tokens.MODE_DARK:
        css += _DARK_CSS.substitute(
            **base,
            glass_bg=GLASS_BG,
            glass_border=GLASS_BORDER,
            glass_blur=GLASS_BLUR,
            glass_shadow=GLASS_SHADOW,
        )
    return css


def palette_css_values(mode: str) -> dict[str, str]:
    """Token values plus the derived values the template needs."""
    values: dict[str, Any] = dict(tokens.palette(mode))
    values.update(
        {
            "scheme": mode,
            "radius": tokens.RADIUS["lg"],
            "content_width": tokens.CONTENT_MAX_WIDTH,
        }
    )
    return values


LIGHT_CSS = build_css(tokens.MODE_LIGHT)
DARK_CSS = build_css(tokens.MODE_DARK)
#: Legacy alias: the stylesheet callers used to inject.
TERMINAL_CSS = DARK_CSS


def clear_theme_cache() -> None:
    """Drop the built stylesheets (test helper; rebuilt lazily on next call)."""
    global LIGHT_CSS, DARK_CSS, TERMINAL_CSS
    LIGHT_CSS = build_css(tokens.MODE_LIGHT)
    DARK_CSS = build_css(tokens.MODE_DARK)
    TERMINAL_CSS = DARK_CSS


# --------------------------------------------------------------------------- #
# Entry points.
# --------------------------------------------------------------------------- #


def load_theme(mode: str | None = None) -> str:
    """Activate ``mode`` and inject its stylesheet. Returns the active mode."""
    import streamlit as st

    resolved = tokens.set_active_mode(mode)
    st.markdown(build_css(resolved), unsafe_allow_html=True)
    return resolved


def inject_terminal_css(mode: str | None = None) -> str:
    """Legacy alias for :func:`load_theme`."""
    return load_theme(mode)


#: Streamlit swaps Plotly's default template for its own on import, and that
#: template's colourway is ten placeholder hues the frontend substitutes at
#: render time. Figures handed to ``st.plotly_chart`` must not carry them —
#: otherwise a second palette authority paints our data. They are adopted onto
#: the active colourway like any other foreign default.
_STREAMLIT_PLACEHOLDERS: Final = (
    "#000001",
    "#000002",
    "#000003",
    "#000004",
    "#000005",
    "#000006",
    "#000007",
    "#000008",
    "#000009",
    "#000010",
)

#: Plotly's own built-in qualitative sequence, carried by bare ``px`` figures
#: created before Streamlit is imported (importing Streamlit swaps the default
#: template, so test processes see one sequence and production the other).
_PLOTLY_DEFAULT_SEQUENCE: Final = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
)


def _resolve_mode(mode: str | None) -> str:
    return mode if mode in tokens.MODES else tokens.active_mode()


def _apply_color_table(fig: Any, table: dict[str, str]) -> Any:
    import plotly.graph_objects as go

    if not table:
        return fig
    payload = json.loads(fig.to_json())
    return go.Figure(_remap_tree(payload, table))


def _remap_table(mode: str) -> dict[str, str]:
    """Inactive-palette colour -> active-palette colour, keyed by colour value.

    Safe because both palettes carry unique values under an identical key set.
    """
    active = tokens.palette(mode)
    inactive = tokens.PALETTES[tokens.MODE_DARK if mode == tokens.MODE_LIGHT else tokens.MODE_LIGHT]
    return {
        inactive[key].upper(): active[key]
        for key in tokens.COLOR_KEYS
        if inactive[key] != active[key]
    }


def _remap_tree(node: Any, table: dict[str, str]) -> Any:
    if isinstance(node, str):
        return table.get(node.upper(), node)
    if isinstance(node, (list, tuple)):
        return [_remap_tree(item, table) for item in node]
    if isinstance(node, dict):
        return {key: _remap_tree(value, table) for key, value in node.items()}
    return node


def normalize_to_mode(fig: Any, mode: str | None = None) -> Any:
    """Re-map colours authored against the other palette onto the active one."""
    return _apply_color_table(fig, _remap_table(_resolve_mode(mode)))


def adopt_colorway(fig: Any, mode: str | None = None) -> Any:
    """Adopt foreign default hues onto the active colourway.

    A factory that passes no colour argument still renders in the design system
    instead of leaking a third palette into the dashboard — whether the figure
    was built under Plotly's defaults (``#636EFA``, …) or under Streamlit's
    import-time template (``#000001``, … placeholders).
    """
    resolved = _resolve_mode(mode)
    way = tokens.colorway(resolved)
    table = {}
    for sequence in (_PLOTLY_DEFAULT_SEQUENCE, _STREAMLIT_PLACEHOLDERS):
        for i, default in enumerate(sequence):
            table[default] = way[i % len(way)]
    return _apply_color_table(fig, table)


def apply_chart_theme(
    fig: Any,
    title: str | None = None,
    height: int = CHART_HEIGHT,
    show_legend: bool | None = None,
) -> Any:
    """Apply the shared chart theme. Every figure factory ends here.

    ``show_legend`` is left alone when ``None`` so single-series charts keep
    Plotly's "no legend needed" behaviour instead of rendering a one-item legend.
    A figure title is re-anchored top-left, matching the page header hierarchy.
    """
    resolved = tokens.active_mode()
    ink = tokens.palette(resolved)
    dark = resolved == tokens.MODE_DARK

    # Adopt order matters: bare-px defaults become active-colourway values first,
    # then (off the default mode only) the inactive palette is re-mapped.
    fig = adopt_colorway(fig, resolved)
    if resolved != tokens.DEFAULT_MODE:
        fig = normalize_to_mode(fig, resolved)

    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=ink["surface"],
        plot_bgcolor=ink["surface"],
        height=height,
        colorway=tokens.colorway(resolved),
        font={"family": tokens.FONT_STACK, "size": 12, "color": ink["muted"]},
        margin={"l": 52, "r": 28, "t": 64, "b": 48},
        hoverlabel={
            "bgcolor": ink["text"] if not dark else ink["surface_alt"],
            "font_color": ink["surface"] if not dark else ink["text"],
            "font_size": 12,
            "bordercolor": ink["border"],
        },
    )

    caption = title if title is not None else getattr(fig.layout.title, "text", None)
    if caption:
        fig.update_layout(
            title={
                "text": caption,
                "x": 0.01,
                "xanchor": "left",
                "y": 0.98,
                "yanchor": "top",
                "font": {"size": 15, "color": ink["text"]},
            }
        )

    if show_legend is not None:
        fig.update_layout(showlegend=show_legend)
    if len(fig.data) > 1:
        fig.update_layout(
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
                "title": None,
                "font": {"size": 11, "color": ink["muted"]},
                "bgcolor": "rgba(0,0,0,0)",
            }
        )

    fig.update_xaxes(
        showgrid=False,
        linecolor=ink["border"],
        tickfont={"color": ink["muted"]},
        title_font={"color": ink["muted"]},
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=ink["grid"],
        zeroline=False,
        linecolor=ink["border"],
        tickfont={"color": ink["muted"]},
        title_font={"color": ink["muted"]},
    )

    # Separate adjacent markers with a hairline in the surface tone.
    separator = ink["background"] if dark else ink["surface"]
    for trace in fig.data:
        marker = getattr(trace, "marker", None)
        if marker is None:
            continue
        try:
            marker.update(line_color=separator, line_width=1.2)
        except (ValueError, AttributeError):
            continue  # trace types without marker lines (heatmaps, Sankeys, ...)

    return fig


def style_plotly_fig(fig: Any, height: int = CHART_HEIGHT) -> Any:
    """Legacy alias for :func:`apply_chart_theme`."""
    return apply_chart_theme(fig, height=height)


def styled(fn: Any = None, *, height: int = CHART_HEIGHT) -> Any:
    """Decorator routing a Plotly factory's figure through :func:`apply_chart_theme`."""

    def wrap(factory: Any) -> Any:
        @functools.wraps(factory)
        def inner(*args: Any, **kwargs: Any) -> Any:
            return apply_chart_theme(factory(*args, **kwargs), height=height)

        return inner

    return wrap(fn) if fn is not None else wrap

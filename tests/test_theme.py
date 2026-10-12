"""Terminal theme tests: styling wrapper, decorator, glass tokens, contrast."""

import pandas as pd
import plotly.express as px

from visualization.theme import (
    CARD,
    GLASS_BG,
    GLASS_BORDER,
    GLASS_RADIUS,
    GLASS_SHADOW,
    MUTED,
    OBSIDIAN,
    TERMINAL_CSS,
    VOLT,
    inject_terminal_css,
    style_plotly_fig,
    styled,
)


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(fg: str, bg: str) -> float:
    lighter, darker = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_style_plotly_fig_dark_terminal() -> None:
    """Styled figures are transparent dark-mode with muted Inter type."""
    fig = style_plotly_fig(px.bar(pd.DataFrame({"a": ["x"], "b": [1]}), x="a", y="b"))
    layout = fig.layout
    assert layout.template is not None
    assert layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert layout.font.color == MUTED
    assert layout.height == 380


def test_styled_decorator_preserves_behavior() -> None:
    """The decorator styles output without changing the wrapped function."""

    @styled
    def make():
        return px.bar(pd.DataFrame({"a": ["x"], "b": [1]}), x="a", y="b")

    assert make.__name__ == "make"
    assert make().layout.paper_bgcolor == "rgba(0,0,0,0)"

    @styled(height=200)
    def small():
        return px.bar(pd.DataFrame({"a": ["x"], "b": [1]}), x="a", y="b")

    assert small().layout.height == 200


def test_css_contract() -> None:
    """CSS compresses padding, defines cards, caps tables, highlights KPIs in volt."""
    assert ".block-container" in TERMINAL_CSS
    assert ".dashboard-card" in TERMINAL_CSS and CARD in TERMINAL_CSS
    assert "max-height: 420px" in TERMINAL_CSS
    assert VOLT in TERMINAL_CSS and OBSIDIAN in TERMINAL_CSS
    assert callable(inject_terminal_css)


def test_glass_tokens_present_in_css() -> None:
    """Frosted-card tokens (translucency, blur, hairline border, glow canvas) ship."""
    for token in (GLASS_BG, GLASS_BORDER, GLASS_RADIUS, GLASS_SHADOW):
        assert token in TERMINAL_CSS
    assert "backdrop-filter: blur(14px)" in TERMINAL_CSS
    assert "color-scheme: dark" in TERMINAL_CSS
    assert "tabular-nums" in TERMINAL_CSS


def test_text_contrast_on_obsidian() -> None:
    """Volt primary ≥ 7:1, muted secondary ≥ 4.5:1 on obsidian (computed, not eyeballed)."""
    assert _contrast(VOLT, OBSIDIAN) >= 7.0
    assert _contrast(MUTED, OBSIDIAN) >= 4.5

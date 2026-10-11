"""Terminal theme tests: styling wrapper, decorator, and CSS contract."""

import pandas as pd
import plotly.express as px

from visualization.theme import (
    CARD,
    MUTED,
    OBSIDIAN,
    TERMINAL_CSS,
    VOLT,
    inject_terminal_css,
    style_plotly_fig,
    styled,
)


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

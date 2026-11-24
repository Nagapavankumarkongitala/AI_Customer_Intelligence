"""Empty states and chart sections: loading/empty discipline for every view.

Pages never render a bare warning and stop ad hoc — ``empty_state`` is the one
spelling for "nothing to show yet". ``chart_section`` frames a figure with its
title and caption; the card chrome comes from the shared stylesheet, so sections
stay frameless here by design.
"""

from __future__ import annotations

import html
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import streamlit as st

from visualization import tokens

#: Heading of the ``chart_section`` currently rendering, used by the PNG export
#: helper so downloaded charts carry the same title the operator sees on screen.
chart_title: ContextVar[str | None] = ContextVar("chart_title", default=None)


def empty_state_html(title: str, body: str) -> str:
    """Centered empty-state markup: glyph, title, and next-step body."""
    return (
        "<div style='text-align:center;padding:2rem 1rem;"
        f"border:1px dashed {tokens.color('border')};"
        f"border-radius:{tokens.RADIUS['lg']};"
        f"background:{tokens.color('surface_alt')};'>"
        f"<div style='font-size:1.6rem;color:{tokens.color('muted_soft')};'>◌</div>"
        f"<div style='font-weight:700;margin:0.35rem 0 0.2rem 0;color:{tokens.color('text')};'>"
        f"{html.escape(title)}</div>"
        f"<div style='color:{tokens.color('muted')};font-size:0.88rem;'>"
        f"{html.escape(body)}</div></div>"
    )


def empty_state(title: str, body: str) -> None:
    """Render an empty state."""
    st.markdown(empty_state_html(title, body), unsafe_allow_html=True)


def chart_section_html(title: str, caption: str | None = None) -> str:
    """Section title plus an optional muted caption line."""
    out = (
        "<div style='font-weight:700;font-size:0.95rem;margin:0.75rem 0 0.1rem 0;"
        f"color:{tokens.color('text')};'>"
        f"{html.escape(title)}</div>"
    )
    if caption:
        out += (
            f"<div style='color:{tokens.color('muted')};font-size:0.8rem;margin-bottom:0.4rem;'>"
            f"{html.escape(caption)}</div>"
        )
    return out


@contextmanager
def chart_section(title: str, caption: str | None = None) -> Iterator[None]:
    """Frame a figure (or a pair) with its title and caption, then yield.

    The heading is also published through :data:`chart_title` so a
    :func:`components.chart_export.plotly_chart` call inside the block can title
    its PNG export with the exact text shown above the chart.
    """
    token = chart_title.set(title)
    st.markdown(chart_section_html(title, caption), unsafe_allow_html=True)
    try:
        yield
    finally:
        chart_title.reset(token)

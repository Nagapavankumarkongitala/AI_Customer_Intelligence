"""Insight boxes: the narrative layer between charts and conclusions.

Every box states one observation plus its business consequence in plain words —
never a bare number, never a causal claim about a model's impact. Tones come
from ``tokens.tone_badge``; the border takes the tone's ink so the box reads as
one state.
"""

from __future__ import annotations

import html

import streamlit as st

from visualization import tokens

TONES = ("info", "positive", "warning", "critical")


def insight_box_html(title: str, body: str, tone: str = "info") -> str:
    """Insight markup: bolded observation line over a muted consequence line."""
    resolved = tone if tone in TONES else "info"
    bg, fg = tokens.tone_badge(resolved, tokens.active_mode())
    return (
        "<div class='insight-box' "
        f"style='background:{bg};color:{fg};border-left-color:{fg};'>"
        f"<strong>{html.escape(title)}</strong>"
        f"<p style='color:{fg};'>{html.escape(body)}</p></div>"
    )


def insight_box(title: str, body: str, tone: str = "info") -> None:
    """Render an insight box."""
    st.markdown(insight_box_html(title, body, tone), unsafe_allow_html=True)

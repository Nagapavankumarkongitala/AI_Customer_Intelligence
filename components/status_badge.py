"""Status badges: one pill per state, coloured from tokens.

``risk_badge`` resolves churn bands through ``tokens.risk_badge`` so band hues
cannot drift between badges, tables, and charts. ``status_badge`` covers every
other pill (tones follow ``tokens.tone_badge``). Builders are pure; ``render_*``
wrappers touch Streamlit.
"""

from __future__ import annotations

import html

import streamlit as st

from visualization import tokens

TONES = ("info", "positive", "warning", "critical")


def status_badge_html(text: str, tone: str = "info") -> str:
    """Pill markup for a tone."""
    resolved = tone if tone in TONES else "info"
    bg, fg = tokens.tone_badge(resolved, tokens.active_mode())
    return (
        "<span class='status-badge' "
        f"style='background:{bg};color:{fg};'>{html.escape(text)}</span>"
    )


def status_badge(text: str, tone: str = "info") -> None:
    """Render a tone pill."""
    st.markdown(status_badge_html(text, tone), unsafe_allow_html=True)


def risk_badge_html(band: str) -> str:
    """Pill markup for a churn risk band (low/medium/high/critical)."""
    bg, fg = tokens.risk_badge(band, tokens.active_mode())
    label = str(band).strip().lower()
    return (
        "<span class='status-badge' "
        f"style='background:{bg};color:{fg};'>{html.escape(label)}</span>"
    )


def risk_badge(band: str) -> None:
    """Render a risk-band pill."""
    st.markdown(risk_badge_html(band), unsafe_allow_html=True)

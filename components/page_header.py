"""Page and section headers: one hierarchy for every view.

``page_header`` opens a page (eyebrow badge, title, subtitle); ``section_header``
opens a section inside it. HTML builders stay pure so tests can assert the
hierarchy without a Streamlit session.
"""

from __future__ import annotations

import html

import streamlit as st

from visualization import tokens


def page_header_html(title: str, subtitle: str | None = None, badge: str | None = None) -> str:
    """Header markup: badge eyebrow, title, and muted subtitle."""
    parts = []
    if badge:
        bg, fg = tokens.tone_badge("info", tokens.active_mode())
        parts.append(
            "<div style='display:inline-block;border-radius:999px;"
            f"background:{bg};color:{fg};"
            "font-size:0.72rem;font-weight:700;letter-spacing:0.08em;"
            f"text-transform:uppercase;padding:4px 12px;margin-bottom:0.5rem;'>"
            f"{html.escape(badge)}</div>"
        )
    parts.append(
        "<h1 style='margin:0 0 0.25rem 0;letter-spacing:-0.02em;'>" f"{html.escape(title)}</h1>"
    )
    if subtitle:
        parts.append(
            f"<p style='color:{tokens.color('muted')};font-size:0.95rem;margin:0 0 0.25rem 0;'>"
            f"{html.escape(subtitle)}</p>"
        )
    return "\n".join(parts)


def page_header(title: str, subtitle: str | None = None, badge: str | None = None) -> None:
    """Render the page header followed by a divider."""
    st.markdown(page_header_html(title, subtitle, badge), unsafe_allow_html=True)
    st.divider()


def section_header_html(title: str, subtitle: str | None = None) -> str:
    """Section markup: compact title with an optional muted subtitle."""
    out = (
        "<h2 style='margin:1.25rem 0 0.15rem 0;font-size:1.15rem;letter-spacing:-0.01em;'>"
        f"{html.escape(title)}</h2>"
    )
    if subtitle:
        out += (
            f"<p style='color:{tokens.color('muted')};font-size:0.85rem;margin:0 0 0.5rem 0;'>"
            f"{html.escape(subtitle)}</p>"
        )
    return out


def section_header(title: str, subtitle: str | None = None) -> None:
    """Render a section header."""
    st.markdown(section_header_html(title, subtitle), unsafe_allow_html=True)

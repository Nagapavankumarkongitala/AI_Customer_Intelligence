"""Explainer cards for AI-generated segment groups.

The cards answer "what is Premium Active A, and what should we do with it?"
without sending the reader back to a profile table. They sit under the
traditional → AI customer flow on page 06 and are reused on the dashboard's
Customer Trends tab, so the same group name carries the same reading in both
places.

Copy and profile numbers come from ``services.segment_explain_service``; this
module only renders. Colours come from ``visualization.tokens`` so an AI group
keeps its violet identity and the card follows the active light/dark mode.
"""

from __future__ import annotations

import html

import streamlit as st

from services.segment_explain_service import SegmentExplanation
from utils import formatting
from visualization import tokens

#: Cards per row: three keeps the behaviour footer readable without crowding.
DEFAULT_COLUMNS = 3


def _stat(label: str, value: str) -> str:
    """One compact stat block inside a card."""
    value_color = tokens.color("text")
    return (
        "<div>"
        f"<div style='font-size:0.66rem;text-transform:uppercase;letter-spacing:0.03em;"
        f"color:{tokens.color('muted')};'>{html.escape(label)}</div>"
        f"<div style='font-size:1.02rem;font-weight:700;font-variant-numeric:tabular-nums;"
        f"color:{value_color};'>{html.escape(value)}</div>"
        "</div>"
    )


def _behaviour(segment: SegmentExplanation) -> str:
    """Median recency/frequency/spend line, skipping columns that are absent."""
    parts: list[str] = []
    if segment.median_recency_days is not None:
        parts.append(
            f"{formatting.fmt_int(round(segment.median_recency_days))} days since last order"
        )
    if segment.median_frequency is not None:
        parts.append(f"{formatting.fmt_float(segment.median_frequency, 1)} orders")
    if segment.median_monetary_value is not None:
        parts.append(f"{formatting.fmt_money_compact(segment.median_monetary_value)} spend")
    return " · ".join(parts)


def segment_card_html(segment: SegmentExplanation) -> str:
    """Card markup: group name, tier reading, profile, and next action."""
    accent = tokens.segment_color(segment.name)
    text = tokens.color("text")
    muted = tokens.color("muted")
    border = tokens.color("border")
    surface = tokens.color("surface")
    behaviour = _behaviour(segment)
    behaviour_html = (
        f"<div style='font-size:0.72rem;color:{muted};border-top:1px solid {border};"
        f"padding-top:0.4rem;overflow-wrap:break-word;'>Typical customer: "
        f"{html.escape(behaviour)}</div>"
        if behaviour
        else ""
    )
    return (
        "<div style='"
        f"background:{surface};border:1px solid {border};border-top:3px solid {accent};"
        f"border-radius:{tokens.RADIUS['lg']};box-shadow:{tokens.shadow('card')};"
        "padding:0.85rem 0.95rem;display:flex;flex-direction:column;gap:0.35rem;height:100%;'>"
        "<div style='display:flex;justify-content:space-between;align-items:baseline;gap:0.5rem;'>"
        f"<div style='font-weight:700;font-size:1rem;color:{text};overflow-wrap:break-word;'>"
        f"{html.escape(segment.name)}</div>"
        f"<div style='font-size:0.62rem;font-weight:600;letter-spacing:0.04em;"
        f"text-transform:uppercase;color:{muted};white-space:nowrap;'>AI group</div>"
        "</div>"
        f"<div style='font-size:0.72rem;font-weight:600;color:{accent};"
        f"overflow-wrap:break-word;'>{html.escape(segment.tier_label)}</div>"
        f"<div style='font-size:0.80rem;color:{muted};overflow-wrap:break-word;'>"
        f"{html.escape(segment.summary)}</div>"
        f"<div style='font-size:0.78rem;color:{muted};overflow-wrap:break-word;'>"
        f"<strong style='color:{text};'>Action</strong> {html.escape(segment.action)}</div>"
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;"
        "border-top:1px solid "
        f"{border};padding-top:0.45rem;'>"
        f"{_stat('Customers', f'{formatting.fmt_int(segment.customers)} ({formatting.fmt_pct100(segment.customer_pct)})')}"
        f"{_stat('Revenue', f'{formatting.fmt_money_compact(segment.revenue)} ({formatting.fmt_pct100(segment.revenue_pct)})')}"
        "</div>"
        f"{behaviour_html}"
        "</div>"
    )


def render_segment_cards(
    segments: list[SegmentExplanation],
    width: int = DEFAULT_COLUMNS,
) -> None:
    """Render AI segment explainers as rows of cards (Streamlit sink)."""
    for start in range(0, len(segments), width):
        row = segments[start : start + width]
        columns = st.columns(len(row))
        for column, segment in zip(columns, row, strict=True):
            with column:
                st.markdown(segment_card_html(segment), unsafe_allow_html=True)


__all__ = ["DEFAULT_COLUMNS", "render_segment_cards", "segment_card_html"]

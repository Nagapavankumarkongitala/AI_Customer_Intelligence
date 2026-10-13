"""KPI cards: labelled values with delta pills, in rows of at most five.

Values arrive preformatted (``utils.formatting``); cards never format numbers
themselves. ``metric_row`` chunks wide inputs into compliant rows. The legacy
``kpi_row`` in ``components.metric_cards`` is untouched — pages migrate to this
module one view at a time.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st

from visualization import tokens

#: Hard cap: no more than five KPI cards share a row (spec §12).
MAX_CARDS_PER_ROW = 5

#: Delta semantics; the caller decides the business meaning (churn up is bad,
#: revenue up is good) and passes the matching tone.
TONES = ("positive", "negative", "neutral", "info")

_TONE_COLOR = {
    "positive": "positive",
    "negative": "danger",
    "neutral": "muted",
    "info": "primary",
}


@dataclass(frozen=True)
class Metric:
    """One KPI card: label, preformatted value, optional delta and tone."""

    label: str
    value: str
    delta: str | None = None
    tone: str = "neutral"
    help: str | None = None


def metric_card_html(metric: Metric) -> str:
    """Card markup: label, tabular value, and an optional delta pill."""
    tone = metric.tone if metric.tone in TONES else "neutral"
    accent = tokens.color(_TONE_COLOR[tone])
    title_attr = f" title='{html.escape(metric.help)}'" if metric.help else ""
    delta_html = ""
    if metric.delta:
        arrow = {"positive": "▲", "negative": "▼"}.get(tone, "●")
        delta_html = (
            "<span style='display:inline-block;border-radius:999px;"
            f"background:{tokens.color('surface_alt')};color:{accent};"
            "font-size:0.72rem;font-weight:700;padding:2px 10px;margin-top:0.45rem;'>"
            f"{arrow} {html.escape(metric.delta)}</span>"
        )
    return (
        "<div style='"
        f"background:{tokens.color('surface')};"
        f"border:1px solid {tokens.color('border')};"
        f"border-top:3px solid {accent};"
        f"border-radius:{tokens.RADIUS['lg']};"
        f"box-shadow:{tokens.shadow('card')};"
        "padding:0.85rem 0.95rem;min-height:6.5rem;'>"
        f"<div style='color:{tokens.color('muted')};font-size:0.70rem;font-weight:600;"
        f"letter-spacing:0.03em;text-transform:uppercase;overflow-wrap:break-word;'{title_attr}>"
        f"{html.escape(metric.label)}</div>"
        "<div style='font-size:1.35rem;font-weight:700;font-variant-numeric:tabular-nums;"
        f"line-height:1.25;margin-top:0.15rem;overflow-wrap:break-word;color:{tokens.color('text')};'>"
        f"{html.escape(metric.value)}</div>"
        f"{delta_html}</div>"
    )


def metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    tone: str = "neutral",
    help: str | None = None,
) -> None:
    """Render one KPI card."""
    st.markdown(metric_card_html(Metric(label, value, delta, tone, help)), unsafe_allow_html=True)


def chunk_metrics(metrics: list[Metric], width: int = MAX_CARDS_PER_ROW) -> list[list[Metric]]:
    """Split cards into compliant rows (pure; tested without Streamlit)."""
    if width < 1:
        raise ValueError(f"row width must be >= 1, got {width}.")
    return [metrics[i : i + width] for i in range(0, len(metrics), width)]


def metric_row(metrics: list[Metric], width: int = MAX_CARDS_PER_ROW) -> None:
    """Render cards in rows of at most ``width`` cards."""
    for row in chunk_metrics(metrics, width):
        columns = st.columns(len(row))
        for column, metric in zip(columns, row, strict=True):
            with column:
                st.markdown(metric_card_html(metric), unsafe_allow_html=True)

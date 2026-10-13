"""Plotly figure factories for RFM/LRFM pages. Pure functions (no Streamlit calls)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from visualization import tokens
from visualization.theme import PRIMARY_DISCRETE, PRIMARY_SCALE, empty_figure, styled


def _segment_map_or_sequence(values: object) -> dict:
    """Keyword args painting a segment column with stable hues (pure helper)."""
    mapped = tokens.segment_map(values)
    if mapped:
        return {"color_discrete_map": mapped}
    return {"color_discrete_sequence": PRIMARY_DISCRETE}


@styled
def segment_distribution_bar(profile: pd.DataFrame) -> px.Figure:
    """Customer counts per segment, largest first, in segment hues."""
    if profile.empty:
        return empty_figure("Customers per segment")
    return px.bar(
        profile,
        x="segment",
        y="customers",
        color="segment",
        **_segment_map_or_sequence(profile["segment"]),
        title="Customers per segment",
        labels={"segment": "Segment", "customers": "Customers"},
    )


@styled
def revenue_by_segment_bar(profile: pd.DataFrame) -> px.Figure:
    """Revenue contribution per segment, in segment hues."""
    if profile.empty:
        return empty_figure("Revenue per segment")
    return px.bar(
        profile,
        x="segment",
        y="revenue",
        color="segment",
        **_segment_map_or_sequence(profile["segment"]),
        title="Revenue per segment",
        labels={"segment": "Segment", "revenue": "Revenue"},
    )


@styled
def recency_frequency_scatter(features: pd.DataFrame, color_col: str = "segment") -> px.Figure:
    """Recency vs frequency, colored by segment when available."""
    if features.empty:
        return empty_figure("Recency vs frequency")
    data = features.copy()
    color = color_col if color_col in data.columns else None
    color_kwargs = (
        _segment_map_or_sequence(data[color])
        if color
        else {"color_discrete_sequence": PRIMARY_DISCRETE}
    )
    return px.scatter(
        data,
        x="recency_days",
        y="frequency",
        color=color,
        **color_kwargs,
        hover_data=["customer_id", "monetary_value"],
        title="Recency vs frequency",
        labels={"recency_days": "Recency (days)", "frequency": "Frequency"},
    )


@styled
def frequency_monetary_scatter(features: pd.DataFrame, color_col: str = "segment") -> px.Figure:
    """Frequency vs monetary value, colored by segment when available."""
    if features.empty:
        return empty_figure("Frequency vs monetary value")
    data = features.copy()
    color = color_col if color_col in data.columns else None
    color_kwargs = (
        _segment_map_or_sequence(data[color])
        if color
        else {"color_discrete_sequence": PRIMARY_DISCRETE}
    )
    return px.scatter(
        data,
        x="frequency",
        y="monetary_value",
        color=color,
        **color_kwargs,
        hover_data=["customer_id", "recency_days"],
        title="Frequency vs monetary value",
        labels={"frequency": "Frequency", "monetary_value": "Monetary value"},
    )


@styled
def rfm_heatmap(scored: pd.DataFrame) -> px.Figure:
    """Customer counts across R (rows) × F (columns)."""
    if scored.empty:
        return empty_figure("Customer counts by R × F score")
    matrix = pd.crosstab(scored["R"], scored["F"])
    return px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale=PRIMARY_SCALE,
        title="Customer counts by R × F score",
        labels={"x": "F score", "y": "R score", "color": "Customers"},
    )

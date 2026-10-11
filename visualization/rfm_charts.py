"""Plotly figure factories for RFM/LRFM pages. Pure functions (no Streamlit calls)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

COLOR_SCALE = "Blues"


def segment_distribution_bar(profile: pd.DataFrame) -> px.Figure:
    """Customer counts per segment, largest first."""
    return px.bar(
        profile,
        x="segment",
        y="customers",
        color="customers",
        color_continuous_scale=COLOR_SCALE,
        title="Customers per segment",
        labels={"segment": "Segment", "customers": "Customers"},
    )


def revenue_by_segment_bar(profile: pd.DataFrame) -> px.Figure:
    """Revenue contribution per segment."""
    return px.bar(
        profile,
        x="segment",
        y="revenue",
        color="revenue",
        color_continuous_scale="Greens",
        title="Revenue per segment",
        labels={"segment": "Segment", "revenue": "Revenue"},
    )


def recency_frequency_scatter(features: pd.DataFrame, color_col: str = "segment") -> px.Figure:
    """Recency vs frequency, colored by segment when available."""
    data = features.copy()
    color = color_col if color_col in data.columns else None
    return px.scatter(
        data,
        x="recency_days",
        y="frequency",
        color=color,
        hover_data=["customer_id", "monetary_value"],
        title="Recency vs frequency",
        labels={"recency_days": "Recency (days)", "frequency": "Frequency"},
    )


def frequency_monetary_scatter(features: pd.DataFrame, color_col: str = "segment") -> px.Figure:
    """Frequency vs monetary value, colored by segment when available."""
    data = features.copy()
    color = color_col if color_col in data.columns else None
    return px.scatter(
        data,
        x="frequency",
        y="monetary_value",
        color=color,
        hover_data=["customer_id", "recency_days"],
        title="Frequency vs monetary value",
        labels={"frequency": "Frequency", "monetary_value": "Monetary value"},
    )


def rfm_heatmap(scored: pd.DataFrame) -> px.Figure:
    """Customer counts across R (rows) × F (columns)."""
    matrix = pd.crosstab(scored["R"], scored["F"])
    return px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale=COLOR_SCALE,
        title="Customer counts by R × F score",
        labels={"x": "F score", "y": "R score", "color": "Customers"},
    )

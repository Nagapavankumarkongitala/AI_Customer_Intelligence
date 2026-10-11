"""Plotly figure factories for the executive dashboard. No Streamlit calls."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def revenue_trend_fig(table: pd.DataFrame) -> px.Figure:
    """Monthly revenue line."""
    return px.line(
        table,
        x="month",
        y="revenue",
        markers=True,
        title="Revenue trend",
        labels={"month": "Month", "revenue": "Revenue"},
    )


def activity_trend_fig(table: pd.DataFrame) -> px.Figure:
    """Monthly active customers line."""
    return px.line(
        table,
        x="month",
        y="active_customers",
        markers=True,
        title="Customer activity trend",
        labels={"month": "Month", "active_customers": "Active customers"},
        color_discrete_sequence=["#1F6FEB"],
    )


def new_returning_fig(table: pd.DataFrame) -> px.Figure:
    """Monthly new vs returning stacked bars."""
    long = table.melt(
        id_vars="month",
        value_vars=["new_customers", "returning_customers"],
        var_name="cohort",
        value_name="customers",
    )
    return px.bar(
        long,
        x="month",
        y="customers",
        color="cohort",
        title="New vs returning customers",
        color_discrete_map={"new_customers": "green", "returning_customers": "#1F6FEB"},
    )


def portfolio_bar_fig(portfolio: pd.DataFrame, segment_col: str = "segment") -> px.Figure:
    """Customer vs revenue share per segment (grouped bars)."""
    col = segment_col if segment_col in portfolio.columns else portfolio.columns[0]
    long = portfolio.melt(
        id_vars=col,
        value_vars=["customer_share", "revenue_share"],
        var_name="share",
        value_name="pct",
    )
    return px.bar(
        long,
        x=col,
        y="pct",
        color="share",
        barmode="group",
        title="Customer vs revenue share by segment",
    )

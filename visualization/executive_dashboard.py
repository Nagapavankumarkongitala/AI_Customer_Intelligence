"""Plotly figure factories for the executive dashboard. No Streamlit calls."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from visualization.theme import PRIMARY, PRIMARY_DISCRETE, RISK_GREEN, empty_figure, styled


@styled
def revenue_trend_fig(table: pd.DataFrame) -> px.Figure:
    """Monthly revenue line."""
    if table.empty:
        return empty_figure("Revenue trend")
    return px.line(
        table,
        x="month",
        y="revenue",
        markers=True,
        color_discrete_sequence=[PRIMARY],
        title="Revenue trend",
        labels={"month": "Month", "revenue": "Revenue"},
    )


@styled
def activity_trend_fig(table: pd.DataFrame) -> px.Figure:
    """Monthly active customers line."""
    if table.empty:
        return empty_figure("Customer activity trend")
    return px.line(
        table,
        x="month",
        y="active_customers",
        markers=True,
        title="Customer activity trend",
        labels={"month": "Month", "active_customers": "Active customers"},
        color_discrete_sequence=[PRIMARY],
    )


@styled
def new_returning_fig(table: pd.DataFrame) -> px.Figure:
    """Monthly new vs returning stacked bars."""
    if table.empty:
        return empty_figure("New vs returning customers")
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
        color_discrete_map={"new_customers": RISK_GREEN, "returning_customers": PRIMARY},
    )


@styled
def portfolio_bar_fig(portfolio: pd.DataFrame, segment_col: str = "segment") -> px.Figure:
    """Customer vs revenue share per segment (grouped bars)."""
    if portfolio.empty:
        return empty_figure("Customer vs revenue share by segment")
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
        color_discrete_sequence=PRIMARY_DISCRETE,
        title="Customer vs revenue share by segment",
    )

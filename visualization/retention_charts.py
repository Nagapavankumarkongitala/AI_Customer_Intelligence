"""Plotly figure factories for retention pages. No Streamlit calls."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def value_risk_matrix(table: pd.DataFrame) -> px.Figure:
    """Historical value vs churn probability, sized by revenue at risk, colored by action."""
    hover = [
        c for c in ("customer_id", "customer_value_tier", "risk_band") if c in table.columns
    ]
    return px.scatter(
        table,
        x="churn_probability",
        y="expected_value_proxy_historical",
        size="revenue_at_risk",
        color="recommended_action",
        hover_data=hover or None,
        title="Value–risk matrix (size = revenue at risk)",
        labels={
            "churn_probability": "Churn probability",
            "expected_value_proxy_historical": "Historical value (proxy, not CLV)",
        },
    )


def revenue_at_risk_bar(table: pd.DataFrame) -> px.Figure:
    """Revenue at risk per recommended action."""
    grouped = table.groupby("recommended_action")["revenue_at_risk"].sum().reset_index()
    return px.bar(
        grouped,
        x="recommended_action",
        y="revenue_at_risk",
        color="recommended_action",
        title="Revenue at risk per action",
    )


def strategy_distribution_bar(table: pd.DataFrame) -> px.Figure:
    """Customer counts per recommended action."""
    counts = table["recommended_action"].value_counts().reset_index()
    counts.columns = ["action", "customers"]
    return px.bar(
        counts,
        x="action",
        y="customers",
        color="action",
        title="Customers per recommended action",
    )


def priority_histogram(table: pd.DataFrame) -> px.Figure:
    """Distribution of retention priority scores."""
    return px.histogram(
        table["retention_priority_score"],
        nbins=20,
        title="Priority score distribution",
        labels={"value": "Priority score", "count": "Customers"},
        color_discrete_sequence=["#1F6FEB"],
    )

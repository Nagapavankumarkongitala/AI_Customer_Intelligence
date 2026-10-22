"""Plotly figure factories for the executive dashboard. No Streamlit calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization import tokens
from visualization.theme import (
    MUTED,
    PRIMARY,
    PRIMARY_DISCRETE,
    RISK_GREEN,
    empty_figure,
    styled,
)


@styled
def revenue_area_fig(table: pd.DataFrame) -> px.Figure:
    """Monthly revenue area (filled)."""
    if table.empty:
        return empty_figure("Revenue trend")
    return px.area(
        table,
        x="month",
        y="revenue",
        markers=True,
        color_discrete_sequence=[PRIMARY],
        title="Revenue trend",
        labels={"month": "Month", "revenue": "Revenue"},
    )


@styled
def revenue_trend_fig(
    table: pd.DataFrame,
    secondary: str | None = None,
    forecast_periods: int = 0,
) -> go.Figure:
    """Monthly revenue line, with optional context and a projected tail.

    A single revenue line says nothing about *why* it moved. ``secondary`` adds a
    second, right-hand axis for a companion series — ``"active_customers"`` (is
    the change volume?) or ``"churn_rate"`` (is it risk?) — so value and its driver
    read together. ``forecast_periods`` extends a dashed projection past the last
    month using a least-squares line over the observed months, so the trajectory
    is visible without implying a model. Shade quarters with
    :func:`shade_quarters` when a caller wants seasonal bands.
    """
    if table.empty or "month" not in table.columns:
        return empty_figure("Revenue trend")
    fig = make_subplots(specs=[[{"secondary_y": bool(secondary)}]])
    fig.add_trace(
        go.Scatter(
            x=table["month"],
            y=table["revenue"],
            mode="lines+markers",
            name="Revenue",
            line={"color": PRIMARY},
            fill="tozeroy",
            fillcolor=_rgba(PRIMARY, 0.15),
        ),
        secondary_y=False,
    )
    if forecast_periods > 0:
        projection = _linear_forecast(table["month"], table["revenue"], forecast_periods)
        if not projection.empty:
            fig.add_trace(
                go.Scatter(
                    x=projection["month"],
                    y=projection["revenue"],
                    mode="lines+markers",
                    name="Projected",
                    line={"color": MUTED, "dash": "dash"},
                ),
                secondary_y=False,
            )
    if secondary and secondary in table.columns:
        fig.add_trace(
            go.Scatter(
                x=table["month"],
                y=table[secondary],
                mode="lines+markers",
                name=secondary.replace("_", " ").title(),
                line={"color": tokens.color("secondary"), "width": 2},
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text=secondary.replace("_", " ").title(), secondary_y=True)
    fig.update_layout(title="Revenue trend")
    fig.update_yaxes(title_text="Revenue", secondary_y=False)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    """Translucent form of a palette hex (non-hex passthrough)."""
    value = str(hex_color).lstrip("#")
    if len(value) != 6:
        return str(hex_color)
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _linear_forecast(months: pd.Series, values: pd.Series, periods: int) -> pd.DataFrame:
    """Least-squares projection of ``periods`` months past the last observed month.

    A transparent straight-line extension, deliberately the simplest defensible
    projection: it shows direction, not a claim about future revenue. Floors at
    zero so a downward slope never draws negative revenue. Empty when there are
    fewer than two points or the next period label cannot be derived.
    """
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    labels = pd.Series(months).astype(str)
    mask = series.notna()
    series, labels = series[mask], labels[mask]
    if len(series) < 2:
        return pd.DataFrame(columns=["month", "revenue"])
    try:
        last_period = pd.Period(str(labels.iloc[-1]), freq="M")
    except (ValueError, TypeError):
        return pd.DataFrame(columns=["month", "revenue"])
    x = np.arange(len(series), dtype=float)
    slope, intercept = np.polyfit(x, series.to_numpy(dtype=float), 1)
    future_x = np.arange(len(series), len(series) + periods, dtype=float)
    projected = np.clip(intercept + slope * future_x, 0, None)
    future_months = [str(last_period + step) for step in range(1, periods + 1)]
    return pd.DataFrame({"month": future_months, "revenue": projected.round(2)})


def shade_quarters(fig: go.Figure, months: pd.Series, secondary_y: bool = False) -> go.Figure:
    """Add faint quarter bands behind a monthly series (alternating shading).

    Seasonal structure is much easier to see against a background that marks the
    quarter boundaries than against a blank grid. Bands alternate so adjacent
    quarters never merge visually; odd quarters are shaded, even ones left clear.
    """
    labels = [str(month) for month in months]
    if not labels:
        return fig
    quarters: dict[str, int] = {}
    for label in labels:
        try:
            period = pd.Period(label, freq="M")
        except (ValueError, TypeError):
            continue
        quarters[label] = period.quarter
    if not quarters:
        return fig
    band: list[str] = []
    for label in labels:
        band.append("shade" if quarters.get(label, 0) % 2 else "clear")
    start = None
    for index, _label in enumerate(labels):
        if band[index] == "shade" and start is None:
            start = index
        if (band[index] == "clear" or index == len(labels) - 1) and start is not None:
            end = index if band[index] == "clear" else index + 1
            fig.add_vrect(
                x0=labels[start],
                x1=labels[min(end, len(labels) - 1)],
                fillcolor=tokens.color("grid"),
                opacity=0.35,
                layer="below",
                line_width=0,
            )
            start = None
    return fig


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

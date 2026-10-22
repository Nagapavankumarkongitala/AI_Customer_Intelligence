"""Plotly figure factories for churn pages. No Streamlit calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, roc_curve

from visualization import tokens
from visualization.theme import (
    MUTED,
    PRIMARY,
    PRIMARY_SCALE,
    empty_figure,
    styled,
)

#: Band order for risk visuals (config.yaml churn.risk_bands keys).
RISK_BAND_ORDER = ("low", "medium", "high", "critical")


@styled
def roc_curve_fig(y_true: np.ndarray, proba: np.ndarray) -> go.Figure:
    """ROC curve with chance diagonal."""
    if len(np.asarray(y_true)) == 0:
        return empty_figure("ROC curve")
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, mode="lines", line={"color": PRIMARY}, name=f"AUC {auc(fpr, tpr):.3f}"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": MUTED}, name="chance"
        )
    )
    fig.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR")
    return fig


@styled
def calibration_curve_fig(table: pd.DataFrame) -> go.Figure:
    """Observed rate vs mean predicted per bin, sized by bin count."""
    if table.empty:
        return empty_figure("Calibration curve")
    fig = px.scatter(
        table,
        x="mean_predicted",
        y="observed_rate",
        size="count",
        color_discrete_sequence=[PRIMARY],
        title="Calibration curve",
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": MUTED}, name="ideal"
        )
    )
    return fig


@styled
def confusion_fig(confusion: dict) -> go.Figure:
    """2×2 confusion matrix heatmap."""
    if not confusion:
        return empty_figure("Confusion matrix")
    matrix = [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]]
    return px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale=PRIMARY_SCALE,
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        title="Confusion matrix",
    )


@styled
def gain_chart_fig(table: pd.DataFrame) -> go.Figure:
    """Cumulative gain by decile with random-targeting baseline."""
    if table.empty:
        return empty_figure("Cumulative gain")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=table["decile"],
            y=table["cum_gain"],
            mode="lines+markers",
            line={"color": PRIMARY},
            name="model",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table["decile"],
            y=table["decile"] / table["decile"].max(),
            mode="lines",
            line={"dash": "dash", "color": MUTED},
            name="random",
        )
    )
    fig.update_layout(
        title="Cumulative gain", xaxis_title="Decile", yaxis_title="Share of churners"
    )
    return fig


@styled
def importance_bar_fig(importance: pd.DataFrame, top_n: int = 15) -> px.Figure:
    """Top permutation importances."""
    if importance.empty:
        return empty_figure("Global feature importance")
    return px.bar(
        importance.head(top_n),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=PRIMARY_SCALE,
        title="Global feature importance",
    )


@styled
def risk_band_bar(scores: pd.DataFrame, revenue_col: str | None = None) -> go.Figure:
    """Customer counts per risk band in band order (semantic risk hues).

    Values ride on the bars as ``count (share%)`` so the card reads exactly
    without a hover, and the x tick labels are dropped because the bars are
    already labelled. When ``revenue_col`` is present the money exposed in each
    band rides a second, right-hand axis as a muted line, so headcount and
    financial impact read together instead of the count alone.
    """
    if scores.empty or "risk_band" not in scores.columns:
        return empty_figure("Customers per risk band")
    counts = scores["risk_band"].value_counts().reindex(RISK_BAND_ORDER, fill_value=0)
    total = int(counts.sum())
    shares = (counts / total * 100) if total else counts.astype(float)
    has_revenue = revenue_col is not None and revenue_col in scores.columns

    if not has_revenue:
        fig = px.bar(
            x=counts.index.tolist(),
            y=counts.tolist(),
            color=counts.index.tolist(),
            color_discrete_map={band: tokens.risk_color(band) for band in RISK_BAND_ORDER},
            title="Customers per risk band",
            labels={"x": "Risk band", "y": "Customers"},
        )
        fig.update_traces(
            text=[
                f"{count:,}<br>{share:.0f}%" for count, share in zip(counts, shares, strict=True)
            ],
            textposition="outside",
            textfont_size=11,
            cliponaxis=False,
        )
        return fig

    revenue = (
        scores.groupby("risk_band")[revenue_col]
        .sum()
        .reindex(RISK_BAND_ORDER, fill_value=0)
        .astype(float)
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=counts.index.tolist(),
            y=counts.tolist(),
            name="Customers",
            marker_color=[tokens.risk_color(band) for band in RISK_BAND_ORDER],
            text=[
                f"{count:,}<br>{share:.0f}%" for count, share in zip(counts, shares, strict=True)
            ],
            textposition="outside",
            textfont_size=11,
            cliponaxis=False,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=revenue.index.tolist(),
            y=revenue.tolist(),
            name="Revenue at risk",
            mode="lines+markers",
            line={"color": MUTED, "width": 2, "dash": "dot"},
            marker={"size": 7, "color": MUTED},
            hovertemplate="%{x}<br>%{y:,.0f} at risk<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(title="Customers per risk band", xaxis_title="Risk band")
    fig.update_yaxes(title_text="Customers", secondary_y=False)
    fig.update_yaxes(title_text="Revenue at risk", secondary_y=True)
    return fig

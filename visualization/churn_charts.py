"""Plotly figure factories for churn pages. No Streamlit calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import auc, roc_curve

from visualization.theme import (
    MUTED,
    PRIMARY,
    PRIMARY_SCALE,
    RISK_AMBER,
    RISK_GREEN,
    RISK_RED,
    styled,
)


@styled
def roc_curve_fig(y_true: np.ndarray, proba: np.ndarray) -> go.Figure:
    """ROC curve with chance diagonal."""
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
def risk_band_bar(scores: pd.DataFrame) -> px.Figure:
    """Customer counts per risk band in band order (semantic risk hues)."""
    order = ["low", "medium", "high", "critical"]
    counts = scores["risk_band"].value_counts().reindex(order, fill_value=0)
    return px.bar(
        x=counts.index.tolist(),
        y=counts.tolist(),
        color=counts.index.tolist(),
        color_discrete_map={
            "low": RISK_GREEN,
            "medium": RISK_AMBER,
            "high": "#FF8C00",
            "critical": RISK_RED,
        },
        title="Customers per risk band",
        labels={"x": "Risk band", "y": "Customers"},
    )

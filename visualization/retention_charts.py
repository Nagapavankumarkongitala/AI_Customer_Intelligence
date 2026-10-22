"""Plotly figure factories for retention pages. No Streamlit calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.formatting import currency_symbol, fmt_compact
from visualization import tokens
from visualization.theme import (
    MUTED,
    PRIMARY,
    PRIMARY_DISCRETE,
    empty_figure,
    styled,
)

#: Default churn-probability cutoff for the "act first" quadrant on the
#: value–risk matrix. Matches the ``high`` risk-band floor in config.yaml.
DEFAULT_CHURN_QUADRANT = 0.60


def _action_map_or_sequence(values: object) -> dict:
    """Keyword args painting an action column with stable hues (pure helper)."""
    mapped = tokens.action_map(values)
    if mapped:
        return {"color_discrete_map": mapped}
    return {"color_discrete_sequence": PRIMARY_DISCRETE}


@styled
def value_risk_matrix(
    table: pd.DataFrame,
    churn_threshold: float = DEFAULT_CHURN_QUADRANT,
    value_threshold: float | None = None,
) -> px.Figure:
    """Historical value vs churn probability, sized by revenue at risk, colored by action.

    Two dashed reference lines frame the priority quadrant: a vertical line at
    ``churn_threshold`` (the ``high`` risk-band floor) and a horizontal line at
    ``value_threshold`` (defaults to the median historical value). Customers in
    the top-right are both high-value and high-risk — the "act first" zone.
    """
    if table.empty:
        return empty_figure("Value–risk matrix (size = revenue at risk)")
    hover = [c for c in ("customer_id", "customer_value_tier", "risk_band") if c in table.columns]
    fig = px.scatter(
        table,
        x="churn_probability",
        y="expected_value_proxy_historical",
        size="revenue_at_risk",
        color="recommended_action",
        **_action_map_or_sequence(table["recommended_action"]),
        hover_data=hover or None,
        title="Value–risk matrix (size = revenue at risk)",
        labels={
            "churn_probability": "Churn probability",
            "expected_value_proxy_historical": "Historical value (proxy, not CLV)",
        },
    )
    if value_threshold is None:
        value_threshold = float(table["expected_value_proxy_historical"].median())
    fig.add_vline(x=churn_threshold, line_dash="dash", line_color=MUTED, line_width=1, opacity=0.7)
    fig.add_hline(y=value_threshold, line_dash="dash", line_color=MUTED, line_width=1, opacity=0.7)
    fig.add_annotation(
        x=1.0,
        y=1.0,
        xref="paper",
        yref="paper",
        text="Act first",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font={"color": MUTED, "size": 11},
    )
    return fig


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Split a ``#RRGGBB`` colour into integer channels."""
    text = value.strip().lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _blend(colors: list[str], fraction: float) -> str:
    """Interpolate across a hex colour list at ``fraction`` in [0, 1]."""
    stops = len(colors) - 1
    position = max(0.0, min(1.0, fraction)) * stops
    index = min(int(position), stops - 1)
    offset = position - index
    start = _hex_to_rgb(colors[index])
    end = _hex_to_rgb(colors[index + 1])
    mixed = tuple(round(a + (b - a) * offset) for a, b in zip(start, end, strict=True))
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def _compact_money(value: float) -> str:
    """Compact money for dense cell labels (₹105M), using the configured symbol."""
    return f"{currency_symbol()}{fmt_compact(value)}"


#: Quadrant labels keyed by (high value, high risk) — the strategic reading.
_QUADRANT_LABELS: dict[tuple[bool, bool], str] = {
    (True, True): "Act first",
    (True, False): "Protect",
    (False, True): "Monitor",
    (False, False): "Maintain",
}


@styled
def value_risk_heatmap(
    table: pd.DataFrame,
    churn_threshold: float = DEFAULT_CHURN_QUADRANT,
    value_threshold: float | None = None,
) -> px.Figure:
    """Strategic 2×2 priority heatmap: customers and revenue at risk per quadrant.

    Rather than plotting one bubble per customer, customers are aggregated into
    the four quadrants bounded by the churn-probability threshold (the ``high``
    risk-band floor) and the value threshold (defaults to the median historical
    value). Each cell encodes three things at once — bubble **size** is the
    customer count, **colour intensity** is revenue at risk, and the in-cell
    label repeats both numbers — so the quadrant that is simultaneously crowded
    and expensive reads without decoding a legend. Hover carries the full
    sentence.
    """
    required = ("churn_probability", "expected_value_proxy_historical")
    if table.empty or not set(required).issubset(table.columns):
        return empty_figure("Strategic priority heatmap")
    if value_threshold is None:
        value_threshold = float(table["expected_value_proxy_historical"].median())

    risk = pd.to_numeric(table["churn_probability"], errors="coerce")
    value = pd.to_numeric(table["expected_value_proxy_historical"], errors="coerce")
    if "revenue_at_risk" in table.columns:
        at_risk = pd.to_numeric(table["revenue_at_risk"], errors="coerce").fillna(0.0)
    else:
        at_risk = pd.Series(0.0, index=table.index)
    frame = pd.DataFrame({"risk": risk, "value": value, "at_risk": at_risk}).dropna(
        subset=["risk", "value"]
    )
    if frame.empty:
        return empty_figure("Strategic priority heatmap")

    high_value = frame["value"] >= value_threshold
    high_risk = frame["risk"] >= churn_threshold
    value_floor = _compact_money(value_threshold)
    x_cats = [f"Low value<br>< {value_floor}", f"High value<br>≥ {value_floor}"]
    y_cats = [f"Low risk<br>< {churn_threshold:.2f}", f"High risk<br>≥ {churn_threshold:.2f}"]

    rows: list[dict] = []
    for risk_index, is_high_risk in enumerate((False, True)):
        for value_index, is_high_value in enumerate((False, True)):
            mask = (high_risk if is_high_risk else ~high_risk) & (
                high_value if is_high_value else ~high_value
            )
            segment = frame[mask]
            rows.append(
                {
                    "x": x_cats[value_index],
                    "y": y_cats[risk_index],
                    "count": int(len(segment)),
                    "revenue": float(segment["at_risk"].sum()),
                    "quadrant": _QUADRANT_LABELS[(is_high_value, is_high_risk)],
                }
            )
    cells = pd.DataFrame(rows)

    zmax = float(cells["revenue"].max()) or 1.0
    scale = tokens.sequential_scale()
    # Per-cell ink: the sequential scale runs light→dark, so each label picks the
    # colour that stays legible on its own cell instead of a single trace-wide hue.
    inks = [tokens.readable_ink(_blend(scale, revenue / zmax)) for revenue in cells["revenue"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cells["x"],
            y=cells["y"],
            mode="markers+text",
            marker={
                "size": cells["count"],
                "sizemode": "area",
                "sizeref": 2.0 * (float(cells["count"].max()) or 1) / (60.0**2),
                "sizemin": 18,
                "color": cells["revenue"],
                "colorscale": scale,
                "showscale": True,
                "colorbar": {"title": "Revenue<br>at risk", "thickness": 10},
                "line": {"color": tokens.color("border"), "width": 1},
            },
            text=[
                f"{row.quadrant}<br>{row.count:,}<br>{_compact_money(row.revenue)}"
                for row in cells.itertuples(index=False)
            ],
            textposition="middle center",
            textfont={"size": 10, "color": inks},
            customdata=[
                [row.quadrant, row.count, _compact_money(row.revenue)]
                for row in cells.itertuples(index=False)
            ],
            hovertemplate="%{customdata[0]}<br>%{customdata[1]:,} customers<br>"
            "%{customdata[2]} at risk<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title="Strategic priority heatmap",
        xaxis_title="Historical value (proxy, not CLV)",
        xaxis={"gridcolor": tokens.color("grid"), "zeroline": False},
        yaxis={"gridcolor": tokens.color("grid"), "zeroline": False},
    )
    return fig


@styled
def revenue_at_risk_bar(table: pd.DataFrame) -> px.Figure:
    """Revenue at risk per recommended action, largest first."""
    if table.empty:
        return empty_figure("Revenue at risk per action")
    grouped = (
        table.groupby("recommended_action")["revenue_at_risk"]
        .sum()
        .reset_index()
        .sort_values("revenue_at_risk", ascending=False)
        .reset_index(drop=True)
    )
    fig = px.bar(
        grouped,
        x="recommended_action",
        y="revenue_at_risk",
        color="recommended_action",
        **_action_map_or_sequence(grouped["recommended_action"]),
        title="Revenue at risk per action",
    )
    # The legend already names every action, so the category labels only repeat it.
    fig.update_xaxes(title_text="", showticklabels=False)
    return fig


@styled
def strategy_distribution_bar(table: pd.DataFrame) -> px.Figure:
    """Customer counts per recommended action, largest first."""
    if table.empty:
        return empty_figure("Customers per recommended action")
    counts = table["recommended_action"].value_counts().reset_index()
    counts.columns = ["action", "customers"]
    # value_counts is already descending; sort defensively so order never drifts.
    counts = counts.sort_values("customers", ascending=False).reset_index(drop=True)
    fig = px.bar(
        counts,
        x="action",
        y="customers",
        color="action",
        **_action_map_or_sequence(counts["action"]),
        title="Customers per recommended action",
    )
    # The legend already names every action, so the category labels only repeat it.
    fig.update_xaxes(title_text="", showticklabels=False)
    return fig


@styled
def priority_histogram(table: pd.DataFrame) -> go.Figure:
    """Distribution of retention priority scores with a cumulative (Pareto) share.

    Bars show customers per priority-score bin; the line on the secondary axis
    shows the cumulative share of customers from the lowest bin upward, so the
    concentrated "top tail" of opportunities reads at a glance.
    """
    if table.empty:
        return empty_figure("Priority score distribution")
    scores = pd.to_numeric(table["retention_priority_score"], errors="coerce").dropna()
    if scores.empty:
        return empty_figure("Priority score distribution")
    counts, edges = np.histogram(scores, bins=20)
    centers = ((edges[:-1] + edges[1:]) / 2).round(4)
    cumulative = (np.cumsum(counts) / max(int(counts.sum()), 1) * 100).round(1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=centers, y=counts, name="Customers", marker_color=PRIMARY),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=cumulative,
            name="Cumulative %",
            mode="lines",
            line={"color": tokens.LIGHT["secondary"], "width": 2},
        ),
        secondary_y=True,
    )
    fig.update_layout(title="Priority score distribution")
    fig.update_xaxes(title_text="Priority score")
    fig.update_yaxes(title_text="Customers", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 100])
    return fig

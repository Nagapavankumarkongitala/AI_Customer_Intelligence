"""Plotly figure factories for RFM/LRFM pages. Pure functions (no Streamlit calls)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from visualization import tokens
from visualization.theme import PRIMARY_DISCRETE, PRIMARY_SCALE, empty_figure, styled


def _segment_map_or_sequence(values: object) -> dict:
    """Keyword args painting a segment column with stable hues (pure helper)."""
    mapped = tokens.segment_map(values)
    if mapped:
        return {"color_discrete_map": mapped}
    return {"color_discrete_sequence": PRIMARY_DISCRETE}


@styled
def segment_distribution_bar(profile: pd.DataFrame, segment_col: str = "segment") -> px.Figure:
    """Customer counts per segment, largest first, in segment hues.

    Values ride on the bars and the legend names every segment, so both axes
    stay clean: no y-axis, no x tick labels. The shared theme still supplies
    fonts, grid, and margins.
    """
    if profile.empty:
        return empty_figure("Customers per segment")
    col = segment_col if segment_col in profile.columns else profile.columns[0]
    fig = px.bar(
        profile,
        x=col,
        y="customers",
        color=col,
        text="customers",
        **_segment_map_or_sequence(profile[col]),
        title="Customers per segment",
        labels={col: "Segment", "customers": "Customers"},
    )
    fig.update_traces(texttemplate="%{y:,}", textposition="outside", textfont_size=13)
    top = float(profile["customers"].max())
    fig.update_yaxes(visible=False, range=[0, top * 1.12 if top > 0 else 1])
    fig.update_xaxes(title_text="", showticklabels=False)
    fig.update_layout(bargap=0.1)
    return fig


@styled
def revenue_by_segment_bar(profile: pd.DataFrame, segment_col: str = "segment") -> px.Figure:
    """Revenue contribution per segment, in segment hues.

    Values ride on the bars and the legend names every segment, so both axes
    stay clean: no y-axis, no x tick labels. The shared theme still supplies
    fonts, grid, and margins.
    """
    if profile.empty:
        return empty_figure("Revenue per segment")
    col = segment_col if segment_col in profile.columns else profile.columns[0]
    fig = px.bar(
        profile,
        x=col,
        y="revenue",
        color=col,
        text="revenue",
        **_segment_map_or_sequence(profile[col]),
        title="Revenue per segment",
        labels={col: "Segment", "revenue": "Revenue"},
    )
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside", textfont_size=13)
    top = float(profile["revenue"].max())
    fig.update_yaxes(visible=False, range=[0, top * 1.12 if top > 0 else 1])
    fig.update_xaxes(title_text="", showticklabels=False)
    fig.update_layout(bargap=0.1)
    return fig


@styled
def segment_share_donut(profile: pd.DataFrame, segment_col: str = "segment") -> px.Figure:
    """Customer share per segment as a donut, in segment hues.

    Slices under a few percent stay thin but legible through hover values;
    the hole centers the total customer count.
    """
    if profile.empty:
        return empty_figure("Customers per segment")
    col = segment_col if segment_col in profile.columns else profile.columns[0]
    fig = px.pie(
        profile,
        names=col,
        values="customers",
        hole=0.45,
        color=col,
        **_segment_map_or_sequence(profile[col]),
        title="Customers per segment",
        labels={col: "Segment", "customers": "Customers"},
    )
    fig.update_traces(textinfo="percent")
    total = int(profile["customers"].sum())
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=f"{total:,}<br>customers",
        showarrow=False,
        font={"size": 14},
    )
    return fig


@styled(height=480)
def rfm_scatter_matrix(
    features: pd.DataFrame, color_col: str = "segment", max_points: int = 2000
) -> px.Figure:
    """Pairwise R/F/M relationships as a scatter matrix, colored by segment.

    Points are a deterministic random sample past ``max_points`` so 25k-row
    frames stay interactive; the page captions the sampling when it applies.
    """
    dims = [c for c in ("recency_days", "frequency", "monetary_value") if c in features.columns]
    if features.empty or not dims:
        return empty_figure("Pairwise RFM relationships")
    data = features.copy()
    if len(data) > max_points:
        data = data.sample(n=max_points, random_state=42).reset_index(drop=True)
    color = color_col if color_col in data.columns else None
    color_kwargs = (
        _segment_map_or_sequence(data[color])
        if color
        else {"color_discrete_sequence": PRIMARY_DISCRETE}
    )
    hover = [c for c in ("customer_id",) if c in data.columns] or None
    return px.scatter_matrix(
        data,
        dimensions=dims,
        color=color,
        **color_kwargs,
        hover_data=hover,
        opacity=0.6,
        title="Pairwise RFM relationships",
        labels={
            "recency_days": "Recency (days)",
            "frequency": "Frequency",
            "monetary_value": "Monetary value",
        },
    )


@styled
def rfm_parallel_plot(features: pd.DataFrame) -> go.Figure:
    """Recency, frequency, and monetary value as parallel coordinates.

    One quiet line per customer, colored by monetary value: value
    concentration across recency and frequency reads as color bands.
    Replaces the two overplotted 25k-point scatters with a single view.
    """
    dims = [c for c in ("recency_days", "frequency", "monetary_value") if c in features.columns]
    if features.empty or not dims:
        return empty_figure("Recency, frequency and value")
    data = features[dims].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        return empty_figure("Recency, frequency and value")
    color = data["monetary_value"] if "monetary_value" in data.columns else data[dims[-1]]
    fig = go.Figure(
        go.Parcoords(
            dimensions=[{"label": c.replace("_", " "), "values": data[c]} for c in dims],
            line={
                "color": color,
                "colorscale": PRIMARY_SCALE,
                "showscale": True,
                "colorbar": {"title": "Value"},
            },
            labelangle=0,
        )
    )
    fig.update_layout(title="Recency, frequency and value")
    return fig


@styled
def rfm_bubble(scored: pd.DataFrame) -> px.Figure:
    """Customer counts across R × F as bubbles sized and colored by count."""
    if scored.empty:
        return empty_figure("Customer counts by R × F score")
    matrix = pd.crosstab(scored["R"], scored["F"])
    long = matrix.reset_index().melt(id_vars="R", var_name="F", value_name="customers")
    long = long[long["customers"] > 0]
    if long.empty:
        return empty_figure("Customer counts by R × F score")
    fig = px.scatter(
        long,
        x="F",
        y="R",
        size="customers",
        color="customers",
        color_continuous_scale=PRIMARY_SCALE,
        size_max=36,
        title="Customer counts by R × F score",
        labels={"F": "F score", "R": "R score", "customers": "Customers"},
        hover_data={"F": True, "R": True, "customers": True},
    )
    ticks = sorted(set(long["F"].tolist()) | set(long["R"].tolist()))
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=1)
    if ticks:
        lo, hi = min(ticks) - 0.6, max(ticks) + 0.6
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(range=[lo, hi])
    return fig

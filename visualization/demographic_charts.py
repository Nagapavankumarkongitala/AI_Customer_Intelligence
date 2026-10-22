"""Plotly figure factories for demographic/geographic pages. No Streamlit calls."""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization.theme import (
    MUTED,
    PRIMARY,
    PRIMARY_DISCRETE,
    PRIMARY_SCALE,
    empty_figure,
    styled,
)

#: Age-band labels carry their bounds ("18–24", "65+"). Sorting by the numeric
#: lower bound puts bands in lifetime order regardless of how the labels are
#: punctuated (en dash, hyphen) or whether the top band is open-ended.
_AGE_BOUND = re.compile(r"(\d+)")


def _age_band_sort_key(label: object) -> tuple[int, str]:
    """Chronological sort key for an age-band label; unparseable labels last."""
    match = _AGE_BOUND.search(str(label))
    return (int(match.group(1)), str(label)) if match else (10**6, str(label))


def plural_geo_level(level: str) -> str:
    """English plural for a geographic level (city→cities, country→countries)."""
    if level.endswith("y"):
        return f"{level[:-1]}ies"
    if level.endswith("s"):
        return f"{level}es"
    return f"{level}s"


@styled
def age_histogram(ages: pd.Series) -> px.Figure:
    """Distribution of customer ages."""
    clean = pd.to_numeric(ages, errors="coerce").dropna()
    if clean.empty:
        return empty_figure("Age distribution")
    return px.histogram(
        clean,
        nbins=20,
        title="Age distribution",
        labels={"value": "Age", "count": "Customers"},
        color_discrete_sequence=[PRIMARY],
    )


@styled
def age_band_bar(age_bands: pd.DataFrame, value_col: str | None = None) -> go.Figure:
    """Customers per age band, ordered youngest to oldest.

    Bands sort by their numeric lower bound ("18–24" before "25–34") rather
    than by count, so the chart always reads left-to-right across the lifetime.
    Values ride on the bars as ``count (share%)``. When ``value_col`` is present
    (e.g. average order value) it rides a second, right-hand axis as a muted
    line, so a band's size and its spend read together.
    """
    if age_bands.empty:
        return empty_figure("Customers per age band")
    frame = age_bands.copy()
    frame["_sort"] = frame["age_band"].map(_age_band_sort_key)
    frame = frame.sort_values("_sort", kind="stable").drop(columns="_sort").reset_index(drop=True)
    total = int(frame["customers"].sum())
    shares = (frame["customers"] / total * 100) if total else frame["customers"].astype(float)
    texts = [f"{c:,}<br>{s:.0f}%" for c, s in zip(frame["customers"], shares, strict=True)]
    has_value = value_col is not None and value_col in frame.columns

    if not has_value:
        fig = px.bar(
            frame,
            x="age_band",
            y="customers",
            color="customers",
            color_continuous_scale=PRIMARY_SCALE,
            title="Customers per age band",
        )
        fig.update_traces(text=texts, textposition="outside", textfont_size=11, cliponaxis=False)
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=frame["age_band"].tolist(),
            y=frame["customers"].tolist(),
            name="Customers",
            marker={"color": frame["customers"].tolist(), "colorscale": PRIMARY_SCALE},
            text=texts,
            textposition="outside",
            textfont_size=11,
            cliponaxis=False,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["age_band"].tolist(),
            y=frame[value_col].astype(float).tolist(),
            name=value_col.replace("_", " ").title(),
            mode="lines+markers",
            line={"color": MUTED, "width": 2, "dash": "dot"},
            marker={"size": 7, "color": MUTED},
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(title="Customers per age band")
    fig.update_yaxes(title_text="Customers", secondary_y=False)
    fig.update_yaxes(title_text=value_col.replace("_", " ").title(), secondary_y=True)
    return fig


@styled
def gender_bar(gender: pd.DataFrame) -> px.Figure:
    """Customer counts by gender."""
    if gender.empty:
        return empty_figure("Customers by gender")
    return px.bar(
        gender,
        x="gender",
        y="customers",
        color="gender",
        color_discrete_sequence=PRIMARY_DISCRETE,
        title="Customers by gender",
        labels={"gender": "Gender", "customers": "Customers"},
    )


@styled
def top_geo_bar(geo: pd.DataFrame, level: str, top_n: int = 10) -> px.Figure:
    """Top values for a geographic level."""
    plural = plural_geo_level(level)
    if geo.empty:
        return empty_figure(f"Top {plural} by customers")
    return px.bar(
        geo.head(top_n),
        x=level,
        y="customers",
        color="customers",
        color_continuous_scale=PRIMARY_SCALE,
        title=f"Top {plural} by customers",
    )


@styled
def revenue_geo_bar(
    geo: pd.DataFrame,
    level: str,
    top_n: int = 10,
    horizontal: bool = False,
    secondary: str | None = None,
) -> go.Figure:
    """Top values for a geographic level by revenue, vertical or horizontal.

    Horizontal bars are always sorted largest-first, so the ranking reads without
    scanning. ``secondary`` names a companion measure (e.g. average order value)
    drawn as a muted dot on a second, right-hand axis beside each bar, so high
    revenue from a few big orders is distinguishable from high revenue from many.
    """
    plural = plural_geo_level(level)
    title = f"Top {plural} by revenue"
    if geo.empty:
        return empty_figure(title)
    if not horizontal:
        return px.bar(
            geo.head(top_n),
            x=level,
            y="revenue",
            color="revenue",
            color_continuous_scale=PRIMARY_SCALE,
            title=title,
        )
    ordered = geo.head(top_n).sort_values("revenue", ascending=True)
    has_secondary = secondary is not None and secondary in ordered.columns
    if not has_secondary:
        return px.bar(
            ordered,
            x="revenue",
            y=level,
            orientation="h",
            color="revenue",
            color_continuous_scale=PRIMARY_SCALE,
            title=title,
            labels={"revenue": "Revenue", level: level.capitalize()},
        )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=ordered["revenue"],
            y=ordered[level],
            orientation="h",
            name="Revenue",
            marker={"color": ordered["revenue"], "colorscale": PRIMARY_SCALE},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ordered[secondary],
            y=ordered[level],
            mode="markers",
            name=secondary.replace("_", " ").title(),
            marker={"size": 9, "color": MUTED, "symbol": "diamond"},
            hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(title=title, showlegend=False)
    fig.update_xaxes(title_text="")
    # Horizontal bars put their value scale on Y, so ``secondary_y`` splits the
    # y axes (yaxis = revenue, yaxis2 = the companion measure) while the
    # category axis stays shared.
    fig.update_yaxes(title_text="Revenue", secondary_y=False)
    fig.update_yaxes(title_text=secondary.replace("_", " ").title(), secondary_y=True)
    return fig


#: Values kept per level before the tail is folded into ``Other``. Near-unique
#: levels are common (a demo city column can hold one name per customer), and an
#: uncapped level emits tens of thousands of tiles that lock the browser tab.
TREEMAP_MAX_VALUES_PER_LEVEL = 12

#: Label for the folded tail of a capped level.
TREEMAP_OTHER = "Other"


def _cap_level(values: pd.Series, keep: int = TREEMAP_MAX_VALUES_PER_LEVEL) -> pd.Series:
    """Keep the ``keep`` most common values of one level; fold the tail into Other.

    Folding (rather than dropping rows) keeps every customer counted exactly
    once, so parent tiles stay true totals.
    """
    if values.nunique() <= keep:
        return values
    top = set(values.value_counts().head(keep - 1).index)
    return values.where(values.isin(top), TREEMAP_OTHER)


@styled
def geo_treemap(features: pd.DataFrame, path_cols: list[str]) -> px.Figure:
    """Nested customer share across geographic levels (outer → inner tile).

    One drill-down treemap showing how customers concentrate from
    region/country down to city. Needs at least two present levels; missing
    values are grouped as "Unknown" so sparse uploads still render. Tiles color
    by customer count.

    Each level keeps at most ``TREEMAP_MAX_VALUES_PER_LEVEL`` values and folds
    its tail into "Other", so a near-unique city column cannot emit tens of
    thousands of tiles.
    """
    cols = [c for c in path_cols if c in features.columns]
    title = "Customer hierarchy" + (f" ({' → '.join(cols)})" if cols else "")
    if len(cols) < 2 or features.empty:
        return empty_figure(title)
    data = pd.DataFrame(
        {col: _cap_level(features[col].astype("string").fillna("Unknown")) for col in cols}
    )
    grouped = data.groupby(cols, observed=True).size().reset_index(name="customers")
    if grouped.empty:
        return empty_figure(title)
    return px.treemap(
        grouped,
        path=cols,
        values="customers",
        color="customers",
        color_continuous_scale=PRIMARY_SCALE,
        title=title,
    )


@styled
def stacked_composition_bar(composition: pd.DataFrame, column: str) -> px.Figure:
    """100% stacked bars of a demographic column across segments."""
    if composition.empty:
        return empty_figure(f"{column} mix by segment")
    long = composition.reset_index().melt(id_vars="segment", var_name=column, value_name="pct")
    return px.bar(
        long,
        x="segment",
        y="pct",
        color=column,
        color_discrete_sequence=PRIMARY_DISCRETE,
        title=f"{column} mix by segment",
        labels={"pct": "Share (%)", "segment": "Segment"},
    )

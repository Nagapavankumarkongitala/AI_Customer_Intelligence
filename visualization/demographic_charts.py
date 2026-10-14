"""Plotly figure factories for demographic/geographic pages. No Streamlit calls."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from visualization.theme import PRIMARY, PRIMARY_DISCRETE, PRIMARY_SCALE, empty_figure, styled


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
def age_band_bar(age_bands: pd.DataFrame) -> px.Figure:
    """Customers per age band."""
    if age_bands.empty:
        return empty_figure("Customers per age band")
    return px.bar(
        age_bands,
        x="age_band",
        y="customers",
        color="customers",
        color_continuous_scale=PRIMARY_SCALE,
        title="Customers per age band",
    )


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
    geo: pd.DataFrame, level: str, top_n: int = 10, horizontal: bool = False
) -> px.Figure:
    """Top values for a geographic level by revenue, vertical or horizontal."""
    plural = plural_geo_level(level)
    if geo.empty:
        return empty_figure(f"Top {plural} by revenue")
    if horizontal:
        ordered = geo.head(top_n).sort_values("revenue")
        return px.bar(
            ordered,
            x="revenue",
            y=level,
            orientation="h",
            color="revenue",
            color_continuous_scale=PRIMARY_SCALE,
            title=f"Top {plural} by revenue",
            labels={"revenue": "Revenue", level: level.capitalize()},
        )
    return px.bar(
        geo.head(top_n),
        x=level,
        y="revenue",
        color="revenue",
        color_continuous_scale=PRIMARY_SCALE,
        title=f"Top {plural} by revenue",
    )


@styled
def geo_treemap(geo: pd.DataFrame, level: str) -> px.Figure:
    """Treemap of customer share for one geographic level."""
    if geo.empty:
        return empty_figure(f"Customer share by {level}")
    return px.treemap(
        geo,
        path=[level],
        values="customers",
        color="customers",
        color_continuous_scale=PRIMARY_SCALE,
        title=f"Customer share by {level}",
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

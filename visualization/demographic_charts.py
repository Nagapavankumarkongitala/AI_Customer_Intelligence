"""Plotly figure factories for demographic/geographic pages. No Streamlit calls."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from visualization.theme import VOLT, VOLT_DISCRETE, VOLT_SCALE, styled


@styled
def age_histogram(ages: pd.Series) -> px.Figure:
    """Distribution of customer ages."""
    return px.histogram(
        pd.to_numeric(ages, errors="coerce").dropna(),
        nbins=20,
        title="Age distribution",
        labels={"value": "Age", "count": "Customers"},
        color_discrete_sequence=[VOLT],
    )


@styled
def age_band_bar(age_bands: pd.DataFrame) -> px.Figure:
    """Customers per age band."""
    return px.bar(
        age_bands,
        x="age_band",
        y="customers",
        color="customers",
        color_continuous_scale=VOLT_SCALE,
        title="Customers per age band",
    )


@styled
def gender_bar(gender: pd.DataFrame) -> px.Figure:
    """Customer counts by gender."""
    return px.bar(
        gender,
        x="gender",
        y="customers",
        color="gender",
        color_discrete_sequence=VOLT_DISCRETE,
        title="Customers by gender",
        labels={"gender": "Gender", "customers": "Customers"},
    )


@styled
def top_geo_bar(geo: pd.DataFrame, level: str, top_n: int = 10) -> px.Figure:
    """Top values for a geographic level."""
    return px.bar(
        geo.head(top_n),
        x=level,
        y="customers",
        color="customers",
        color_continuous_scale=VOLT_SCALE,
        title=f"Top {level}s by customers",
    )


@styled
def revenue_geo_bar(geo: pd.DataFrame, level: str, top_n: int = 10) -> px.Figure:
    """Top values for a geographic level by revenue."""
    return px.bar(
        geo.head(top_n),
        x=level,
        y="revenue",
        color="revenue",
        color_continuous_scale=VOLT_SCALE,
        title=f"Top {level}s by revenue",
    )


@styled
def geo_treemap(geo: pd.DataFrame, level: str) -> px.Figure:
    """Treemap of customer share for one geographic level."""
    return px.treemap(
        geo,
        path=[level],
        values="customers",
        color="customers",
        color_continuous_scale=VOLT_SCALE,
        title=f"Customer share by {level}",
    )


@styled
def stacked_composition_bar(composition: pd.DataFrame, column: str) -> px.Figure:
    """100% stacked bars of a demographic column across segments."""
    long = composition.reset_index().melt(id_vars="segment", var_name=column, value_name="pct")
    return px.bar(
        long,
        x="segment",
        y="pct",
        color=column,
        color_discrete_sequence=VOLT_DISCRETE,
        title=f"{column} mix by segment",
        labels={"pct": "Share (%)", "segment": "Segment"},
    )

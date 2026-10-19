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

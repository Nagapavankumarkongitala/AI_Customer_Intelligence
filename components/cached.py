"""Cached compute: Streamlit-cached wrappers over pure service/analytics calls.

Switching pages re-executes the whole script, and several views recompute
expensive frames on every visit (dashboard KPIs/trends, quality scoring,
demographic profiles, retention tables). These thin wrappers memoize those
calls per session so repeat visits are instant.

Rules:

* Only wrap pure functions (same inputs → same outputs, no session writes).
* Invalidation is by content hash: new data means new cache keys, so callers
  must never mutate an argument after passing it in (pages copy first).
* ``show_spinner=False``: first computes happen inside page flows that already
  narrate progress; cache hits stay silent.
* UI-bound work (buttons, downloads) and model training stay uncached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from analytics.demographics import build_demographic_profile, segment_composition
from analytics.statistical_tests import run_segment_tests
from segmentation.segment_profiling import build_cluster_profile
from services.dashboard_service import (
    apply_filters,
    build_master,
    compute_kpis,
    high_value_high_risk,
    prepare_transactions,
    segment_portfolio,
    trend_tables,
)
from services.data_dictionary_service import build_dictionary
from services.feature_service import run_segment_history, segment_profile
from services.profiling_service import compute_quality
from services.retention_service import build_retention_table, export_campaign


@st.cache_data(show_spinner=False)
def cached_prepare_transactions(clean_data: pd.DataFrame) -> pd.DataFrame:
    """Parsed transaction grain (dates, amounts, month)."""
    return prepare_transactions(clean_data)


@st.cache_data(show_spinner=False)
def cached_build_master(
    features: pd.DataFrame,
    rfm: pd.DataFrame | None,
    segmented: pd.DataFrame | None,
    scores: pd.DataFrame | None,
    retention: pd.DataFrame | None,
) -> pd.DataFrame:
    """All customer grains left-joined onto the feature table."""
    return build_master(features, rfm=rfm, segmented=segmented, scores=scores, retention=retention)


@st.cache_data(show_spinner=False)
def cached_apply_filters(
    txns: pd.DataFrame, master: pd.DataFrame, filters: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter-consistent (transactions, customers) pair."""
    return apply_filters(txns, master, filters)


@st.cache_data(show_spinner=False)
def cached_compute_kpis(txns: pd.DataFrame, master: pd.DataFrame) -> dict:
    """Row-1 KPI cards from the filtered pair."""
    return compute_kpis(txns, master)


@st.cache_data(show_spinner=False)
def cached_trend_tables(txns: pd.DataFrame, master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Monthly revenue, activity, and new-vs-returning tables."""
    return trend_tables(txns, master)


@st.cache_data(show_spinner=False)
def cached_segment_portfolio(master: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Customer and revenue share per segment."""
    return segment_portfolio(master, segment_col)


@st.cache_data(show_spinner=False)
def cached_high_value_high_risk(master: pd.DataFrame) -> pd.DataFrame:
    """Masked high-tier + high/critical-band customers by revenue at risk."""
    return high_value_high_risk(master)


@st.cache_data(show_spinner=False)
def cached_run_segment_history(
    clean_data: pd.DataFrame, granularity: str, quantiles: int
) -> pd.DataFrame:
    """Period × model × segment customer counts for trend ribbons."""
    return run_segment_history(clean_data, granularity=granularity, quantiles=quantiles)


@st.cache_data(show_spinner=False)
def cached_segment_profile(enriched: pd.DataFrame, monetary_col: str = "monetary_value"):
    """Segment profile table for the scored frame."""
    return segment_profile(enriched, monetary_col=monetary_col)


@st.cache_data(show_spinner=False)
def cached_compute_quality(
    raw: pd.DataFrame,
    mapping: dict,
    age_min: float,
    age_max: float,
    future_tolerance_days: int,
    iqr_multiplier: float,
):
    """Quality scoring (completeness/validity/uniqueness + issues)."""
    return compute_quality(
        raw,
        mapping,
        age_min=age_min,
        age_max=age_max,
        future_tolerance_days=future_tolerance_days,
        iqr_multiplier=iqr_multiplier,
    )


@st.cache_data(show_spinner=False)
def cached_build_dictionary(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """One documented row per mapped canonical field."""
    return build_dictionary(df, mapping)


@st.cache_data(show_spinner=False)
def cached_build_demographic_profile(
    features: pd.DataFrame, segments: pd.DataFrame | None, age_band_spec: str | None
):
    """Demographic/geographic profile tables plus notes."""
    return build_demographic_profile(features, segments, age_band_spec=age_band_spec)


@st.cache_data(show_spinner=False)
def cached_segment_composition(
    features: pd.DataFrame, segments: pd.DataFrame, column: str
) -> pd.DataFrame:
    """100% stacked composition of a demographic column across segments."""
    return segment_composition(features, segments, column)


@st.cache_data(show_spinner=False)
def cached_run_segment_tests(
    features: pd.DataFrame,
    segments: pd.Series,
    continuous: list,
    categorical: list,
) -> pd.DataFrame:
    """Segment significance tests with Holm correction."""
    return run_segment_tests(features, segments, continuous, categorical)


@st.cache_data(show_spinner=False)
def cached_build_cluster_profile(
    features: pd.DataFrame, labels: np.ndarray, feature_names: list, id_col: str = "customer_id"
) -> pd.DataFrame:
    """Per-cluster profile table for the current run."""
    return build_cluster_profile(features, labels, feature_names, id_col=id_col)


@st.cache_data(show_spinner=False)
def cached_build_retention_table(
    scores: pd.DataFrame,
    features: pd.DataFrame,
    matrix,
    value_basis: str,
    strategic_weights: float,
    value_quantiles: int,
) -> pd.DataFrame:
    """Ranked retention actions with traceable reasons."""
    return build_retention_table(
        scores,
        features,
        matrix,
        value_basis=value_basis,
        strategic_weights=strategic_weights,
        value_quantiles=value_quantiles,
    )


@st.cache_data(show_spinner=False)
def cached_export_campaign(
    table: pd.DataFrame, top_n: int, min_priority: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Campaign audience plus channel summary for the filtered view."""
    return export_campaign(table, top_n=top_n, min_priority=min_priority)

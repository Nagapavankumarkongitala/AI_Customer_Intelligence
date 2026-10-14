"""Cached compute tests: wrappers match direct calls and repeat cheaply.

Each test runs the cached wrapper twice (warming the cache) and asserts the
result equals the direct service call. Streamlit's cache works without a live
session, so these run in the plain suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from components import cached
from services.dashboard_service import (
    apply_filters,
    build_master,
    compute_kpis,
    high_value_high_risk,
    prepare_transactions,
    segment_portfolio,
    trend_tables,
)


def _txns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2", "C3"],
            "transaction_date": ["2024-01-10", "2024-02-10", "2024-02-15", "2024-03-10"],
            "amount_spent": [100.0, 200.0, 300.0, 400.0],
            "country": ["India"] * 4,
            "region": ["West", "West", "North", "West"],
        }
    )


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "first_transaction_date": ["2024-01-10", "2024-02-15", "2024-03-10"],
            "last_transaction_date": ["2024-02-10", "2024-02-15", "2024-03-10"],
            "monetary_value": [300.0, 300.0, 400.0],
            "frequency": [2, 1, 1],
            "recency_days": [30, 25, 10],
            "gender": ["Male", "Female", "Male"],
            "age": [25, 40, 60],
        }
    )


def test_dashboard_path_parity() -> None:
    """Prepare/master/filter/KPIs/trends/portfolio match direct calls twice."""
    clean = _txns()
    txns = cached.cached_prepare_transactions(clean)
    pd.testing.assert_frame_equal(txns, prepare_transactions(clean))

    master = cached.cached_build_master(_features(), None, None, None, None)
    pd.testing.assert_frame_equal(
        master, build_master(_features(), rfm=None, segmented=None, scores=None, retention=None)
    )

    filters: dict = {}
    txns_f, master_f = cached.cached_apply_filters(txns, master, filters)
    direct_f, direct_m = apply_filters(txns, master, filters)
    pd.testing.assert_frame_equal(txns_f, direct_f)
    pd.testing.assert_frame_equal(master_f, direct_m)

    assert cached.cached_compute_kpis(txns_f, master_f) == compute_kpis(txns_f, master_f)
    for name, table in cached.cached_trend_tables(txns_f, master_f).items():
        pd.testing.assert_frame_equal(table, trend_tables(txns_f, master_f)[name])

    rfm = pd.DataFrame(
        {"customer_id": ["C1", "C2", "C3"], "segment": ["Champions", "Lost", "Champions"]}
    )
    master_seg = cached.cached_build_master(_features(), rfm, None, None, None)
    pd.testing.assert_frame_equal(
        cached.cached_segment_portfolio(master_seg, "rfm_segment"),
        segment_portfolio(master_seg, "rfm_segment"),
    )
    pd.testing.assert_frame_equal(
        cached.cached_high_value_high_risk(master_seg), high_value_high_risk(master_seg)
    )
    # Second pass exercises cache hits.
    assert cached.cached_compute_kpis(txns_f, master_f) == compute_kpis(txns_f, master_f)


def test_quality_and_dictionary_parity() -> None:
    """Quality scoring and dictionary match direct calls."""
    from services.data_dictionary_service import build_dictionary
    from services.profiling_service import compute_quality

    mapping = {
        "customer_id": "customer_id",
        "transaction_date": "transaction_date",
        "amount_spent": "amount_spent",
    }
    direct = compute_quality(_txns(), mapping)
    cached_result = cached.cached_compute_quality(_txns(), mapping, 0, 120, 0, 1.5)
    assert cached_result.overall == direct.overall
    assert cached_result.counts == direct.counts

    pd.testing.assert_frame_equal(
        cached.cached_build_dictionary(_txns(), mapping), build_dictionary(_txns(), mapping)
    )


def test_demographics_and_tests_parity() -> None:
    """Profile, composition, and significance tests match direct calls."""
    from analytics.demographics import build_demographic_profile, segment_composition
    from analytics.statistical_tests import run_segment_tests

    segments = pd.DataFrame({"customer_id": ["C1", "C2", "C3"], "segment": ["A", "B", "A"]})
    direct = build_demographic_profile(_features(), segments)
    cached_profile = cached.cached_build_demographic_profile(_features(), segments, None)
    assert set(cached_profile.tables) == set(direct.tables)
    assert cached_profile.notes == direct.notes

    pd.testing.assert_frame_equal(
        cached.cached_segment_composition(_features(), segments, "gender"),
        segment_composition(_features(), segments, "gender"),
    )
    seg_series = pd.Series(["A", "B", "A"])
    pd.testing.assert_frame_equal(
        cached.cached_run_segment_tests(_features(), seg_series, ["age"], ["gender"]),
        run_segment_tests(_features(), seg_series, ["age"], ["gender"]),
    )


def test_cluster_profile_parity() -> None:
    """Cluster profiling matches the direct call."""
    from segmentation.segment_profiling import build_cluster_profile

    labels = np.array([0, 1, 0])
    pd.testing.assert_frame_equal(
        cached.cached_build_cluster_profile(_features(), labels, ["age", "monetary_value"]),
        build_cluster_profile(_features(), labels, ["age", "monetary_value"]),
    )


def test_retention_and_campaign_parity() -> None:
    """Retention table and campaign export match direct calls."""
    from services.retention_service import (
        build_retention_table,
        default_matrix_report,
        export_campaign,
    )

    scores = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(6)],
            "churn_probability": [0.9, 0.7, 0.5, 0.3, 0.1, 0.05],
            "risk_band": ["critical", "high", "medium", "medium", "low", "low"],
        }
    )
    features = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(6)],
            "monetary_value": [5000.0, 4000.0, 3000.0, 2000.0, 1000.0, 500.0],
        }
    )
    matrix, _ = default_matrix_report()
    direct = build_retention_table(scores, features, matrix)
    pd.testing.assert_frame_equal(
        cached.cached_build_retention_table(scores, features, matrix, "monetary_value", 1.0, 3),
        direct,
    )
    campaign, summary = cached.cached_export_campaign(direct, top_n=4, min_priority=0.0)
    direct_campaign, direct_summary = export_campaign(direct, top_n=4, min_priority=0.0)
    pd.testing.assert_frame_equal(campaign, direct_campaign)
    pd.testing.assert_frame_equal(summary, direct_summary)


def test_segment_profile_parity() -> None:
    """Segment profile matches the direct call."""
    from services.feature_service import segment_profile

    enriched = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "segment": ["A", "B", "A"],
            "monetary_value": [300.0, 300.0, 400.0],
            "recency_days": [30, 25, 10],
            "frequency": [2, 1, 1],
        }
    )
    pd.testing.assert_frame_equal(
        cached.cached_segment_profile(enriched), segment_profile(enriched)
    )

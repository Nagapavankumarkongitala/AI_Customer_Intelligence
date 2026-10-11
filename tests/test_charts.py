"""Phase 8 tests: every Plotly factory renders on tiny frames (no Streamlit)."""

import numpy as np
import pandas as pd
import pytest

from deep_learning.latent_space_visualization import latent_scatter
from visualization import (
    churn_charts,
    demographic_charts,
    executive_dashboard,
    retention_charts,
    rfm_charts,
    segmentation_charts,
)


def _seg() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "segment": ["A", "A", "B", "B"],
            "customers": [2, 2, 2, 2],
            "revenue": [100.0, 100.0, 200.0, 200.0],
            "recency_days": [5, 10, 50, 60],
            "frequency": [10, 9, 2, 1],
            "monetary_value": [100.0, 120.0, 300.0, 320.0],
            "R": [5, 4, 2, 1],
            "F": [5, 4, 2, 1],
        }
    )


def test_rfm_charts_render() -> None:
    """RFM factories produce figures with data traces."""
    seg = _seg()
    profile = pd.DataFrame({"segment": ["A", "B"], "customers": [2, 2], "revenue": [220.0, 620.0]})
    for fig in (
        rfm_charts.segment_distribution_bar(profile),
        rfm_charts.revenue_by_segment_bar(profile),
        rfm_charts.recency_frequency_scatter(seg),
        rfm_charts.frequency_monetary_scatter(seg),
        rfm_charts.rfm_heatmap(seg),
    ):
        assert len(fig.data) > 0


def test_demographic_charts_render() -> None:
    """Demographic factories render without optional-field crashes."""
    bands = pd.DataFrame({"age_band": ["18–24", "25–34"], "customers": [2, 2]})
    gender = pd.DataFrame({"gender": ["Male", "Female"], "customers": [2, 2]})
    geo = pd.DataFrame({"city": ["Pune", "Delhi"], "customers": [3, 1], "revenue": [500.0, 340.0]})
    comp = pd.DataFrame(
        {"segment": ["A", "B"], "Male": [50.0, 50.0], "Female": [50.0, 50.0]}
    ).set_index("segment")
    for fig in (
        demographic_charts.age_histogram(pd.Series([25, 34, 45])),
        demographic_charts.age_band_bar(bands),
        demographic_charts.gender_bar(gender),
        demographic_charts.top_geo_bar(geo, "city"),
        demographic_charts.revenue_geo_bar(geo, "city"),
        demographic_charts.geo_treemap(geo, "city"),
        demographic_charts.stacked_composition_bar(comp, "gender"),
    ):
        assert len(fig.data) > 0


def test_segmentation_charts_render() -> None:
    """PCA/heatmap/size/Sankey/metric charts render; UMAP guides when absent."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 4))
    labels = np.array([0] * 10 + [1] * 10)
    profile = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "business_name": ["Core Active", "Light Dormant"],
            "customers": [10, 10],
            "median_recency_days": [5.0, 60.0],
            "median_frequency": [9.0, 1.5],
        }
    )
    assert len(segmentation_charts.pca_scatter(X, labels).data) > 0
    assert len(segmentation_charts.cluster_size_bar(profile).data) > 0
    assert len(segmentation_charts.profile_heatmap(profile).data) > 0
    ct = pd.DataFrame([[8, 2], [1, 9]], index=["A", "B"], columns=["X", "Y"])
    assert len(segmentation_charts.sankey_mapping(ct, "RFM", "AI").data) > 0
    table = pd.DataFrame({"model": ["m1", "m2"], "silhouette": [0.5, 0.6]})
    assert len(segmentation_charts.metric_comparison_bar(table, "silhouette", "t").data) > 0
    try:
        import umap  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="umap-learn"):
            segmentation_charts.umap_scatter(X, labels)
    else:
        assert len(segmentation_charts.umap_scatter(X, labels).data) > 0


def test_churn_retention_executive_charts_render() -> None:
    """Churn, retention, executive, and latent factories render."""
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.9])
    importance = pd.DataFrame({"feature": ["a", "b"], "importance": [0.5, 0.3]})
    scores = pd.DataFrame({"risk_band": ["low", "high", "critical", "medium"]})
    table = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "churn_probability": [0.9, 0.2],
            "expected_value_proxy_historical": [5000.0, 300.0],
            "revenue_at_risk": [4500.0, 60.0],
            "recommended_action": ["VIP", "Nurture"],
            "retention_priority_score": [0.8, 0.05],
        }
    )
    trend = pd.DataFrame(
        {
            "month": ["2024-01", "2024-02"],
            "revenue": [100.0, 200.0],
            "active_customers": [2, 3],
            "new_customers": [2, 1],
            "returning_customers": [0, 2],
        }
    )
    portfolio = pd.DataFrame(
        {"seg": ["A", "B"], "customer_share": [50.0, 50.0], "revenue_share": [30.0, 70.0]}
    )
    for fig in (
        churn_charts.roc_curve_fig(y, proba),
        churn_charts.confusion_fig({"tn": 1, "fp": 1, "fn": 0, "tp": 2}),
        churn_charts.gain_chart_fig(pd.DataFrame({"decile": [1, 2], "cum_gain": [0.8, 1.0]})),
        churn_charts.importance_bar_fig(importance),
        churn_charts.risk_band_bar(scores),
        retention_charts.value_risk_matrix(table),
        retention_charts.revenue_at_risk_bar(table),
        retention_charts.strategy_distribution_bar(table),
        retention_charts.priority_histogram(table),
        executive_dashboard.revenue_trend_fig(trend),
        executive_dashboard.activity_trend_fig(trend),
        executive_dashboard.new_returning_fig(trend),
        executive_dashboard.portfolio_bar_fig(portfolio, "seg"),
        latent_scatter(np.random.default_rng(1).normal(size=(10, 4)), np.array([0] * 5 + [1] * 5)),
    ):
        assert len(fig.data) > 0
    cal = pd.DataFrame(
        {"mean_predicted": [0.2, 0.8], "observed_rate": [0.25, 0.75], "count": [2, 2]}
    )
    assert len(churn_charts.calibration_curve_fig(cal).data) > 0

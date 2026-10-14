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


def _trace_colors(fig) -> dict:
    """Trace name → marker color for per-category figures."""
    return {trace.name: trace.marker.color for trace in fig.data}


def test_segment_charts_use_stable_hues() -> None:
    """Known segments render in their vocabulary hues on every chart."""
    from visualization import tokens

    profile = pd.DataFrame(
        {
            "segment": ["Champions", "Lost Customers"],
            "customers": [3, 1],
            "revenue": [900.0, 100.0],
        }
    )
    assert _trace_colors(rfm_charts.segment_distribution_bar(profile)) == {
        "Champions": tokens.SEGMENT_COLORS["Champions"],
        "Lost Customers": tokens.SEGMENT_COLORS["Lost Customers"],
    }
    assert _trace_colors(rfm_charts.revenue_by_segment_bar(profile)) == {
        "Champions": tokens.SEGMENT_COLORS["Champions"],
        "Lost Customers": tokens.SEGMENT_COLORS["Lost Customers"],
    }

    renamed = profile.rename(columns={"segment": "rfm_segment"})
    assert _trace_colors(rfm_charts.segment_distribution_bar(renamed, "rfm_segment")) == {
        "Champions": tokens.SEGMENT_COLORS["Champions"],
        "Lost Customers": tokens.SEGMENT_COLORS["Lost Customers"],
    }


def test_segment_share_donut() -> None:
    """Donut carries segment hues, a hole, and the total in its center."""
    from visualization import tokens

    profile = pd.DataFrame(
        {
            "segment": ["Champions", "Lost Customers"],
            "customers": [3, 1],
            "revenue": [900.0, 100.0],
        }
    )
    fig = rfm_charts.segment_share_donut(profile)
    assert len(fig.data) > 0
    assert fig.data[0].hole == 0.45
    assert fig.layout.title.text == "Customers per segment"
    assert list(fig.data[0].marker.colors) == [
        tokens.SEGMENT_COLORS["Champions"],
        tokens.SEGMENT_COLORS["Lost Customers"],
    ]
    assert "4<br>customers" in fig.layout.annotations[0].text
    assert len(rfm_charts.segment_share_donut(pd.DataFrame()).data) == 0

    frame = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "segment": ["Champions", "Lost Customers"],
            "recency_days": [5, 60],
            "frequency": [10, 1],
            "monetary_value": [100.0, 300.0],
        }
    )
    assert _trace_colors(rfm_charts.recency_frequency_scatter(frame)) == {
        "Champions": tokens.SEGMENT_COLORS["Champions"],
        "Lost Customers": tokens.SEGMENT_COLORS["Lost Customers"],
    }

    sizes = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "business_name": ["Premium Active", "Light Dormant"],
            "customers": [30, 10],
        }
    )
    assert _trace_colors(segmentation_charts.cluster_size_bar(sizes)) == {
        "Premium Active": tokens.SEGMENT_COLORS["Premium Active"],
        "Light Dormant": tokens.SEGMENT_COLORS["Light Dormant"],
    }

    pca_colors = _trace_colors(
        segmentation_charts.pca_scatter(
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            np.array([0, 1]),
            names={0: "Premium Active", 1: "Core Steady"},
        )
    )
    assert pca_colors == {
        "Premium Active": tokens.SEGMENT_COLORS["Premium Active"],
        "Core Steady": tokens.SEGMENT_COLORS["Core Steady"],
    }


def test_unknown_categories_fall_back_to_neutral_sequence() -> None:
    """Unmapped labels never collapse to grey; they take the neutral sequence."""
    from visualization.theme import PRIMARY_DISCRETE

    profile = pd.DataFrame({"segment": ["A", "B"], "customers": [2, 2], "revenue": [1.0, 2.0]})
    colors = _trace_colors(rfm_charts.segment_distribution_bar(profile))
    assert set(colors.values()) == set(PRIMARY_DISCRETE[:2])


def test_action_charts_use_stable_hues() -> None:
    """Known retention actions render in their matrix hues."""
    from visualization import tokens

    table = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "churn_probability": [0.9, 0.2],
            "expected_value_proxy_historical": [5000.0, 300.0],
            "revenue_at_risk": [4500.0, 60.0],
            "recommended_action": [
                "Personal outreach (VIP intervention)",
                "Routine nurture and monitoring",
            ],
        }
    )
    expected = {
        "Personal outreach (VIP intervention)": tokens.action_color(
            "Personal outreach (VIP intervention)"
        ),
        "Routine nurture and monitoring": tokens.action_color("Routine nurture and monitoring"),
    }
    assert _trace_colors(retention_charts.value_risk_matrix(table)) == expected
    assert _trace_colors(retention_charts.revenue_at_risk_bar(table)) == expected
    assert _trace_colors(retention_charts.strategy_distribution_bar(table)) == expected


def test_geo_titles_use_english_plurals() -> None:
    """Level names pluralize correctly in chart titles (no 'citys')."""
    geo = pd.DataFrame({"city": ["Pune"], "customers": [3], "revenue": [500.0]})
    assert demographic_charts.top_geo_bar(geo, "city").layout.title.text == (
        "Top cities by customers"
    )
    assert demographic_charts.revenue_geo_bar(geo, "city").layout.title.text == (
        "Top cities by revenue"
    )
    country = pd.DataFrame({"country": ["India"], "customers": [3]})
    assert "countries" in demographic_charts.top_geo_bar(country, "country").layout.title.text
    region = pd.DataFrame({"region": ["West"], "customers": [3]})
    assert "regions" in demographic_charts.top_geo_bar(region, "region").layout.title.text


def test_sankey_has_a_title() -> None:
    """The flow chart names its endpoints instead of Plotly's placeholder."""
    contingency = pd.DataFrame({"B": [5]}, index=pd.Index(["A"], name="src"))
    fig = segmentation_charts.sankey_mapping(contingency, "RFM", "AI")
    assert fig.layout.title.text == "RFM → AI flow"


def test_revenue_area_and_horizontal_geo_render() -> None:
    """Area revenue trend and horizontal geo bars render with data."""
    trend = pd.DataFrame({"month": ["2024-01", "2024-02"], "revenue": [100.0, 200.0]})
    area = executive_dashboard.revenue_area_fig(trend)
    assert len(area.data) > 0
    assert area.layout.title.text == "Revenue trend"
    assert executive_dashboard.revenue_area_fig(pd.DataFrame()).layout.title.text == (
        "Revenue trend"
    )

    geo = pd.DataFrame({"region": ["West", "East"], "revenue": [500.0, 300.0]})
    hbar = demographic_charts.revenue_geo_bar(geo, "region", horizontal=True)
    assert len(hbar.data) > 0
    assert hbar.layout.title.text == "Top regions by revenue"
    assert hbar.data[0].orientation == "h"


def test_empty_inputs_return_titled_placeholders() -> None:
    """Empty inputs render a titled empty state instead of crashing or blanking."""
    from visualization.theme import empty_figure

    cases = [
        (rfm_charts.segment_distribution_bar(pd.DataFrame()), "Customers per segment"),
        (rfm_charts.revenue_by_segment_bar(pd.DataFrame()), "Revenue per segment"),
        (rfm_charts.recency_frequency_scatter(pd.DataFrame()), "Recency vs frequency"),
        (
            rfm_charts.frequency_monetary_scatter(pd.DataFrame()),
            "Frequency vs monetary value",
        ),
        (rfm_charts.rfm_heatmap(pd.DataFrame()), "Customer counts by R × F score"),
        (churn_charts.roc_curve_fig(np.array([]), np.array([])), "ROC curve"),
        (churn_charts.calibration_curve_fig(pd.DataFrame()), "Calibration curve"),
        (churn_charts.confusion_fig({}), "Confusion matrix"),
        (churn_charts.gain_chart_fig(pd.DataFrame()), "Cumulative gain"),
        (churn_charts.importance_bar_fig(pd.DataFrame()), "Global feature importance"),
        (churn_charts.risk_band_bar(pd.DataFrame()), "Customers per risk band"),
        (
            retention_charts.value_risk_matrix(pd.DataFrame()),
            "Value–risk matrix (size = revenue at risk)",
        ),
        (retention_charts.revenue_at_risk_bar(pd.DataFrame()), "Revenue at risk per action"),
        (
            retention_charts.strategy_distribution_bar(pd.DataFrame()),
            "Customers per recommended action",
        ),
        (retention_charts.priority_histogram(pd.DataFrame()), "Priority score distribution"),
        (executive_dashboard.revenue_trend_fig(pd.DataFrame()), "Revenue trend"),
        (executive_dashboard.activity_trend_fig(pd.DataFrame()), "Customer activity trend"),
        (executive_dashboard.new_returning_fig(pd.DataFrame()), "New vs returning customers"),
        (
            executive_dashboard.portfolio_bar_fig(pd.DataFrame()),
            "Customer vs revenue share by segment",
        ),
        (demographic_charts.age_histogram(pd.Series([], dtype=float)), "Age distribution"),
        (demographic_charts.age_band_bar(pd.DataFrame()), "Customers per age band"),
        (demographic_charts.gender_bar(pd.DataFrame()), "Customers by gender"),
        (demographic_charts.top_geo_bar(pd.DataFrame(), "city"), "Top cities by customers"),
        (demographic_charts.revenue_geo_bar(pd.DataFrame(), "city"), "Top cities by revenue"),
        (demographic_charts.geo_treemap(pd.DataFrame(), "city"), "Customer share by city"),
        (
            demographic_charts.stacked_composition_bar(pd.DataFrame(), "gender"),
            "gender mix by segment",
        ),
        (
            segmentation_charts.pca_scatter(np.empty((0, 3)), np.array([])),
            "Clusters (PCA 2D)",
        ),
        (segmentation_charts.cluster_size_bar(pd.DataFrame()), "Customers per cluster"),
        (
            segmentation_charts.profile_heatmap(pd.DataFrame()),
            "Cluster profile (median, z-scored per feature)",
        ),
        (
            segmentation_charts.sankey_mapping(pd.DataFrame(), "A", "B"),
            "A → B flow",
        ),
        (
            segmentation_charts.metric_comparison_bar(
                pd.DataFrame({"silhouette": []}), "silhouette", "Silhouette"
            ),
            "Silhouette",
        ),
    ]
    assert len(cases) == 31
    for fig, title in cases:
        assert len(fig.data) == 0, title
        assert fig.layout.title.text == title, title
        assert fig.layout.annotations, title

    direct = empty_figure("Direct", "Custom message.")
    assert direct.layout.title.text == "Direct"
    assert direct.layout.annotations[0].text == "Custom message."

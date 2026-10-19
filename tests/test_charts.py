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
    history = pd.DataFrame(
        {
            "period": ["2024-Q1", "2024-Q1", "2024-Q2", "2024-Q2"],
            "model": ["RFM"] * 4,
            "segment": ["Champions", "Lost Customers", "Champions", "Lost Customers"],
            "customers": [10, 5, 12, 4],
        }
    )
    for fig in (
        rfm_charts.segment_distribution_bar(profile),
        rfm_charts.revenue_by_segment_bar(profile),
        rfm_charts.rfm_parallel_plot(seg),
        rfm_charts.segment_trend_ribbon(history),
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
    dist = rfm_charts.segment_distribution_bar(profile)
    assert dist.layout.xaxis.title.text == ""
    assert dist.layout.yaxis.visible is False
    assert _trace_colors(rfm_charts.revenue_by_segment_bar(profile)) == {
        "Champions": tokens.SEGMENT_COLORS["Champions"],
        "Lost Customers": tokens.SEGMENT_COLORS["Lost Customers"],
    }

    renamed = profile.rename(columns={"segment": "rfm_segment"})
    assert _trace_colors(rfm_charts.segment_distribution_bar(renamed, "rfm_segment")) == {
        "Champions": tokens.SEGMENT_COLORS["Champions"],
        "Lost Customers": tokens.SEGMENT_COLORS["Lost Customers"],
    }
    for fig in (
        rfm_charts.segment_distribution_bar(profile),
        rfm_charts.revenue_by_segment_bar(profile),
    ):
        assert fig.layout.xaxis.showticklabels is False
        assert fig.layout.xaxis.title.text == ""
        assert fig.layout.bargap == 0.1
        assert fig.data[0].textfont.size == 13


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
    parcoords = rfm_charts.rfm_parallel_plot(frame)
    assert len(parcoords.data) > 0
    assert parcoords.layout.title.text == "Recency, frequency and value"

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


def test_scatter_matrix_samples_and_maps_segments() -> None:
    """Matrix renders segment hues and caps points deterministically."""
    rng = np.random.default_rng(42)
    big = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(2500)],
            "segment": ["Champions" if i % 2 else "Lost Customers" for i in range(2500)],
            "recency_days": rng.integers(1, 300, 2500),
            "frequency": rng.integers(1, 20, 2500),
            "monetary_value": rng.uniform(10.0, 900.0, 2500).round(2),
        }
    )
    fig = rfm_charts.rfm_scatter_matrix(big)
    assert len(fig.data) > 0
    assert fig.layout.title.text == "Pairwise RFM relationships"
    assert fig.layout.height == 480
    rendered = sum(len(trace.x) for trace in fig.data if getattr(trace, "x", None) is not None)
    assert rendered <= 2000 * 9
    names = {trace.name for trace in fig.data if trace.name}
    assert names <= {"Champions", "Lost Customers"}

    small = big.head(10)
    assert len(rfm_charts.rfm_scatter_matrix(small).data) > 0
    assert len(rfm_charts.rfm_scatter_matrix(pd.DataFrame()).data) == 0


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
        (rfm_charts.rfm_parallel_plot(pd.DataFrame()), "Recency, frequency and value"),
        (rfm_charts.segment_trend_ribbon(pd.DataFrame()), "Customer segments over time"),
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
        (rfm_charts.segment_trend_share(pd.DataFrame()), "Segment share over time"),
        (
            rfm_charts.segment_trend_facets(pd.DataFrame()),
            "Segment trends (one panel per segment)",
        ),
        (rfm_charts.segment_trend_heatmap(pd.DataFrame()), "Segment activity by period"),
    ]
    assert len(cases) == 32
    for fig, title in cases:
        assert len(fig.data) == 0, title
        assert fig.layout.title.text == title, title
        assert fig.layout.annotations, title

    direct = empty_figure("Direct", "Custom message.")
    assert direct.layout.title.text == "Direct"
    assert direct.layout.annotations[0].text == "Custom message."


def _retention_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "churn_probability": [0.9, 0.2, 0.7],
            "expected_value_proxy_historical": [5000.0, 300.0, 2000.0],
            "revenue_at_risk": [4500.0, 60.0, 1400.0],
            "recommended_action": ["VIP", "Nurture", "Reactivation"],
            "retention_priority_score": [0.8, 0.05, 0.5],
        }
    )


def test_revenue_at_risk_bar_sorted_descending() -> None:
    """Actions order largest-first so the biggest exposure leads the chart."""
    fig = retention_charts.revenue_at_risk_bar(_retention_table())
    names = [trace.name for trace in fig.data]
    # Nurture 60, Reactivation 1400, VIP 4500 -> ascending input, descending output.
    assert names == ["VIP", "Reactivation", "Nurture"]


def test_value_risk_matrix_has_priority_quadrant() -> None:
    """Matrix draws churn/value reference lines framing the 'act first' zone."""
    fig = retention_charts.value_risk_matrix(_retention_table(), churn_threshold=0.6)
    assert len(fig.layout.shapes) == 2  # one vertical, one horizontal
    assert any("Act first" in (a.text or "") for a in fig.layout.annotations)


def test_value_risk_heatmap_aggregates_quadrants() -> None:
    """Heatmap bins customers into four quadrants with count + revenue labels."""
    table = _retention_table()
    fig = retention_charts.value_risk_heatmap(table, churn_threshold=0.6)
    heatmap = next(trace for trace in fig.data if trace.type == "heatmap")
    z = _trace_values(heatmap.z)
    # Rows: Low risk, High risk. Cols: Low value, High value (median value 2,000).
    # Act first holds C1 (0.9 / 5,000) and C3 (0.7 / 2,000); Maintain holds C2.
    assert z.tolist() == [[1, 0], [0, 2]]
    annotations = [a.text for a in fig.layout.annotations]
    assert any("Act first" in (text or "") for text in annotations)
    assert any("Maintain" in (text or "") for text in annotations)
    hover = [str(value) for value in np.asarray(heatmap.customdata).ravel()]
    assert any("at risk" in value for value in hover)
    assert len(retention_charts.value_risk_heatmap(pd.DataFrame()).data) == 0


def _trace_values(node) -> np.ndarray:
    """Numeric values from a trace field, decoding Plotly's binary form if used."""
    if isinstance(node, dict) and "bdata" in node:
        import base64

        return np.frombuffer(base64.b64decode(node["bdata"]), dtype=np.dtype(node["dtype"]))
    return np.asarray(node)


def test_priority_histogram_adds_cumulative_share() -> None:
    """Histogram pairs per-bin bars with a cumulative (Pareto) share line."""
    fig = retention_charts.priority_histogram(_retention_table())
    assert len(fig.data) == 2
    assert fig.data[0].type == "bar"
    assert fig.data[1].mode == "lines"
    assert fig.data[1].name == "Cumulative %"
    # Cumulative share tops out at 100 on the secondary axis.
    assert _trace_values(fig.data[1].y).max() == pytest.approx(100.0)
    assert list(fig.layout.yaxis2.range) == [0, 100]


def test_segment_share_donut_stays_a_donut_with_labels() -> None:
    """Every segment count keeps the donut; names ride on the slices."""
    few = pd.DataFrame(
        {"segment": ["Champions", "Lost Customers"], "customers": [3, 1], "revenue": [9.0, 1.0]}
    )
    assert rfm_charts.segment_share_donut(few).data[0].type == "pie"

    many = pd.DataFrame(
        {
            "segment": [f"S{i}" for i in range(7)],
            "customers": list(range(1, 8)),
            "revenue": [float(i) for i in range(1, 8)],
        }
    )
    fig = rfm_charts.segment_share_donut(many)
    assert fig.data[0].type == "pie"
    assert fig.data[0].hole == 0.45
    # Slice labels carry the names, so the wrapping legend is suppressed;
    # otherwise ten long names crush the ring into unreadable slivers.
    assert fig.data[0].textinfo == "label+percent"
    assert fig.layout.showlegend is False


def test_segment_trend_lines_renders_with_segment_hues() -> None:
    """Multi-line alternative to the stacked ribbon, colored by segment."""
    from visualization import tokens

    history = pd.DataFrame(
        {
            "period": ["2024-Q1", "2024-Q1", "2024-Q2", "2024-Q2"],
            "segment": ["Champions", "Lost Customers", "Champions", "Lost Customers"],
            "customers": [10, 5, 12, 4],
        }
    )
    fig = rfm_charts.segment_trend_lines(history)
    assert fig.layout.title.text == "Segment trends"
    line_colors = {t.name: t.line.color for t in fig.data}
    assert line_colors == {
        "Champions": tokens.SEGMENT_COLORS["Champions"],
        "Lost Customers": tokens.SEGMENT_COLORS["Lost Customers"],
    }
    assert len(rfm_charts.segment_trend_lines(pd.DataFrame()).data) == 0


def test_segment_trend_views_answer_different_questions() -> None:
    """Share, small multiples, and heatmap each reframe the same history."""
    history = pd.DataFrame(
        {
            "period": ["2024-Q1", "2024-Q1", "2024-Q2", "2024-Q2"],
            "segment": ["Champions", "Lost Customers", "Champions", "Lost Customers"],
            "customers": [10, 5, 12, 4],
        }
    )

    share = rfm_charts.segment_trend_share(history)
    assert share.layout.title.text == "Segment share over time"
    assert share.layout.yaxis.ticksuffix == "%"
    champions, lost = (_trace_values(trace.y) for trace in share.data)
    assert list(champions) == pytest.approx([200 / 3, 75.0])
    assert list(lost) == pytest.approx([100 / 3, 25.0])

    facets = rfm_charts.segment_trend_facets(history)
    assert [a.text for a in facets.layout.annotations] == ["Champions", "Lost Customers"]
    assert facets.layout.showlegend is False
    assert facets.layout.yaxis2.matches is None  # each panel keeps its own scale

    heat = rfm_charts.segment_trend_heatmap(history)
    assert list(heat.data[0].x) == ["2024-Q1", "2024-Q2"]
    assert list(heat.data[0].y) == ["Champions", "Lost Customers"]  # biggest on top
    assert list(_trace_values(heat.data[0].z).ravel()) == [10, 12, 5, 4]
    assert heat.data[0].texttemplate == "%{z}"


def test_rfm_scatter_3d_renders_samples_and_falls_back() -> None:
    """3D scatter colors by segment, samples deterministically, guards empty/short input."""
    from visualization import tokens

    rng = np.random.default_rng(42)
    big = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(2500)],
            "segment": ["Champions" if i % 2 else "Lost Customers" for i in range(2500)],
            "recency_days": rng.integers(1, 300, 2500),
            "frequency": rng.integers(1, 20, 2500),
            "monetary_value": rng.uniform(10.0, 900.0, 2500),
        }
    )
    fig = rfm_charts.rfm_scatter_3d(big)
    assert fig.data[0].type == "scatter3d"
    assert _trace_colors(fig) == {
        "Champions": tokens.SEGMENT_COLORS["Champions"],
        "Lost Customers": tokens.SEGMENT_COLORS["Lost Customers"],
    }
    assert len(rfm_charts.rfm_scatter_3d(pd.DataFrame()).data) == 0
    # Fewer than three RFM dims cannot form a 3D scatter.
    two_dims = big[["customer_id", "segment", "frequency", "monetary_value"]]
    assert len(rfm_charts.rfm_scatter_3d(two_dims).data) == 0


def test_geo_treemap_nests_levels_and_guards_short_paths() -> None:
    """Treemap builds a hierarchy from two levels; a single level falls back."""
    features = pd.DataFrame(
        {
            "country": ["India", "India", "US"],
            "city": ["Pune", "Delhi", "Austin"],
        }
    )
    fig = demographic_charts.geo_treemap(features, ["country", "city"])
    assert fig.data[0].type == "treemap"
    assert "country" in fig.layout.title.text and "city" in fig.layout.title.text

    single = demographic_charts.geo_treemap(features, ["country"])
    assert len(single.data) == 0
    assert "country" in single.layout.title.text


def test_geo_treemap_caps_near_unique_levels() -> None:
    """A near-unique level folds into Other instead of emitting thousands of tiles.

    Real uploads carry one city name per customer (the demo set has 15,516 cities
    for 24,911 customers). Uncapped, that built a ~24,000-tile figure and locked
    the browser tab; the cap keeps every customer counted exactly once.
    """
    n = 4000
    features = pd.DataFrame(
        {
            "region": ["R1"] * n,
            "city": [f"city-{i}" for i in range(n)],
        }
    )
    fig = demographic_charts.geo_treemap(features, ["region", "city"])
    labels = list(fig.data[0].labels)
    assert labels.count(demographic_charts.TREEMAP_OTHER) == 1
    # R1 + the kept cities + Other, far below one tile per customer.
    assert len(fig.data[0].ids) <= demographic_charts.TREEMAP_MAX_VALUES_PER_LEVEL + 2
    assert len(fig.data[0].ids) < n / 100

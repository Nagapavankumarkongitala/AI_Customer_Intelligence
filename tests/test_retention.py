"""Phase 7 tests: tiers, scoring math, matrix coverage, actions, campaign."""

import pandas as pd
import pytest

from retention_engine.campaign_generator import build_campaign
from retention_engine.next_best_action import assign_actions
from retention_engine.priority_scoring import (
    normalize_value,
    priority_score,
    revenue_at_risk,
)
from retention_engine.strategy_mapper import load_matrix, validate_matrix
from retention_engine.value_tiers import assign_value_tiers, expected_value_proxy
from services.retention_service import (
    build_retention_table,
    default_matrix_report,
    retention_summary,
)

TIERS = ["Low", "Medium", "High"]
BANDS = ["low", "medium", "high", "critical"]


def _scores_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(12)],
            "churn_probability": [
                0.9,
                0.85,
                0.7,
                0.65,
                0.5,
                0.45,
                0.3,
                0.25,
                0.15,
                0.1,
                0.05,
                0.95,
            ],
            "risk_band": [
                "critical",
                "critical",
                "high",
                "high",
                "medium",
                "medium",
                "medium",
                "low",
                "low",
                "low",
                "low",
                "critical",
            ],
        }
    )
    features = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(12)],
            "monetary_value": [
                5000.0,
                4000.0,
                3000.0,
                2500.0,
                2000.0,
                1500.0,
                1200.0,
                1000.0,
                800.0,
                500.0,
                300.0,
                4500.0,
            ],
            "region": ["West"] * 6 + ["North"] * 6,
        }
    )
    return scores, features


def test_default_matrix_covers_every_cell() -> None:
    """The shipped YAML matrix leaves no tier×band combination without a rule."""
    matrix, report = default_matrix_report()
    assert report["uncovered"] == []
    assert len(matrix.rules) >= 5
    assert matrix.disclaimer


def test_matrix_rejects_causal_claims() -> None:
    """Matrices promising retention outcomes are rejected at load time."""
    import tempfile

    payload = (
        "rules:\n"
        "  - name: bad\n"
        "    value_tiers: [High]\n"
        "    risk_bands: [high]\n"
        "    action: Call\n"
        "    channel: Phone\n"
        "    offer_type: X\n"
        "    reason: This will retain them guaranteed.\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(payload)
        path = fh.name
    with pytest.raises(ValueError, match="causal"):
        load_matrix(path)


def test_tiers_split_and_edge_cases() -> None:
    """Quantiles split evenly; constant input collapses without crashing."""
    tiers = assign_value_tiers(pd.Series([float(v) for v in range(1, 13)]))
    assert tiers.tolist().count("Low") == tiers.tolist().count("High") == 4
    flat = assign_value_tiers(pd.Series([100.0] * 5))
    assert flat.nunique() == 1
    proxy = expected_value_proxy(pd.Series([10.0, None]))
    assert proxy.tolist() == [10.0, 0.0]
    assert "historical" in proxy.name


def test_scoring_math_exact() -> None:
    """Revenue at risk and priority follow the spec formulas exactly."""
    prob = pd.Series([0.5, 1.0])
    value = pd.Series([200.0, 300.0])
    assert revenue_at_risk(prob, value).tolist() == [100.0, 300.0]
    normed = normalize_value(value)
    assert normed.tolist() == [0.0, 1.0]
    priority = priority_score(normed, prob, 2.0)
    assert priority.tolist() == [0.0, 2.0]


def test_actions_traceable_with_reasons() -> None:
    """Every action carries channel, offer, reason, and its rule name."""
    scores, features = _scores_features()
    matrix, _ = default_matrix_report()
    table = build_retention_table(scores, features, matrix)
    assert len(table) == 12
    for col in (
        "recommended_action",
        "recommended_channel",
        "suggested_offer_type",
        "action_reason",
        "strategy_rule",
        "revenue_at_risk",
        "retention_priority_score",
        "customer_value_tier",
    ):
        assert table[col].notna().all(), col
    vip = table.loc[table["customer_id"] == "C0"].iloc[0]
    assert vip["strategy_rule"] == "vip_intervention"
    assert "High" in vip["action_reason"] and "critical" in vip["action_reason"]
    assert table["revenue_at_risk"].sum() == round(
        (table["churn_probability"] * table["monetary_value"]).sum(), 2
    )


def test_ranking_and_summary() -> None:
    """Table ranks by priority descending; summary totals reconcile."""
    scores, features = _scores_features()
    matrix, _ = default_matrix_report()
    table = build_retention_table(scores, features, matrix)
    assert table["retention_priority_score"].is_monotonic_decreasing
    summary = retention_summary(table)
    assert summary["customers"] == 12
    assert summary["total_revenue_at_risk"] == round(table["revenue_at_risk"].sum(), 2)
    assert sum(summary["strategy_mix"].values()) == 12


def test_campaign_export() -> None:
    """Campaign takes the top-N, respects floor, and summarizes channels."""
    scores, features = _scores_features()
    matrix, _ = default_matrix_report()
    table = build_retention_table(scores, features, matrix)
    campaign, summary = build_campaign(table, top_n=5, min_priority=0.1)
    assert len(campaign) <= 5
    assert (campaign["retention_priority_score"] >= 0.1).all()
    assert summary["customers"].sum() == len(campaign)
    if "revenue_at_risk" in campaign.columns:
        assert summary["revenue_at_risk"].sum() == round(campaign["revenue_at_risk"].sum(), 2)


def test_assign_actions_validates_input() -> None:
    """Missing tier/band columns fail fast with the column names."""
    matrix, _ = default_matrix_report()
    with pytest.raises(KeyError):
        assign_actions(pd.DataFrame({"customer_id": ["C1"]}), matrix)
    _ = validate_matrix(matrix, TIERS, BANDS)

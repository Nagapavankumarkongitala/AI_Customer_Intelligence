"""Phase 8 tests: masking, filters, KPIs, trends, config, model cards."""

import pandas as pd
import pytest
import yaml

from retention_engine.strategy_mapper import load_matrix
from services.dashboard_service import (
    apply_filters,
    build_master,
    compute_kpis,
    high_value_high_risk,
    mask_customer_id,
    period_deltas,
    prepare_transactions,
    segment_activity_trend,
    segment_mom_bridge,
    segment_portfolio,
    strategy_segment_customer_counts,
    trend_tables,
)
from utils.config import load_config, validate_config
from utils.exceptions import ConfigurationError
from utils.model_card import (
    churn_card_metadata,
    segmentation_card_metadata,
    write_model_card,
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


def _master() -> pd.DataFrame:
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
            "region": ["West", "North", "West"],
            "rfm_segment": ["Champions", "Lost Customers", "Champions"],
            "ai_segment": ["Premium Active", "Core Dormant", "Premium Active"],
            "churn_probability": [0.1, 0.9, 0.2],
            "risk_band": ["low", "critical", "low"],
            "customer_value_tier": ["High", "Medium", "High"],
            "revenue_at_risk": [30.0, 270.0, 80.0],
            "retention_priority_score": [0.05, 0.7, 0.1],
            "recommended_action": ["Recognition", "VIP", "Recognition"],
        }
    )


def test_masking_format() -> None:
    """IDs hide the middle; short IDs collapse fully."""
    assert mask_customer_id("C00001") == "C•••01"
    assert mask_customer_id("AB") == "•••"


def test_filters_restrict_consistently() -> None:
    """Date/region/segment filters narrow txns and customers together."""
    txns, master = prepare_transactions(_txns()), build_master(_master())
    assert "age_band" in master.columns
    txns_f, master_f = apply_filters(
        txns,
        master,
        {
            "date_start": None,
            "date_end": None,
            "region": ["West"],
            "rfm_segment": ["Champions"],
            "country": [],
            "state": [],
            "city": [],
            "gender": [],
            "age_band": [],
            "ai_segment": [],
            "risk_band": [],
            "value_tier": [],
        },
    )
    assert set(master_f["customer_id"]) == {"C1", "C3"}
    assert set(txns_f["customer_id"]) <= set(master_f["customer_id"])


def test_kpis_add_up() -> None:
    """Revenue, AOV, and churn rate reconcile with inputs."""
    txns, master = prepare_transactions(_txns()), build_master(_master())
    txns_f, master_f = apply_filters(txns, master, {})
    kpis = compute_kpis(txns_f, master_f)
    assert kpis["total_customers"] == 3
    assert kpis["total_revenue"] == 1000.0
    assert kpis["average_order_value"] == 250.0
    assert kpis["predicted_churn_rate"] == round((0.1 + 0.9 + 0.2) / 3, 4)
    assert kpis["revenue_at_risk"] == 380.0


def test_trends_and_portfolio() -> None:
    """Monthly tables cover all months; portfolio shares sum to 100."""
    txns, master = prepare_transactions(_txns()), build_master(_master())
    trends = trend_tables(txns, master)
    assert trends["revenue"]["revenue"].sum() == 1000.0
    assert trends["cohort"]["new_customers"].sum() == 3
    portfolio = segment_portfolio(master, "rfm_segment")
    assert portfolio["customer_share"].sum() == 100.0
    assert portfolio["revenue_share"].sum() == 100.0


def test_segment_activity_trend_counts_fixed_labels_by_month() -> None:
    """Monthly activity follows each customer's current label, not a re-score."""
    txns, master = prepare_transactions(_txns()), build_master(_master())
    trend = segment_activity_trend(txns, master, "rfm_segment")
    assert set(trend["month"]) == {"2024-01", "2024-02", "2024-03"}
    feb = trend[trend["month"] == "2024-02"].set_index("rfm_segment")["active_customers"]
    assert feb["Champions"] == 1  # C1
    assert feb["Lost Customers"] == 1  # C2
    # A segment column that is absent yields the empty contract, not a crash.
    assert segment_activity_trend(txns, master, "missing_segment").empty


def test_segment_mom_bridge_walks_previous_to_selected_month() -> None:
    """The bridge starts at the prior total, sums segment deltas, ends at the new total."""
    txns, master = prepare_transactions(_txns()), build_master(_master())
    activity = segment_activity_trend(txns, master, "rfm_segment")
    bridge = segment_mom_bridge(activity, "2024-02", "rfm_segment")
    assert list(bridge["measure"]) == ["absolute", "relative", "relative", "total"]
    assert bridge.loc[0, "label"] == "2024-01 total"
    assert bridge.loc[0, "value"] == 1
    relative = bridge[bridge["measure"] == "relative"].set_index("label")["value"]
    assert relative["Lost Customers"] == 1
    assert relative["Champions"] == 0
    assert bridge.iloc[-1]["value"] == 2
    # No earlier month, or a single-month table, has nothing to bridge.
    assert segment_mom_bridge(activity, "2024-01", "rfm_segment").empty
    assert segment_mom_bridge(activity.head(1), "2024-02", "rfm_segment").empty


def test_period_deltas_compare_windows() -> None:
    """KPI deltas split at the latest date and compare equal-length windows."""
    txns = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C1", "C2", "C3"],
            "transaction_date": [
                "2024-02-01",
                "2024-02-02",
                "2024-03-01",
                "2024-03-02",
                "2024-03-03",
            ],
            "amount_spent": [100.0, 100.0, 200.0, 200.0, 200.0],
        }
    )
    prepared = prepare_transactions(txns)
    deltas = period_deltas(prepared, build_master(_master()), days=30)
    # Current window (after 2024-02-02): three 200s = 600. Previous: two 100s = 200.
    assert deltas["total_revenue"] == pytest.approx(2.0)  # 600 vs 200
    assert deltas["total_customers"] == pytest.approx(0.5)  # 3 vs 2
    assert deltas["average_order_value"] == pytest.approx(1.0)  # 200/order vs 100/order
    assert deltas["days"] == 30


def test_period_deltas_empty_without_baseline() -> None:
    """No earlier window means no delta, not a divide-by-zero or a fake zero."""
    single = prepare_transactions(_txns().head(1))
    assert period_deltas(single, build_master(_master()), days=30) == {}
    assert period_deltas(pd.DataFrame(), build_master(_master())) == {}


def test_high_risk_empty_gracefully() -> None:
    """No High/high overlap yields an empty frame, not an error."""
    master = build_master(_master())
    assert high_value_high_risk(master).empty


def test_high_risk_overlap_case() -> None:
    """A High-tier critical-band customer appears masked at the top."""
    master = build_master(_master())
    master.loc[master["customer_id"] == "C2", "customer_value_tier"] = "High"
    out = high_value_high_risk(master)
    assert len(out) == 1
    assert out["customer_id"].iloc[0] == "•••"
    assert out["revenue_at_risk"].iloc[0] == 270.0


def test_strategy_segment_customer_counts_are_long_form() -> None:
    """Every strategy keeps segment-level customer counts in playbook order."""
    matrix = load_matrix("retention_strategies.yaml")
    master = build_master(_master())
    retention = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "strategy_rule": ["recognition_growth", "recognition_growth", "vip_intervention"],
        }
    )
    rfm = strategy_segment_customer_counts(retention, master, matrix, "rfm_segment")
    assert list(rfm.columns) == ["strategy", "rule", "segment", "customers"]
    assert list(dict.fromkeys(rfm["rule"])) == [rule.name for rule in matrix.rules]
    recognition = rfm.loc[rfm["rule"] == "recognition_growth"].set_index("segment")["customers"]
    assert recognition.to_dict() == {"Champions": 1, "Lost Customers": 1}
    vip = rfm.loc[rfm["rule"] == "vip_intervention"].set_index("segment")["customers"]
    assert vip.to_dict() == {"Champions": 1}
    unused = rfm.loc[rfm["rule"] == "routine_nurture"].iloc[0]
    assert unused["segment"] == "No customers" and unused["customers"] == 0

    ai = strategy_segment_customer_counts(retention, master, matrix, "ai_segment")
    ai_recognition = ai.loc[ai["rule"] == "recognition_growth"].set_index("segment")["customers"]
    assert ai_recognition.to_dict() == {"Premium Active": 1, "Core Dormant": 1}
    missing = strategy_segment_customer_counts(retention, master, matrix, "no_such_segment")
    assert missing.empty and "customers" in missing.columns


def test_config_validation() -> None:
    """Shipped config passes; broken values fail with the offending key named."""
    validate_config(load_config())
    cfg = load_config()
    cfg["rfm"]["quantiles"] = 7
    with pytest.raises(ConfigurationError, match="rfm.quantiles"):
        validate_config(cfg)
    cfg = load_config()
    cfg["churn"]["risk_bands"]["low"] = [0.0, 0.5]
    with pytest.raises(ConfigurationError, match="risk_bands"):
        validate_config(cfg)


def test_model_cards_written() -> None:
    """Segmentation and churn cards render all required sections to disk."""
    seg = segmentation_card_metadata("kmeans", ["recency_days"], {"silhouette": 0.5}, {"k": 3})
    card = write_model_card(
        "reports/model_cards/test_seg.md", version="kmeans_v1", training_period="2024", **seg
    )
    text = card.read_text(encoding="utf-8")
    assert "## Fairness considerations" in text and "kmeans_v1" in text
    churn = churn_card_metadata("logreg", ["recency_days"], {"roc_auc": 0.9}, 0.5, "2024-06-01", 90)
    card2 = write_model_card("reports/model_cards/test_churn.md", version="logreg_v1", **churn)
    assert "Leakage-safe" in card2.read_text(encoding="utf-8")
    assert (
        "recency_days"
        in yaml.safe_load(open("config.yaml", encoding="utf-8"))["segmentation"]["default_features"]
    )

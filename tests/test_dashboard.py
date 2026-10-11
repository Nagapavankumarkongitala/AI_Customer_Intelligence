"""Phase 8 tests: masking, filters, KPIs, trends, config, model cards."""

import pandas as pd
import pytest
import yaml

from services.dashboard_service import (
    apply_filters,
    build_master,
    compute_kpis,
    high_value_high_risk,
    mask_customer_id,
    prepare_transactions,
    segment_portfolio,
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

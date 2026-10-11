"""Phase 7 orchestration: scores + features → ranked retention table + campaign."""

from __future__ import annotations

import pandas as pd

from retention_engine.campaign_generator import build_campaign
from retention_engine.next_best_action import assign_actions
from retention_engine.priority_scoring import (
    normalize_value,
    priority_score,
    revenue_at_risk,
)
from retention_engine.strategy_mapper import StrategyMatrix, load_matrix, validate_matrix
from retention_engine.value_tiers import assign_value_tiers, expected_value_proxy
from utils.exceptions import InsufficientDataError


def build_retention_table(
    churn_scores: pd.DataFrame,
    features: pd.DataFrame,
    matrix: StrategyMatrix,
    value_basis: str = "monetary_value",
    strategic_weights: dict[str, float] | float = 1.0,
    value_quantiles: int = 3,
) -> pd.DataFrame:
    """Merge value, risk, priority, and traceable actions into one ranked table."""
    if churn_scores.empty or features.empty:
        raise InsufficientDataError("Retention needs churn scores and customer features.")
    if value_basis not in features.columns:
        raise KeyError(f"Value basis '{value_basis}' is not in the customer table.")
    base = churn_scores[["customer_id", "churn_probability", "risk_band"]].merge(
        features[["customer_id", value_basis]], on="customer_id", how="inner"
    )
    if base.empty:
        raise InsufficientDataError("No customers overlap between scores and features.")
    table = base.copy()
    table["customer_value_tier"] = assign_value_tiers(
        table[value_basis], n_quantiles=value_quantiles
    ).tolist()
    table["expected_value_proxy_historical"] = expected_value_proxy(table[value_basis])
    table["revenue_at_risk"] = revenue_at_risk(table["churn_probability"], table[value_basis])
    normed = normalize_value(table[value_basis])
    if isinstance(strategic_weights, dict):
        seg_map: dict = {}
        if "segment" in features.columns:
            seg_map = features.set_index("customer_id")["segment"].to_dict()
        weight = base["customer_id"].map(
            lambda cid: float(strategic_weights.get(seg_map.get(cid, ""), 1.0))
        )
    else:
        weight = float(strategic_weights)
    table["retention_priority_score"] = priority_score(normed, table["churn_probability"], weight)
    actions = assign_actions(table, matrix)
    table = table.merge(actions, on="customer_id", how="left")
    return table.sort_values("retention_priority_score", ascending=False).reset_index(drop=True)


def retention_summary(table: pd.DataFrame) -> dict:
    """Totals for KPI cards: revenue at risk, priority counts, strategy mix."""
    return {
        "customers": len(table),
        "total_revenue_at_risk": round(float(table["revenue_at_risk"].sum()), 2),
        "priority_customers": int((table["retention_priority_score"] > 0).sum()),
        "strategy_mix": table["recommended_action"].value_counts().to_dict(),
        "band_mix": table["risk_band"].value_counts().to_dict(),
    }


def default_matrix_report(
    matrix_path: str = "retention_strategies.yaml",
    tiers: tuple[str, ...] = ("Low", "Medium", "High"),
    bands: tuple[str, ...] = ("low", "medium", "high", "critical"),
) -> tuple[StrategyMatrix, dict]:
    """Load the YAML matrix and prove full tier×band coverage."""
    matrix = load_matrix(matrix_path)
    return matrix, validate_matrix(matrix, list(tiers), list(bands))


def export_campaign(
    table: pd.DataFrame, top_n: int = 200, min_priority: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Campaign rows + channel summary for the ranked retention table."""
    return build_campaign(table, top_n=top_n, min_priority=min_priority)

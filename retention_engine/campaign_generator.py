"""Campaign-ready export: rank by priority, cut to budget, group by channel."""

from __future__ import annotations

import pandas as pd


def build_campaign(
    actions: pd.DataFrame,
    top_n: int = 200,
    min_priority: float = 0.0,
    priority_col: str = "retention_priority_score",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (campaign rows, channel summary) for the highest-priority customers.

    Required action columns: customer_id, recommended_action, recommended_channel,
    suggested_offer_type, action_reason, strategy_rule (+ priority_col, revenue_at_risk
    when available for value-sorted export).
    """
    required = {
        "customer_id",
        "recommended_action",
        "recommended_channel",
        "suggested_offer_type",
        "action_reason",
        "strategy_rule",
    }
    missing = required - set(actions.columns)
    if missing:
        raise KeyError(f"Campaign export is missing column(s): {sorted(missing)}.")
    pool = actions.copy()
    if priority_col in pool.columns:
        pool = pool[pool[priority_col] >= min_priority]
        pool = pool.sort_values(priority_col, ascending=False)
    campaign = pool.head(max(top_n, 0)).reset_index(drop=True)
    grouped = campaign.groupby("recommended_channel")
    summary = grouped.agg(customers=("customer_id", "size")).reset_index()
    if "revenue_at_risk" in campaign.columns:
        revenue = grouped["revenue_at_risk"].sum().round(2).reset_index()
        summary = summary.merge(revenue, on="recommended_channel")
    return campaign, summary

"""Next-best-action assignment: value tier × risk band → traceable recommendation.

Every output row carries the matched rule name and a fully rendered reason, so
any recommendation can be traced back to its rule and audited.
"""

from __future__ import annotations

import pandas as pd

from retention_engine.strategy_mapper import StrategyMatrix


def assign_actions(
    customers: pd.DataFrame,
    matrix: StrategyMatrix,
    value_col: str = "expected_value_proxy_historical",
) -> pd.DataFrame:
    """Attach recommended_action/channel/offer/reason/rule to each customer row.

    Required columns: customer_id, customer_value_tier, risk_band, churn_probability,
    plus value_col for reason rendering.
    """
    required = {"customer_id", "customer_value_tier", "risk_band", "churn_probability"}
    missing = required - set(customers.columns)
    if missing:
        raise KeyError(f"Action assignment is missing column(s): {sorted(missing)}.")
    rows: list[dict] = []
    for _, row in customers.iterrows():
        rule = matrix.match(str(row["customer_value_tier"]), str(row["risk_band"]))
        value = row[value_col] if value_col in customers.columns else "n/a"
        try:
            rendered = rule.reason.format(
                value=value,
                band=row["risk_band"],
                probability=round(float(row["churn_probability"]), 3),
            )
        except (KeyError, IndexError, ValueError):
            rendered = rule.reason
        rows.append(
            {
                "customer_id": row["customer_id"],
                "recommended_action": rule.action,
                "recommended_channel": rule.channel,
                "suggested_offer_type": rule.offer_type,
                "action_reason": rendered,
                "strategy_rule": rule.name,
            }
        )
    return pd.DataFrame(rows)

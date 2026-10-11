"""Revenue-at-risk and retention-priority scoring.

Revenue at Risk = churn_probability × selected customer value.
Priority = min-max normalized value × churn_probability × strategic weight.
Weights default to 1.0 (config strategic_weight_default) and may vary per segment.
"""

from __future__ import annotations

import pandas as pd


def normalize_value(values: pd.Series) -> pd.Series:
    """Min-max to [0, 1]. Zero range (all equal) maps to 1.0: equally valuable."""
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    span = numeric.max() - numeric.min()
    if span == 0:
        return pd.Series(1.0, index=numeric.index)
    return ((numeric - numeric.min()) / span).round(4)


def revenue_at_risk(probability: pd.Series, value: pd.Series) -> pd.Series:
    """Expected revenue exposed per customer: p(churn) × value basis."""
    return (
        (
            pd.to_numeric(probability, errors="coerce").fillna(0.0)
            * pd.to_numeric(value, errors="coerce").fillna(0.0)
        )
        .round(2)
        .rename("revenue_at_risk")
    )


def priority_score(
    normalized_value: pd.Series,
    probability: pd.Series,
    weight: pd.Series | float = 1.0,
) -> pd.Series:
    """Priority in [0, ~max weight]: normalized value × churn probability × weight."""
    if isinstance(weight, (int, float)):
        weight = pd.Series(float(weight), index=normalized_value.index)
    return (
        (
            pd.to_numeric(normalized_value, errors="coerce").fillna(0.0)
            * pd.to_numeric(probability, errors="coerce").fillna(0.0)
            * pd.to_numeric(weight, errors="coerce").fillna(1.0)
        )
        .round(4)
        .rename("retention_priority_score")
    )

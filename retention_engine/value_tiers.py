"""Customer value tiers and the historical-value proxy.

Tiers come from quantiles over historical monetary value (config value_quantiles,
default 3 → Low/Medium/High). The proxy is labeled historical everywhere it
surfaces: it is past spend, never predicted future CLV.
"""

from __future__ import annotations

import pandas as pd

TIER_LABELS = ("Low", "Medium", "High")


def assign_value_tiers(monetary: pd.Series, n_quantiles: int = 3) -> pd.Series:
    """Quantile tiers over historical monetary value (duplicates collapse tiers)."""
    if n_quantiles < 2:
        raise ValueError("Need at least 2 value quantiles.")
    labels = (
        list(TIER_LABELS[:n_quantiles])
        if n_quantiles <= 3
        else [f"Q{i + 1}" for i in range(n_quantiles)]
    )
    try:
        tiers = pd.qcut(monetary, q=n_quantiles, labels=labels, duplicates="drop")
    except ValueError:
        tiers = pd.Series([labels[len(labels) // 2]] * len(monetary), index=monetary.index)
    observed = [str(v) for v in tiers.dropna().unique()]
    collapsed = tiers.astype(str)
    if len(observed) < n_quantiles:
        collapsed = collapsed.where(
            collapsed.isin(observed), other=observed[len(observed) // 2] if observed else "Medium"
        )
    return collapsed.rename("customer_value_tier")


def expected_value_proxy(monetary: pd.Series) -> pd.Series:
    """Historical monetary value as the value proxy — explicitly NOT predicted CLV."""
    return (
        pd.to_numeric(monetary, errors="coerce")
        .fillna(0.0)
        .rename("expected_value_proxy_historical")
    )

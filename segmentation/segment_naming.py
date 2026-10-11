"""Post-hoc business naming: profile first, label second.

Clusters are ranked by value (median monetary percentile rank) and activity
(inverted recency rank averaged with frequency rank), split into tertiles, and
mapped onto a value × activity grid. Collisions get letter suffixes. Cluster IDs
and business names remain separate fields downstream.
"""

from __future__ import annotations

import pandas as pd

GRID = {
    ("Premium", "Active"): "Premium Active",
    ("Premium", "Steady"): "Premium Steady",
    ("Premium", "Dormant"): "Premium Dormant",
    ("Core", "Active"): "Core Active",
    ("Core", "Steady"): "Core Steady",
    ("Core", "Dormant"): "Core Dormant",
    ("Light", "Active"): "Light Active",
    ("Light", "Steady"): "Light Steady",
    ("Light", "Dormant"): "Light Dormant",
}


def _tertile(rank: float) -> str:
    if rank >= 2 / 3:
        return "high"
    if rank >= 1 / 3:
        return "mid"
    return "low"


def propose_names(
    profile: pd.DataFrame,
    monetary_col: str = "median_monetary_value",
    recency_col: str = "median_recency_days",
    frequency_col: str = "median_frequency",
) -> pd.DataFrame:
    """Map each non-noise cluster to a grid name. Returns id/name/tier table."""
    work = profile[~profile["is_noise"]].copy() if "is_noise" in profile else profile.copy()
    n = len(work)
    value_rank = work[monetary_col].rank(pct=True) if monetary_col in work else 0.5
    rec_rank = 1 - work[recency_col].rank(pct=True) if recency_col in work else 0.5
    freq_rank = work[frequency_col].rank(pct=True) if frequency_col in work else 0.5
    activity = (rec_rank + freq_rank) / 2 if n > 1 else 0.5
    value_tier = value_rank.map(
        lambda r: {"high": "Premium", "mid": "Core", "low": "Light"}[_tertile(r)]
    )
    activity_tier = activity.map(
        lambda r: {"high": "Active", "mid": "Steady", "low": "Dormant"}[_tertile(r)]
    )
    names = [GRID[(v, a)] for v, a in zip(value_tier, activity_tier, strict=True)]
    seen: dict[str, int] = {}
    final: list[str] = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        final.append(
            name if seen[name] == 1 else f"{name} {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[seen[name] - 2]}"
        )
    return pd.DataFrame(
        {
            "cluster_id": work["cluster_id"].tolist(),
            "business_name": final,
            "value_tier": value_tier.tolist(),
            "activity_tier": activity_tier.tolist(),
        }
    )

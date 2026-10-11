"""Per-cluster profiling over original (unscaled) customer features.

Cluster IDs stay numeric here; business names are assigned separately in
segment_naming.py only after these profiles are reviewed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from segmentation.common import NOISE_LABEL
from utils.validators import require_min_rows


def build_cluster_profile(
    original: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
    monetary_col: str = "monetary_value",
    id_col: str = "customer_id",
) -> pd.DataFrame:
    """Aggregate count, share, medians, means, and revenue per cluster (noise kept)."""
    require_min_rows(original, 1, "customers")
    frame = original.copy()
    frame["_cluster_id"] = np.asarray(labels, dtype=int)
    rows: list[dict] = []
    for cluster_id, group in frame.groupby("_cluster_id"):
        entry: dict = {
            "cluster_id": int(cluster_id),
            "is_noise": bool(cluster_id == NOISE_LABEL),
            "customers": len(group),
            "pct": round(len(group) / len(frame) * 100, 2),
        }
        for col in feature_cols:
            numeric = pd.to_numeric(group[col], errors="coerce")
            entry[f"median_{col}"] = round(float(numeric.median()), 3)
            entry[f"mean_{col}"] = round(float(numeric.mean()), 3)
        if monetary_col in frame.columns:
            revenue = float(pd.to_numeric(group[monetary_col], errors="coerce").sum())
            entry["revenue"] = round(revenue, 2)
        rows.append(entry)
    profile = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    if "revenue" in profile.columns and profile["revenue"].sum():
        profile["revenue_pct"] = (profile["revenue"] / profile["revenue"].sum() * 100).round(2)
    if id_col in frame.columns:
        members = frame.groupby("_cluster_id")[id_col].apply(list).to_dict()
        profile["member_ids"] = profile["cluster_id"].map({int(k): v for k, v in members.items()})
    return profile

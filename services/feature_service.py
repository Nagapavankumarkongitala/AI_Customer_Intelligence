"""Phase 2 orchestration: customer features -> RFM/LRFM runs with validation."""

from __future__ import annotations

import pandas as pd

from analytics.customer_features import CustomerFeatureResult, build_customer_features
from analytics.lrfm import score_lrfm
from analytics.rfm import VALID_QUANTILES, score_rfm
from utils.validators import require_min_rows


def compute_customer_features(
    clean_df: pd.DataFrame,
    snapshot_date: str | pd.Timestamp | None = None,
    recent_days: int = 90,
    snapshot_offset_days: int = 1,
) -> CustomerFeatureResult:
    """Validate the cleaned canonical frame and build the customer feature table."""
    require_min_rows(clean_df, 1, "cleaned transactions")
    for col in ("customer_id", "transaction_date", "amount_spent"):
        if col not in clean_df.columns:
            raise KeyError(f"Clean data is missing required column: {col}")
    if recent_days < 1:
        raise ValueError("Recent window must be at least 1 day.")
    return build_customer_features(
        clean_df,
        snapshot_date=snapshot_date,
        recent_days=recent_days,
        snapshot_offset_days=snapshot_offset_days,
    )


def run_rfm(features: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Score RFM segments for a customer feature table."""
    if quantiles not in VALID_QUANTILES:
        raise ValueError(f"quantiles must be one of {VALID_QUANTILES}, got {quantiles}.")
    return score_rfm(features, quantiles=quantiles)


def run_lrfm(features: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Score LRFM segments for a customer feature table."""
    if quantiles not in VALID_QUANTILES:
        raise ValueError(f"quantiles must be one of {VALID_QUANTILES}, got {quantiles}.")
    return score_lrfm(features, quantiles=quantiles)


def segment_profile(scored: pd.DataFrame, monetary_col: str = "monetary_value") -> pd.DataFrame:
    """Aggregate segment sizes, revenue share, and mean raw metrics for display."""
    require_min_rows(scored, 1, "scored customers")
    profile = (
        scored.groupby("segment")
        .agg(
            customers=("segment", "size"),
            revenue=(monetary_col, "sum"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=(monetary_col, "mean"),
        )
        .reset_index()
    )
    profile["customer_pct"] = (profile["customers"] / profile["customers"].sum() * 100).round(2)
    total_revenue = profile["revenue"].sum()
    profile["revenue_pct"] = (
        (profile["revenue"] / total_revenue * 100).round(2) if total_revenue else 0.0
    )
    return profile.sort_values("customers", ascending=False).reset_index(drop=True)

"""Phase 2 orchestration: customer features -> RFM/LRFM runs with validation."""

from __future__ import annotations

from typing import Final

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


PERIOD_GRANULARITIES: Final = {"quarterly": 8, "monthly": 12}


def period_end_dates(max_date: pd.Timestamp, granularity: str, periods: int) -> list[pd.Timestamp]:
    """Period-end snapshots at or before the data edge, oldest first.

    Quarterly uses calendar quarter ends, monthly uses month ends. Partial
    trailing periods are excluded: every snapshot is fully observed data.
    """
    if granularity == "quarterly":
        ends = pd.date_range(end=max_date, periods=periods, freq="QE")
    elif granularity == "monthly":
        ends = pd.date_range(end=max_date, periods=periods, freq="ME")
    else:
        raise ValueError(f"granularity must be quarterly or monthly, got {granularity!r}.")
    return [ts for ts in ends if ts.date() <= max_date.date()]


def period_label(end: pd.Timestamp, granularity: str) -> str:
    """Sortable period label (alphabetical order is chronological)."""
    if granularity == "quarterly":
        return f"{end.year}-Q{end.quarter}"
    return end.strftime("%Y-%m")


def run_segment_history(
    clean_df: pd.DataFrame, granularity: str = "quarterly", quantiles: int = 5
) -> pd.DataFrame:
    """Score RFM/LRFM at successive period ends: period × model × segment counts.

    Each period sees only transactions on or before its snapshot date, so
    history can never leak future behavior into past segments. Periods too
    small to score are skipped. Returns long-frame columns
    (period, model, segment, customers) with chronological period order.
    """
    if quantiles not in VALID_QUANTILES:
        raise ValueError(f"quantiles must be one of {VALID_QUANTILES}, got {quantiles}.")
    if granularity not in PERIOD_GRANULARITIES:
        raise ValueError(f"granularity must be quarterly or monthly, got {granularity!r}.")
    txn_dates = pd.to_datetime(clean_df["transaction_date"], errors="coerce")
    max_date = txn_dates.max()
    snapshots = period_end_dates(max_date, granularity, PERIOD_GRANULARITIES[granularity])
    rows: list[dict] = []
    for snap in snapshots:
        frame = clean_df[txn_dates <= snap].copy()
        if frame.empty:
            continue
        try:
            features = compute_customer_features(frame, snapshot_date=snap).features
            if len(features) < quantiles:
                continue
            label = period_label(snap, granularity)
            for model, scored in (
                ("RFM", run_rfm(features, quantiles=quantiles)),
                ("LRFM", run_lrfm(features, quantiles=quantiles)),
            ):
                counts = scored["segment"].value_counts()
                for segment, customers in counts.items():
                    rows.append(
                        {
                            "period": label,
                            "model": model,
                            "segment": segment,
                            "customers": int(customers),
                        }
                    )
        except (ValueError, KeyError):
            continue
    history = pd.DataFrame(rows, columns=["period", "model", "segment", "customers"])
    order = [period_label(s, granularity) for s in snapshots]
    history["period"] = pd.Categorical(history["period"], categories=order, ordered=True)
    return history.sort_values(["period", "model", "segment"]).reset_index(drop=True)

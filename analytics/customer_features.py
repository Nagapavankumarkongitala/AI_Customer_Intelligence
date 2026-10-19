"""Customer-level feature engineering from cleaned canonical transactions.

Snapshot default: one day after the last valid transaction (configurable offset).
Demographics/geography: latest non-null observed value per customer (never arbitrary
first); customers with conflicting values are listed in the conflicts report.

Implementation: fully vectorized groupby aggregations (no per-customer Python
loop). Output values, column order, and row order (customer first-appearance)
match the legacy row loop exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from utils.validators import require_min_rows

DEMOGRAPHIC_FIELDS = ("gender", "age", "city", "state", "country", "region")


@dataclass
class CustomerFeatureResult:
    """Feature table plus data-quality side reports."""

    features: pd.DataFrame
    demographic_conflicts: dict[str, dict[str, list]] = field(default_factory=dict)
    dropped_invalid_dates: int = 0


def _latest_valid_values(ordered: pd.DataFrame, column: str) -> tuple[pd.Series, dict[str, list]]:
    """Latest non-blank value per customer for one demographic column.

    ``ordered`` must be date-ordered within each customer (ties keep input
    order). Returns (latest value, conflicts for multi-value customers only).
    """
    series = ordered[column]
    not_null = series.notna()
    stripped = series[not_null].astype("string").str.strip()
    valid = not_null & (stripped != "").reindex(ordered.index, fill_value=False)
    picked = (
        ordered.loc[valid]
        .groupby("customer_id", sort=False)
        .tail(1)
        .set_index("customer_id")[column]
    )
    sub = ordered.loc[valid]
    distinct_counts = (
        sub[column].astype(str).str.strip().groupby(sub["customer_id"], sort=False).nunique()
    )
    conflicts: dict[str, list] = {}
    multi = distinct_counts[distinct_counts > 1]
    if not multi.empty:
        holders = ordered.loc[valid & ordered["customer_id"].isin(set(multi.index))]
        for customer_id, group in holders.groupby("customer_id", sort=False):
            distinct = sorted(set(group[column].astype(str).str.strip()))
            conflicts[str(customer_id)] = [str(value) for value in distinct]
    return picked, conflicts


def build_customer_features(
    clean_df: pd.DataFrame,
    snapshot_date: str | pd.Timestamp | None = None,
    recent_days: int = 90,
    snapshot_offset_days: int = 1,
) -> CustomerFeatureResult:
    """Aggregate one row per customer with behavioral, value, and trend features."""
    require_min_rows(clean_df, 1, "cleaned transactions")
    for col in ("customer_id", "transaction_date", "amount_spent"):
        if col not in clean_df.columns:
            raise KeyError(f"Clean data is missing required column: {col}")

    df = clean_df.copy()
    df["_txn_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    dropped = int(df["_txn_date"].isna().sum())
    df = df.dropna(subset=["_txn_date"]).reset_index(drop=True)
    require_min_rows(df, 1, "transactions with valid dates")
    df["_amount"] = pd.to_numeric(df["amount_spent"], errors="coerce").fillna(0.0)

    max_txn = df["_txn_date"].max()
    snapshot = (
        pd.Timestamp(snapshot_date)
        if snapshot_date is not None
        else max_txn + pd.Timedelta(days=snapshot_offset_days)
    )
    recent_start = snapshot - pd.Timedelta(days=recent_days)
    previous_start = snapshot - pd.Timedelta(days=2 * recent_days)

    df["_year_month"] = df["_txn_date"].dt.to_period("M").astype(str)
    order = list(df["customer_id"].drop_duplicates())
    ordered = df.sort_values(["customer_id", "_txn_date"], kind="mergesort").reset_index(drop=True)
    key = ordered["customer_id"]

    first = ordered.groupby(key, sort=False)["_txn_date"].min()
    last = ordered.groupby(key, sort=False)["_txn_date"].max()
    frequency = ordered.groupby(key, sort=False).size()
    monetary = ordered.groupby(key, sort=False)["_amount"].sum()
    active_months = ordered.groupby(key, sort=False)["_year_month"].nunique()

    ordered["_gap_days"] = (
        ordered.groupby(key, sort=False)["_txn_date"].diff().dt.total_seconds() / 86400.0
    )
    gap_stats = ordered.groupby(key, sort=False)["_gap_days"].agg(["mean", "std"])

    recent_mask = ordered["_txn_date"] > recent_start
    previous_mask = (ordered["_txn_date"] > previous_start) & (ordered["_txn_date"] <= recent_start)
    recent_freq = recent_mask.groupby(key, sort=False).sum()
    previous_freq = previous_mask.groupby(key, sort=False).sum()
    recent_monetary = ordered["_amount"].where(recent_mask, 0.0).groupby(key, sort=False).sum()
    previous_monetary = ordered["_amount"].where(previous_mask, 0.0).groupby(key, sort=False).sum()

    frame = pd.DataFrame(
        {
            "first_transaction_date": first.dt.date.astype(str),
            "last_transaction_date": last.dt.date.astype(str),
            "recency_days": (snapshot - last).dt.days,
            "frequency": frequency,
            "monetary_value": monetary.round(2),
            "relationship_length_days": (last - first).dt.days,
            "average_order_value": (monetary / frequency).round(2),
            "active_months": active_months,
            "transactions_per_active_month": (frequency / active_months).round(3),
            "average_interpurchase_days": gap_stats["mean"],
            "interpurchase_std_days": gap_stats["std"],
            "recent_period_frequency": recent_freq,
            "previous_period_frequency": previous_freq,
            "frequency_change": recent_freq - previous_freq,
            "recent_period_monetary": recent_monetary.round(2),
            "previous_period_monetary": previous_monetary.round(2),
            "monetary_change": (recent_monetary - previous_monetary).round(2),
        }
    )

    conflicts: dict[str, dict[str, list]] = {}
    for col in DEMOGRAPHIC_FIELDS:
        if col not in ordered.columns:
            frame[col] = None
            continue
        picked, col_conflicts = _latest_valid_values(ordered, col)
        # Legacy contract: complete columns keep their native dtype (so numeric
        # demographics stay model-eligible); columns with gaps become object
        # with None — never pandas-NA, which sklearn imputers reject.
        aligned = picked.reindex(frame.index)
        if aligned.isna().any():
            aligned = aligned.astype(object).where(aligned.notna(), None)
        frame[col] = aligned
        for customer_id, distinct in col_conflicts.items():
            conflicts.setdefault(customer_id, {})[col] = distinct

    frame = frame.reindex(order)
    frame.insert(0, "customer_id", frame.index)
    frame.index = pd.RangeIndex(len(frame))
    return CustomerFeatureResult(
        features=frame,
        demographic_conflicts=conflicts,
        dropped_invalid_dates=dropped,
    )

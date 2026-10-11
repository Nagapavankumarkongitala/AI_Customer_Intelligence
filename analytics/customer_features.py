"""Customer-level feature engineering from cleaned canonical transactions.

Snapshot default: one day after the last valid transaction (configurable offset).
Demographics/geography: latest non-null observed value per customer (never arbitrary
first); customers with conflicting values are listed in the conflicts report.
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
    rows: list[dict] = []
    conflicts: dict[str, dict[str, list]] = {}
    for customer_id, group in df.groupby("customer_id", sort=False):
        group = group.sort_values("_txn_date")
        dates = group["_txn_date"]
        amounts = group["_amount"]
        first, last = dates.iloc[0], dates.iloc[-1]
        frequency = len(group)
        monetary = float(amounts.sum())
        relationship = (last - first).days
        if frequency >= 2:
            gaps = dates.diff().dt.total_seconds().dropna() / 86400.0
            avg_gap = float(gaps.mean())
            std_gap = float(gaps.std()) if frequency >= 3 else float("nan")
        else:
            avg_gap, std_gap = float("nan"), float("nan")
        recent_mask = dates > recent_start
        previous_mask = (dates > previous_start) & (dates <= recent_start)
        recent_freq = int(recent_mask.sum())
        previous_freq = int(previous_mask.sum())
        recent_monetary = float(amounts[recent_mask].sum())
        previous_monetary = float(amounts[previous_mask].sum())
        active_months = int(group["_year_month"].nunique())

        row: dict = {
            "customer_id": customer_id,
            "first_transaction_date": first.date().isoformat(),
            "last_transaction_date": last.date().isoformat(),
            "recency_days": (snapshot - last).days,
            "frequency": frequency,
            "monetary_value": round(monetary, 2),
            "relationship_length_days": relationship,
            "average_order_value": round(monetary / frequency, 2),
            "active_months": active_months,
            "transactions_per_active_month": round(frequency / active_months, 3),
            "average_interpurchase_days": avg_gap,
            "interpurchase_std_days": std_gap,
            "recent_period_frequency": recent_freq,
            "previous_period_frequency": previous_freq,
            "frequency_change": recent_freq - previous_freq,
            "recent_period_monetary": round(recent_monetary, 2),
            "previous_period_monetary": round(previous_monetary, 2),
            "monetary_change": round(recent_monetary - previous_monetary, 2),
        }
        customer_conflicts: dict[str, list] = {}
        for col in DEMOGRAPHIC_FIELDS:
            if col not in group:
                row[col] = None
                continue
            observed = group[col].dropna()
            observed = observed[observed.astype("string").str.strip() != ""]
            if observed.empty:
                row[col] = None
            else:
                row[col] = observed.iloc[-1]
                distinct = sorted(str(v) for v in observed.astype(str).str.strip().unique())
                if len(distinct) > 1:
                    customer_conflicts[col] = distinct
        if customer_conflicts:
            conflicts[str(customer_id)] = customer_conflicts
        rows.append(row)

    features = pd.DataFrame(rows)
    return CustomerFeatureResult(
        features=features,
        demographic_conflicts=conflicts,
        dropped_invalid_dates=dropped,
    )

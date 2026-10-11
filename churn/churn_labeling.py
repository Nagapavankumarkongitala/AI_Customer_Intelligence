"""Time-aware churn labeling without target leakage.

Label rule: churn = 1 when a customer has NO transaction in (cutoff, cutoff + horizon],
0 otherwise. Predictors may only use transactions on or before cutoff. The outcome
window must exist in the data: fully observed is ideal, partially observed warns,
absent raises.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from utils.exceptions import ChurnLeakageError


@dataclass
class LabelingResult:
    """Per-customer labels plus build metadata."""

    labels: pd.DataFrame  # customer_id, churn
    cutoff: pd.Timestamp
    horizon_days: int
    outcome_end: pd.Timestamp
    observed_outcome_days: int
    partial_window: bool
    excluded_short_history: int


def validate_outcome_window(
    max_transaction_date: pd.Timestamp, cutoff: pd.Timestamp, horizon_days: int
) -> tuple[pd.Timestamp, int, bool]:
    """Check outcome coverage. Raises when no outcome data exists past cutoff."""
    outcome_end = cutoff + pd.Timedelta(days=horizon_days)
    if max_transaction_date <= cutoff:
        raise ChurnLeakageError(
            "The cutoff leaves no outcome window: the latest transaction "
            f"({max_transaction_date.date()}) is on or before the cutoff "
            f"({cutoff.date()}). Move the cutoff earlier or extend the data."
        )
    observed = max(0, (max_transaction_date - cutoff).days)
    return outcome_end, observed, observed < horizon_days


def label_churn(
    clean_df: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    horizon_days: int = 90,
    min_history_days: int = 30,
    observation_start: str | pd.Timestamp | None = None,
) -> LabelingResult:
    """Label customers from post-cutoff activity; predictors must be built ≤ cutoff."""
    cutoff = pd.Timestamp(cutoff)
    if horizon_days < 1:
        raise ValueError("Outcome horizon must be at least 1 day.")
    work = clean_df.copy()
    work["_txn_date"] = pd.to_datetime(work["transaction_date"], errors="coerce")
    work = work.dropna(subset=["_txn_date"])
    if observation_start is not None:
        work = work[work["_txn_date"] >= pd.Timestamp(observation_start)]
    if work.empty:
        raise ChurnLeakageError("No transactions remain in the observation window.")

    max_txn = work["_txn_date"].max()
    outcome_end, observed_days, partial = validate_outcome_window(max_txn, cutoff, horizon_days)

    first_txn = work.groupby("customer_id")["_txn_date"].min()
    eligible = first_txn[(cutoff - first_txn).dt.days >= min_history_days].index
    excluded = int(first_txn.shape[0] - len(eligible))

    post = work[(work["_txn_date"] > cutoff) & (work["_txn_date"] <= outcome_end)]
    active = set(post.loc[post["customer_id"].isin(eligible), "customer_id"])
    labels = pd.DataFrame(
        {
            "customer_id": list(eligible),
            "churn": [0 if cid in active else 1 for cid in eligible],
        }
    )
    return LabelingResult(
        labels=labels,
        cutoff=cutoff,
        horizon_days=horizon_days,
        outcome_end=outcome_end,
        observed_outcome_days=observed_days,
        partial_window=partial,
        excluded_short_history=excluded,
    )

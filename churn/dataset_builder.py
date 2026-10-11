"""Churn dataset builder: leakage-safe predictors + labels in one audited frame.

Predictors reuse the customer feature pipeline on transactions ≤ cutoff with
snapshot = cutoff, so no post-cutoff information can leak. Categorical fields
ride along for one-hot encoding in the model pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analytics.customer_features import build_customer_features
from churn.churn_labeling import LabelingResult, label_churn

CATEGORICAL_FEATURES = ("gender", "city", "state", "country", "region")


@dataclass
class ChurnDataset:
    """Modeling frame plus the full leakage-audit trail."""

    frame: pd.DataFrame
    feature_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    labeling: LabelingResult | None = None
    predictor_max_date: pd.Timestamp | None = None
    leakage_ok: bool = False


def build_churn_dataset(
    clean_df: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    horizon_days: int = 90,
    min_history_days: int = 30,
    observation_start: str | pd.Timestamp | None = None,
    recent_days: int = 90,
    snapshot_offset_days: int = 0,
) -> ChurnDataset:
    """Build predictors (≤ cutoff) and labels (post-cutoff) with a leakage proof."""
    labeling = label_churn(
        clean_df,
        cutoff,
        horizon_days,
        min_history_days,
        observation_start,
    )
    cutoff = labeling.cutoff
    predictors = clean_df.copy()
    predictors["_txn_date"] = pd.to_datetime(predictors["transaction_date"], errors="coerce")
    predictors = predictors[predictors["_txn_date"] <= cutoff]
    predictor_max = predictors["_txn_date"].max()

    result = build_customer_features(
        predictors.drop(columns=["_txn_date"]),
        snapshot_date=cutoff,
        recent_days=recent_days,
        snapshot_offset_days=snapshot_offset_days,
    )
    features = result.features
    features = features[features["customer_id"].isin(set(labeling.labels["customer_id"]))]
    frame = features.merge(labeling.labels, on="customer_id", how="inner")

    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c in frame.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in frame.columns]
    return ChurnDataset(
        frame=frame.reset_index(drop=True),
        feature_cols=numeric_cols,
        categorical_cols=categorical_cols,
        labeling=labeling,
        predictor_max_date=predictor_max,
        leakage_ok=bool(predictor_max <= cutoff),
    )

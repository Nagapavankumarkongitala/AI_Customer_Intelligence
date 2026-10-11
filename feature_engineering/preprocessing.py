"""Reusable numeric preprocessing pipeline: impute -> skew-triggered log1p -> scale.

Variance and correlation review are reported (only near-zero-variance columns are
dropped — they cannot carry cluster signal). The single fitted pipeline (impute,
log choice, scaler) is returned for persistence so scoring replays identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from feature_engineering.scaling import SCALERS, make_scaler
from utils.validators import require_min_rows

CORRELATION_FLAG = 0.9
SKEW_TRIGGER = 1.0


class SkewedLogTransformer(BaseEstimator, TransformerMixin):
    """log1p columns that are nonnegative with skew > 1 (decided at fit time)."""

    def __init__(self, enabled: bool = True, skew_trigger: float = SKEW_TRIGGER):
        self.enabled = enabled
        self.skew_trigger = skew_trigger
        self.columns_: list[int] = []

    def fit(self, X: np.ndarray, y=None) -> SkewedLogTransformer:
        self.columns_ = []
        if self.enabled:
            frame = pd.DataFrame(np.asarray(X, dtype=float))
            for i, col in enumerate(frame.columns):
                values = frame[col].to_numpy()
                if np.isnan(values).all() or (values[~np.isnan(values)] < 0).any():
                    continue
                if float(pd.Series(values).skew()) > self.skew_trigger:
                    self.columns_.append(i)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.asarray(X, dtype=float).copy()
        for i in self.columns_:
            out[:, i] = np.log1p(np.nan_to_num(out[:, i], nan=0.0))
        return out


@dataclass
class PreparedData:
    """Processed matrix plus the fitted pipeline and review report."""

    matrix: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    pipeline: Pipeline | None = None
    dropped_zero_variance: list[str] = field(default_factory=list)
    log1p_applied: list[str] = field(default_factory=list)
    high_correlations: list[dict] = field(default_factory=list)


def build_preprocessing_pipeline(scaler: str = "robust", log1p: bool = True) -> Pipeline:
    """Build the unfitted impute -> skew-log -> scale pipeline."""
    if scaler not in SCALERS:
        raise ValueError(f"Unknown scaler '{scaler}'. Choose from {SCALERS}.")
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("skew_log", SkewedLogTransformer(enabled=log1p)),
            ("scale", make_scaler(scaler)),
        ]
    )


def prepare_features(
    df: pd.DataFrame,
    feature_names: list[str],
    scaler: str = "robust",
    log1p: bool = True,
    variance_threshold: float = 1e-9,
) -> PreparedData:
    """Validate, review, transform, and return the clustering matrix + report."""
    require_min_rows(df, 1, "customers")
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise KeyError(f"Selected feature(s) not in the customer table: {', '.join(missing)}.")
    frame = df[feature_names].apply(pd.to_numeric, errors="coerce")

    variances = frame.var(numeric_only=True).fillna(0.0)
    dropped = [c for c in frame.columns if variances[c] < variance_threshold]
    frame = frame.drop(columns=dropped)
    if frame.empty:
        raise ValueError("No usable features remain after review. Select more features.")

    high_corr: list[dict] = []
    if len(frame.columns) >= 2:
        corr = frame.corr(numeric_only=True).abs()
        for i, col_a in enumerate(corr.columns):
            for col_b in corr.columns[i + 1 :]:
                if corr.loc[col_a, col_b] > CORRELATION_FLAG:
                    high_corr.append(
                        {
                            "feature_a": col_a,
                            "feature_b": col_b,
                            "abs_correlation": round(float(corr.loc[col_a, col_b]), 3),
                        }
                    )

    pipeline = build_preprocessing_pipeline(scaler=scaler, log1p=log1p)
    matrix = pipeline.fit_transform(frame.to_numpy(dtype=float))
    skew_step: SkewedLogTransformer = pipeline.named_steps["skew_log"]
    applied = [frame.columns[i] for i in skew_step.columns_]
    return PreparedData(
        matrix=np.asarray(matrix, dtype=float),
        feature_names=list(frame.columns),
        pipeline=pipeline,
        dropped_zero_variance=dropped,
        log1p_applied=applied,
        high_correlations=high_corr,
    )

"""Churn classification metrics, threshold analysis, and lift/gain tables.

Single-class test slices yield None for ranking metrics instead of exceptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Thresholded + ranking metrics for one evaluated split."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    predicted = (proba >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) > 1
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4) if two_classes else None,
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4) if two_classes else None,
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predicted)), 4),
        "log_loss": round(float(log_loss(y_true, np.clip(proba, 1e-9, 1 - 1e-9))), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
    }


def suggest_threshold(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Max-F1 threshold scanned on a validation split (grid step 0.05)."""
    best = {"threshold": 0.5, "f1": -1.0}
    for threshold in np.arange(0.05, 1.0, 0.05):
        score = f1_score(y_true, (np.asarray(proba) >= threshold).astype(int), zero_division=0)
        if score > best["f1"]:
            best = {"threshold": round(float(threshold), 2), "f1": round(float(score), 4)}
    return best


def calibration_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed rate per probability bin for calibration curves."""
    frame = pd.DataFrame({"p": np.asarray(proba, dtype=float), "y": np.asarray(y_true, dtype=int)})
    frame["bin"] = pd.qcut(
        frame["p"], q=min(n_bins, max(frame["p"].nunique(), 1)), duplicates="drop"
    )
    grouped = frame.groupby("bin", observed=True)
    return pd.DataFrame(
        {
            "mean_predicted": grouped["p"].mean().round(4).tolist(),
            "observed_rate": grouped["y"].mean().round(4).tolist(),
            "count": grouped.size().tolist(),
        }
    )


def lift_gain_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift and cumulative gain for targeting analysis."""
    frame = pd.DataFrame({"p": np.asarray(proba, dtype=float), "y": np.asarray(y_true, dtype=int)})
    frame = frame.sort_values("p", ascending=False).reset_index(drop=True)
    frame["decile"] = (
        pd.qcut(frame.index, q=min(n_bins, len(frame)), labels=False, duplicates="drop") + 1
    )
    total_pos = frame["y"].sum()
    grouped = frame.groupby("decile", observed=True)["y"].agg(["size", "sum"])
    table = grouped.rename(columns={"size": "customers", "sum": "churners"}).reset_index()
    table["churn_rate"] = (table["churners"] / table["customers"]).round(4)
    table["cum_churners"] = table["churners"].cumsum()
    table["cum_gain"] = (table["cum_churners"] / max(total_pos, 1)).round(4)
    base_rate = frame["y"].mean()
    table["lift"] = (table["churn_rate"] / base_rate).round(3) if base_rate else 0.0
    return table

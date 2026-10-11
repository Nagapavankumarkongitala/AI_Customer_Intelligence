"""Model cards: purpose, data, features, metrics, limits, fairness, version."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any


def write_model_card(
    path: str | Path,
    purpose: str,
    training_period: str,
    features: list[str],
    preprocessing: str,
    hyperparameters: dict[str, Any],
    metrics: dict[str, Any],
    limitations: str,
    fairness_notes: str,
    version: str,
) -> Path:
    """Write a Markdown model card. Returns the card path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    card = f"""# Model Card — {version}

Date: {date.today().isoformat()}

## Purpose
{purpose}

## Training data
{training_period}

## Input features
{", ".join(features)}

## Preprocessing
{preprocessing}

## Hyperparameters
{hyperparameters}

## Evaluation metrics
{metrics}

## Limitations
{limitations}

## Fairness considerations
{fairness_notes}
"""
    dest.write_text(card, encoding="utf-8")
    return dest


def segmentation_card_metadata(
    model: str, feature_names: list[str], metrics: dict, params: dict
) -> dict:
    """Assemble card fields for a segmentation run (caller adds period/version)."""
    return {
        "purpose": f"Unsupervised customer segmentation with {model}.",
        "features": feature_names,
        "preprocessing": "Median imputation, skew-triggered log1p, robust scaling.",
        "hyperparameters": params,
        "metrics": {
            k: metrics.get(k)
            for k in (
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "n_clusters",
                "noise_pct",
                "min_cluster_size",
            )
        },
        "limitations": "Clusters describe past behavior; cluster IDs are arbitrary until "
        "profiled; small/noisy clusters need review.",
        "fairness_notes": "Demographics used for profiling only, never as clustering "
        "inputs unless explicitly selected; review segment composition "
        "for disparate impact before targeted action.",
    }


def churn_card_metadata(
    model: str,
    feature_cols: list[str],
    metrics: dict,
    threshold: float,
    cutoff: str,
    horizon_days: int,
) -> dict:
    """Assemble card fields for a churn model (caller adds period/version)."""
    return {
        "purpose": f"Leakage-safe churn classification with {model}.",
        "training_period": f"Predictors ≤ {cutoff}; outcome window {horizon_days}d after cutoff.",
        "features": feature_cols,
        "preprocessing": "Median/most-frequent imputation, one-hot encoding, "
        "scaling for logreg; balanced class weights.",
        "hyperparameters": {"model": model, "threshold": threshold},
        "metrics": {
            k: metrics.get(k)
            for k in (
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "log_loss",
                "brier",
            )
        },
        "limitations": "Predicts inactivity, not intent; performance degrades as behavior "
        "drifts — revalidate on fresh outcome windows.",
        "fairness_notes": "Demographics are optional predictors; check score calibration "
        "across gender/region slices before operational use.",
    }

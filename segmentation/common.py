"""Shared segmentation helpers: optional-dependency guard and label summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.exceptions import CustomerIntelligenceError

NOISE_LABEL = -1


def require_optional(package: str, pip_name: str = "") -> None:
    """Raise a business-readable install-guidance error when an optional dep is absent."""
    try:
        __import__(package)
    except ImportError as exc:
        pip_name = pip_name or package
        raise CustomerIntelligenceError(
            f"This model needs the optional '{pip_name}' package, which is not installed. "
            f"Install it with 'pip install {pip_name}' and run again."
        ) from exc


def describe_labels(labels: np.ndarray) -> dict:
    """Cluster sizes, proportions, noise share, and minimum non-noise cluster size."""
    labels = np.asarray(labels)
    total = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    noise = int(counts[unique == NOISE_LABEL].sum()) if NOISE_LABEL in unique else 0
    non_noise = counts[unique != NOISE_LABEL]
    return {
        "n_clusters": int((unique != NOISE_LABEL).sum()),
        "cluster_ids": [int(u) for u in unique if u != NOISE_LABEL],
        "sizes": {int(u): int(c) for u, c in zip(unique, counts, strict=True) if u != NOISE_LABEL},
        "proportions": {
            int(u): round(float(c) / total, 4)
            for u, c in zip(unique, counts, strict=True)
            if u != NOISE_LABEL
        },
        "noise_count": noise,
        "noise_pct": round(noise / total * 100, 2) if total else 0.0,
        "min_cluster_size": int(non_noise.min()) if len(non_noise) else 0,
    }


def labels_to_series(labels: np.ndarray, index: pd.Index, name: str = "cluster_id") -> pd.Series:
    """Attach integer labels to customer rows."""
    return pd.Series(np.asarray(labels, dtype=int), index=index, name=name)

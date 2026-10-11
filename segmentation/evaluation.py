"""Clustering quality metrics. Noise-aware: scores use non-noise points only.

Silhouette / Davies-Bouldin / Calinski-Harabasz are None when fewer than two
non-noise clusters exist (or fewer points than clusters).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from segmentation.common import NOISE_LABEL, describe_labels


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """Score a labeling; combine internal metrics with size/noise summary."""
    labels = np.asarray(labels, dtype=int)
    X = np.asarray(X, dtype=float)
    summary = describe_labels(labels)
    keep = labels != NOISE_LABEL
    usable = bool(summary["n_clusters"] >= 2 and keep.sum() > summary["n_clusters"])
    metrics: dict = {
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if usable:
        X_use, lab_use = X[keep], labels[keep]
        try:
            metrics["silhouette"] = round(float(silhouette_score(X_use, lab_use)), 4)
        except ValueError:
            metrics["silhouette"] = None
        try:
            metrics["davies_bouldin"] = round(float(davies_bouldin_score(X_use, lab_use)), 4)
        except ValueError:
            metrics["davies_bouldin"] = None
        try:
            metrics["calinski_harabasz"] = round(float(calinski_harabasz_score(X_use, lab_use)), 4)
        except ValueError:
            metrics["calinski_harabasz"] = None
    return {**metrics, **summary}

"""Agglomerative (hierarchical) segmentation with Ward linkage."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

MODEL_TYPE = "k_based"
MODEL_NAME = "hierarchical"


def fit_predict(
    X: np.ndarray, n_clusters: int = 5, random_state: int = 42, **kwargs
) -> tuple[np.ndarray, AgglomerativeClustering]:
    """Fit Ward-linkage agglomerative clustering; random_state kept for uniformity."""
    _ = random_state  # deterministic algorithm; accepted for a uniform interface.
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", **kwargs)
    return np.asarray(model.fit_predict(X), dtype=int), model

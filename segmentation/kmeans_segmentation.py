"""K-Means segmentation. MODEL_TYPE k_based: honors n_clusters."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

MODEL_TYPE = "k_based"
MODEL_NAME = "kmeans"


def fit_predict(
    X: np.ndarray, n_clusters: int = 5, random_state: int = 42, **kwargs
) -> tuple[np.ndarray, KMeans]:
    """Fit K-Means and return integer labels plus the fitted estimator."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, **kwargs)
    return np.asarray(model.fit_predict(X), dtype=int), model

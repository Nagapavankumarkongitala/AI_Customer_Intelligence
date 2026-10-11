"""Gaussian Mixture segmentation (soft clustering via hard assignments)."""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

MODEL_TYPE = "k_based"
MODEL_NAME = "gmm"


def fit_predict(
    X: np.ndarray, n_clusters: int = 5, random_state: int = 42, **kwargs
) -> tuple[np.ndarray, GaussianMixture]:
    """Fit a GMM and return argmax-component labels plus the fitted estimator."""
    model = GaussianMixture(n_components=n_clusters, random_state=random_state, **kwargs)
    return np.asarray(model.fit_predict(X), dtype=int), model

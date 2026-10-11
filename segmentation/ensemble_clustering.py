"""Consensus (ensemble) clustering via co-association.

Base labelings vote pairwise: two points that share a cluster in many base runs
are close in consensus space. Noise (-1) never agrees — not even with other
noise. Final clusters come from average-linkage agglomerative clustering on the
consensus distance. Never a raw mode of arbitrary labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from segmentation import gmm_segmentation, hierarchical_segmentation, kmeans_segmentation

MODEL_TYPE = "k_based"
MODEL_NAME = "ensemble"

BASE_MODELS = (kmeans_segmentation, gmm_segmentation, hierarchical_segmentation)


def consensus_matrix(labelings: list[np.ndarray]) -> np.ndarray:
    """Fraction of base runs placing each pair in the same non-noise cluster."""
    labelings = [np.asarray(lab, dtype=int) for lab in labelings]
    n = len(labelings[0])
    agreement = np.zeros((n, n), dtype=float)
    for lab in labelings:
        same = lab[:, None] == lab[None, :]
        non_noise = (lab[:, None] != -1) & (lab[None, :] != -1)
        agreement += (same & non_noise).astype(float)
    return agreement / max(len(labelings), 1)


def fit_predict(
    X: np.ndarray, n_clusters: int = 5, random_state: int = 42, **kwargs
) -> tuple[np.ndarray, dict]:
    """Build consensus from base models and cut to n_clusters. Returns bundle."""
    labelings = []
    for module in BASE_MODELS:
        labels, _ = module.fit_predict(X, n_clusters=n_clusters, random_state=random_state)
        labelings.append(labels)
    distance = 1.0 - consensus_matrix(labelings)
    model = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average", **kwargs
    )
    labels = np.asarray(model.fit_predict(distance), dtype=int)
    return labels, {"model": model, "base_labelings": labelings}

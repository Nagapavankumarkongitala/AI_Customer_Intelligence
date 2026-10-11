"""HDBSCAN density segmentation. MODEL_TYPE density: finds k itself, flags noise (-1).

Optional dependency: raises a business-readable install message when absent.
"""

from __future__ import annotations

import numpy as np

from segmentation.common import require_optional

MODEL_TYPE = "density"
MODEL_NAME = "hdbscan"


def fit_predict(
    X: np.ndarray, min_cluster_size: int = 15, random_state: int = 42, **kwargs
) -> tuple[np.ndarray, object]:
    """Fit HDBSCAN and return labels (-1 = noise) plus the fitted clusterer."""
    require_optional("hdbscan", "hdbscan")
    import hdbscan

    _ = random_state  # HDBSCAN is deterministic; accepted for a uniform interface.
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, **kwargs)
    return np.asarray(model.fit_predict(X), dtype=int), model

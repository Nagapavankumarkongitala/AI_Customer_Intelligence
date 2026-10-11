"""Autoencoder-embedding segmentation (TensorFlow backend): compress, then cluster.

Trains a dense autoencoder on the scaled features, embeds to the latent space,
and clusters the codes with K-Means. Graceful degradation: every entry point
raises a business-readable install message when TensorFlow is absent.
"""

from __future__ import annotations

import numpy as np

from deep_learning.autoencoder import train_autoencoder
from segmentation.common import NOISE_LABEL
from segmentation.kmeans_segmentation import fit_predict as _kmeans_fit

MODEL_TYPE = "k_based"
MODEL_NAME = "autoencoder"


def fit_predict(
    X: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
    latent_dim: int = 8,
    epochs: int = 100,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Train the autoencoder, embed, and K-Means the latent codes.

    Returns labels plus a bundle dict (encoder, history, latent table, kmeans model).
    """
    result = train_autoencoder(
        X, latent_dim=latent_dim, epochs=epochs, random_state=random_state, **kwargs
    )
    labels, kmeans = _kmeans_fit(result.latent, n_clusters=n_clusters, random_state=random_state)
    _ = NOISE_LABEL  # autoencoder path never emits noise; referenced for interface parity.
    bundle = {
        "encoder": result.encoder,
        "history": result.history,
        "latent": result.latent,
        "kmeans": kmeans,
        "latent_dim": result.latent_dim,
    }
    return labels, bundle

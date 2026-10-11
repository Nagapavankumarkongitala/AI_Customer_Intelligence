"""Latent-space embedding helpers (backend-agnostic: any encoder with .predict)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def embed_with_encoder(encoder: object, X: np.ndarray) -> np.ndarray:
    """Project scaled features through a trained encoder to latent codes."""
    codes = encoder.predict(np.asarray(X, dtype=float), verbose=0)  # type: ignore[union-attr]
    return np.asarray(codes, dtype=float)


def latent_table(codes: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Latent codes as a labeled frame (latent_0, latent_1, ...) for export."""
    return pd.DataFrame(
        np.asarray(codes, dtype=float),
        index=index,
        columns=[f"latent_{i}" for i in range(np.asarray(codes).shape[1])],
    )

"""Latent-space visualization: PCA scatter of autoencoder codes. No Streamlit calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


def latent_scatter(
    codes: np.ndarray, labels: np.ndarray, names: dict[int, str] | None = None
) -> px.Figure:
    """2D PCA of latent codes colored by cluster (business names when given)."""
    coords = PCA(n_components=2, random_state=42).fit_transform(np.asarray(codes, dtype=float))
    frame = pd.DataFrame(coords, columns=["dim_1", "dim_2"])
    frame["cluster"] = [names.get(int(lab), str(lab)) if names else str(lab) for lab in labels]
    return px.scatter(
        frame,
        x="dim_1",
        y="dim_2",
        color="cluster",
        title="Latent space (PCA of autoencoder codes)",
    )

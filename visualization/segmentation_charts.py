"""Plotly figure factories for segmentation pages. No Streamlit calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


def pca_scatter(
    X: np.ndarray, labels: np.ndarray, names: dict[int, str] | None = None
) -> px.Figure:
    """2D PCA of the scaled clustering matrix colored by cluster."""
    coords = PCA(n_components=2, random_state=42).fit_transform(np.asarray(X, dtype=float))
    frame = pd.DataFrame(coords, columns=["pc_1", "pc_2"])
    frame["cluster"] = [names.get(int(lab), str(lab)) if names else str(lab) for lab in labels]
    return px.scatter(
        frame,
        x="pc_1",
        y="pc_2",
        color="cluster",
        title="Clusters (PCA 2D)",
    )


def umap_scatter(
    X: np.ndarray, labels: np.ndarray, names: dict[int, str] | None = None
) -> px.Figure:
    """2D UMAP of the clustering matrix. Raises install guidance without umap-learn."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP view needs the optional 'umap-learn' package. "
            "Install it with 'pip install umap-learn'."
        ) from exc
    coords = umap.UMAP(n_components=2, random_state=42).fit_transform(np.asarray(X, dtype=float))
    frame = pd.DataFrame(coords, columns=["umap_1", "umap_2"])
    frame["cluster"] = [names.get(int(lab), str(lab)) if names else str(lab) for lab in labels]
    return px.scatter(frame, x="umap_1", y="umap_2", color="cluster", title="Clusters (UMAP 2D)")


def cluster_size_bar(profile: pd.DataFrame) -> px.Figure:
    """Customer counts per cluster (business names when present)."""
    data = profile.copy()
    data["label"] = data.get("business_name", data["cluster_id"].astype(str))
    return px.bar(
        data,
        x="label",
        y="customers",
        color="customers",
        color_continuous_scale="Blues",
        title="Customers per cluster",
    )


def profile_heatmap(profile: pd.DataFrame) -> px.Figure:
    """Cluster medians (z-scored per feature) as a heatmap."""
    medians = profile.filter(regex="^median_").copy()
    medians.columns = [c.replace("median_", "") for c in medians.columns]
    standardized = (medians - medians.mean()) / medians.std().replace(0, 1)
    labels = (
        profile["business_name"].tolist()
        if "business_name" in profile
        else profile["cluster_id"].astype(str).tolist()
    )
    return px.imshow(
        standardized.to_numpy(),
        x=list(standardized.columns),
        y=labels,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="Cluster profile (median, z-scored per feature)",
        labels={"x": "Feature", "y": "Cluster", "color": "z"},
    )

"""Plotly figure factories for segmentation pages. No Streamlit calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from visualization.theme import VOLT_DISCRETE, VOLT_SCALE, styled


@styled
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
        color_discrete_sequence=VOLT_DISCRETE,
        title="Clusters (PCA 2D)",
    )


@styled
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
    return px.scatter(
        frame,
        x="umap_1",
        y="umap_2",
        color="cluster",
        color_discrete_sequence=VOLT_DISCRETE,
        title="Clusters (UMAP 2D)",
    )


@styled
def cluster_size_bar(profile: pd.DataFrame) -> px.Figure:
    """Customer counts per cluster (business names when present)."""
    data = profile.copy()
    data["label"] = data.get("business_name", data["cluster_id"].astype(str))
    return px.bar(
        data,
        x="label",
        y="customers",
        color="customers",
        color_continuous_scale=VOLT_SCALE,
        title="Customers per cluster",
    )


@styled
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


@styled
def sankey_mapping(contingency: pd.DataFrame, source_name: str, target_name: str) -> go.Figure:
    """Flow of customers from source segments (left) to target segments (right)."""
    sources = [f"{source_name}: {idx}" for idx in contingency.index]
    targets = [f"{target_name}: {col}" for col in contingency.columns]
    labels = sources + targets
    link_source, link_target, values = [], [], []
    for i in range(len(contingency.index)):
        for j in range(len(contingency.columns)):
            value = float(contingency.iloc[i, j])
            if value > 0:
                link_source.append(i)
                link_target.append(len(sources) + j)
                values.append(value)
    return go.Figure(
        go.Sankey(
            node={"label": labels, "pad": 12, "thickness": 16, "color": "#9DB800"},
            link={
                "source": link_source,
                "target": link_target,
                "value": values,
                "color": "rgba(204,255,0,0.25)",
            },
        )
    )


@styled
def metric_comparison_bar(table: pd.DataFrame, metric: str, title: str) -> px.Figure:
    """One metric across candidate models (NaN metrics are dropped from view)."""
    data = table.dropna(subset=[metric])
    return px.bar(
        data,
        x="model",
        y=metric,
        color="model",
        color_discrete_sequence=VOLT_DISCRETE,
        title=title,
        labels={"model": "Model", metric: metric},
    )

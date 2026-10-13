"""Plotly figure factories for segmentation pages. No Streamlit calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from visualization import tokens
from visualization.theme import PRIMARY_DISCRETE, empty_figure, styled


def _segment_map_or_sequence(values: object) -> dict:
    """Keyword args painting a cluster column with stable hues (pure helper)."""
    mapped = tokens.segment_map(values)
    if mapped:
        return {"color_discrete_map": mapped}
    return {"color_discrete_sequence": PRIMARY_DISCRETE}


@styled
def pca_scatter(
    X: np.ndarray, labels: np.ndarray, names: dict[int, str] | None = None
) -> px.Figure:
    """2D PCA of the scaled clustering matrix colored by cluster."""
    if np.asarray(X).size == 0:
        return empty_figure("Clusters (PCA 2D)")
    coords = PCA(n_components=2, random_state=42).fit_transform(np.asarray(X, dtype=float))
    frame = pd.DataFrame(coords, columns=["pc_1", "pc_2"])
    frame["cluster"] = [names.get(int(lab), str(lab)) if names else str(lab) for lab in labels]
    return px.scatter(
        frame,
        x="pc_1",
        y="pc_2",
        color="cluster",
        **_segment_map_or_sequence(frame["cluster"]),
        title="Clusters (PCA 2D)",
        labels={"pc_1": "PC 1", "pc_2": "PC 2", "cluster": "Cluster"},
    )


@styled
def umap_scatter(
    X: np.ndarray, labels: np.ndarray, names: dict[int, str] | None = None
) -> px.Figure:
    """2D UMAP of the clustering matrix. Raises install guidance without umap-learn."""
    if np.asarray(X).size == 0:
        return empty_figure("Clusters (UMAP 2D)")
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
        **_segment_map_or_sequence(frame["cluster"]),
        title="Clusters (UMAP 2D)",
        labels={"umap_1": "UMAP 1", "umap_2": "UMAP 2", "cluster": "Cluster"},
    )


@styled
def cluster_size_bar(profile: pd.DataFrame) -> px.Figure:
    """Customer counts per cluster (business names when present)."""
    if profile.empty:
        return empty_figure("Customers per cluster")
    data = profile.copy()
    data["label"] = data.get("business_name", data["cluster_id"].astype(str))
    return px.bar(
        data,
        x="label",
        y="customers",
        color="label",
        **_segment_map_or_sequence(data["label"]),
        title="Customers per cluster",
        labels={"label": "Cluster", "customers": "Customers"},
    )


@styled
def profile_heatmap(profile: pd.DataFrame) -> px.Figure:
    """Cluster medians (z-scored per feature) as a heatmap."""
    medians = profile.filter(regex="^median_").copy()
    if profile.empty or medians.empty:
        return empty_figure("Cluster profile (median, z-scored per feature)")
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
    if contingency.empty:
        return empty_figure(f"{source_name} → {target_name} flow")
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
            node={
                "label": labels,
                "pad": 12,
                "thickness": 16,
                "color": tokens.color("secondary_dim"),
            },
            link={
                "source": link_source,
                "target": link_target,
                "value": values,
                "color": tokens.with_alpha(tokens.color("primary"), 0.25),
            },
        )
    )


@styled
def metric_comparison_bar(table: pd.DataFrame, metric: str, title: str) -> px.Figure:
    """One metric across candidate models (NaN metrics are dropped from view)."""
    data = table.dropna(subset=[metric])
    if data.empty:
        return empty_figure(title)
    return px.bar(
        data,
        x="model",
        y=metric,
        color="model",
        color_discrete_sequence=PRIMARY_DISCRETE,
        title=title,
        labels={"model": "Model", metric: metric},
    )

"""Phase 4 orchestration: prepare features, run/sweep models, evaluate, persist."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering.preprocessing import PreparedData, prepare_features
from segmentation import (
    autoencoder_model,
    ensemble_clustering,
    gmm_segmentation,
    hdbscan_segmentation,
    hierarchical_segmentation,
    kmeans_segmentation,
)
from segmentation.common import require_optional
from segmentation.evaluation import evaluate_clustering
from segmentation.segment_naming import propose_names
from segmentation.segment_profiling import build_cluster_profile
from segmentation.stability import bootstrap_stability, seed_stability
from utils.exceptions import CustomerIntelligenceError
from utils.model_utils import save_model
from utils.validators import require_min_rows

MODEL_MODULES = {
    "kmeans": kmeans_segmentation,
    "gmm": gmm_segmentation,
    "hierarchical": hierarchical_segmentation,
    "hdbscan": hdbscan_segmentation,
    "autoencoder": autoencoder_model,
    "ensemble": ensemble_clustering,
}
OPTIONAL_PIPS = {"hdbscan": "hdbscan", "autoencoder": "tensorflow"}


def model_availability() -> dict[str, str | None]:
    """Per-model None (ready) or install-guidance message (optional dep missing)."""
    status: dict[str, str | None] = {}
    for name, pip in OPTIONAL_PIPS.items():
        try:
            require_optional(name, pip)
            status[name] = None
        except Exception as exc:  # message string for the UI, not a crash
            status[name] = str(exc)
    for name in MODEL_MODULES:
        status.setdefault(name, None)
    return status


def prepare_segmentation_features(
    features_df: pd.DataFrame,
    feature_names: list[str],
    scaler: str = "robust",
    log1p: bool = True,
    minimum_customers: int = 50,
) -> PreparedData:
    """Validate size, then build the persisted preprocessing pipeline + matrix."""
    require_min_rows(features_df, minimum_customers, "customers")
    return prepare_features(features_df, feature_names, scaler=scaler, log1p=log1p)


def run_model(
    model_name: str,
    X: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
    **kwargs,
) -> tuple[np.ndarray, object]:
    """Fit one model by registry name. Density models take min_cluster_size instead."""
    if model_name not in MODEL_MODULES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(MODEL_MODULES)}.")
    module = MODEL_MODULES[model_name]
    if module.MODEL_TYPE == "density":
        return module.fit_predict(X, random_state=random_state, **kwargs)
    return module.fit_predict(X, n_clusters=n_clusters, random_state=random_state, **kwargs)


def sweep_clusters(
    model_name: str,
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate k-based models across a cluster range for model/k selection."""
    module = MODEL_MODULES[model_name]
    if module.MODEL_TYPE != "k_based":
        raise ValueError(f"Cluster sweep needs a k-based model, not '{model_name}'.")
    rows: list[dict] = []
    for k in range(max(2, k_min), max(2, k_max) + 1):
        labels, _ = module.fit_predict(X, n_clusters=k, random_state=random_state)
        metrics = evaluate_clustering(X, labels)
        rows.append(
            {
                "k": k,
                **{
                    m: metrics[m]
                    for m in (
                        "silhouette",
                        "davies_bouldin",
                        "calinski_harabasz",
                        "n_clusters",
                        "min_cluster_size",
                        "noise_pct",
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def assess_stability(model_name: str, X: np.ndarray, random_state: int = 42, **kwargs) -> dict:
    """Seed ARI for every model; bootstrap ARI when the fitted model has .predict."""
    module = MODEL_MODULES[model_name]
    result = dict(seed_stability(module.fit_predict, X, **kwargs))
    _, fitted = module.fit_predict(X, random_state=random_state)
    if hasattr(fitted, "predict"):
        try:
            result["bootstrap"] = bootstrap_stability(module.fit_predict, X)
        except (ValueError, AttributeError) as exc:
            result["bootstrap"] = {"error": str(exc)}
    else:
        result["bootstrap"] = None  # no .predict (hierarchical/HDBSCAN/ensemble/AE bundle)
    return result


def train_all_models(
    X: np.ndarray,
    customer_ids: list,
    feature_names: list[str],
    original_features: pd.DataFrame | None = None,
    n_clusters: int = 5,
    random_state: int = 42,
    min_cluster_size: int = 15,
    latent_dim: int = 8,
    epochs: int = 100,
) -> list[dict]:
    """Train every registry model; failures record as skipped, never abort the batch.

    Returned run dicts match the page-06 record shape (model, params, metrics,
    stability, names, feature_names, customer_ids, labels). Seed-only stability
    (no bootstrap) keeps batch time bounded.
    """
    runs: list[dict] = []
    for name, module in MODEL_MODULES.items():
        extra: dict = {}
        if name == "hdbscan":
            extra["min_cluster_size"] = min_cluster_size
        if name == "autoencoder":
            extra.update({"latent_dim": latent_dim, "epochs": epochs})
        params = {"n_clusters": n_clusters, "seed": random_state, **extra}
        try:
            labels, _ = run_model(
                name, X, n_clusters=n_clusters, random_state=random_state, **extra
            )
        except (CustomerIntelligenceError, ValueError, ImportError) as exc:
            runs.append({"model": name, "status": "skipped", "reason": str(exc), "params": params})
            continue
        metrics = evaluate_clustering(X, labels)
        stability = dict(seed_stability(module.fit_predict, X, **extra))
        names: dict[int, str] = {}
        if original_features is not None:
            try:
                profile = build_cluster_profile(original_features, labels, feature_names)
                name_table = propose_names(profile)
                names = {
                    int(k): str(v)
                    for k, v in zip(
                        name_table["cluster_id"], name_table["business_name"], strict=True
                    )
                }
            except (KeyError, ValueError, AttributeError):
                names = {}  # naming needs monetary/recency/frequency medians; run stays valid
        runs.append(
            {
                "model": name,
                "status": "ok",
                "params": params,
                "metrics": metrics,
                "stability": stability,
                "names": names,
                "feature_names": list(feature_names),
                "customer_ids": list(customer_ids),
                "labels": [int(v) for v in labels],
            }
        )
    return runs


def save_segmentation_run(
    model: object,
    pipeline: object,
    metadata: dict,
    run_name: str,
) -> str:
    """Persist model + pipeline with a metadata sidecar. Returns the model path."""
    model_path = save_model(model, f"models/segmentation/{run_name}.joblib", metadata)
    save_model(
        pipeline,
        f"models/segmentation/{run_name}.pipeline.joblib",
        {"run": run_name, "kind": "preprocessing_pipeline"},
    )
    from utils.model_card import segmentation_card_metadata, write_model_card

    card_fields = segmentation_card_metadata(
        str(metadata.get("model", run_name)),
        list(metadata.get("feature_names", [])),
        metadata.get("metrics", {}),
        metadata.get("params", {}),
    )
    write_model_card(
        f"reports/model_cards/{run_name}.md",
        version=run_name,
        training_period=str(metadata.get("training_period", "recorded in run audit")),
        **card_fields,
    )
    return str(model_path)

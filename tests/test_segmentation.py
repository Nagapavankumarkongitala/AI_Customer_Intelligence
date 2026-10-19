"""Phase 4 tests: pipeline, models, evaluation, stability, naming, graceful fallback."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from feature_engineering.preprocessing import prepare_features
from segmentation import (
    ensemble_clustering,
    gmm_segmentation,
    hdbscan_segmentation,
    hierarchical_segmentation,
    kmeans_segmentation,
)
from segmentation.common import describe_labels
from segmentation.ensemble_clustering import consensus_matrix
from segmentation.evaluation import evaluate_clustering
from segmentation.segment_naming import propose_names
from segmentation.segment_profiling import build_cluster_profile
from segmentation.stability import bootstrap_stability, seed_stability
from services.segmentation_service import (
    labeling_series,
    run_model,
    sweep_clusters,
    train_all_models,
)
from utils.exceptions import CustomerIntelligenceError
from utils.model_utils import load_model, save_model


def _blobs() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
    return np.asarray(X, dtype=float), y


def test_prepare_imputes_drops_constant_and_persists() -> None:
    """NaNs imputed, zero-variance dropped, skewed log1p'd, pipeline replays."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "skewed": [1.0, 2.0, 3.0, 4.0, 5.0, 500.0],
            "flat": [7.0] * 6,
        }
    )
    prepared = prepare_features(df, ["a", "skewed", "flat"])
    assert prepared.dropped_zero_variance == ["flat"]
    assert "skewed" in prepared.log1p_applied
    assert not np.isnan(prepared.matrix).any()
    replay = prepared.pipeline.transform(df[prepared.feature_names].to_numpy(dtype=float))
    assert np.allclose(replay, prepared.matrix)
    save_model(prepared.pipeline, "models/metadata/test_pipeline.joblib", {"t": 1})
    assert np.allclose(
        load_model("models/metadata/test_pipeline.joblib").transform(
            df[prepared.feature_names].to_numpy(dtype=float)
        ),
        prepared.matrix,
    )


def test_prepare_flags_high_correlation() -> None:
    """Near-duplicate columns are reported, not silently dropped."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    prepared = prepare_features(df, ["a", "b"])
    assert prepared.high_correlations
    assert prepared.high_correlations[0]["abs_correlation"] > 0.9


def test_classical_models_recover_blobs() -> None:
    """K-Means, GMM, hierarchical each find 3 well-separated clusters."""
    X, _ = _blobs()
    for module in (kmeans_segmentation, gmm_segmentation, hierarchical_segmentation):
        labels, _ = module.fit_predict(X, n_clusters=3, random_state=42)
        metrics = evaluate_clustering(X, labels)
        assert metrics["n_clusters"] == 3
        assert metrics["silhouette"] is not None and metrics["silhouette"] > 0.5
        assert metrics["noise_pct"] == 0.0


def test_evaluation_degrades_gracefully() -> None:
    """Single-cluster labelings yield None metrics instead of exceptions."""
    X, _ = _blobs()
    metrics = evaluate_clustering(X, np.zeros(len(X), dtype=int))
    assert metrics["silhouette"] is None
    assert metrics["n_clusters"] == 1


def test_hdbscan_missing_gives_install_guidance() -> None:
    """Absent optional dep raises a business-readable error, never ImportError."""
    X, _ = _blobs()
    try:
        import hdbscan  # noqa: F401
    except ImportError:
        with pytest.raises(CustomerIntelligenceError, match="pip install hdbscan"):
            hdbscan_segmentation.fit_predict(X)
    else:
        labels, _ = hdbscan_segmentation.fit_predict(X, min_cluster_size=10)
        assert len(labels) == len(X)


def test_consensus_matrix_properties_and_ensemble() -> None:
    """Consensus is symmetric with unit diagonal; ensemble covers every point."""
    X, _ = _blobs()
    labelings = [
        kmeans_segmentation.fit_predict(X, n_clusters=3, random_state=s)[0] for s in (1, 2)
    ]
    consensus = consensus_matrix(labelings)
    assert np.allclose(consensus, consensus.T)
    assert np.allclose(np.diag(consensus), 1.0)
    labels, bundle = ensemble_clustering.fit_predict(X, n_clusters=3, random_state=42)
    assert len(bundle["base_labelings"]) == 3
    assert describe_labels(labels)["n_clusters"] == 3


def test_stability_bounds_and_determinism() -> None:
    """ARI stays in [-1, 1]; deterministic models score exactly 1."""
    X, _ = _blobs()
    seed = seed_stability(hierarchical_segmentation.fit_predict, X)
    assert seed["mean_pairwise_ari"] == 1.0
    seed_km = seed_stability(kmeans_segmentation.fit_predict, X, seeds=(42, 43))
    assert -1.0 <= seed_km["mean_pairwise_ari"] <= 1.0
    boot = bootstrap_stability(kmeans_segmentation.fit_predict, X, n_boot=2)
    assert -1.0 <= boot["mean_ari"] <= 1.0


def test_profile_then_name() -> None:
    """Profiles aggregate correctly; every non-noise cluster gets a distinct-ready name."""
    X, _ = _blobs()
    labels, _ = kmeans_segmentation.fit_predict(X, n_clusters=3, random_state=42)
    original = pd.DataFrame(X, columns=["f0", "f1"])
    original["customer_id"] = [f"C{i}" for i in range(len(X))]
    original["monetary_value"] = np.abs(X[:, 0]) * 100 + 50
    profile = build_cluster_profile(original, labels, ["f0", "f1"])
    assert profile["customers"].sum() == len(X)
    profile = profile.assign(
        median_monetary_value=profile["revenue"] / profile["customers"],
        median_recency_days=profile["median_f0"].abs(),
        median_frequency=profile["median_f1"].abs() + 1,
    )
    names = propose_names(profile)
    assert len(names) == 3 and names["business_name"].notna().all()
    assert set(names["cluster_id"]) == set(profile["cluster_id"])


def test_sweep_table_shape() -> None:
    """K sweep returns one metrics row per k."""
    X, _ = _blobs()
    table = sweep_clusters("kmeans", X, k_min=2, k_max=4)
    assert list(table["k"]) == [2, 3, 4]
    assert table["silhouette"].notna().all()


def test_run_model_rejects_unknown() -> None:
    """Unknown model names fail fast with the registry in the message."""
    X, _ = _blobs()
    with pytest.raises(ValueError, match="kmeans"):
        run_model("nope", X)


def test_autoencoder_trains_when_available() -> None:
    """Tiny TF run produces latent codes and a loss history (skipped without TF)."""
    tf = pytest.importorskip("tensorflow")
    _ = tf  # backend presence is the point; training API is exercised below.
    from deep_learning.autoencoder import train_autoencoder
    from deep_learning.feature_embeddings import embed_with_encoder, latent_table

    rng = np.random.default_rng(42)
    X = rng.normal(size=(220, 6))
    result = train_autoencoder(X, latent_dim=3, epochs=3, batch_size=32, patience=1, hidden=(16, 8))
    assert result.latent.shape == (220, 3)
    assert "loss" in result.history and result.epochs_run >= 1
    codes = embed_with_encoder(result.encoder, X)
    assert codes.shape == (220, 3)
    table = latent_table(codes, pd.RangeIndex(220))
    assert list(table.columns) == ["latent_0", "latent_1", "latent_2"]


def test_train_all_models_records_and_skips() -> None:
    """Batch trains every available model; missing optionals record as skipped."""
    X, _ = _blobs()
    ids = [f"C{i}" for i in range(len(X))]
    original = pd.DataFrame(X, columns=["f0", "f1"])
    original["customer_id"] = ids
    original["monetary_value"] = np.abs(X[:, 0]) * 100 + 50
    runs = train_all_models(
        X,
        ids,
        ["f0", "f1"],
        original_features=original,
        n_clusters=3,
        epochs=2,
    )
    assert {r["model"] for r in runs} == {
        "kmeans",
        "gmm",
        "hierarchical",
        "hdbscan",
        "autoencoder",
        "ensemble",
    }
    ok_runs = [r for r in runs if r.get("status") == "ok"]
    assert len(ok_runs) >= 4  # classical models + ensemble (+ autoencoder when TF present)
    for run in ok_runs:
        assert len(run["labels"]) == len(X)
        assert run["metrics"]["n_clusters"] >= 2
        assert set(run) >= {
            "model",
            "params",
            "metrics",
            "stability",
            "names",
            "feature_names",
            "customer_ids",
            "labels",
        }
    try:
        import hdbscan  # noqa: F401
    except ImportError:
        skipped = [r for r in runs if r.get("status") == "skipped"]
        assert any(r["model"] == "hdbscan" and "reason" in r for r in skipped)


def test_labeling_series_uses_run_names_and_noise_label() -> None:
    """Saved runs map to business names, falling back to IDs and to "Noise"."""
    run = {
        "model": "kmeans",
        "customer_ids": ["a", "b", "c"],
        "labels": [0, 7, -1],
        "names": {"0": "Core Active"},
    }
    series = labeling_series(run)
    assert series.name == "segment"
    assert series.tolist() == ["Core Active", "7", "Noise"]
    assert labeling_series({"customer_ids": ["a"], "labels": [3]}).tolist() == ["3"]

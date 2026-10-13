"""Default-data bootstrap: data_edited.csv loads with the app, fully modeled.

Builds the whole pipeline once (clean -> features -> RFM/LRFM -> demographics
-> segmentation -> churn -> retention), caches artifacts under
``data/processed/default_run/``, and reloads them into session state on every
app start so all pages show results immediately.

Heavy O(n^2) segmentation models (hierarchical, ensemble) are recorded as
skipped in the default run — the data has ~25k customers and Ward linkage on
a 25k x 25k matrix is not boot-safe. They remain runnable from page 05.
Optional-dependency models (hdbscan, autoencoder, lightgbm, xgboost) are
likewise skipped with their install guidance when unavailable.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from analytics.demographics import build_demographic_profile
from services import churn_service, retention_service, segmentation_service
from services.cleaning_service import CleaningConfig, clean
from services.column_detection import validate_mapping
from services.data_dictionary_service import build_dictionary
from services.data_service import load_upload_bytes
from services.feature_service import run_lrfm, run_rfm
from services.profiling_service import compute_quality
from utils.constants import RANDOM_SEED

log = logging.getLogger(__name__)

DEFAULT_SOURCE = Path("data_edited.csv")
CACHE_DIR = Path("data/processed/default_run")
MANIFEST = CACHE_DIR / "manifest.json"

# Fast, scalable models for the unattended default run.
DEFAULT_SEGMENTATION_MODELS = ("kmeans", "gmm")
SKIPPED_SEGMENTATION_REASONS = {
    "hierarchical": "skipped in default run: O(n^2) Ward linkage is not boot-safe at ~25k customers (run manually on page 05)",
    "ensemble": "skipped in default run: consensus matrix is O(n^2) (run manually on page 05)",
}
DEFAULT_CHURN_MODELS = ("logreg", "random_forest")
DEFAULT_N_CLUSTERS = 5


def _config() -> dict:
    return yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))


def _write_frame(frame: pd.DataFrame, name: str) -> None:
    frame.to_parquet(CACHE_DIR / f"{name}.parquet", index=False)


def _read_frame(name: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE_DIR / f"{name}.parquet")


def _write_json(obj: dict, name: str) -> None:
    (CACHE_DIR / f"{name}.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _read_json(name: str) -> dict:
    return json.loads((CACHE_DIR / f"{name}.json").read_text(encoding="utf-8"))


def cache_is_fresh() -> bool:
    """True when the manifest exists and matches the current source file."""
    if not MANIFEST.exists() or not DEFAULT_SOURCE.exists():
        return False
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    stat = DEFAULT_SOURCE.stat()
    return (
        manifest.get("source_mtime") == stat.st_mtime
        and manifest.get("source_size") == stat.st_size
    )


def build_default_run(force: bool = False) -> dict:
    """Run the full pipeline on data_edited.csv and cache every artifact.

    Returns the manifest summary dict. Reuses the cache unless ``force`` or
    the source file changed.
    """
    if not force and cache_is_fresh():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not DEFAULT_SOURCE.exists():
        raise FileNotFoundError(
            "Default dataset 'data_edited.csv' not found at the repository root."
        )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _config()
    started = datetime.now().isoformat()

    # 1-2. Load + mapping (auto-detect; data_edited.csv normalizes cleanly).
    preview = load_upload_bytes(DEFAULT_SOURCE.read_bytes(), DEFAULT_SOURCE.name)
    mapping = dict(preview.proposal.mapping)
    validate_mapping(mapping, preview.normalized_columns)
    raw = preview.frame
    _write_frame(raw, "raw")

    # 3. Quality + clean + dictionary (defaults = least destructive valid).
    quality = compute_quality(raw, mapping)
    cleaning_cfg = CleaningConfig()
    cleaned, audit = clean(raw, mapping, cleaning_cfg, source_filename=DEFAULT_SOURCE.name)
    dictionary = build_dictionary(cleaned, mapping)
    _write_frame(cleaned, "clean")
    _write_json(
        {
            "audit": audit,
            "cleaning_config": cleaning_cfg.as_dict(),
            "column_mapping": mapping,
            "quality_overall": quality.overall,
        },
        "cleaning",
    )
    _write_json(dictionary.to_dict(orient="records"), "dictionary")

    # 4-5. Features + RFM/LRFM (snapshot = max transaction + 1 day).
    from analytics.customer_features import build_customer_features

    snapshot = pd.to_datetime(cleaned["transaction_date"]).max() + pd.Timedelta(days=1)
    features = build_customer_features(
        cleaned,
        snapshot_date=snapshot,
        snapshot_offset_days=cfg["rfm"]["snapshot_offset_days"],
    ).features
    quantiles = int(cfg["rfm"]["quantiles"])
    rfm = run_rfm(features, quantiles=quantiles)
    lrfm = run_lrfm(features, quantiles=quantiles)
    _write_frame(features, "features")
    _write_frame(rfm, "rfm")
    _write_frame(lrfm, "lrfm")

    # 6. Demographics (same record shape as page 04).
    profile = build_demographic_profile(features, rfm[["customer_id", "segment"]])
    _write_json(
        {
            "tables": {
                name: table.to_dict(orient="records") for name, table in profile.tables.items()
            },
            "missing_pct": dict(profile.missing_pct),
            "notes": list(profile.notes),
            "source": DEFAULT_SOURCE.name,
        },
        "demographics",
    )

    # 7. Segmentation: fast models trained, O(n^2)/optional ones recorded skipped.
    seg_features = [f for f in cfg["segmentation"]["default_features"] if f in features.columns]
    prepared = segmentation_service.prepare_segmentation_features(
        features,
        seg_features,
        minimum_customers=cfg["segmentation"]["minimum_customers"],
    )
    customer_ids = features["customer_id"].tolist()
    runs: list[dict] = []
    for name in DEFAULT_SEGMENTATION_MODELS:
        try:
            labels, _ = segmentation_service.run_model(
                name,
                prepared.matrix,
                n_clusters=DEFAULT_N_CLUSTERS,
                random_state=RANDOM_SEED,
            )
        except Exception as exc:  # optional dep missing -> record, don't abort
            runs.append(
                {
                    "model": name,
                    "status": "skipped",
                    "reason": str(exc),
                    "params": {"n_clusters": DEFAULT_N_CLUSTERS, "seed": RANDOM_SEED},
                }
            )
            continue
        from segmentation.evaluation import evaluate_clustering
        from segmentation.segment_naming import propose_names
        from segmentation.segment_profiling import build_cluster_profile
        from segmentation.stability import seed_stability

        metrics = evaluate_clustering(prepared.matrix, labels)
        stability = dict(
            seed_stability(segmentation_service.MODEL_MODULES[name].fit_predict, prepared.matrix)
        )
        try:
            cluster_profile = build_cluster_profile(features, labels, seg_features)
            name_table = propose_names(cluster_profile)
            names = {
                int(k): str(v)
                for k, v in zip(name_table["cluster_id"], name_table["business_name"], strict=True)
            }
        except (KeyError, ValueError, AttributeError):
            names = {}
        runs.append(
            {
                "model": name,
                "status": "ok",
                "params": {"n_clusters": DEFAULT_N_CLUSTERS, "seed": RANDOM_SEED},
                "metrics": metrics,
                "stability": stability,
                "names": names,
                "feature_names": list(seg_features),
                "customer_ids": list(customer_ids),
                "labels": [int(v) for v in labels],
            }
        )
    for name, reason in SKIPPED_SEGMENTATION_REASONS.items():
        runs.append(
            {
                "model": name,
                "status": "skipped",
                "reason": reason,
                "params": {"n_clusters": DEFAULT_N_CLUSTERS, "seed": RANDOM_SEED},
            }
        )
    for name in ("hdbscan", "autoencoder"):
        availability = segmentation_service.model_availability().get(name)
        if availability:
            runs.append(
                {
                    "model": name,
                    "status": "skipped",
                    "reason": availability,
                    "params": {"n_clusters": DEFAULT_N_CLUSTERS, "seed": RANDOM_SEED},
                }
            )
    ok_runs = [r for r in runs if r.get("status") == "ok"]
    if not ok_runs:
        raise RuntimeError("Default segmentation produced no successful runs.")
    best = max(
        ok_runs,
        key=lambda r: (
            r["metrics"].get("silhouette") is not None,
            r["metrics"].get("silhouette") or -9,
        ),
    )
    labeled = features[["customer_id"]].copy()
    label_map = dict(zip(best["customer_ids"], best["labels"], strict=True))
    labeled["cluster_id"] = labeled["customer_id"].map(label_map).fillna(-1).astype(int)
    labeled["business_name"] = (
        labeled["cluster_id"].map({int(k): v for k, v in best["names"].items()}).fillna("Cluster")
    )
    _write_frame(labeled, "segmented")
    _write_json(
        {
            "runs": [
                {**r, "customer_ids": None, "labels": None} if r.get("status") == "ok" else r
                for r in runs
            ],
            "selected": best["model"],
        },
        "segmentation_index",
    )
    # Full labels live in the segmented frame; keep a compact copy per model.
    _write_json({r["model"]: r["labels"] for r in ok_runs}, "segmentation_labels")
    _write_json({r["model"]: r["customer_ids"] for r in ok_runs}, "segmentation_ids")
    saved_seg_path = segmentation_service.save_segmentation_run(
        segmentation_service.MODEL_MODULES[best["model"]].fit_predict(
            prepared.matrix, n_clusters=DEFAULT_N_CLUSTERS, random_state=RANDOM_SEED
        )[1],
        prepared.pipeline,
        {**best, "metrics": best["metrics"]},
        run_name=f"default_{best['model']}_k{DEFAULT_N_CLUSTERS}",
    )

    # 8-9. Churn dataset + train + score.
    from churn.dataset_builder import build_churn_dataset

    txn_dates = pd.to_datetime(cleaned["transaction_date"])
    min_date, max_date = txn_dates.min(), txn_dates.max()
    horizon = int(cfg["churn"]["default_outcome_horizon_days"])
    cutoff = (max_date - pd.Timedelta(days=horizon)).date().isoformat()
    dataset = build_churn_dataset(
        cleaned,
        cutoff,
        horizon_days=horizon,
        min_history_days=30,
        observation_start=min_date,
    )
    bundles: dict[str, dict] = {}
    for model_name in DEFAULT_CHURN_MODELS:
        try:
            bundles[model_name] = churn_service.train_churn_model(
                dataset, model_name, random_state=RANDOM_SEED, calibrate=True
            )
        except Exception as exc:
            log.warning("Default churn model %s failed: %s", model_name, exc)
    if not bundles:
        raise RuntimeError("Default churn training produced no successful models.")
    best_churn = max(bundles, key=lambda m: bundles[m]["test_metrics"]["roc_auc"] or -1)
    bundle = bundles[best_churn]
    scoring_features = build_customer_features(
        cleaned, snapshot_date=max_date + pd.Timedelta(days=1)
    ).features
    scores = churn_service.score_all_customers(
        bundle,
        scoring_features,
        cfg["churn"]["risk_bands"],
        bundle["suggested_threshold"],
        best_churn,
    )
    _write_frame(scores, "scores")
    _write_frame(dataset.frame, "churn_frame")
    _write_json(
        {
            "cutoff": str(dataset.labeling.cutoff.date()),
            "horizon_days": dataset.labeling.horizon_days,
            "churn_rate": round(float(dataset.frame["churn"].mean()), 4),
            "customers": len(dataset.frame),
            "excluded_short_history": dataset.labeling.excluded_short_history,
            "leakage_ok": bool(dataset.leakage_ok),
        },
        "churn_summary",
    )
    _write_json(
        {
            name: {
                "model": name,
                "threshold": b["suggested_threshold"],
                "metrics": {k: v for k, v in b["test_metrics"].items() if k != "confusion"},
                "calibration": b["calibration"],
                "cutoff": b["cutoff"],
                "horizon_days": b["horizon_days"],
            }
            for name, b in bundles.items()
        },
        "churn_metadata",
    )
    saved_churn_path = churn_service.save_churn_run(
        bundle, f"default_{best_churn}_{bundle['cutoff']}"
    )

    # 10. Retention table (page-08 record shape: dict of records + summary).
    matrix = retention_service.default_matrix_report()[0]
    retention_table = retention_service.build_retention_table(scores, features, matrix)
    _write_frame(retention_table, "retention")
    _write_json(retention_service.retention_summary(retention_table), "retention_summary")

    stat = DEFAULT_SOURCE.stat()
    manifest = {
        "source": DEFAULT_SOURCE.name,
        "source_mtime": stat.st_mtime,
        "source_size": stat.st_size,
        "built_at": started,
        "rows": len(raw),
        "customers": len(features),
        "cleaned_rows": len(cleaned),
        "quality_overall": quality.overall,
        "rfm_segments": int(rfm["segment"].nunique()),
        "lrfm_segments": int(lrfm["segment"].nunique()),
        "segmentation_models": [r["model"] for r in ok_runs],
        "selected_segmentation_model": best["model"],
        "churn_models": sorted(bundles),
        "selected_churn_model": best_churn,
        "churn_auc": bundles[best_churn]["test_metrics"]["roc_auc"],
        "churn_rate": round(float(dataset.frame["churn"].mean()), 4),
        "retention_customers": len(retention_table),
        "segmentation_path": str(saved_seg_path),
        "churn_path": str(saved_churn_path),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Default run built: %s", manifest)
    return manifest


def load_default_into_state(state: dict) -> dict | None:
    """Load cached default artifacts into session state (no-op when fresh data exists).

    Does not overwrite keys the user already populated this session. Returns
    the manifest summary, or None when no cache exists.
    """
    if not MANIFEST.exists():
        return None
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if state.get("raw_data") is not None:
        return manifest
    cleaning = _read_json("cleaning")
    state["raw_data"] = _read_frame("raw")
    state["clean_data"] = _read_frame("clean")
    state["source_filename"] = manifest["source"]
    state["column_mapping"] = cleaning["column_mapping"]
    state["cleaning_config"] = cleaning["cleaning_config"]
    state["cleaning_audit"] = cleaning["audit"]
    state["data_dictionary"] = _read_json("dictionary")
    state["customer_features"] = _read_frame("features")
    state["rfm_results"] = _read_frame("rfm")
    state["lrfm_results"] = _read_frame("lrfm")
    state["demographic_profile"] = _read_json("demographics")
    seg_index = _read_json("segmentation_index")
    seg_labels = _read_json("segmentation_labels")
    seg_ids = _read_json("segmentation_ids")
    runs = []
    for record in seg_index["runs"]:
        if record.get("status") != "ok":
            runs.append(record)
            continue
        model = record["model"]
        runs.append({**record, "labels": seg_labels[model], "customer_ids": seg_ids[model]})
    state["segmentation_runs"] = runs
    state["selected_segmentation_model"] = seg_index["selected"]
    state["segmented_customers"] = _read_frame("segmented")
    state["churn_dataset"] = _read_json("churn_summary")
    churn_meta = _read_json("churn_metadata")
    best = manifest["selected_churn_model"]
    state["churn_model_metadata"] = {
        "model": best,
        "threshold": churn_meta[best]["threshold"],
        "metrics": churn_meta[best]["metrics"],
        "calibration": churn_meta[best]["calibration"],
        "cutoff": churn_meta[best]["cutoff"],
        "horizon_days": churn_meta[best]["horizon_days"],
    }
    # Restore the persisted pipeline object so downstream code can reuse it.
    from utils.model_utils import load_model

    try:
        state["churn_model"], _ = load_model(manifest["churn_path"])
    except (FileNotFoundError, OSError, ValueError):
        state["churn_model"] = None
    state["churn_scores"] = _read_frame("scores")
    retention_table = _read_frame("retention")
    state["retention_results"] = {
        "table": retention_table.to_dict(orient="records"),
        "summary": _read_json("retention_summary"),
    }
    return manifest

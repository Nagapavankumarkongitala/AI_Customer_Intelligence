"""Finish a partial default run: churn -> retention -> manifest from cache.

Usage:
    python scripts/finish_default_run.py

For when a full `build_default_run.py --force` stalls after segmentation:
every artifact through `segmentation_ids` is reused from
`data/processed/default_run/` and only the churn/retention tail recomputes
in a fresh short-lived process. Writes the same manifest shape (with
``cache_version``) so the app and `test_default_bootstrap.py` see a
complete, consistent cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import churn_service, retention_service
from services.bootstrap_service import (
    CACHE_DIR,
    CACHE_VERSION,
    DEFAULT_CHURN_MODELS,
    DEFAULT_HISTORY_GRANULARITY,
    DEFAULT_N_CLUSTERS,
    MANIFEST,
    RANDOM_SEED,
    _config,
    _read_frame,
    _read_json,
    _write_frame,
    _write_json,
)

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for name in ("clean", "features", "rfm", "segmented", "segment_history"):
        if not (CACHE_DIR / f"{name}.parquet").exists():
            raise SystemExit(f"Cache artifact missing: {name}.parquet — rerun the full build.")
    cfg = _config()
    cleaned = _read_frame("clean")
    features = _read_frame("features")

    log.info("finish stage: churn dataset")
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
    log.info("finish stage: churn train")
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

    log.info("finish stage: churn scoring features")
    from analytics.customer_features import build_customer_features

    scoring_features = build_customer_features(
        cleaned, snapshot_date=max_date + pd.Timedelta(days=1)
    ).features
    log.info("finish stage: churn scoring")
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

    log.info("finish stage: retention")
    matrix = retention_service.default_matrix_report()[0]
    retention_table = retention_service.build_retention_table(scores, features, matrix)
    _write_frame(retention_table, "retention")
    _write_json(retention_service.retention_summary(retention_table), "retention_summary")

    log.info("finish stage: manifest")
    from services.bootstrap_service import DEFAULT_SOURCE

    raw = _read_frame("raw")
    rfm = _read_frame("rfm")
    lrfm = _read_frame("lrfm")
    history = _read_frame("segment_history")
    cleaning = _read_json("cleaning")
    seg_index = _read_json("segmentation_index")
    seg_path = Path(
        f"models/segmentation/default_{seg_index['selected']}_k{DEFAULT_N_CLUSTERS}.joblib"
    )
    if not seg_path.exists():
        raise SystemExit(f"Segmentation artifact missing: {seg_path}")
    stat = DEFAULT_SOURCE.stat()
    manifest = {
        "source": DEFAULT_SOURCE.name,
        "cache_version": CACHE_VERSION,
        "source_mtime": stat.st_mtime,
        "source_size": stat.st_size,
        "built_at": datetime.now().isoformat(),
        "rows": len(raw),
        "customers": len(features),
        "cleaned_rows": len(cleaned),
        "quality_overall": cleaning["quality_overall"],
        "rfm_segments": int(rfm["segment"].nunique()),
        "lrfm_segments": int(lrfm["segment"].nunique()),
        "segment_history_granularity": DEFAULT_HISTORY_GRANULARITY,
        "segment_history_periods": int(history["period"].nunique()) if not history.empty else 0,
        "segmentation_models": [r["model"] for r in seg_index["runs"] if r.get("status") == "ok"],
        "selected_segmentation_model": seg_index["selected"],
        "churn_models": sorted(bundles),
        "selected_churn_model": best_churn,
        "churn_auc": bundles[best_churn]["test_metrics"]["roc_auc"],
        "churn_rate": round(float(dataset.frame["churn"].mean()), 4),
        "retention_customers": len(retention_table),
        "segmentation_path": str(seg_path),
        "churn_path": str(saved_churn_path),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("finish complete: %s", manifest["churn_path"])
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

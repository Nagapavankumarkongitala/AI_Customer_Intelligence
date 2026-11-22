"""Phase 6 orchestration: time-aware datasets, training, calibration, scoring.

Splits are chronological: customers ordered by last observed transaction —
oldest 70% train, next 15% validation, newest 15% test. Oversampling touches
training data only. Calibration learns on validation only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from churn.baseline_models import BASELINE_ESTIMATORS, build_model_pipeline, oversample_minority
from churn.calibration import calibrate_on_validation
from churn.dataset_builder import ChurnDataset
from churn.evaluation import (
    calibration_table,
    classification_metrics,
    lift_gain_table,
    suggest_threshold,
)
from churn.explainability import global_importance
from churn.predict import score_customers
from churn.train_lightgbm import MODEL_NAME as LGBM_NAME
from churn.train_lightgbm import build_lgbm_pipeline
from churn.train_rf import MODEL_NAME as RF_NAME
from churn.train_rf import build_rf_pipeline
from churn.train_xgb import MODEL_NAME as XGB_NAME
from churn.train_xgb import build_xgb_pipeline, set_balance
from segmentation.common import require_optional
from utils.exceptions import ChurnLeakageError, InsufficientDataError
from utils.model_utils import save_model

MODEL_BUILDERS = {
    "logreg": ("baseline", None),
    "histgb": ("baseline", None),
    RF_NAME: ("rf", None),
    LGBM_NAME: ("lgbm", "lightgbm"),
    XGB_NAME: ("xgb", "xgboost"),
}


def time_split(
    dataset: ChurnDataset, train_frac: float = 0.70, val_frac: float = 0.15
) -> dict[str, pd.DataFrame]:
    """Chronological cohort split by first transaction date.

    Cohort (not recency) ordering: classes mix naturally because churn depends on
    recent activity while cohorts span all behaviors. Leakage-safe regardless of
    split — every predictor uses transactions ≤ cutoff. Oldest 70% train, next
    15% validation, newest 15% test.
    """
    frame = dataset.frame.copy()
    if "first_transaction_date" not in frame.columns:
        raise ChurnLeakageError("Churn frame lacks first_transaction_date for time splits.")
    order = frame.sort_values("first_transaction_date").reset_index(drop=True)
    order = frame.sort_values("first_transaction_date").reset_index(drop=True)
    n = len(order)
    if n < 10:
        raise InsufficientDataError(
            f"Only {n} labeled customers; at least 10 are needed for train/val/test splits."
        )
    cut_train = int(n * train_frac)
    cut_val = cut_train + int(n * val_frac)
    return {
        "train": order.iloc[:cut_train].reset_index(drop=True),
        "val": order.iloc[cut_train:cut_val].reset_index(drop=True),
        "test": order.iloc[cut_val:].reset_index(drop=True),
    }


def build_pipeline(
    model_name: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int = 42,
):
    """Instantiate a training pipeline; optional deps raise install guidance."""
    if model_name in BASELINE_ESTIMATORS:
        return build_model_pipeline(model_name, numeric_cols, categorical_cols, random_state)
    if model_name == RF_NAME:
        return build_rf_pipeline(numeric_cols, categorical_cols, random_state)
    if model_name == LGBM_NAME:
        return build_lgbm_pipeline(numeric_cols, categorical_cols, random_state)
    if model_name == XGB_NAME:
        return build_xgb_pipeline(numeric_cols, categorical_cols, random_state)
    raise ValueError(f"Unknown churn model '{model_name}'. Choose from {sorted(MODEL_BUILDERS)}.")


def train_churn_model(
    dataset: ChurnDataset,
    model_name: str,
    random_state: int = 42,
    oversample: bool = False,
    calibrate: bool = True,
) -> dict:
    """Train on train split, tune threshold on val, report test metrics. Returns bundle."""
    splits = time_split(dataset)
    feature_cols = dataset.feature_cols + dataset.categorical_cols
    train, val, test = (splits[k] for k in ("train", "val", "test"))
    for name, split in (("train", train), ("val", val), ("test", test)):
        if split["churn"].nunique() < 2:
            raise InsufficientDataError(
                f"The {name} split has a single class; adjust cutoff/horizon "
                "so both churners and retained customers appear in every split."
            )
    pipeline = build_pipeline(
        model_name, dataset.feature_cols, dataset.categorical_cols, random_state
    )
    X_train = train[feature_cols]
    y_train = train["churn"].to_numpy()
    if model_name == XGB_NAME:
        set_balance(pipeline, y_train)
    if oversample:
        flat = pipeline.named_steps["preprocess"].fit_transform(X_train)
        X_over, y_over = oversample_minority(flat, y_train, random_state)
        pipeline.named_steps["model"].fit(X_over, y_over)
    else:
        pipeline.fit(X_train, train["churn"])

    val_proba = np.asarray(pipeline.predict_proba(val[feature_cols])[:, 1])
    suggestion = suggest_threshold(val["churn"].to_numpy(), val_proba)
    estimator = pipeline
    calibration_note = "off"
    if calibrate:
        try:
            estimator = calibrate_on_validation(pipeline, val[feature_cols], val["churn"])
            calibration_note = "sigmoid on validation"
        except (ValueError, IndexError) as exc:
            calibration_note = f"skipped ({exc})"
    test_proba = np.asarray(estimator.predict_proba(test[feature_cols])[:, 1])
    metrics = classification_metrics(
        test["churn"].to_numpy(), test_proba, threshold=suggestion["threshold"]
    )
    val_metrics = classification_metrics(
        val["churn"].to_numpy(), val_proba, threshold=suggestion["threshold"]
    )
    importance = global_importance(pipeline, train[feature_cols], train["churn"])
    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "calibrated": estimator,
        "calibration": calibration_note,
        "splits": {k: len(v) for k, v in splits.items()},
        "suggested_threshold": suggestion["threshold"],
        "test_metrics": metrics,
        "val_metrics": val_metrics,
        "test_proba": test_proba,
        "test_y": test["churn"].to_numpy(),
        "importance": importance,
        "feature_cols": feature_cols,
        "cutoff": str(dataset.labeling.cutoff.date()) if dataset.labeling else "",
        "horizon_days": dataset.labeling.horizon_days if dataset.labeling else 0,
    }


def model_selection_rows(models: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize trained or persisted churn models into comparable selection rows.

    Accepts in-session training bundles (``test_metrics`` + ``suggested_threshold``)
    and persisted metadata (``metrics`` + ``threshold``), so the training page,
    the cached default run, and the report all present the same candidate set.
    """
    rows: list[dict[str, Any]] = []
    for name in sorted(models):
        record = models[name]
        if not isinstance(record, Mapping):
            continue
        metrics = record.get("test_metrics") or record.get("metrics") or record
        if not isinstance(metrics, Mapping):
            metrics = record
        rows.append(
            {
                "model": str(name),
                "roc_auc": _optional_float(metrics.get("roc_auc")),
                "pr_auc": _optional_float(metrics.get("pr_auc")),
                "f1": _optional_float(metrics.get("f1")),
                "recall": _optional_float(metrics.get("recall")),
                "threshold": _optional_float(
                    record.get("suggested_threshold", record.get("threshold"))
                ),
            }
        )
    return rows


def _optional_float(value: Any) -> float | None:
    """Return a finite float or None for optional metric fields."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    """Recursively coerce numpy scalars/arrays into JSON-serializable values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def evaluation_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Serializable evaluation detail for reports, exports, and model cards.

    Captures the curves, confusion counts, and feature importances the churn
    page renders, so every downstream view can show the same evaluation without
    re-training or holding the estimator. Storing this payload in
    ``churn_model_metadata['evaluation']`` is what keeps the report current when
    a model is retrained or a different candidate is scored.
    """
    raw_y = bundle.get("test_y")
    raw_proba = bundle.get("test_proba")
    test_y = np.asarray([] if raw_y is None else raw_y, dtype=int)
    test_proba = np.asarray([] if raw_proba is None else raw_proba, dtype=float)
    roc_points: list[dict[str, float]] = []
    calibration_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    if len(test_y) and len(test_y) == len(test_proba):
        try:
            calibration_rows = [
                {str(key): _json_safe(value) for key, value in row.items()}
                for row in calibration_table(test_y, test_proba).to_dict(orient="records")
            ]
            gain_rows = [
                {str(key): _json_safe(value) for key, value in row.items()}
                for row in lift_gain_table(test_y, test_proba).to_dict(orient="records")
            ]
            if len(np.unique(test_y)) > 1:
                from sklearn.metrics import roc_curve

                fpr, tpr, _ = roc_curve(test_y, test_proba)
                # Evenly thin the curve so cached metadata stays small; the
                # endpoints and overall shape are preserved for the report.
                if len(fpr) > 250:
                    picks = np.linspace(0, len(fpr) - 1, 250).round().astype(int)
                    fpr, tpr = fpr[picks], tpr[picks]
                roc_points = [
                    {"fpr": round(float(x), 5), "tpr": round(float(y), 5)}
                    for x, y in zip(fpr, tpr, strict=True)
                ]
        except (ValueError, TypeError):
            # Sparse or single-class test slices leave the curves empty; the
            # metrics and confusion counts still travel.
            calibration_rows, gain_rows, roc_points = [], [], []
    importance = bundle.get("importance")
    importance_rows: list[dict[str, Any]] = []
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        keep = [column for column in ("feature", "importance") if column in importance.columns]
        importance_rows = [
            {str(key): _json_safe(value) for key, value in row.items()}
            for row in importance[keep].head(25).to_dict(orient="records")
        ]
    return {
        "model": str(bundle.get("model_name") or ""),
        "threshold": _optional_float(bundle.get("suggested_threshold")),
        "calibration": str(bundle.get("calibration") or ""),
        "cutoff": str(bundle.get("cutoff") or ""),
        "horizon_days": int(bundle.get("horizon_days") or 0),
        "splits": _json_safe(dict(bundle.get("splits") or {})),
        "test_metrics": _json_safe(dict(bundle.get("test_metrics") or {})),
        "val_metrics": _json_safe(dict(bundle.get("val_metrics") or {})),
        "roc": roc_points,
        "calibration_curve": calibration_rows,
        "gain": gain_rows,
        "importance": importance_rows,
    }


def score_all_customers(
    bundle: dict,
    scoring_frame: pd.DataFrame,
    bands: dict | None,
    threshold: float,
    model_version_name: str,
) -> pd.DataFrame:
    """Score a full customer frame with the trained (uncalibrated) pipeline."""
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]
    return score_customers(
        pipeline,
        scoring_frame[feature_cols],
        scoring_frame["customer_id"],
        bundle["importance"],
        model_version_name,
        threshold=threshold,
        bands=bands,
    )


def save_churn_run(bundle: dict, run_name: str) -> str:
    """Persist the trained pipeline with evaluation metadata sidecar."""
    metadata = {
        "run": run_name,
        "model": bundle["model_name"],
        "threshold": bundle["suggested_threshold"],
        "test_metrics": bundle["test_metrics"],
        "calibration": bundle["calibration"],
        "cutoff": bundle["cutoff"],
        "horizon_days": bundle["horizon_days"],
        "evaluation": evaluation_payload(bundle),
    }
    path = save_model(bundle["pipeline"], f"models/churn/{run_name}.joblib", metadata)
    from utils.model_card import churn_card_metadata, write_model_card

    write_model_card(
        f"reports/model_cards/{run_name}.md",
        version=run_name,
        **churn_card_metadata(
            bundle["model_name"],
            bundle["feature_cols"],
            bundle["test_metrics"],
            bundle["suggested_threshold"],
            bundle["cutoff"],
            bundle["horizon_days"],
        ),
    )
    return str(path)


def available_models() -> dict[str, str | None]:
    """Per-model None (ready) or install-guidance message."""
    status: dict[str, str | None] = {m: None for m in (*BASELINE_ESTIMATORS, RF_NAME, XGB_NAME)}
    try:
        require_optional("lightgbm", "lightgbm")
    except Exception as exc:
        status[LGBM_NAME] = str(exc)
    else:
        status[LGBM_NAME] = None
    return status

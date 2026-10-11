"""Phase 6 orchestration: time-aware datasets, training, calibration, scoring.

Splits are chronological: customers ordered by last observed transaction —
oldest 70% train, next 15% validation, newest 15% test. Oversampling touches
training data only. Calibration learns on validation only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.baseline_models import BASELINE_ESTIMATORS, build_model_pipeline, oversample_minority
from churn.calibration import calibrate_on_validation
from churn.dataset_builder import ChurnDataset
from churn.evaluation import classification_metrics, suggest_threshold
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
    }
    return str(save_model(bundle["pipeline"], f"models/churn/{run_name}.joblib", metadata))


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

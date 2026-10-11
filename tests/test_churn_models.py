"""Phase 6 tests: training, calibration, explainability, scoring, graceful paths."""

import numpy as np
import pandas as pd
import pytest

from churn.baseline_models import oversample_minority
from churn.dataset_builder import build_churn_dataset
from churn.evaluation import (
    calibration_table,
    classification_metrics,
    lift_gain_table,
    suggest_threshold,
)
from churn.explainability import customer_risk_drivers, shap_values
from churn.predict import assign_band, score_customers
from services.churn_service import available_models, time_split, train_churn_model
from utils.exceptions import CustomerIntelligenceError, InsufficientDataError

EXPECTED_SCORE_COLS = [
    "customer_id",
    "churn_probability",
    "risk_band",
    "predicted_class",
    "threshold",
    "primary_risk_drivers",
    "model_version",
    "score_timestamp",
]


def _separable_txns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(100):  # retained: weekly from January through December
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 30)))
        day = start
        while day <= pd.Timestamp("2024-12-31"):
            amount = max(10.0, round(float(rng.normal(120, 15)), 2))
            rows.append((f"G{i:03d}", day.date().isoformat(), amount))
            day += pd.Timedelta(days=7)
    for i in range(100):  # churned: 8–16 weekly transactions, all ending pre-cutoff
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 30)))
        for week in range(int(rng.integers(8, 17))):
            day = start + pd.Timedelta(weeks=week)
            amount = max(5.0, round(float(rng.normal(40, 10)), 2))
            rows.append((f"B{i:03d}", day.date().isoformat(), amount))
    return pd.DataFrame(rows, columns=["customer_id", "transaction_date", "amount_spent"])
    return pd.DataFrame(rows, columns=["customer_id", "transaction_date", "amount_spent"])


def _dataset():
    return build_churn_dataset(
        _separable_txns(), "2024-06-01", horizon_days=90, min_history_days=30
    )


def test_dataset_labels_match_behavior() -> None:
    """Good customers retained, early-only customers churned, leakage proof holds."""
    dataset = _dataset()
    by_id = dict(zip(dataset.frame["customer_id"], dataset.frame["churn"], strict=True))
    assert all(by_id[f"G{i:03d}"] == 0 for i in range(100))
    assert all(by_id[f"B{i:03d}"] == 1 for i in range(100))
    assert dataset.leakage_ok


def test_time_split_order_and_minimum() -> None:
    """Oldest customers train; tiny datasets are rejected loudly."""
    splits = time_split(_dataset())
    assert set(splits) == {"train", "val", "test"}
    assert (
        splits["train"]["first_transaction_date"].max()
        <= splits["test"]["first_transaction_date"].min()
    )
    for name, split in splits.items():
        assert split["churn"].nunique() == 2, f"{name} split is single-class"
    small = _dataset()
    with pytest.raises(InsufficientDataError):
        time_split(
            small.__class__(
                frame=small.frame.head(5),
                feature_cols=[],
                categorical_cols=[],
                labeling=small.labeling,
            )
        )


def test_baseline_models_separate_cleanly() -> None:
    """Logreg, RF, and HGB all achieve high AUC on separable behavior."""
    dataset = _dataset()
    for model in ("logreg", "random_forest", "histgb"):
        bundle = train_churn_model(dataset, model, calibrate=False)
        assert bundle["test_metrics"]["roc_auc"] is not None
        assert bundle["test_metrics"]["roc_auc"] > 0.85
        assert 0.0 < bundle["suggested_threshold"] < 1.0
        assert len(bundle["importance"]) > 0


def test_calibration_and_oversample_paths() -> None:
    """Calibration learns on validation; oversample balances training only."""
    dataset = _dataset()
    bundle = train_churn_model(dataset, "logreg", calibrate=True)
    assert bundle["calibration"].startswith("sigmoid")
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 0, 1])
    Xo, yo = oversample_minority(X, y, random_state=42)
    assert (pd.Series(yo).value_counts() == 3).all()
    assert len(Xo) == 6


def test_xgb_trains_and_lgbm_guides() -> None:
    """XGBoost trains when installed; LightGBM absence gives install guidance."""
    dataset = _dataset()
    bundle = train_churn_model(dataset, "xgboost", calibrate=False)
    assert bundle["test_metrics"]["roc_auc"] > 0.85
    models = available_models()
    assert models["xgboost"] is None
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        assert models["lightgbm"] is not None and "pip install" in models["lightgbm"]
        from churn.train_lightgbm import build_lgbm_pipeline

        with pytest.raises(CustomerIntelligenceError, match="pip install lightgbm"):
            build_lgbm_pipeline([], [])


def test_shap_absence_guides() -> None:
    """SHAP explanations raise install guidance without the package."""
    try:
        import shap  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install shap"):
            shap_values(None, pd.DataFrame({"a": [1]}))


def test_metrics_tables_sane() -> None:
    """Calibration bins, lift deciles, and threshold suggestion behave."""
    y = np.array([0] * 50 + [1] * 50)
    rng = np.random.default_rng(42)
    proba = np.clip(
        np.concatenate([rng.normal(0.3, 0.1, 50), rng.normal(0.7, 0.1, 50)]), 0.01, 0.99
    )
    metrics = classification_metrics(y, proba, threshold=0.5)
    assert metrics["roc_auc"] > 0.9
    assert metrics["confusion"]["tp"] + metrics["confusion"]["fn"] == 50
    suggestion = suggest_threshold(y, proba)
    assert 0.0 < suggestion["threshold"] < 1.0
    assert len(calibration_table(y, proba)) >= 2
    gain = lift_gain_table(y, proba)
    assert gain["lift"].iloc[0] > 1.0
    assert gain["cum_gain"].iloc[-1] == 1.0


def test_scoring_schema_and_bands() -> None:
    """Score output matches the spec schema; bands follow configured edges."""
    dataset = _dataset()
    bundle = train_churn_model(dataset, "logreg", calibrate=False)
    frame = dataset.frame
    scores = score_customers(
        bundle["pipeline"],
        frame[bundle["feature_cols"]],
        frame["customer_id"],
        bundle["importance"],
        "logreg",
        threshold=0.5,
    )
    assert list(scores.columns) == EXPECTED_SCORE_COLS
    assert scores["risk_band"].isin(["low", "medium", "high", "critical"]).all()
    assert scores["primary_risk_drivers"].str.len().gt(0).all()
    assert assign_band(0.1) == "low" and assign_band(0.95) == "critical"


def test_customer_drivers_pick_risky_side() -> None:
    """Drivers cite features where the customer exceeds the risky side of median."""
    importance = pd.DataFrame({"feature": ["recency_days", "frequency"], "importance": [0.5, 0.4]})
    row = pd.Series({"recency_days": 120.0, "frequency": 30.0})
    medians = pd.Series({"recency_days": 30.0, "frequency": 5.0})
    drivers = customer_risk_drivers(row, medians, importance, {"frequency": -1})
    assert any("recency_days" in d for d in drivers)
    assert not any("frequency" in d for d in drivers)  # high frequency is protective

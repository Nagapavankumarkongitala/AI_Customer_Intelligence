"""Random Forest churn trainer on the shared preprocessed pipeline."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from churn.baseline_models import build_preprocessor

MODEL_NAME = "random_forest"


def build_rf_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int = 42,
    n_estimators: int = 300,
) -> Pipeline:
    """Random Forest with balanced class weights (no scaling needed for trees)."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

"""LightGBM churn trainer. Optional dependency with install guidance."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from churn.baseline_models import build_preprocessor
from segmentation.common import require_optional

MODEL_NAME = "lightgbm"


def build_lgbm_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int = 42,
) -> Pipeline:
    """LightGBM with balanced class weights. Raises guidance text when absent."""
    require_optional("lightgbm", "lightgbm")
    from lightgbm import LGBMClassifier

    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                LGBMClassifier(
                    class_weight="balanced",
                    random_state=random_state,
                    verbose=-1,
                ),
            ),
        ]
    )

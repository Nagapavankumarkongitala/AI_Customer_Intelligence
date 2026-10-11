"""XGBoost churn trainer. Optional dependency with install guidance."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from churn.baseline_models import build_preprocessor
from segmentation.common import require_optional

MODEL_NAME = "xgboost"


def build_xgb_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int = 42,
) -> Pipeline:
    """XGBoost with scale_pos_weight set from the training label balance."""
    require_optional("xgboost", "xgboost")
    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def set_balance(pipeline: Pipeline, y_train) -> Pipeline:
    """Set scale_pos_weight = neg/pos from TRAINING labels only. Returns pipeline."""
    import numpy as np
    import pandas as pd

    y = pd.Series(np.asarray(y_train)).value_counts()
    weight = float(y.get(0, 1) / max(y.get(1, 1), 1))
    pipeline.named_steps["model"].set_params(scale_pos_weight=weight)
    return pipeline

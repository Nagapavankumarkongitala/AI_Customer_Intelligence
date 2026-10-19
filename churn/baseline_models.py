"""Shared sklearn churn pipeline factory plus baseline estimators.

One ColumnTransformer handles numerics (median impute) and categoricals
(most-frequent impute + one-hot). Estimators carry class_weight='balanced'
unless they manage imbalance themselves. Sampling (oversampling) is applied to
training data only, in churn_service — never here.

The one-hot block is sparse for every estimator except
``HistGradientBoostingClassifier``, which rejects sparse input outright; its
preprocessor emits dense output instead (``dense=True``).
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASELINE_ESTIMATORS = ("logreg", "histgb")

#: Estimators that cannot consume sparse input and need a dense one-hot block.
DENSE_INPUT_ESTIMATORS = ("histgb",)


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = False,
    dense: bool = False,
) -> ColumnTransformer:
    """ColumnTransformer for churn modeling frames.

    ``dense=True`` emits a dense one-hot block, for estimators that reject
    sparse input (see ``DENSE_INPUT_ESTIMATORS``).
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_estimator(name: str, random_state: int = 42):
    """Baseline estimator by name: 'logreg' or 'histgb' (both class-balanced)."""
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    if name == "histgb":
        return HistGradientBoostingClassifier(random_state=random_state, class_weight="balanced")
    raise ValueError(f"Unknown baseline estimator '{name}'. Choose from {BASELINE_ESTIMATORS}.")


def build_model_pipeline(
    name: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int = 42,
) -> Pipeline:
    """Full fit/predict pipeline; logreg scales numerics, trees do not."""
    preprocessor = build_preprocessor(
        numeric_cols,
        categorical_cols,
        scale_numeric=(name == "logreg"),
        dense=(name in DENSE_INPUT_ESTIMATORS),
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", make_estimator(name, random_state)),
        ]
    )


def oversample_minority(
    X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Random minority duplication for TRAINING data only. Returns balanced arrays."""
    rng = np.random.default_rng(random_state)
    X, y = np.asarray(X), np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    target = counts.max()
    extra_X, extra_y = [X], [y]
    for cls, count in zip(classes, counts, strict=True):
        need = target - count
        if need > 0:
            idx = rng.choice(np.where(y == cls)[0], size=need, replace=True)
            extra_X.append(X[idx])
            extra_y.append(y[idx])
    order = rng.permutation(sum(len(a) for a in extra_X))
    return np.concatenate(extra_X)[order], np.concatenate(extra_y)[order]

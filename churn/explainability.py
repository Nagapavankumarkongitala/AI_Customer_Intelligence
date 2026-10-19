"""Explainability: coefficients, importances, permutation, SHAP, risk drivers.

Global importance is permutation-based (model-agnostic, same units for every
model). Customer-level drivers use SHAP when installed; otherwise the top global
features where the customer sits on the risky side of the population median.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

N_DRIVERS = 2


class ShapBudgetExceeded(RuntimeError):
    """Exact TreeExplainer work exceeds the safe budget (median drivers instead)."""


#: rows × trees × encoded-features ceiling for exact TreeExplainer. Past this
#: the C extension stalls with zero CPU (observed: 500 rows × 300 trees × ~65
#: features wedged; 200 × 300 × ~20 returns instantly). Callers fall back to
#: median-based drivers, which carry the same output schema.
MAX_TREE_SHAP_WORK = 3_000_000


def feature_names_of(pipeline) -> list[str]:
    """Post-preprocessing feature names for a fitted churn pipeline."""
    preprocess = pipeline.named_steps["preprocess"]
    if hasattr(preprocess, "get_feature_names_out"):
        return list(preprocess.get_feature_names_out())
    return []


def global_importance(
    pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Permutation importance on raw (pre-pipeline) columns, descending."""
    result = permutation_importance(pipeline, X, y, n_repeats=n_repeats, random_state=random_state)
    table = (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": result.importances_mean.round(4),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return table


def logreg_effects(pipeline) -> pd.DataFrame:
    """Coefficients and odds ratios for the logistic baseline (encoded space)."""
    model = pipeline.named_steps["model"]
    names = feature_names_of(pipeline)
    coefs = np.asarray(model.coef_).ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coefs.round(4)})
    table["odds_ratio"] = np.exp(coefs).round(4)
    return table.sort_values("coefficient", ascending=False).reset_index(drop=True)


def _shap_explainer(model, background):
    """TreeExplainer for tree ensembles, LinearExplainer for linear models.

    Newer shap releases no longer accept linear estimators in TreeExplainer, so
    linear churn baselines (logreg et al.) route to LinearExplainer instead.
    Linearity is duck-typed on ``coef_`` (stable across sklearn versions);
    anything else keeps TreeExplainer and its own informative error.
    """
    import shap

    if hasattr(model, "coef_"):
        dense = background.toarray() if hasattr(background, "toarray") else background
        return shap.LinearExplainer(model, dense), dense
    # TreeExplainer validates inputs with np.isnan, which rejects sparse and
    # object-dtype matrices. The churn preprocessor emits sparse output once
    # one-hot cardinality pushes density below the ColumnTransformer threshold
    # (real geo data always does), so densify here like the linear path.
    dense = background.toarray() if hasattr(background, "toarray") else np.asarray(background)
    if not np.issubdtype(dense.dtype, np.number):
        dense = dense.astype(float)
    return shap.TreeExplainer(model), dense


def shap_values(pipeline, X: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame:
    """Per-row SHAP values (binary positive class). Raises guidance when absent."""
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Customer-level SHAP explanations need the optional 'shap' package. "
            "Install it with 'pip install shap'."
        ) from exc
    sample = X.head(max_rows)
    transformed = pipeline.named_steps["preprocess"].transform(sample)
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        n_trees = getattr(model, "n_estimators", 1) or 1
        work = len(sample) * n_trees * np.shape(transformed)[1]
        if work > MAX_TREE_SHAP_WORK:
            raise ShapBudgetExceeded(
                f"SHAP explanations skipped: {work:,} row-tree-feature work exceeds "
                f"the {MAX_TREE_SHAP_WORK:,} budget. Median-based drivers are used instead."
            )
    explainer, background = _shap_explainer(model, transformed)
    values = explainer.shap_values(background)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            # Newer shap returns (rows, features, classes) instead of a per-class list.
            values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    names = feature_names_of(pipeline)
    return pd.DataFrame(
        np.asarray(values), index=sample.index, columns=names[: np.asarray(values).shape[1]]
    )


def customer_risk_drivers(
    customer_row: pd.Series,
    population_medians: pd.Series,
    importance: pd.DataFrame,
    positive_side: dict[str, int],
    n: int = N_DRIVERS,
) -> list[str]:
    """Top global features where this customer sits on the risky side of median."""
    drivers: list[str] = []
    for feature in importance["feature"].tolist():
        if len(drivers) >= n or feature not in customer_row or feature not in population_medians:
            continue
        value, median = customer_row[feature], population_medians[feature]
        if pd.isna(value) or pd.isna(median):
            continue
        side = positive_side.get(feature, 1)
        risky = (value >= median) if side >= 0 else (value <= median)
        if risky:
            direction = "above" if value >= median else "below"
            drivers.append(f"{feature} ({direction} typical: {value} vs median {median})")
    return drivers

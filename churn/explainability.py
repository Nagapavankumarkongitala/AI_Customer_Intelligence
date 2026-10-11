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


def shap_values(pipeline, X: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame:
    """Per-row SHAP values (binary positive class). Raises guidance when absent."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "Customer-level SHAP explanations need the optional 'shap' package. "
            "Install it with 'pip install shap'."
        ) from exc
    sample = X.head(max_rows)
    transformed = pipeline.named_steps["preprocess"].transform(sample)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
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

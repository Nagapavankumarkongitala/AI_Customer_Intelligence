"""Risk scoring: probabilities, bands, classes, drivers in the output schema.

Output columns: customer_id, churn_probability, risk_band, predicted_class,
threshold, primary_risk_drivers, model_version, score_timestamp. Bands come from
config.yaml (churn.risk_bands); drivers from explainability with SHAP preferred.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from churn.explainability import (
    customer_risk_drivers,
    shap_values,
)

DEFAULT_BANDS = {
    "low": [0.0, 0.30],
    "medium": [0.30, 0.60],
    "high": [0.60, 0.80],
    "critical": [0.80, 1.0],
}


def assign_band(probability: float, bands: dict | None = None) -> str:
    """Map a probability to its configured risk band (upper-bound inclusive)."""
    bands = bands or DEFAULT_BANDS
    for name, (lo, hi) in bands.items():
        if lo <= probability <= hi or (hi == 1.0 and probability >= lo):
            return name
    return "medium"


def score_customers(
    pipeline,
    X: pd.DataFrame,
    customer_ids: pd.Series,
    importance: pd.DataFrame,
    model_name: str,
    threshold: float = 0.5,
    bands: dict | None = None,
) -> pd.DataFrame:
    """Score every customer with drivers, bands, and full output schema."""
    proba = np.asarray(pipeline.predict_proba(X)[:, 1], dtype=float)
    medians = X.select_dtypes(include="number").median(numeric_only=True)
    positive_side = {f: 1 for f in importance["feature"].tolist()}
    try:
        shap_frame = shap_values(pipeline, X)
        use_shap = True
    except ImportError:
        shap_frame = None
        use_shap = False
    # shap_values caps rows for interactive latency; rows beyond the cap fall
    # back to the median-based drivers instead of indexing out of bounds.
    n_shap = len(shap_frame) if use_shap and shap_frame is not None else 0
    version = f"{model_name}_v{date.today().isoformat()}"
    stamped = datetime.now(UTC).isoformat()
    rows: list[dict] = []
    for i, cid in enumerate(customer_ids.tolist()):
        if i < n_shap:
            row_shap = shap_frame.iloc[i]
            top = row_shap.abs().sort_values(ascending=False).head(2)
            drivers = [f"{feat} (SHAP {row_shap[feat]:+.3f})" for feat in top.index]
        else:
            drivers = customer_risk_drivers(X.iloc[i], medians, importance, positive_side)
        probability = round(float(proba[i]), 4)
        rows.append(
            {
                "customer_id": cid,
                "churn_probability": probability,
                "risk_band": assign_band(probability, bands),
                "predicted_class": int(probability >= threshold),
                "threshold": threshold,
                "primary_risk_drivers": "; ".join(drivers) if drivers else "no dominant driver",
                "model_version": version,
                "score_timestamp": stamped,
            }
        )
    return pd.DataFrame(rows)

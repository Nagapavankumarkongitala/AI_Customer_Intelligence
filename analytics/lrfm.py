"""LRFM scoring: RFM plus relationship-Length. LRFM score = 1000*L + 100*R + 10*F + M.

Length (relationship_length_days) scores ascending like F and M. Segment naming
reuses the RFM component rules; L differentiates tenure inside profiles.
"""

from __future__ import annotations

import pandas as pd

from analytics.rfm import VALID_QUANTILES, _quantile_scores, name_rfm_segments
from utils.validators import require_min_rows


def score_lrfm(features: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Score length/recency/frequency/monetary and name segments. Returns a new frame."""
    if quantiles not in VALID_QUANTILES:
        raise ValueError(f"quantiles must be one of {VALID_QUANTILES}, got {quantiles}.")
    require_min_rows(features, 1, "customers")
    required = (
        "customer_id",
        "relationship_length_days",
        "recency_days",
        "frequency",
        "monetary_value",
    )
    for col in required:
        if col not in features.columns:
            raise KeyError(f"Customer features are missing required column: {col}")
    scored = features[[*required]].copy()
    scored["L"] = _quantile_scores(scored["relationship_length_days"], quantiles, high_is_good=True)
    scored["R"] = _quantile_scores(scored["recency_days"], quantiles, high_is_good=False)
    scored["F"] = _quantile_scores(scored["frequency"], quantiles, high_is_good=True)
    scored["M"] = _quantile_scores(scored["monetary_value"], quantiles, high_is_good=True)
    scored["lrfm_score"] = 1000 * scored["L"] + 100 * scored["R"] + 10 * scored["F"] + scored["M"]
    scored["segment"] = name_rfm_segments(scored, quantiles)
    return scored

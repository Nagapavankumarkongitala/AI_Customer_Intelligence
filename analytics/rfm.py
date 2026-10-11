"""RFM scoring and transparent segment naming.

Scores use rank-percentiles (ties share the same score) and stay safe for small
or constant inputs. Recency is reversed (lower recency -> higher R).
RFM score = 100*R + 10*F + M. Raw metrics and score components stay separate;
segments are named from components + raw values, never from the combined score.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from utils.validators import require_min_rows

VALID_QUANTILES = (3, 4, 5)

# Order matters: first matching rule wins. Thresholds derive from q so the same
# rules work for 3, 4, and 5 quantile levels.
Rule = tuple[str, Callable[[int, int, int, int], bool]]


def _quantile_scores(values: pd.Series, q: int, high_is_good: bool) -> pd.Series:
    """Rank-percentile scores in 1..q. Ties share ranks, hence scores; constant input is safe."""
    ranks = values.rank(pct=True, method="average")
    raw = np.ceil(ranks * q).clip(1, q).astype(int)
    return raw if high_is_good else (q + 1 - raw)


def default_naming_rules() -> list[Rule]:
    """Transparent (R, F, M, q) rules for the ten business segments."""

    def bands(q: int) -> tuple[int, int]:
        low_max = 2 if q >= 4 else 1
        return low_max, q - 1

    return [
        ("Champions", lambda r, f, m, q: r >= bands(q)[1] and f >= bands(q)[1]),
        ("Loyal Customers", lambda r, f, m, q: f >= bands(q)[1] and r > bands(q)[0]),
        ("New Customers", lambda r, f, m, q: r == q and f <= bands(q)[0] and m <= bands(q)[0]),
        ("Promising", lambda r, f, m, q: r == q and f <= bands(q)[0]),
        ("Potential Loyalists", lambda r, f, m, q: r >= bands(q)[1]),
        ("Cannot Lose Them", lambda r, f, m, q: r <= bands(q)[0] and f >= bands(q)[1]),
        ("At Risk", lambda r, f, m, q: r <= bands(q)[0] and f > bands(q)[0]),
        ("Hibernating", lambda r, f, m, q: r <= bands(q)[0] and m >= bands(q)[1]),
        ("Needs Attention", lambda r, f, m, q: bands(q)[0] < r < bands(q)[1]),
        ("Lost Customers", lambda r, f, m, q: True),
    ]


def name_rfm_segments(scored: pd.DataFrame, q: int, rules: list[Rule] | None = None) -> pd.Series:
    """Apply ordered naming rules to scored rows. Every row is named (catch-all last)."""
    active = rules if rules is not None else default_naming_rules()
    names: list[str] = []
    for _, row in scored.iterrows():
        r, f, m = int(row["R"]), int(row["F"]), int(row["M"])
        names.append(next(name for name, pred in active if pred(r, f, m, q)))
    return pd.Series(names, index=scored.index, name="segment")


def score_rfm(features: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Score recency/frequency/monetary and name segments. Returns a new frame."""
    if quantiles not in VALID_QUANTILES:
        raise ValueError(f"quantiles must be one of {VALID_QUANTILES}, got {quantiles}.")
    require_min_rows(features, 1, "customers")
    for col in ("customer_id", "recency_days", "frequency", "monetary_value"):
        if col not in features.columns:
            raise KeyError(f"Customer features are missing required column: {col}")
    scored = features[["customer_id", "recency_days", "frequency", "monetary_value"]].copy()
    scored["R"] = _quantile_scores(scored["recency_days"], quantiles, high_is_good=False)
    scored["F"] = _quantile_scores(scored["frequency"], quantiles, high_is_good=True)
    scored["M"] = _quantile_scores(scored["monetary_value"], quantiles, high_is_good=True)
    scored["rfm_score"] = 100 * scored["R"] + 10 * scored["F"] + scored["M"]
    scored["segment"] = name_rfm_segments(scored, quantiles)
    return scored

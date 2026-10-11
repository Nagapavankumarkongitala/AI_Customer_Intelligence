"""Segment-difference testing: ANOVA / Welch ANOVA / Kruskal-Wallis, chi-square /
Fisher with Cramer's V, and Holm-Bonferroni correction across batched tests.

Method choice is automatic and reported: Shapiro (subsampled, seeded) for normality
plus Levene for equal variances selects ANOVA; normal-but-unequal selects Welch's
ANOVA (manual formula, see docstring); otherwise Kruskal-Wallis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
_RNG = np.random.default_rng(42)


@dataclass
class SegmentTestResult:
    """One hypothesis test outcome with effect size and plain-English reading."""

    variable: str
    test: str
    statistic: float
    p_value: float
    p_corrected: float | None = None
    reject: bool | None = None
    effect_size: float | None = None
    effect_name: str = ""
    interpretation: str = ""


def welch_anova(*groups: np.ndarray) -> tuple[float, float]:
    """Welch's F-test for k groups with unequal variances.

    F = between-group weighted variance / (1 + 2(k-2)/(k^2-1) * m) with
    df1 = k-1, df2 = (k^2-1)/(3m), m = sum((1-w_i/W)^2/(n_i-1)).
    Requires >= 2 observations and nonzero variance per group.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    if len(arrays) < 2 or any(len(a) < 2 or np.var(a, ddof=1) == 0 for a in arrays):
        raise ValueError("Welch ANOVA needs >= 2 groups with >= 2 distinct values each.")
    k = len(arrays)
    sizes = np.array([len(a) for a in arrays], dtype=float)
    means = np.array([a.mean() for a in arrays])
    variances = np.array([a.var(ddof=1) for a in arrays])
    weights = sizes / variances
    total_weight = weights.sum()
    grand = float((weights * means).sum() / total_weight)
    between = float((weights * (means - grand) ** 2).sum() / (k - 1))
    m_term = float((((1 - weights / total_weight) ** 2) / (sizes - 1)).sum())
    if k == 2:
        correction = 1.0
    else:
        correction = 1 + 2 * (k - 2) / (k**2 - 1) * m_term
    f_stat = between / correction
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * m_term) if m_term > 0 else float("inf")
    return float(f_stat), float(stats.f.sf(f_stat, df1, df2))


def _normal_enough(values: np.ndarray) -> bool:
    sample = values if len(values) <= 500 else _RNG.choice(values, 500, replace=False)
    if len(sample) < 3:
        return False
    return bool(stats.shapiro(sample).pvalue > ALPHA)


def eta_squared(*groups: np.ndarray) -> float:
    """Share of total variance explained by group membership (ANOVA-family effect)."""
    pooled = np.concatenate(groups)
    total = float(((pooled - pooled.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    between = float(sum(len(g) * (g.mean() - pooled.mean()) ** 2 for g in groups))
    return between / total


def compare_continuous_across_segments(values: pd.Series, segments: pd.Series) -> SegmentTestResult:
    """Compare a numeric variable across segments with an automatically chosen test."""
    frame = pd.DataFrame(
        {
            "v": pd.to_numeric(values.reset_index(drop=True), errors="coerce"),
            "s": segments.reset_index(drop=True).astype(str),
        }
    )
    frame = frame.dropna()
    groups = [g["v"].to_numpy() for _, g in frame.groupby("s")]
    variable = str(values.name or "value")
    if len(groups) < 2:
        return SegmentTestResult(
            variable,
            "none",
            float("nan"),
            1.0,
            interpretation="Only one segment present; no test run.",
        )
    if all(_normal_enough(g) for g in groups):
        if len(groups) == 2 or stats.levene(*groups).pvalue > ALPHA:
            stat, p = stats.f_oneway(*groups)
            test, effect, effect_name = "ANOVA", eta_squared(*groups), "eta_squared"
        else:
            stat, p = welch_anova(*groups)
            test, effect, effect_name = "Welch ANOVA", eta_squared(*groups), "eta_squared"
    else:
        stat, p = stats.kruskal(*groups)
        n = sum(len(g) for g in groups)
        effect = float(stat / ((n**2 - 1) / (n + 1))) if n > 1 else 0.0
        test, effect_name = "Kruskal-Wallis", "epsilon_squared"
    return SegmentTestResult(
        variable,
        test,
        round(float(stat), 4),
        float(p),
        effect_size=round(float(effect), 4),
        effect_name=effect_name,
        interpretation=f"{test} p={p:.4f} "
        f"({'differs across segments' if p < ALPHA else 'no detected difference'}).",
    )


def cramers_v(contingency: pd.DataFrame) -> float:
    """Effect size for categorical association in [0, 1]."""
    chi2 = float(stats.chi2_contingency(contingency, correction=False)[0])
    n = float(contingency.to_numpy().sum())
    dims = min(contingency.shape) - 1
    if n == 0 or dims <= 0:
        return 0.0
    return min(1.0, (chi2 / (n * dims)) ** 0.5)


def categorical_association(row_labels: pd.Series, col_labels: pd.Series) -> SegmentTestResult:
    """Chi-square (Fisher exact for 2×2 with small expected counts) + Cramer's V."""
    contingency = pd.crosstab(row_labels.fillna("Unknown"), col_labels.fillna("Unknown"))
    variable = f"{row_labels.name} × {col_labels.name}"
    if contingency.shape == (2, 2):
        _, p, _, expected = stats.chi2_contingency(contingency)
        if (expected < 5).any():
            _, p = stats.fisher_exact(contingency)
            test = "Fisher exact"
        else:
            test = "Chi-square"
        stat = float(stats.chi2_contingency(contingency, correction=False)[0])
    else:
        stat, p, _, _ = stats.chi2_contingency(contingency)
        test = "Chi-square"
    effect = cramers_v(contingency)
    return SegmentTestResult(
        variable,
        test,
        round(float(stat), 4),
        float(p),
        effect_size=round(effect, 4),
        effect_name="cramers_v",
        interpretation=f"{test} p={p:.4f}, Cramer's V={effect:.3f} "
        f"({'associated' if p < ALPHA else 'no detected association'}).",
    )


def holm_reject(p_values: list[float], alpha: float = ALPHA) -> list[bool]:
    """Holm-Bonferroni step-down decisions for an ordered p-value batch."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def run_segment_tests(
    features: pd.DataFrame,
    segments: pd.Series,
    continuous: list[str],
    categorical: list[str],
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Batch all segment-difference tests and apply Holm correction across them."""
    results: list[SegmentTestResult] = []
    available = [c for c in continuous if c in features.columns]
    for col in available:
        results.append(compare_continuous_across_segments(features[col], segments))
    for col in [c for c in categorical if c in features.columns]:
        results.append(categorical_association(segments, features[col]))
    decisions = holm_reject([r.p_value for r in results], alpha=alpha)
    rows = []
    for result, rejected in zip(results, decisions, strict=True):
        result.p_corrected = None  # Holm is a decision procedure; raw p reported.
        result.reject = bool(rejected)
        rows.append(
            {
                "variable": result.variable,
                "test": result.test,
                "statistic": result.statistic,
                "p_value": round(result.p_value, 4),
                "significant_after_holm": result.reject,
                "effect_size": result.effect_size,
                "effect_name": result.effect_name,
                "interpretation": result.interpretation,
            }
        )
    return pd.DataFrame(rows)

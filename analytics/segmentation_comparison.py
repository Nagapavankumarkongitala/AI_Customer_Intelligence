"""Traditional-vs-AI segmentation comparison: agreement, balance, value, stability.

Recommendation policy (deliberately NOT max-silhouette): each dimension crowns its
own winner, and the operational pick must clear stability/size/noise gates first,
preferring interpretability with silhouette as tie-break. The rationale always
names what a silhouette-only pick would have chosen so a human can disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from analytics.statistical_tests import eta_squared

STABILITY_GATE = 0.5
MIN_SHARE_GATE = 0.05
NOISE_GATE_PCT = 15.0


def crosstab_counts(labels_a: pd.Series, labels_b: pd.Series) -> pd.DataFrame:
    """Contingency table of two labelings aligned on index."""
    aligned = pd.DataFrame({"a": labels_a, "b": labels_b}).dropna()
    return pd.crosstab(aligned["a"], aligned["b"])


def agreement_ari(labels_a: pd.Series, labels_b: pd.Series) -> float:
    """Permutation-invariant agreement between two labelings."""
    aligned = pd.DataFrame({"a": labels_a, "b": labels_b}).dropna()
    if aligned.empty:
        return 0.0
    return round(float(adjusted_rand_score(aligned["a"], aligned["b"])), 4)


def size_balance(sizes: dict[int, int]) -> dict:
    """Balance via minimum share and normalized entropy (1 = perfectly even)."""
    counts = np.array(sorted(sizes.values()), dtype=float)
    total = counts.sum()
    if total == 0 or len(counts) < 2:
        return {"min_share": 0.0, "evenness": 0.0}
    shares = counts / total
    entropy = float(-(shares * np.log(shares, where=shares > 0)).sum())
    return {
        "min_share": round(float(shares.min()), 4),
        "evenness": round(entropy / np.log(len(counts)), 4),
    }


def revenue_concentration(revenues: dict[str, float]) -> dict:
    """Top-segment revenue share and HHI across segments (lower = more spread)."""
    values = np.array(sorted(revenues.values()), dtype=float)
    total = values.sum()
    if total == 0 or len(values) == 0:
        return {"top_segment_share": 0.0, "hhi": 0.0}
    shares = values / total
    return {
        "top_segment_share": round(float(shares.max()), 4),
        "hhi": round(float((shares**2).sum()), 4),
    }


def value_separation(values: pd.Series, labels: pd.Series) -> float:
    """Share of monetary variance explained by segment membership (eta squared)."""
    frame = pd.DataFrame(
        {
            "v": pd.to_numeric(values, errors="coerce"),
            "s": labels.astype(str),
        }
    ).dropna()
    groups = [g["v"].to_numpy() for _, g in frame.groupby("s")]
    if len(groups) < 2:
        return 0.0
    return round(float(eta_squared(*groups)), 4)


def churn_rate_by_segment(labels: pd.Series, churn: pd.Series | None) -> pd.DataFrame | None:
    """Mean churn rate per segment. None when churn labels are unavailable (pre-Phase 6)."""
    if churn is None:
        return None
    frame = pd.DataFrame({"s": labels.astype(str), "c": pd.to_numeric(churn, errors="coerce")})
    frame = frame.dropna()
    if frame.empty:
        return None
    out = frame.groupby("s")["c"].agg(customers="size", churn_rate="mean").reset_index()
    out["churn_rate"] = out["churn_rate"].round(4)
    return out.rename(columns={"s": "segment"})


def interpretability_checklist(
    n_segments: int,
    has_names: bool,
    noise_pct: float,
    stability_ari: float | None,
    min_share: float,
) -> list[dict]:
    """Human-readable checklist; machines fill evidence, humans judge fit."""
    stability_ok = stability_ari is not None and stability_ari >= 0.7
    return [
        {"item": "At most 8 segments", "pass": n_segments <= 8, "detail": f"{n_segments} segments"},
        {
            "item": "Business names assigned",
            "pass": bool(has_names),
            "detail": "named" if has_names else "numeric IDs only",
        },
        {"item": "Noise below 10%", "pass": noise_pct < 10.0, "detail": f"{noise_pct}% noise"},
        {
            "item": "Stable across seeds (ARI ≥ 0.7)",
            "pass": stability_ok,
            "detail": f"ARI {stability_ari}" if stability_ari is not None else "not measured",
        },
        {
            "item": "Smallest segment ≥ 5% of base",
            "pass": min_share >= MIN_SHARE_GATE,
            "detail": f"min share {min_share}",
        },
    ]


def recommend_model(candidates: list[dict]) -> dict:
    """Crown per-dimension winners and an operational pick with rationale.

    Each candidate: {name, silhouette|None, davies_bouldin|None, stability_ari|None,
    value_eta2, min_share, noise_pct, interpretability_passes, n_segments}.
    Missing metrics never crash the ranking; they lose tie-breaks.
    """
    if not candidates:
        raise ValueError("No candidate models to recommend from.")
    valid = [c for c in candidates if c.get("silhouette") is not None] or candidates

    def _num(c: dict, key: str, default: float) -> float:
        value = c.get(key)
        return float(value) if value is not None else default

    best_statistical = max(valid, key=lambda c: _num(c, "silhouette", -2.0))["name"]
    stable = [c for c in candidates if _num(c, "stability_ari", -2.0) >= 0.0]
    best_stability = (
        max(stable, key=lambda c: _num(c, "stability_ari", -2.0))["name"] if stable else None
    )
    best_interpretability = max(
        candidates,
        key=lambda c: (c.get("interpretability_passes", 0), -c.get("n_segments", 99)),
    )["name"]
    best_revenue = max(candidates, key=lambda c: _num(c, "value_eta2", -1.0))["name"]

    gated = [
        c
        for c in candidates
        if _num(c, "stability_ari", -2.0) >= STABILITY_GATE
        and _num(c, "min_share", 0.0) >= MIN_SHARE_GATE
        and _num(c, "noise_pct", 100.0) < NOISE_GATE_PCT
    ]
    pool = gated or candidates
    operational = max(
        pool,
        key=lambda c: (c.get("interpretability_passes", 0), _num(c, "silhouette", -2.0)),
    )["name"]
    gate_note = (
        "cleared"
        if gated
        else "no candidate cleared all gates; fell back to most interpretable/stable"
    )
    rationale = (
        f"Best statistical separation: {best_statistical}. Best stability: {best_stability}. "
        f"Best interpretability: {best_interpretability}. Best revenue differentiation: "
        f"{best_revenue}. Operational pick: {operational} ({gate_note} stability ≥ "
        f"{STABILITY_GATE}, min share ≥ {MIN_SHARE_GATE}, noise < {NOISE_GATE_PCT}%). "
        f"A silhouette-only rule would have chosen {best_statistical}; override it below "
        "if the evidence says otherwise."
    )
    return {
        "best_statistical": best_statistical,
        "best_stability": best_stability,
        "best_interpretability": best_interpretability,
        "best_revenue": best_revenue,
        "operational_pick": operational,
        "gated_pool": [c["name"] for c in gated],
        "rationale": rationale,
    }

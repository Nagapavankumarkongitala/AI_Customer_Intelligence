"""Phase 5 tests: agreement, balance, separation, checklists, recommendation."""

import pandas as pd
import pytest

from analytics.segmentation_comparison import (
    agreement_ari,
    churn_rate_by_segment,
    crosstab_counts,
    interpretability_checklist,
    recommend_model,
    revenue_concentration,
    size_balance,
    value_separation,
)


def test_agreement_permutation_invariant() -> None:
    """Relabeled-but-identical partitions score ARI 1; unrelated score near 0."""
    a = pd.Series([0, 0, 1, 1, 2, 2])
    b = pd.Series([2, 2, 0, 0, 1, 1])
    assert agreement_ari(a, b) == 1.0
    c = pd.Series([0, 1, 2, 0, 1, 2])
    assert agreement_ari(a, c) < 0.5
    assert crosstab_counts(a, b).to_numpy().sum() == 6


def test_balance_and_concentration_math() -> None:
    """Even splits score evenness 1; skewed revenue concentrates."""
    assert size_balance({0: 50, 1: 50})["evenness"] == 1.0
    skewed = size_balance({0: 90, 1: 10})
    assert skewed["min_share"] == 0.1 and skewed["evenness"] < 1.0
    conc = revenue_concentration({"A": 900.0, "B": 100.0})
    assert conc["top_segment_share"] == 0.9
    assert conc["hhi"] == round(0.9**2 + 0.1**2, 4)


def test_value_separation_extremes() -> None:
    """Perfectly separated values score ~1; identical values score 0."""
    values = pd.Series([10.0, 10.0, 100.0, 100.0])
    labels = pd.Series(["L", "L", "H", "H"])
    assert value_separation(values, labels) > 0.9
    assert value_separation(pd.Series([5.0] * 4), labels) == 0.0


def test_churn_rates_none_when_absent() -> None:
    """No churn labels (pre-Phase 6) yields None; present labels average per segment."""
    labels = pd.Series(["A", "A", "B", "B"])
    assert churn_rate_by_segment(labels, None) is None
    churn = pd.Series([1, 0, 0, 0])
    out = churn_rate_by_segment(labels, churn)
    assert out is not None
    assert out.loc[out["segment"] == "A", "churn_rate"].iloc[0] == 0.5


def test_checklist_evidence() -> None:
    """Checklist reflects evidence: noisy/unnamed models fail their items."""
    items = interpretability_checklist(3, True, 2.0, 0.9, 0.2)
    assert all(i["pass"] for i in items)
    bad = interpretability_checklist(12, False, 25.0, 0.3, 0.01)
    assert not any(i["pass"] for i in bad)


def _candidate(name: str, **over) -> dict:
    base = {
        "name": name,
        "silhouette": 0.5,
        "davies_bouldin": 1.0,
        "stability_ari": 0.8,
        "value_eta2": 0.5,
        "min_share": 0.2,
        "noise_pct": 0.0,
        "interpretability_passes": 4,
        "n_segments": 4,
    }
    return {**base, **over}


def test_recommendation_not_silhouette_only() -> None:
    """A high-silhouette but unstable/noisy model loses the operational pick."""
    candidates = [
        _candidate(
            "flashy", silhouette=0.9, stability_ari=0.2, noise_pct=30.0, interpretability_passes=1
        ),
        _candidate("steady", silhouette=0.55, stability_ari=0.9, interpretability_passes=5),
    ]
    rec = recommend_model(candidates)
    assert rec["best_statistical"] == "flashy"
    assert rec["best_stability"] == "steady"
    assert rec["operational_pick"] == "steady"
    assert "flashy" in rec["rationale"] and "steady" in rec["rationale"]


def test_recommendation_needs_candidates() -> None:
    """Empty candidate lists are rejected loudly."""
    with pytest.raises(ValueError):
        recommend_model([])


def test_recommendation_handles_missing_metrics() -> None:
    """None metrics never crash ranking; they lose tie-breaks."""
    rec = recommend_model(
        [
            _candidate("a", silhouette=None, stability_ari=None),
            _candidate("b", silhouette=0.4, stability_ari=0.6),
        ]
    )
    assert rec["operational_pick"] == "b"

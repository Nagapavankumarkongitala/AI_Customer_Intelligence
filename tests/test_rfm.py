"""Phase 2 tests: RFM score directions, ties, edge cases, naming coverage."""

import pandas as pd
import pytest

from analytics.rfm import score_rfm


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D", "E"],
            "recency_days": [1, 10, 30, 60, 120],
            "frequency": [50, 30, 15, 5, 1],
            "monetary_value": [5000.0, 3000.0, 1500.0, 500.0, 100.0],
        }
    )


def test_score_directions_and_formula() -> None:
    """Best customer scores 555/Champions; worst scores 111/Lost; formula holds."""
    scored = score_rfm(_features(), quantiles=5)
    best = scored.loc[scored["customer_id"] == "A"].iloc[0]
    worst = scored.loc[scored["customer_id"] == "E"].iloc[0]
    assert (best["R"], best["F"], best["M"]) == (5, 5, 5)
    assert best["segment"] == "Champions"
    assert (worst["R"], worst["F"], worst["M"]) == (1, 1, 1)
    assert worst["segment"] == "Lost Customers"
    assert (scored["rfm_score"] == 100 * scored["R"] + 10 * scored["F"] + scored["M"]).all()


def test_tied_values_share_scores() -> None:
    """Identical raw values always receive identical scores."""
    df = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C"],
            "recency_days": [10, 10, 50],
            "frequency": [5, 5, 5],
            "monetary_value": [100.0, 100.0, 900.0],
        }
    )
    scored = score_rfm(df, quantiles=5)
    assert (
        scored.loc[scored["customer_id"] == "A", "F"].iloc[0]
        == scored.loc[scored["customer_id"] == "B", "F"].iloc[0]
    )
    assert scored["F"].nunique() == 1  # constant column is safe


def test_constant_and_tiny_inputs_are_safe() -> None:
    """All-equal values and single-customer frames score without errors."""
    df = pd.DataFrame(
        {
            "customer_id": ["A", "B"],
            "recency_days": [10, 10],
            "frequency": [5, 5],
            "monetary_value": [100.0, 100.0],
        }
    )
    scored = score_rfm(df, quantiles=5)
    assert scored[["R", "F", "M"]].nunique().max() == 1
    single = score_rfm(df.iloc[[0]], quantiles=3)
    assert ((single[["R", "F", "M"]] >= 1) & (single[["R", "F", "M"]] <= 3)).all().all()
    assert single["segment"].notna().all()


def test_all_quantile_levels_name_everyone() -> None:
    """q=3, 4, 5 each name every customer from the ten-segment vocabulary."""
    df = _features()
    for q in (3, 4, 5):
        scored = score_rfm(df, quantiles=q)
        assert ((scored[["R", "F", "M"]] >= 1) & (scored[["R", "F", "M"]] <= q)).all().all()
        assert scored["segment"].notna().all()
        assert set(scored["segment"]) <= {
            "Champions",
            "Loyal Customers",
            "Potential Loyalists",
            "New Customers",
            "Promising",
            "Needs Attention",
            "At Risk",
            "Cannot Lose Them",
            "Hibernating",
            "Lost Customers",
        }


def test_invalid_quantiles_rejected() -> None:
    """Only 3, 4, 5 quantile levels are accepted."""
    with pytest.raises(ValueError):
        score_rfm(_features(), quantiles=6)

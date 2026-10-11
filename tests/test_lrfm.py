"""Phase 2 tests: LRFM length scoring, formula, and naming."""

import pandas as pd
import pytest

from analytics.lrfm import score_lrfm


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D", "E"],
            "relationship_length_days": [700, 500, 300, 100, 10],
            "recency_days": [1, 10, 30, 60, 120],
            "frequency": [50, 30, 15, 5, 1],
            "monetary_value": [5000.0, 3000.0, 1500.0, 500.0, 100.0],
        }
    )


def test_length_direction_and_formula() -> None:
    """Longest relationship scores L=5; LRFM = 1000L + 100R + 10F + M."""
    scored = score_lrfm(_features(), quantiles=5)
    assert scored.loc[scored["customer_id"] == "A", "L"].iloc[0] == 5
    assert scored.loc[scored["customer_id"] == "E", "L"].iloc[0] == 1
    assert (
        scored["lrfm_score"]
        == 1000 * scored["L"] + 100 * scored["R"] + 10 * scored["F"] + scored["M"]
    ).all()
    assert scored.loc[scored["customer_id"] == "A", "lrfm_score"].iloc[0] == 5555


def test_lrfm_names_everyone() -> None:
    """LRFM reuses RFM naming: full coverage, scores in range."""
    scored = score_lrfm(_features(), quantiles=4)
    assert ((scored[["L", "R", "F", "M"]] >= 1) & (scored[["L", "R", "F", "M"]] <= 4)).all().all()
    assert scored["segment"].notna().all()


def test_lrfm_invalid_quantiles_rejected() -> None:
    """Only 3, 4, 5 quantile levels are accepted."""
    with pytest.raises(ValueError):
        score_lrfm(_features(), quantiles=2)

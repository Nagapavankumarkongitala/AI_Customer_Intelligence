"""Phase 1 tests: quality scores, boundaries, and warning/count separation."""

import pandas as pd

from services.profiling_service import compute_quality

MAPPING = {
    "customer_id": "customer_id",
    "transaction_date": "transaction_date",
    "amount_spent": "amount_spent",
    "age": "age",
}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C1", None, "C3"],
            "transaction_date": ["2024-01-01", "not-a-date", "2024-03-01", "2030-01-01"],
            "amount_spent": ["100", "-50", "bad", "100000"],
            "age": [30, 250, 40, 25],
        }
    )


def test_quality_scores_bounded_and_weighted() -> None:
    """Scores stay in [0, 100] and overall follows the 0.4/0.4/0.2 weights."""
    result = compute_quality(_frame(), MAPPING)
    for score in (result.completeness, result.validity, result.uniqueness, result.overall):
        assert 0.0 <= score <= 100.0
    assert result.overall == round(
        0.40 * result.completeness + 0.40 * result.validity + 0.20 * result.uniqueness, 2
    )


def test_quality_counts_breakdown() -> None:
    """Each issue class is counted: duplicates, bad dates, future dates, ages."""
    result = compute_quality(_frame(), MAPPING)
    assert result.counts["missing_customer_id"] == 1
    assert result.counts["invalid_dates"] == 1
    assert result.counts["future_dates"] == 1
    assert result.counts["invalid_ages"] == 1
    assert result.counts["invalid_records"] >= 3


def test_warnings_do_not_reduce_validity() -> None:
    """Negatives/outliers are warnings; a frame with only them scores 100 validity."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4", "C5"],
            "transaction_date": ["2024-01-01"] * 5,
            "amount_spent": ["-5", "10", "12", "11", "5000"],
            "age": [30, 31, 32, 33, 34],
        }
    )
    result = compute_quality(df, MAPPING)
    assert result.validity == 100.0
    assert result.warnings["negative_amounts"] == 1


def test_perfect_frame_scores_100() -> None:
    """A clean frame scores 100 on every metric."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "transaction_date": ["2024-01-01", "2024-02-01"],
            "amount_spent": ["100", "200"],
            "age": [30, 40],
        }
    )
    result = compute_quality(df, MAPPING)
    assert (result.completeness, result.validity, result.uniqueness, result.overall) == (
        100.0,
        100.0,
        100.0,
        100.0,
    )

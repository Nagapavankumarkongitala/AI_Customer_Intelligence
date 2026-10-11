"""Phase 6 tests: time-aware labeling boundaries, windows, history, leakage proof."""

import pandas as pd
import pytest

from churn.churn_labeling import LabelingResult, label_churn, validate_outcome_window
from churn.dataset_builder import build_churn_dataset
from utils.exceptions import ChurnLeakageError


def _txns() -> pd.DataFrame:
    rows = [
        ("C_active", "2024-05-01", 100.0),
        ("C_active", "2024-07-01", 100.0),  # inside outcome window -> retained
        ("C_churned", "2024-04-01", 50.0),
        ("C_churned", "2024-05-15", 50.0),  # nothing after cutoff -> churned
        ("C_edge_end", "2024-03-01", 70.0),
        ("C_edge_end", "2024-08-30", 70.0),  # exactly at outcome end -> retained
        ("C_edge_cut", "2024-04-01", 80.0),
        ("C_edge_cut", "2024-06-01", 80.0),  # exactly at cutoff -> predictor side only
        ("C_new", "2024-05-25", 90.0),  # 7 days history < 30 -> excluded
    ]
    return pd.DataFrame(rows, columns=["customer_id", "transaction_date", "amount_spent"])


def test_label_boundaries() -> None:
    """Cutoff belongs to predictors; outcome end is inclusive; short history excluded."""
    result = label_churn(_txns(), "2024-06-01", horizon_days=90, min_history_days=30)
    assert isinstance(result, LabelingResult)
    labels = dict(zip(result.labels["customer_id"], result.labels["churn"], strict=True))
    assert labels["C_active"] == 0
    assert labels["C_churned"] == 1
    assert labels["C_edge_end"] == 0
    assert labels["C_edge_cut"] == 1
    assert "C_new" not in labels
    assert result.excluded_short_history == 1
    assert not result.partial_window


def test_missing_outcome_window_raises() -> None:
    """Cutoff at/after the last transaction leaves nothing to label from."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "transaction_date": ["2024-01-01"],
            "amount_spent": [10.0],
        }
    )
    with pytest.raises(ChurnLeakageError):
        label_churn(df, "2024-06-01", horizon_days=90)


def test_partial_window_warns() -> None:
    """Short outcome coverage flags partial instead of failing."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C1"],
            "transaction_date": ["2024-01-01", "2024-06-15"],
            "amount_spent": [10.0, 20.0],
        }
    )
    result = label_churn(df, "2024-06-01", horizon_days=90, min_history_days=0)
    assert result.partial_window
    assert result.observed_outcome_days == 14


def test_validate_outcome_window_math() -> None:
    """Outcome end = cutoff + horizon; observed days counted honestly."""
    end, observed, partial = validate_outcome_window(
        pd.Timestamp("2024-07-01"), pd.Timestamp("2024-06-01"), 90
    )
    assert end == pd.Timestamp("2024-08-30")
    assert observed == 30 and partial


def test_builder_leakage_proof() -> None:
    """Predictors peak at the cutoff and every labeled customer has features."""
    dataset = build_churn_dataset(_txns(), "2024-06-01", horizon_days=90, min_history_days=30)
    assert dataset.leakage_ok
    assert dataset.predictor_max_date <= pd.Timestamp("2024-06-01")
    assert set(dataset.frame["customer_id"]) == set(dataset.labeling.labels["customer_id"])
    assert "churn" in dataset.frame.columns
    assert dataset.frame["churn"].isin([0, 1]).all()

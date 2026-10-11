"""Phase 1 tests: every cleaning treatment plus audit integrity."""

import pandas as pd
import pytest

from services.cleaning_service import CleaningConfig, clean
from utils.exceptions import CleaningError

MAPPING = {
    "customer_id": "customer_id",
    "transaction_date": "transaction_date",
    "amount_spent": "amount_spent",
    "gender": "gender",
    "age": "age",
}


def _dirty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C1", None, "C2", "C3", "C4"],
            "transaction_date": [
                "2024-01-01",
                "2024-01-01",
                "2024-02-01",
                "bad-date",
                "2024-03-01",
                "2024-03-02",
            ],
            "amount_spent": ["100", "100", "50", "70", "-20", None],
            "gender": ["Male", "Male", " Female ", "Male", "Male", "Male"],
            "age": [30, 30, 40, 35, 250, 28],
        }
    )


def test_clean_default_audit_adds_up() -> None:
    """Audit arithmetic holds: output = input - removed; rules recorded per treatment."""
    _, audit = clean(_dirty(), MAPPING, CleaningConfig(), source_filename="dirty.csv")
    assert audit["output_rows"] == audit["input_rows"] - audit["rows_removed"]
    assert audit["source_filename"] == "dirty.csv"
    assert set(audit["rules"]) >= {
        "duplicates",
        "missing_customer_id",
        "invalid_dates",
        "missing_amounts",
        "negative_amounts",
        "monetary_outliers",
        "invalid_ages",
        "whitespace_normalization",
    }
    assert audit["column_mapping"] == MAPPING
    assert audit["timestamp"]


def test_clean_drops_and_counts() -> None:
    """Drop treatments remove exactly the affected rows."""
    frame, _ = clean(_dirty(), MAPPING, CleaningConfig())
    # 6 rows - 1 dup - 1 missing id - 1 bad date = 3 rows left.
    assert len(frame) == 3
    assert frame["customer_id"].isna().sum() == 0


def test_negative_treatments() -> None:
    """keep/drop/missing/abs each behave as documented."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "transaction_date": ["2024-01-01", "2024-01-02"],
            "amount_spent": ["-20", "100"],
            "gender": ["Male", "Female"],
            "age": [30, 40],
        }
    )
    keep, _ = clean(df, MAPPING, CleaningConfig(negatives="keep"))
    assert (keep["amount_spent"] < 0).sum() == 1
    dropped, _ = clean(df, MAPPING, CleaningConfig(negatives="drop"))
    assert len(dropped) == 1
    absoluted, _ = clean(df, MAPPING, CleaningConfig(negatives="abs"))
    assert (absoluted["amount_spent"] < 0).sum() == 0
    assert absoluted["amount_spent"].iloc[0] == 20.0


def test_missing_amount_median_fill() -> None:
    """Median fill replaces blanks with the median of valid amounts."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "transaction_date": ["2024-01-01"] * 3,
            "amount_spent": ["10", "20", None],
            "gender": ["Male"] * 3,
            "age": [30, 31, 32],
        }
    )
    filled, _ = clean(df, MAPPING, CleaningConfig(missing_amounts="median"))
    assert filled["amount_spent"].iloc[2] == 15.0


def test_outlier_treatments() -> None:
    """remove drops extremes; flag keeps rows and marks them; cap winsorizes."""
    amounts = ["10", "11", "12", "13", "12", "11", "10", "5000"]
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(len(amounts))],
            "transaction_date": ["2024-01-01"] * len(amounts),
            "amount_spent": amounts,
            "gender": ["Male"] * len(amounts),
            "age": [30] * len(amounts),
        }
    )
    removed, audit = clean(df, MAPPING, CleaningConfig(outliers="remove"))
    assert len(removed) == len(df) - 1
    assert audit["rules"]["monetary_outliers"]["records_affected"] == 1
    flagged, _ = clean(df, MAPPING, CleaningConfig(outliers="flag"))
    assert len(flagged) == len(df) and flagged["is_outlier"].sum() == 1
    capped, _ = clean(df, MAPPING, CleaningConfig(outliers="cap"))
    assert capped["amount_spent"].max() < 5000


def test_invalid_age_treatments() -> None:
    """missing nulls bad ages; drop removes those rows; keep preserves them."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "transaction_date": ["2024-01-01"] * 2,
            "amount_spent": ["10", "20"],
            "gender": ["Male"] * 2,
            "age": [250, 30],
        }
    )
    nulled, _ = clean(df, MAPPING, CleaningConfig(invalid_ages="missing"))
    assert pd.isna(nulled["age"].iloc[0])
    dropped, _ = clean(df, MAPPING, CleaningConfig(invalid_ages="drop"))
    assert len(dropped) == 1
    kept, _ = clean(df, MAPPING, CleaningConfig(invalid_ages="keep"))
    assert kept["age"].iloc[0] == 250


def test_clean_empty_result_raises_helpfully() -> None:
    """Wiping out every row raises a business-readable error, not an empty frame."""
    df = pd.DataFrame(
        {
            "customer_id": [None],
            "transaction_date": ["bad"],
            "amount_spent": ["bad"],
            "gender": ["Male"],
            "age": [30],
        }
    )
    with pytest.raises(CleaningError):
        clean(df, MAPPING, CleaningConfig())

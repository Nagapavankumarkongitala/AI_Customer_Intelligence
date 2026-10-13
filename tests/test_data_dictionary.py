"""Data dictionary tests: builder rows and state-shape normalization."""

import pandas as pd

from services.data_dictionary_service import build_dictionary, normalize_dictionary_rows


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "transaction_date": ["2024-01-10", "2024-02-10"],
            "amount_spent": [100.0, 200.0],
        }
    )


def test_build_dictionary_returns_documented_rows() -> None:
    """One row per mapped field with counts and descriptions."""
    dictionary = build_dictionary(_frame(), dict.fromkeys(["customer_id", "transaction_date"]))
    assert list(dictionary["canonical_field"]) == ["customer_id", "transaction_date"]
    assert dictionary["required"].tolist() == [True, True]


def test_normalize_accepts_every_state_shape() -> None:
    """Records pass through; legacy caches and frames normalize; junk empties."""
    records = build_dictionary(_frame(), {"customer_id": "customer_id"}).to_dict(orient="records")
    assert normalize_dictionary_rows(records) == records
    assert normalize_dictionary_rows(None) == []
    assert normalize_dictionary_rows([]) == []
    assert normalize_dictionary_rows(42) == []

    legacy = {"source_column": "0  a\n1  b", "canonical_field": "0  x\n1  y"}
    normalized = normalize_dictionary_rows(legacy)
    assert normalized == [
        {"field": "source_column", "detail": "0  a\n1  b"},
        {"field": "canonical_field", "detail": "0  x\n1  y"},
    ]
    # The legacy shape is what crashed page 02: it must build a frame cleanly.
    assert len(pd.DataFrame(normalized)) == 2

    frame = pd.DataFrame(records)
    assert normalize_dictionary_rows(frame) == records

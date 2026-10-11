"""Phase 1 tests: header handling, mapping, parsing, validation."""

import pandas as pd
import pytest

from services.column_detection import (
    auto_map,
    deduplicate_columns,
    normalize_header,
    validate_mapping,
)
from services.data_service import canonical_frame, load_upload_bytes, parse_currency, parse_dates
from utils.exceptions import DataIngestionError, MappingError


def test_normalize_header() -> None:
    """Separators/case collapse to canonical-style names."""
    assert normalize_header("Customer ID") == "customer_id"
    assert normalize_header("  Txn-Date ") == "txn_date"
    assert normalize_header("AMOUNT ($)") == "amount"


def test_deduplicate_columns_suffixes_repeats() -> None:
    """Repeated normalized headers get _2, _3 suffixes."""
    assert deduplicate_columns(["Amount", "amount", "AMOUNT"]) == [
        "amount",
        "amount_2",
        "amount_3",
    ]


def test_auto_map_exact_and_fuzzy() -> None:
    """Exact aliases map directly; close variants map fuzzily; gaps are listed."""
    proposal = auto_map(["cust_id", "order_date", "revenue", "town"])
    assert proposal.mapping["customer_id"] == "cust_id"
    assert proposal.mapping["transaction_date"] == "order_date"
    assert proposal.mapping["amount_spent"] == "revenue"
    assert proposal.mapping["city"] == "town"
    assert proposal.missing_required == []


def test_auto_map_missing_required_reported() -> None:
    """Unmapped required fields are reported, not silently ignored."""
    proposal = auto_map(["cust_id"])
    assert set(proposal.missing_required) == {"transaction_date", "amount_spent"}


def test_validate_mapping_rejects_problems() -> None:
    """Missing required, unknown columns, and double-mapped sources all raise."""
    with pytest.raises(MappingError):
        validate_mapping({"customer_id": "c"}, ["c"])
    with pytest.raises(MappingError):
        validate_mapping(
            {"customer_id": "c", "transaction_date": "t", "amount_spent": "ghost"},
            ["c", "t", "a"],
        )
    with pytest.raises(MappingError):
        validate_mapping(
            {"customer_id": "c", "transaction_date": "c", "amount_spent": "a"},
            ["c", "a"],
        )
    assert validate_mapping(
        {"customer_id": "c", "transaction_date": "t", "amount_spent": "a"}, ["c", "t", "a"]
    ) == {"customer_id": "ok", "transaction_date": "ok", "amount_spent": "ok"}


def test_parse_currency() -> None:
    """Symbols, commas, and parentheses parse; garbage and blanks are NaN."""
    assert parse_currency("₹1,234.56") == 1234.56
    assert parse_currency("$2,000") == 2000.0
    assert parse_currency("(100)") == -100.0
    assert parse_currency(42) == 42.0
    assert pd.isna(parse_currency("not money"))
    assert pd.isna(parse_currency(""))
    assert pd.isna(parse_currency(None))


def test_parse_dates_mixed_formats() -> None:
    """ISO and day-first strings parse; garbage becomes NaT."""
    parsed = parse_dates(pd.Series(["2024-01-15", "15/02/2024", "not-a-date", ""]))
    assert parsed.iloc[0] == pd.Timestamp("2024-01-15")
    assert parsed.iloc[1] == pd.Timestamp("2024-02-15")
    assert pd.isna(parsed.iloc[2])
    assert pd.isna(parsed.iloc[3])


def test_load_upload_bytes_rejects_empty_and_bad_type() -> None:
    """Empty files and unsupported formats fail with business-readable errors."""
    with pytest.raises(DataIngestionError):
        load_upload_bytes(b"", "sales.csv")
    with pytest.raises(DataIngestionError):
        load_upload_bytes(b"a,b\n1,2", "sales.txt")


def test_load_upload_bytes_stats_and_mapping() -> None:
    """A small CSV yields stats, a preview, and a mapping proposal."""
    csv = (
        "cust_id,order_date,revenue,gender\n"
        "C1,2024-01-01,100,Male\n"
        "C1,2024-01-01,100,Male\n"
        "C2,2024-02-01,250,Female\n"
    )
    preview = load_upload_bytes(csv.encode("utf-8"), "sales.csv")
    assert preview.rows == 3
    assert preview.duplicate_rows == 1
    assert preview.proposal.mapping["customer_id"] == "cust_id"
    assert preview.proposal.missing_required == []
    assert len(preview.preview) == 3


def test_canonical_frame_keeps_only_mapped() -> None:
    """Unmapped source columns are excluded from the canonical frame."""
    df = pd.DataFrame({"c": [1], "t": [2], "extra": [3]})
    out = canonical_frame(df, {"customer_id": "c", "transaction_date": "t"})
    assert list(out.columns) == ["customer_id", "transaction_date"]

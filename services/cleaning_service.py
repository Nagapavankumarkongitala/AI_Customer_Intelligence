"""Configurable cleaning with a full audit record. Nothing is dropped silently.

Treatment order (fixed, documented): duplicates -> missing customer_id ->
invalid/future dates -> missing amounts -> negative amounts -> IQR outliers ->
invalid ages -> categorical whitespace normalization (always applied).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Literal

import numpy as np
import pandas as pd

from services.data_service import canonical_frame, parse_currency, parse_dates
from utils.exceptions import CleaningError

DuplicatesTreatment = Literal["drop", "retain"]
MissingIdTreatment = Literal["drop", "retain"]
InvalidDateTreatment = Literal["drop", "as_missing"]
MissingAmountTreatment = Literal["drop", "retain", "zero", "median"]
NegativeTreatment = Literal["keep", "drop", "missing", "abs"]
OutlierTreatment = Literal["flag", "cap", "remove", "ignore"]
InvalidAgeTreatment = Literal["missing", "drop", "keep"]


@dataclass
class CleaningConfig:
    """User-selected treatments; defaults are the least destructive valid choices."""

    duplicates: DuplicatesTreatment = "drop"
    missing_customer_id: MissingIdTreatment = "drop"
    invalid_dates: InvalidDateTreatment = "drop"
    missing_amounts: MissingAmountTreatment = "median"
    negatives: NegativeTreatment = "keep"
    outliers: OutlierTreatment = "ignore"
    invalid_ages: InvalidAgeTreatment = "missing"
    age_min: float = 0
    age_max: float = 120
    future_tolerance_days: int = 0
    iqr_multiplier: float = 1.5

    def as_dict(self) -> dict:
        """JSON-serializable copy for the audit record."""
        return {k: v for k, v in self.__dict__.items()}


def _rule(rules: dict, name: str, action: str, affected: int) -> None:
    rules[name] = {"action": action, "records_affected": int(affected)}


def clean(
    df: pd.DataFrame,
    mapping: dict[str, str],
    config: CleaningConfig,
    source_filename: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Apply treatments in fixed order. Returns (clean canonical frame, audit record)."""
    frame = canonical_frame(df, mapping).copy()
    input_rows = len(frame)
    if input_rows == 0:
        raise CleaningError("There are no rows to clean. Upload a dataset first.")
    rules: dict[str, dict] = {}
    unmapped = [c for c in df.columns if c not in set(mapping.values())]

    # 1. Exact duplicates.
    dup = int(frame.duplicated().sum())
    if config.duplicates == "drop":
        frame = frame[~frame.duplicated()].reset_index(drop=True)
    _rule(rules, "duplicates", config.duplicates, dup)

    # 2. Missing customer_id.
    blank_id = frame["customer_id"].isna() | (
        frame["customer_id"].astype("string").str.strip() == ""
    )
    if config.missing_customer_id == "drop":
        frame = frame[~blank_id].reset_index(drop=True)
    _rule(rules, "missing_customer_id", config.missing_customer_id, int(blank_id.sum()))

    # 3. Transaction dates (future dates count as invalid; tolerance from config).
    dates = parse_dates(frame["transaction_date"])
    horizon = pd.Timestamp(date.today()) + pd.Timedelta(days=config.future_tolerance_days)
    bad_date = dates.isna() | (dates > horizon)
    if config.invalid_dates == "drop":
        frame = frame[~bad_date].reset_index(drop=True)
        dates = dates[~bad_date].reset_index(drop=True)
    else:
        frame["transaction_date"] = dates.where(~bad_date, pd.NaT).astype("string")
        dates = parse_dates(frame["transaction_date"])
    frame["transaction_date"] = dates.astype("string")
    _rule(rules, "invalid_dates", config.invalid_dates, int(bad_date.sum()))

    # 4. Amounts: parse, then treat missing.
    amounts = frame["amount_spent"].map(parse_currency)
    missing_amt = amounts.isna()
    if config.missing_amounts == "drop":
        frame = frame[~missing_amt].reset_index(drop=True)
        amounts = amounts[~missing_amt].reset_index(drop=True)
    elif config.missing_amounts == "zero":
        amounts = amounts.fillna(0.0)
    elif config.missing_amounts == "median":
        median = float(amounts.dropna().median()) if amounts.notna().any() else 0.0
        amounts = amounts.fillna(median)
    frame["amount_spent"] = amounts
    _rule(rules, "missing_amounts", config.missing_amounts, int(missing_amt.sum()))
    if frame.empty:
        raise CleaningError(
            "Cleaning removed every row. Loosen a treatment (e.g. keep invalid dates "
            "as missing instead of dropping them) and try again."
        )

    # 5. Negative amounts (refunds). Absolute value only converts on explicit choice.
    negative = frame["amount_spent"] < 0
    if config.negatives == "drop":
        frame = frame[~negative].reset_index(drop=True)
    elif config.negatives == "missing":
        frame.loc[negative, "amount_spent"] = np.nan
    elif config.negatives == "abs":
        frame.loc[negative, "amount_spent"] = frame.loc[negative, "amount_spent"].abs()
    _rule(rules, "negative_amounts", config.negatives, int(negative.sum()))

    # 6. IQR monetary outliers on the post-treatment amounts.
    valid = frame["amount_spent"].dropna()
    outlier_mask = pd.Series(False, index=frame.index)
    if len(valid) >= 4:
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - config.iqr_multiplier * iqr, q3 + config.iqr_multiplier * iqr
        outlier_mask = (frame["amount_spent"] < low) | (frame["amount_spent"] > high)
        outlier_mask = outlier_mask.fillna(False)
        if config.outliers == "remove":
            frame = frame[~outlier_mask].reset_index(drop=True)
        elif config.outliers == "cap":
            frame.loc[frame["amount_spent"] < low, "amount_spent"] = low
            frame.loc[frame["amount_spent"] > high, "amount_spent"] = high
        elif config.outliers == "flag":
            frame["is_outlier"] = outlier_mask.reset_index(drop=True)
    _rule(rules, "monetary_outliers", config.outliers, int(outlier_mask.sum()))

    # 7. Invalid ages.
    if "age" in frame:
        ages = pd.to_numeric(frame["age"], errors="coerce")
        bad_age = (ages < config.age_min) | (ages > config.age_max)
        bad_age = bad_age.fillna(False)
        if config.invalid_ages == "drop":
            frame = frame[~bad_age].reset_index(drop=True)
        elif config.invalid_ages == "missing":
            frame.loc[bad_age, "age"] = np.nan
        _rule(rules, "invalid_ages", config.invalid_ages, int(bad_age.sum()))

    # 8. Whitespace normalization of categoricals (always; never merges distinct values).
    normalized_cells = 0
    for col in ("gender", "city", "state", "country", "region"):
        if col in frame:
            stripped = frame[col].astype("string").str.strip()
            changed = (stripped != frame[col].astype("string")) & frame[col].notna()
            normalized_cells += int(changed.sum())
            frame[col] = stripped
    _rule(rules, "whitespace_normalization", "always", normalized_cells)

    if unmapped:
        rules["unmapped_columns_dropped"] = {
            "action": "dropped",
            "records_affected": 0,
            "columns": unmapped,
        }

    output_rows = len(frame)
    if output_rows == 0:
        raise CleaningError(
            "Cleaning removed every row. Loosen a treatment (e.g. keep invalid dates "
            "as missing instead of dropping them) and try again."
        )
    audit = {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "rows_removed": input_rows - output_rows,
        "rules": rules,
        "cleaning_config": config.as_dict(),
        "timestamp": datetime.now(UTC).isoformat(),
        "source_filename": source_filename,
        "column_mapping": dict(mapping),
    }
    return frame.reset_index(drop=True), audit


@dataclass
class CleaningAudit:
    """Typed view over the audit dict for UI code (kept in sync with clean())."""

    input_rows: int = 0
    output_rows: int = 0
    rows_removed: int = 0
    rules: dict = field(default_factory=dict)
    timestamp: str = ""
    source_filename: str = ""
    column_mapping: dict = field(default_factory=dict)

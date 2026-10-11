"""Data-quality profiling: completeness / validity / uniqueness + issue breakdown.

Validity failures (reduce the Validity score): missing customer_id, unparseable or
future transaction_date, unparseable amount_spent, out-of-range age.
Warnings (reported and treatable, do NOT reduce Validity): negative amounts,
IQR monetary outliers, category case/whitespace variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from services.data_service import canonical_frame, parse_currency, parse_dates


@dataclass
class QualityResult:
    """Scores plus the raw counts behind them for display and testing."""

    completeness: float
    validity: float
    uniqueness: float
    overall: float
    counts: dict[str, int] = field(default_factory=dict)
    warnings: dict[str, int] = field(default_factory=dict)


def _category_variant_rows(series: pd.Series) -> int:
    """Rows whose value collides with another value after strip().lower()."""
    text = series.dropna().astype(str)
    folded = text.str.strip().str.lower()
    dup_folded = folded[folded.duplicated(keep=False)]
    raw_variants = text[dup_folded.index].nunique()
    return int(dup_folded.shape[0]) if raw_variants > dup_folded.nunique() else 0


def compute_quality(
    df: pd.DataFrame,
    mapping: dict[str, str],
    age_min: float = 0,
    age_max: float = 120,
    future_tolerance_days: int = 0,
    iqr_multiplier: float = 1.5,
) -> QualityResult:
    """Score the mapped raw frame. Never mutates the input."""
    frame = canonical_frame(df, mapping)
    total = len(frame)
    counts: dict[str, int] = {"total_rows": total}

    counts["missing_cells"] = int(frame.isna().sum().sum())
    counts["missing_customer_id"] = int(
        frame["customer_id"].isna().sum()
        + (frame["customer_id"].astype("string").str.strip() == "").sum()
    )
    counts["duplicate_rows"] = int(frame.duplicated().sum())

    dates = parse_dates(frame["transaction_date"]) if "transaction_date" in frame else None
    if dates is not None:
        counts["invalid_dates"] = int(dates.isna().sum())
        horizon = pd.Timestamp(date.today()) + pd.Timedelta(days=future_tolerance_days)
        counts["future_dates"] = int((dates > horizon).sum())
    else:
        counts["invalid_dates"] = 0
        counts["future_dates"] = 0

    amounts = frame["amount_spent"].map(parse_currency) if "amount_spent" in frame else None
    if amounts is not None:
        blank = frame["amount_spent"].isna() | (
            frame["amount_spent"].astype("string").str.strip() == ""
        )
        counts["invalid_amounts"] = int(amounts.isna().sum() - blank.sum())
    else:
        counts["invalid_amounts"] = 0

    ages = pd.to_numeric(frame["age"], errors="coerce") if "age" in frame else None
    if ages is not None:
        counts["invalid_ages"] = int(((ages < age_min) | (ages > age_max)).sum())
    else:
        counts["invalid_ages"] = 0

    warnings: dict[str, int] = {}
    if amounts is not None:
        valid_amounts = amounts.dropna()
        warnings["negative_amounts"] = int((valid_amounts < 0).sum())
        if len(valid_amounts) >= 4:
            q1, q3 = valid_amounts.quantile(0.25), valid_amounts.quantile(0.75)
            iqr = q3 - q1
            warnings["monetary_outliers"] = int(
                (
                    (valid_amounts < q1 - iqr_multiplier * iqr)
                    | (valid_amounts > q3 + iqr_multiplier * iqr)
                ).sum()
            )
        else:
            warnings["monetary_outliers"] = 0
    variant_total = 0
    for col in ("gender", "city", "state", "country", "region"):
        if col in frame:
            variant_total += _category_variant_rows(frame[col])
    warnings["category_variants"] = variant_total

    total_cells = total * len(frame.columns) if len(frame.columns) else 0
    completeness = (1 - counts["missing_cells"] / total_cells) * 100 if total_cells else 0.0

    invalid_mask = pd.Series(False, index=frame.index)
    invalid_mask |= frame["customer_id"].isna() | (
        frame["customer_id"].astype("string").str.strip() == ""
    )
    if dates is not None:
        invalid_mask |= dates.isna() | (dates > horizon)
    if amounts is not None:
        invalid_mask |= amounts.isna()
    if ages is not None:
        invalid_mask |= (ages < age_min) | (ages > age_max)
    validity = (1 - float(invalid_mask.sum()) / total) * 100 if total else 0.0
    uniqueness = (1 - counts["duplicate_rows"] / total) * 100 if total else 0.0
    overall = 0.40 * completeness + 0.40 * validity + 0.20 * uniqueness
    counts["invalid_records"] = int(invalid_mask.sum())
    return QualityResult(
        completeness=round(completeness, 2),
        validity=round(validity, 2),
        uniqueness=round(uniqueness, 2),
        overall=round(overall, 2),
        counts=counts,
        warnings=warnings,
    )

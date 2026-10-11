"""Data dictionary builder: one documented row per canonical field."""

from __future__ import annotations

import pandas as pd

from utils.constants import REQUIRED_FIELDS

DESCRIPTIONS = {
    "customer_id": "Unique customer identifier (never blank in modeling data).",
    "transaction_date": "Date of the transaction; must be parseable, not unexpectedly future.",
    "amount_spent": "Transaction revenue; numeric after cleaning (negatives may be refunds).",
    "gender": "Optional demographic attribute; profiling and fairness review only.",
    "age": "Optional demographic attribute; valid range is configurable (default 0–120).",
    "city": "Optional geographic attribute at city/town granularity.",
    "state": "Optional geographic attribute at state/province granularity.",
    "country": "Optional geographic attribute at country granularity.",
    "region": "Optional geographic attribute at region/zone granularity.",
}


def build_dictionary(
    df: pd.DataFrame, mapping: dict[str, str], stage: str = "clean"
) -> pd.DataFrame:
    """Return a dictionary frame with one row per mapped canonical field."""
    rows: list[dict] = []
    for canonical, source in mapping.items():
        col = df[canonical] if canonical in df.columns else pd.Series(dtype=object)
        non_null = int(col.notna().sum())
        missing = int(col.isna().sum())
        total = len(col)
        entry: dict = {
            "source_column": source,
            "canonical_field": canonical,
            "description": DESCRIPTIONS.get(canonical, ""),
            "required": canonical in REQUIRED_FIELDS,
            "data_type": str(col.dtype),
            "stage": stage,
            "non_null_count": non_null,
            "missing_count": missing,
            "missing_pct": round(missing / total * 100, 2) if total else 0.0,
            "unique_count": int(col.nunique(dropna=True)),
            "examples": ", ".join(map(str, col.dropna().unique()[:3])),
        }
        numeric = pd.to_numeric(col, errors="coerce")
        if numeric.notna().any():
            entry.update(
                {
                    "min": round(float(numeric.min()), 2),
                    "max": round(float(numeric.max()), 2),
                    "mean": round(float(numeric.mean()), 2),
                    "median": round(float(numeric.median()), 2),
                }
            )
        else:
            entry.update({"min": None, "max": None, "mean": None, "median": None})
        rows.append(entry)
    return pd.DataFrame(rows)

"""File helpers: encoding-fallback reads, suffix checks, safe dataframe exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.constants import CSV_ENCODINGS, SUPPORTED_UPLOAD_SUFFIXES
from utils.exceptions import DataIngestionError


def ensure_supported_suffix(filename: str) -> str:
    """Return the lowercase suffix or raise a business-readable error."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise DataIngestionError(
            f"File type '{suffix or '(none)'}' is not supported. "
            f"Upload one of: {', '.join(SUPPORTED_UPLOAD_SUFFIXES)}."
        )
    return suffix


def read_upload(path: str | Path, filename: str) -> pd.DataFrame:
    """Read an uploaded file with CSV encoding fallback (UTF-8 BOM → UTF-8 → CP1252 → Latin-1)."""
    suffix = ensure_supported_suffix(filename)
    src = Path(path)
    if src.stat().st_size == 0:
        raise DataIngestionError(
            "The uploaded file is empty (0 bytes). Upload a file containing transaction rows."
        )
    if suffix == ".csv":
        last: Exception | None = None
        for enc in CSV_ENCODINGS:
            try:
                return pd.read_csv(src, encoding=enc)
            except Exception as exc:  # try next encoding
                last = exc
        raise DataIngestionError(f"Could not decode the CSV with any known encoding: {last}")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(src)
    return pd.read_parquet(src)


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Write df to CSV/Excel/Parquet based on suffix, creating parent dirs."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix.lower() == ".csv":
        df.to_csv(dest, index=False)
    elif dest.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(dest, index=False)
    elif dest.suffix.lower() == ".parquet":
        df.to_parquet(dest, index=False)
    else:
        raise DataIngestionError(f"Cannot export to '{dest.suffix}'. Use CSV, Excel, or Parquet.")
    return dest

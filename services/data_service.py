"""Upload ingestion: read bytes, normalize headers, preview stats, propose mapping.

Parsing helpers (currency, dates) live here so profiling and cleaning share them.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from services.column_detection import (
    MappingProposal,
    auto_map,
    deduplicate_columns,
)
from utils.constants import CSV_ENCODINGS
from utils.exceptions import DataIngestionError
from utils.file_utils import ensure_supported_suffix

_CURRENCY_SYMBOLS = ("$", "₹", "€", "£", "¥")


@dataclass
class UploadPreview:
    """Everything Page 01 needs: stats, preview rows, and a mapping proposal."""

    filename: str
    rows: int
    columns: int
    memory_mb: float
    duplicate_rows: int
    normalized_columns: list[str] = field(default_factory=list)
    preview: pd.DataFrame = field(default_factory=pd.DataFrame)
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    proposal: MappingProposal = field(default_factory=MappingProposal)


def parse_currency(value: object) -> float:
    """Parse '₹1,234.56', '$1,234', '(100)' (parentheses = negative) to float; NaN if unparseable.

    European '1.234,56' notation is NOT supported and reports as invalid by design.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    negative = text.startswith("(") and text.endswith(")")
    text = text[1:-1] if negative else text
    for symbol in _CURRENCY_SYMBOLS:
        text = text.replace(symbol, "")
    text = text.replace(",", "").strip()
    try:
        amount = float(text)
    except ValueError:
        return np.nan
    return -amount if negative else amount


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse mixed date strings; day-first retry for leftovers; NaT when unparseable."""
    cleaned = series.astype("string").str.strip()
    parsed = pd.to_datetime(cleaned, errors="coerce")
    still_missing = parsed.isna() & cleaned.notna() & (cleaned != "")
    if still_missing.any():
        retry = pd.to_datetime(cleaned[still_missing], errors="coerce", dayfirst=True)
        parsed.loc[still_missing] = retry
    return parsed


def load_upload_bytes(data: bytes, filename: str) -> UploadPreview:
    """Read an uploaded file from bytes with CSV encoding fallback."""
    suffix = ensure_supported_suffix(filename)
    if not data:
        raise DataIngestionError(
            "The uploaded file is empty (0 bytes). Upload a file containing transaction rows."
        )
    buffer = io.BytesIO(data)
    if suffix == ".csv":
        last: Exception | None = None
        frame: pd.DataFrame | None = None
        for enc in CSV_ENCODINGS:
            try:
                buffer.seek(0)
                frame = pd.read_csv(buffer, encoding=enc)
                break
            except Exception as exc:
                last = exc
        if frame is None:
            raise DataIngestionError(f"Could not decode the CSV with any known encoding: {last}")
    elif suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(buffer)
    elif suffix == ".parquet":
        frame = pd.read_parquet(buffer)
    else:  # pragma: no cover — guarded by ensure_supported_suffix
        raise DataIngestionError(f"Unsupported file type '{suffix}'.")
    if frame.empty:
        raise DataIngestionError(
            "The file has headers but no data rows. Upload a file with transaction records."
        )
    normalized = deduplicate_columns(list(frame.columns))
    frame.columns = normalized
    return UploadPreview(
        filename=filename,
        rows=len(frame),
        columns=len(normalized),
        memory_mb=float(frame.memory_usage(deep=True).sum() / 1e6),
        duplicate_rows=int(frame.duplicated().sum()),
        normalized_columns=normalized,
        preview=frame.head(10).copy(),
        frame=frame.copy(),
        proposal=auto_map(normalized),
    )


def canonical_frame(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename mapped source columns to canonical names; keep only mapped columns."""
    return df[[mapping[f] for f in mapping]].rename(columns={v: k for k, v in mapping.items()})

"""Download serialization: CSV / Excel / JSON bytes for Streamlit download buttons."""

from __future__ import annotations

import io
import json

import pandas as pd

from utils.exceptions import DataIngestionError


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a frame to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    """Serialize a frame to XLSX bytes (requires openpyxl)."""
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise DataIngestionError(
            "Excel export needs the 'openpyxl' package. Install it with "
            "'pip install openpyxl' and try again."
        ) from exc
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def to_json_bytes(payload: dict | list) -> bytes:
    """Serialize mapping/audit payloads to indented JSON bytes."""
    return json.dumps(payload, indent=2, default=str).encode("utf-8")

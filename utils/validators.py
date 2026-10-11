"""Lightweight input validators used by services before heavy computation."""

from __future__ import annotations

import pandas as pd

from utils.constants import REQUIRED_FIELDS
from utils.exceptions import InsufficientDataError, MappingError


def require_mapped_columns(df: pd.DataFrame, mapping: dict[str, str]) -> None:
    """Ensure every required canonical field has a mapped source column present in df."""
    missing = [f for f in REQUIRED_FIELDS if f not in mapping or mapping[f] not in df.columns]
    if missing:
        raise MappingError(
            "Required field(s) not mapped to a dataset column: "
            + ", ".join(missing)
            + ". Map them on the Data Upload page before continuing."
        )


def require_min_rows(df: pd.DataFrame, minimum: int, what: str = "rows") -> None:
    """Raise a business-readable error when the frame is too small to analyze."""
    if len(df) < minimum:
        raise InsufficientDataError(
            f"Only {len(df)} {what} available; at least {minimum} are needed. "
            "Upload a larger dataset or widen the selected filters."
        )

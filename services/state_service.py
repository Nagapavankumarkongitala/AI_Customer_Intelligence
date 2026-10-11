"""Session-state contract owner: initialize, validate, reset, and report status.

Operates on any MutableMapping so it is unit-testable without Streamlit;
in pages, pass st.session_state directly.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pandas as pd

from utils.constants import SESSION_KEYS

_DEFAULTS: dict[str, Any] = {
    "raw_data": None,
    "clean_data": None,
    "column_mapping": {},
    "cleaning_config": {},
    "cleaning_audit": {},
    "data_dictionary": {},
    "customer_features": None,
    "rfm_results": None,
    "lrfm_results": None,
    "demographic_profile": {},
    "segmentation_runs": [],
    "selected_segmentation_model": None,
    "segmented_customers": None,
    "churn_dataset": {},
    "churn_model": None,
    "churn_model_metadata": {},
    "churn_scores": None,
    "retention_results": {},
    "global_filters": {},
}

assert set(_DEFAULTS) == set(SESSION_KEYS), "state defaults drifted from SESSION_KEYS"


def _fresh(key: str) -> Any:
    value = _DEFAULTS[key]
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def initialize(state: MutableMapping) -> None:
    """Fill absent keys with defaults; never overwrite existing values."""
    for key in SESSION_KEYS:
        if key not in state:
            state[key] = _fresh(key)


def reset(state: MutableMapping, keys: list[str] | None = None) -> None:
    """Reset given keys (default: all) back to defaults."""
    for key in keys or list(SESSION_KEYS):
        if key in _DEFAULTS:
            state[key] = _fresh(key)


def validate(state: MutableMapping) -> list[str]:
    """Return contract keys entirely absent from state (empty = contract satisfied)."""
    return [key for key in SESSION_KEYS if key not in state]


def _describe(value: Any) -> str:
    if value is None:
        return "empty"
    if isinstance(value, pd.DataFrame):
        return f"{len(value)} rows × {len(value.columns)} cols" if not value.empty else "empty"
    if isinstance(value, dict):
        return f"{len(value)} entries" if value else "empty"
    if isinstance(value, list):
        return f"{len(value)} items" if value else "empty"
    return "set"


def status_report(state: MutableMapping) -> dict[str, str]:
    """Human-readable per-key status for sidebars and diagnostics."""
    initialize(state)
    return {key: _describe(state[key]) for key in SESSION_KEYS}

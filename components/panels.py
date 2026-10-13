"""Panels: the three framed blocks pages compose views from.

``filter_panel`` is the "Controls & Filters" expander for compute-driving
configuration (re-exported from ``components.filters`` so panels are the one
import a page needs). View slicers belong in ``sidebar_filters`` instead.
``status_panel`` renders pipeline-status facts. ``download_panel`` groups the
CSV/Excel/JSON exports for one artefact. All three are thin compositions over
existing, tested primitives — no new download or filter logic lives here.
"""

from __future__ import annotations

import html
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
import streamlit as st

from components.export_buttons import (
    csv_download_button,
    excel_download_button,
    json_download_button,
)
from components.filters import controls_expander
from visualization import tokens

__all__ = ["filter_panel", "status_panel", "download_panel", "active_filter_chips_html"]


@contextmanager
def filter_panel(title: str = "Controls & Filters") -> Iterator[None]:
    """Collapsed expander holding all of a page's controls."""
    with controls_expander(title):
        yield


def active_filter_chips_html(active: dict[str, str]) -> str:
    """Inline summary of the filters currently narrowing the view (pure)."""
    if not active:
        return ""
    chips = " ".join(
        "<span class='status-badge' "
        f"style='background:{tokens.color('surface_alt')};"
        f"color:{tokens.color('muted')};border:1px solid {tokens.color('border')};'>"
        f"{html.escape(str(label))}: {html.escape(str(value))}</span>"
        for label, value in active.items()
    )
    return f"<div style='margin:0.25rem 0 0.75rem 0;'>Filtered by {chips}</div>"


def status_panel(items: list[tuple[str, str]]) -> None:
    """Two-column fact grid (label → value) for pipeline status blocks."""
    for label, value in items:
        left, right = st.columns([1, 2])
        left.markdown(
            f"<span style='color:{tokens.color('muted')};font-size:0.85rem;'>"
            f"{html.escape(label)}</span>",
            unsafe_allow_html=True,
        )
        right.markdown(
            f"<span style='font-size:0.85rem;font-variant-numeric:tabular-nums;'>"
            f"{html.escape(value)}</span>",
            unsafe_allow_html=True,
        )


def download_panel(
    df: pd.DataFrame, basename: str, key_prefix: str, payload: dict | list | None = None
) -> None:
    """CSV + Excel buttons for a frame (plus JSON when a metadata payload ships)."""
    buttons = st.columns(3 if payload is not None else 2)
    with buttons[0]:
        csv_download_button(df, f"{basename}.csv", "Download CSV", key=f"{key_prefix}_csv")
    with buttons[1]:
        excel_download_button(df, f"{basename}.xlsx", "Download Excel", key=f"{key_prefix}_xlsx")
    if payload is not None:
        with buttons[2]:
            json_download_button(
                payload, f"{basename}.json", "Download JSON", key=f"{key_prefix}_json"
            )

"""Shared sidebar controls: grouped slicers keep the main canvas scroll-free.

Every filter input lives in the sidebar; pages only render KPIs, charts, and
collapsed data inspectors. Group related controls with control_group().
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
import streamlit as st


@contextmanager
def control_group(title: str) -> Iterator[None]:
    """A titled sidebar section with a divider, e.g. 'Global Slicers'."""
    st.sidebar.markdown(f"### {title}")
    yield
    st.sidebar.markdown("---")


def multiselect_filter(label: str, options: list, key: str) -> list:
    """Sidebar multiselect defaulting to all options; empty means no filtering."""
    if not options:
        return []
    return st.sidebar.multiselect(label, options, default=options, key=key)


def segment_select(label: str, options: list, key: str, default: str = "All") -> str:
    """Sidebar single-select slicer with an All option first."""
    choices = ["All", *[str(o) for o in options]]
    return st.sidebar.selectbox(
        label, choices, index=choices.index(default) if default in choices else 0, key=key
    )


def range_slider(label: str, lo: float, hi: float, value: tuple, key: str) -> tuple:
    """Sidebar range slider; guards degenerate bounds."""
    if hi <= lo:
        return (lo, hi)
    return st.sidebar.slider(label, float(lo), float(hi), value, key=key)


def date_range_filter(dates: pd.Series) -> tuple:
    """Sidebar start/end date inputs bounded by the data range."""
    clean = pd.to_datetime(dates, errors="coerce").dropna()
    if clean.empty:
        return None, None
    lo, hi = clean.min().date(), clean.max().date()
    start = st.sidebar.date_input("From", value=lo, min_value=lo, max_value=hi, key="f_from")
    end = st.sidebar.date_input("To", value=hi, min_value=lo, max_value=hi, key="f_to")
    return start, end

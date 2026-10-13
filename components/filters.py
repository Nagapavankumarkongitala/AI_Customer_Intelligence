"""Shared page controls: compute-driving configuration in collapsed expanders.

View slicers live in the sidebar (``components.sidebar.sidebar_filters``);
everything here drives computation and stays in-page next to the analysis,
grouped with control_group().
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
import streamlit as st


@contextmanager
def controls_expander(title: str = "Controls & Filters") -> Iterator[None]:
    """Single collapsed expander holding all of a page's controls."""
    with st.expander(title, expanded=False):
        yield


@contextmanager
def control_group(title: str) -> Iterator[None]:
    """A titled control section with a divider, e.g. 'Global Slicers'."""
    st.markdown(f"### {title}")
    yield
    st.markdown("---")


def multiselect_filter(label: str, options: list, key: str) -> list:
    """Multiselect defaulting to all options; empty means no filtering."""
    if not options:
        return []
    return st.multiselect(label, options, default=options, key=key)


def segment_select(label: str, options: list, key: str, default: str = "All") -> str:
    """Single-select slicer with an All option first."""
    choices = ["All", *[str(o) for o in options]]
    return st.selectbox(
        label, choices, index=choices.index(default) if default in choices else 0, key=key
    )


def range_slider(label: str, lo: float, hi: float, value: tuple, key: str) -> tuple:
    """Range slider; guards degenerate bounds."""
    if hi <= lo:
        return (lo, hi)
    return st.slider(label, float(lo), float(hi), value, key=key)


def date_range_filter(dates: pd.Series) -> tuple:
    """Start/end date inputs bounded by the data range."""
    clean = pd.to_datetime(dates, errors="coerce").dropna()
    if clean.empty:
        return None, None
    lo, hi = clean.min().date(), clean.max().date()
    start = st.date_input("From", value=lo, min_value=lo, max_value=hi, key="f_from")
    end = st.date_input("To", value=hi, min_value=lo, max_value=hi, key="f_to")
    return start, end

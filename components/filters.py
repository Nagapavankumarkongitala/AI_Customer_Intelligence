"""Shared global-filter widgets for dashboard-style pages."""

import pandas as pd
import streamlit as st


def multiselect_filter(label: str, options: list, key: str) -> list:
    """Multiselect defaulting to all options; empty selection means no filtering."""
    if not options:
        return []
    return st.sidebar.multiselect(label, options, default=options, key=key)


def date_range_filter(dates: pd.Series) -> tuple:
    """Start/end date inputs bounded by the data range."""
    clean = pd.to_datetime(dates, errors="coerce").dropna()
    if clean.empty:
        return None, None
    lo, hi = clean.min().date(), clean.max().date()
    start = st.sidebar.date_input("From", value=lo, min_value=lo, max_value=hi, key="f_from")
    end = st.sidebar.date_input("To", value=hi, min_value=lo, max_value=hi, key="f_to")
    return start, end

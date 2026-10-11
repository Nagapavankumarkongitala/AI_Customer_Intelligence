"""Reusable KPI card row for Streamlit pages."""

import streamlit as st


def kpi_row(metrics: list[tuple[str, str]]) -> None:
    """Render label/value KPI cards across columns. Values must be preformatted strings."""
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics, strict=True):
        column.metric(label, value)

"""Shared download buttons: CSV / Excel / JSON for any page."""

import pandas as pd
import streamlit as st

from services.export_service import to_csv_bytes, to_excel_bytes, to_json_bytes
from utils.exceptions import CustomerIntelligenceError


def csv_download_button(df: pd.DataFrame, filename: str, label: str, key: str) -> None:
    """CSV download button for a dataframe."""
    st.download_button(label, data=to_csv_bytes(df), file_name=filename, mime="text/csv", key=key)


def excel_download_button(df: pd.DataFrame, filename: str, label: str, key: str) -> None:
    """Excel download button; shows install guidance when openpyxl is absent."""
    try:
        data = to_excel_bytes(df)
    except CustomerIntelligenceError as exc:
        st.error(str(exc))
        return
    st.download_button(
        label,
        data=data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
    )


def json_download_button(payload: dict | list, filename: str, label: str, key: str) -> None:
    """JSON download button for mapping/audit/metadata payloads."""
    st.download_button(
        label, data=to_json_bytes(payload), file_name=filename, mime="application/json", key=key
    )

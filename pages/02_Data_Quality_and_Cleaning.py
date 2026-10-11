"""Page 02 — Data quality scoring, configurable cleaning, audit, and dictionary."""

from pathlib import Path

import streamlit as st
import yaml

from services import state_service
from services.cleaning_service import CleaningConfig, clean
from services.data_dictionary_service import build_dictionary
from services.export_service import to_csv_bytes, to_excel_bytes, to_json_bytes
from services.profiling_service import compute_quality
from utils.exceptions import CustomerIntelligenceError
from utils.logging_config import configure_logging

configure_logging()
state_service.initialize(st.session_state)

st.title("02 · Data Quality & Cleaning")
st.markdown(
    "Scores: **Completeness** (no missing cells), **Validity** (no invalid records), "
    "**Uniqueness** (no duplicate rows); "
    "overall = 0.40 × completeness + 0.40 × validity + 0.20 × uniqueness. "
    "Every cleaning action is recorded in the audit below — nothing is dropped silently."
)

raw = st.session_state.get("raw_data")
mapping = st.session_state.get("column_mapping") or {}
if raw is None or not mapping:
    st.info("Complete 01 · Data Upload & Mapping first (upload a file and save the mapping).")
    st.stop()

cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

st.subheader("Quality scores")
try:
    quality = compute_quality(
        raw,
        mapping,
        age_min=cfg["quality"]["age_min"],
        age_max=cfg["quality"]["age_max"],
        future_tolerance_days=cfg["quality"]["future_date_tolerance_days"],
        iqr_multiplier=cfg["quality"]["outlier_iqr_multiplier"],
    )
except CustomerIntelligenceError as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Completeness", f"{quality.completeness}%")
c2.metric("Validity", f"{quality.validity}%")
c3.metric("Uniqueness", f"{quality.uniqueness}%")
c4.metric("Overall", f"{quality.overall}%")

with st.expander("Issue breakdown", expanded=True):
    st.write({k: v for k, v in quality.counts.items() if k != "total_rows"})
    st.markdown("**Warnings** (reported and treatable, do not lower the validity score):")
    st.write(quality.warnings)

st.subheader("Cleaning treatments")
treatments = st.columns(3)
with treatments[0]:
    duplicates = st.selectbox("Duplicate rows", ["drop", "retain"], index=0)
    missing_ids = st.selectbox("Missing customer IDs", ["drop", "retain"], index=0)
    invalid_dates = st.selectbox(
        "Invalid / future dates",
        ["drop", "as_missing"],
        format_func=lambda v: {"drop": "Drop rows", "as_missing": "Keep as missing"}[v],
    )
with treatments[1]:
    missing_amounts = st.selectbox(
        "Missing amounts",
        ["median", "zero", "drop", "retain"],
        format_func=lambda v: {
            "median": "Fill with median",
            "zero": "Fill with zero",
            "drop": "Drop rows",
            "retain": "Keep as missing",
        }[v],
    )
    negatives = st.selectbox(
        "Negative amounts (may be refunds)",
        ["keep", "abs", "missing", "drop"],
        format_func=lambda v: {
            "keep": "Keep as-is",
            "abs": "Convert to absolute value",
            "missing": "Set to missing",
            "drop": "Drop rows",
        }[v],
        help="Negatives are only converted when you explicitly choose absolute value.",
    )
with treatments[2]:
    outliers = st.selectbox(
        "Monetary outliers (IQR)",
        ["ignore", "flag", "cap", "remove"],
        format_func=lambda v: {
            "ignore": "Ignore",
            "flag": "Flag with is_outlier column",
            "cap": "Cap at IQR bounds",
            "remove": "Remove rows",
        }[v],
    )
    invalid_ages = st.selectbox(
        "Invalid ages",
        ["missing", "drop", "keep"],
        format_func=lambda v: {
            "missing": "Set to missing",
            "drop": "Drop rows",
            "keep": "Keep as-is",
        }[v],
    )

cleaning_config = CleaningConfig(
    duplicates=duplicates,
    missing_customer_id=missing_ids,
    invalid_dates=invalid_dates,
    missing_amounts=missing_amounts,
    negatives=negatives,
    outliers=outliers,
    invalid_ages=invalid_ages,
    age_min=cfg["quality"]["age_min"],
    age_max=cfg["quality"]["age_max"],
    future_tolerance_days=cfg["quality"]["future_date_tolerance_days"],
    iqr_multiplier=cfg["quality"]["outlier_iqr_multiplier"],
)

if st.button("Apply cleaning", type="primary"):
    try:
        clean_data, audit = clean(
            raw,
            mapping,
            cleaning_config,
            source_filename=st.session_state.get("source_filename", ""),
        )
        st.session_state["clean_data"] = clean_data
        st.session_state["cleaning_config"] = cleaning_config.as_dict()
        st.session_state["cleaning_audit"] = audit
        st.session_state["data_dictionary"] = build_dictionary(clean_data, mapping).to_dict(
            orient="records"
        )
        st.success(
            f"Cleaning complete: {audit['input_rows']} → {audit['output_rows']} rows "
            f"({audit['rows_removed']} removed)."
        )
    except CustomerIntelligenceError as exc:
        st.error(str(exc))

audit = st.session_state.get("cleaning_audit") or {}
if audit:
    st.subheader("Audit record")
    a1, a2, a3 = st.columns(3)
    a1.metric("Input rows", audit["input_rows"])
    a2.metric("Output rows", audit["output_rows"])
    a3.metric("Rows removed", audit["rows_removed"])
    st.write(audit["rules"])
    st.caption(f"Ran at {audit['timestamp']} on {audit.get('source_filename', '(unknown file)')}.")

clean_data = st.session_state.get("clean_data")
if clean_data is not None:
    st.subheader("Cleaned preview")
    st.dataframe(clean_data.head(20), use_container_width=True)

    st.subheader("Data dictionary")
    dictionary_rows = st.session_state.get("data_dictionary") or []
    if dictionary_rows:
        import pandas as pd

        st.dataframe(pd.DataFrame(dictionary_rows), use_container_width=True)

    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Clean data (CSV)",
            data=to_csv_bytes(clean_data),
            file_name="clean_transactions.csv",
            mime="text/csv",
        )
    with d2:
        try:
            st.download_button(
                "Clean data (Excel)",
                data=to_excel_bytes(clean_data),
                file_name="clean_transactions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except CustomerIntelligenceError as exc:
            st.error(str(exc))
    with d3:
        st.download_button(
            "Audit record (JSON)",
            data=to_json_bytes(audit),
            file_name="cleaning_audit.json",
            mime="application/json",
        )

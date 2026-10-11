"""Page 01 — Data upload, automatic column detection, and manual mapping."""

import pandas as pd
import streamlit as st

from services import state_service
from services.column_detection import validate_mapping
from services.data_service import load_upload_bytes
from services.export_service import to_json_bytes
from utils.constants import CANONICAL_FIELDS, REQUIRED_FIELDS, SUPPORTED_UPLOAD_SUFFIXES
from utils.exceptions import CustomerIntelligenceError
from utils.logging_config import configure_logging

configure_logging()
state_service.initialize(st.session_state)

st.title("01 · Data Upload & Mapping")
st.markdown(
    "Upload a transaction file, review the automatic column mapping, correct it if "
    "needed, then continue to **02 · Data Quality & Cleaning**. "
    "Required fields: `customer_id`, `transaction_date`, `amount_spent`."
)

uploaded = st.file_uploader(
    "Transaction file", type=[s.lstrip(".") for s in SUPPORTED_UPLOAD_SUFFIXES]
)
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Load synthetic demo data instead"):
        try:
            with open("data/sample/sample_customer_transactions.csv", "rb") as fh:
                data, name = fh.read(), "sample_customer_transactions.csv"
            preview = load_upload_bytes(data, name)
            st.session_state["raw_data"] = preview.frame
            st.session_state["source_filename"] = name
            st.session_state["column_mapping"] = dict(preview.proposal.mapping)
            st.success(
                f"Loaded demo data: {preview.rows} rows, {preview.columns} columns "
                f"({preview.memory_mb:.1f} MB, {preview.duplicate_rows} duplicate rows)."
            )
        except CustomerIntelligenceError as exc:
            st.error(str(exc))
with col_b:
    if uploaded is not None:
        try:
            preview = load_upload_bytes(uploaded.getvalue(), uploaded.name)
            st.session_state["raw_data"] = preview.frame
            st.session_state["source_filename"] = uploaded.name
            st.session_state["column_mapping"] = dict(preview.proposal.mapping)
            st.success(
                f"Loaded {uploaded.name}: {preview.rows} rows, {preview.columns} columns "
                f"({preview.memory_mb:.1f} MB, {preview.duplicate_rows} duplicate rows)."
            )
        except CustomerIntelligenceError as exc:
            st.error(str(exc))

raw = st.session_state.get("raw_data")
if raw is None:
    st.info("Upload a file or load the demo data to begin mapping.")
    st.stop()

st.subheader("File preview")
st.dataframe(raw.head(10), use_container_width=True)

st.subheader("Column mapping")
st.markdown(
    "Each business field maps to exactly one file column. Required fields "
    "must all be mapped before cleaning."
)
sources = list(raw.columns)
saved = dict(st.session_state.get("column_mapping") or {})
mapping: dict[str, str] = {}
for canonical in CANONICAL_FIELDS:
    options = ["(not mapped)", *sources]
    default = saved.get(canonical, "(not mapped)")
    choice = st.selectbox(
        f"{canonical}{' *' if canonical in REQUIRED_FIELDS else ''}",
        options,
        index=options.index(default) if default in options else 0,
        key=f"map_{canonical}",
    )
    if choice != "(not mapped)":
        mapping[canonical] = choice

missing_required = [f for f in REQUIRED_FIELDS if f not in mapping]
if missing_required:
    st.warning("Still unmapped (required): " + ", ".join(missing_required))
else:
    try:
        validate_mapping(mapping, sources)
        st.success("All required fields are mapped and the mapping is valid.")
    except CustomerIntelligenceError as exc:
        st.error(str(exc))

if st.button("Save mapping", disabled=bool(missing_required)):
    try:
        validate_mapping(mapping, sources)
        st.session_state["column_mapping"] = dict(mapping)
        # Downstream results depend on the mapping; invalidate them.
        state_service.reset(
            st.session_state,
            keys=[
                "clean_data",
                "cleaning_audit",
                "data_dictionary",
                "customer_features",
                "rfm_results",
                "lrfm_results",
            ],
        )
        st.session_state["column_mapping"] = dict(mapping)
        st.success("Mapping saved. Continue to 02 · Data Quality & Cleaning.")
    except CustomerIntelligenceError as exc:
        st.error(str(exc))

if st.session_state.get("column_mapping"):
    st.download_button(
        "Download mapping (JSON)",
        data=to_json_bytes(pd.Series(st.session_state["column_mapping"]).to_dict()),
        file_name="column_mapping.json",
        mime="application/json",
    )

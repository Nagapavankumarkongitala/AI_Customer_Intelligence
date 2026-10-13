"""Page 01 — Data upload, automatic column detection, and manual mapping."""

import pandas as pd
import streamlit as st

from components.empty_state import empty_state
from components.filters import control_group, controls_expander
from components.page_header import page_header
from components.sidebar import render_sidebar_footer, render_sidebar_nav
from services import state_service
from services.column_detection import validate_mapping
from services.data_service import load_upload_bytes
from services.export_service import to_json_bytes
from utils.constants import CANONICAL_FIELDS, REQUIRED_FIELDS, SUPPORTED_UPLOAD_SUFFIXES
from utils.exceptions import CustomerIntelligenceError
from utils.logging_config import configure_logging
from visualization.theme import inject_terminal_css

configure_logging()
state_service.initialize(st.session_state)
inject_terminal_css()
render_sidebar_nav("pages/01_Data_Upload_and_Mapping.py")

page_header(
    "Data Upload & Mapping",
    "Upload a transaction file, review the automatic column mapping, correct it if "
    "needed, then continue to Data Quality & Cleaning. "
    "Required fields: customer_id, transaction_date, amount_spent.",
)

with controls_expander():
    with control_group("Data Source"):
        uploaded = st.file_uploader(
            "Transaction file", type=[s.lstrip(".") for s in SUPPORTED_UPLOAD_SUFFIXES]
        )
        use_demo = st.button("Load synthetic demo data")
        use_default = st.button("Load default dataset (data_edited.csv)")
    with control_group("Column Mapping (auto-detected, correct as needed)"):
        raw_preview = st.session_state.get("raw_data")
        if raw_preview is not None:
            sources = list(raw_preview.columns)
            saved = dict(st.session_state.get("column_mapping") or {})
            mapping = {}
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
            st.session_state["_mapping_draft"] = mapping

if use_demo:
    try:
        with open("data/sample/sample_customer_transactions.csv", "rb") as fh:
            data, name = fh.read(), "sample_customer_transactions.csv"
        preview = load_upload_bytes(data, name)
        st.session_state["raw_data"] = preview.frame
        st.session_state["source_filename"] = name
        st.session_state["column_mapping"] = dict(preview.proposal.mapping)
    except CustomerIntelligenceError as exc:
        st.error(str(exc))
if use_default:
    try:
        from services import bootstrap_service

        state_service.reset(st.session_state)
        manifest = bootstrap_service.load_default_into_state(st.session_state)
        if manifest is None:
            st.error(
                "Default cache not built yet. Run `python scripts/build_default_run.py` "
                "once, then retry."
            )
        else:
            st.success(
                f"Default dataset {manifest['source']} loaded with pre-trained runs: "
                f"{manifest['customers']:,} customers, "
                f"churn {manifest['selected_churn_model']} AUC {manifest['churn_auc']:.3f}."
            )
    except CustomerIntelligenceError as exc:
        st.error(str(exc))
if uploaded is not None:
    try:
        preview = load_upload_bytes(uploaded.getvalue(), uploaded.name)
        st.session_state["raw_data"] = preview.frame
        st.session_state["source_filename"] = uploaded.name
        st.session_state["column_mapping"] = dict(preview.proposal.mapping)
    except CustomerIntelligenceError as exc:
        st.error(str(exc))

raw = st.session_state.get("raw_data")
if raw is None:
    empty_state("No data yet", "Upload a file or load the demo data to begin mapping.")
    st.stop()

st.success(
    f"{st.session_state.get('source_filename', 'Dataset')}: {len(raw)} rows, "
    f"{len(raw.columns)} columns."
)
with st.expander("Inspect file preview (top 10)", expanded=False):
    st.dataframe(raw.head(10), width="stretch")

mapping = st.session_state.get("_mapping_draft") or dict(
    st.session_state.get("column_mapping") or {}
)
sources = list(raw.columns)
missing_required = [f for f in REQUIRED_FIELDS if f not in mapping]
if missing_required:
    st.warning("Still unmapped (required): " + ", ".join(missing_required))
else:
    try:
        validate_mapping(mapping, sources)
        st.success("All required fields are mapped and the mapping is valid.")
    except CustomerIntelligenceError as exc:
        st.error(str(exc))

col_a, col_b = st.columns(2)
with col_a:
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
            st.success("Mapping saved. Continue to Data Quality & Cleaning.")
        except CustomerIntelligenceError as exc:
            st.error(str(exc))
with col_b:
    if st.session_state.get("column_mapping"):
        st.download_button(
            "Download mapping (JSON)",
            data=to_json_bytes(pd.Series(st.session_state["column_mapping"]).to_dict()),
            file_name="column_mapping.json",
            mime="application/json",
        )

render_sidebar_footer()

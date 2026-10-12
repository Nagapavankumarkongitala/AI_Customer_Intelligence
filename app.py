"""Landing page: purpose, input schema, navigation, and current build status."""

import streamlit as st

from components.nav_tabs import render_nav_tabs
from services import state_service
from utils.config import load_config
from utils.exceptions import ConfigurationError
from utils.logging_config import configure_logging
from visualization.theme import inject_terminal_css

configure_logging()

try:
    load_config()
except ConfigurationError as exc:
    st.error(f"Startup blocked: {exc}")
    st.stop()

state_service.initialize(st.session_state)

st.set_page_config(page_title="AI Customer Intelligence", layout="wide")
inject_terminal_css()
render_nav_tabs("app.py")
st.title("AI Customer Intelligence Platform")

st.markdown("""
Upload customer transactions once, then move through the sidebar pages in order:
**Upload & Mapping → Data Quality & Cleaning → RFM/LRFM → Demographics →
Advanced Segmentation → Comparison → Churn Prediction → Retention → Executive Dashboard.**
Each page reads from shared session state; nothing is recomputed unless inputs change.
""")

st.subheader("Expected input fields")
st.markdown("""
**Required:** `customer_id` (text, never blank), `transaction_date` (parseable date),
`amount_spent` (number).

**Optional (used when present):** `gender`, `age`, `city`, `state`, `country`, `region`.

Column names do not need to match exactly — the upload page auto-detects common
variants (e.g. `order_date`, `revenue`, `cust_id`) and lets you correct the mapping.
""")

st.subheader("Current build status")
st.markdown("""
- **Phase 0 (this build):** repository initialization — config, logging, sample data,
  startup scripts, and test/lint tooling. Analytical pages are stubs under construction.
- **Next:** Phase 1 data foundation (upload, mapping, quality, cleaning, audit, export).
- See `PROJECT_STATUS.md` for the live phase tracker and `OPEN_CODE_BUILD_INSTRUCTIONS.md`
  for the full phase specification.
""")

st.info(
    "No data yet? Generate the synthetic demo file with "
    "`python utils/sample_data.py` and upload `data/sample/sample_customer_transactions.csv`."
)

st.subheader("Pipeline status")
status = state_service.status_report(st.session_state)
stages = [
    ("01 Upload", "raw_data"),
    ("02 Clean", "clean_data"),
    ("03 Features", "customer_features"),
    ("03 RFM", "rfm_results"),
    ("05 AI segments", "segmented_customers"),
    ("07 Scores", "churn_scores"),
    ("08 Actions", "retention_results"),
]
cols = st.columns(len(stages))
for col, (label, key) in zip(cols, stages, strict=True):
    col.metric(label, "done" if status.get(key) != "empty" else "—")

"""Landing page: purpose, input schema, navigation, and current build status."""

import streamlit as st

from utils.logging_config import configure_logging

configure_logging()

st.set_page_config(page_title="AI Customer Intelligence", layout="wide")
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

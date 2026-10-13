"""Landing page: purpose, input schema, navigation, and current build status."""

import streamlit as st

from components.metric_card import Metric, metric_row
from components.page_header import page_header, section_header
from components.sidebar import render_sidebar_footer, render_sidebar_nav
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

try:
    from services import bootstrap_service

    default_manifest = bootstrap_service.load_default_into_state(st.session_state)
except Exception as exc:  # missing cache or corrupt artifact: app stays usable
    default_manifest = None
    st.warning(f"Default dataset unavailable ({exc}). Upload a file to begin.")

st.set_page_config(page_title="AI Customer Intelligence", layout="wide")
inject_terminal_css()
render_sidebar_nav("app.py")

page_header(
    "AI Customer Intelligence",
    "Upload customer transactions once, then move through the sidebar pages in order: "
    "Upload & Mapping → Data Quality & Cleaning → RFM/LRFM → Demographics → "
    "Advanced Segmentation → Comparison → Churn Prediction → Retention → Executive "
    "Dashboard. Each page reads from shared session state; nothing is recomputed "
    "unless inputs change.",
)

section_header("Expected input fields")
st.markdown("""
**Required:** `customer_id` (text, never blank), `transaction_date` (parseable date),
`amount_spent` (number).

**Optional (used when present):** `gender`, `age`, `city`, `state`, `country`, `region`.

Column names do not need to match exactly — the upload page auto-detects common
variants (e.g. `order_date`, `revenue`, `cust_id`) and lets you correct the mapping.
""")

section_header("Current build status")
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

section_header("Default dataset & pre-trained runs")
if default_manifest:
    st.success(
        f"**{default_manifest['source']}** auto-loaded: "
        f"{default_manifest['rows']:,} transactions, "
        f"{default_manifest['customers']:,} customers "
        f"(built {default_manifest['built_at']}). All pages below show its results."
    )
    kpis = [
        Metric("Quality score", f"{default_manifest['quality_overall']:.1f}"),
        Metric(
            "RFM / LRFM segments",
            f"{default_manifest['rfm_segments']} / {default_manifest['lrfm_segments']}",
        ),
        Metric(
            "Segmentation",
            f"{default_manifest['selected_segmentation_model']} "
            f"(+{len(default_manifest['segmentation_models'])} trained)",
        ),
        Metric(
            f"Churn {default_manifest['selected_churn_model']}",
            f"AUC {default_manifest['churn_auc']:.3f} · rate {default_manifest['churn_rate']:.1%}",
        ),
        Metric("Retention ranked", f"{default_manifest['retention_customers']:,}"),
    ]
    metric_row(kpis)

    with st.expander("RFM segment mix (default run)", expanded=False):
        rfm_default = st.session_state.get("rfm_results")
        if rfm_default is not None:
            mix = (
                rfm_default["segment"]
                .value_counts()
                .rename_axis("segment")
                .reset_index(name="customers")
            )
            st.dataframe(mix, width="stretch")
    with st.expander("Churn test metrics (default run)", expanded=False):
        meta = st.session_state.get("churn_model_metadata") or {}
        if meta.get("metrics"):
            st.write(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in meta["metrics"].items()
                }
            )
            st.caption(
                f"Model {meta.get('model')} · cutoff {meta.get('cutoff')} · "
                f"horizon {meta.get('horizon_days')}d · calibration {meta.get('calibration')}."
            )
    with st.expander("Retention summary (default run)", expanded=False):
        retention_default = st.session_state.get("retention_results") or {}
        if retention_default.get("summary"):
            st.write(retention_default["summary"])
    st.caption(
        "Hierarchical/ensemble segmentation are skipped in the default run (O(n²) at "
        "~25k customers) — train them on page 05. Rebuild after data changes with "
        "`python scripts/build_default_run.py --force`."
    )
else:
    st.info(
        "Default cache not built yet. Run `python scripts/build_default_run.py` once, "
        "then reload — or upload any file on Data Upload & Mapping."
    )
section_header("Pipeline status")
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
metric_row(
    [
        Metric(
            label,
            "done" if status.get(key) != "empty" else "—",
            tone="positive" if status.get(key) != "empty" else "neutral",
        )
        for label, key in stages
    ]
)

render_sidebar_footer()

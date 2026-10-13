"""Page 08 — Retention intelligence: value at risk, actions, campaign export."""

import streamlit as st

from components.filters import controls_expander, multiselect_filter
from components.metric_cards import kpi_row
from components.nav_tabs import render_nav_tabs
from retention_engine.offer_recommender import describe_offer
from services import state_service
from services.export_service import to_csv_bytes
from services.retention_service import (
    build_retention_table,
    default_matrix_report,
    export_campaign,
    retention_summary,
)
from utils.exceptions import CustomerIntelligenceError
from utils.logging_config import configure_logging
from visualization.retention_charts import (
    priority_histogram,
    revenue_at_risk_bar,
    strategy_distribution_bar,
    value_risk_matrix,
)
from visualization.theme import inject_terminal_css

configure_logging()
state_service.initialize(st.session_state)
inject_terminal_css()
render_nav_tabs("pages/08_Retention_Strategies.py")

st.markdown(
    "Decision support, not promises: historical value × modeled churn risk becomes "
    "a ranked, fully reasoned action list. No causal retention impact is claimed."
)

scores = st.session_state.get("churn_scores")
features = st.session_state.get("customer_features")
if scores is None or features is None:
    st.info("Score customers on Churn Prediction first (needs churn scores + features).")
    st.stop()

features = features.copy()
with controls_expander("Segment Source"):
    segment_source = st.selectbox(
        "Segment labels for filtering",
        [
            s
            for s in ("RFM", "AI")
            if s == "RFM"
            and st.session_state.get("rfm_results") is not None
            or s == "AI"
            and st.session_state.get("segmented_customers") is not None
        ]
        or ["(none)"],
    )
if segment_source == "RFM":
    seg = st.session_state["rfm_results"].set_index("customer_id")["segment"]
    features["segment"] = features["customer_id"].map(seg)
elif segment_source == "AI":
    seg = st.session_state["segmented_customers"].set_index("customer_id")["business_name"]
    features["segment"] = features["customer_id"].map(seg)

value_options = [c for c in ("monetary_value", "average_order_value") if c in features.columns]
with controls_expander("Value Config"):
    value_basis = st.selectbox("Value basis (historical, not CLV)", value_options)
    weight = st.number_input(
        "Strategic weight",
        min_value=0.0,
        value=1.0,
        step=0.25,
        help="Multiplies every priority score; per-segment weights "
        "stay 1.0 unless tuned in code.",
    )
    quantiles = st.selectbox(
        "Value tiers",
        [3],
        format_func=lambda _: "Low/Med/High",
        help="Tier count follows config value_quantiles.",
    )

try:
    matrix, report = default_matrix_report()
except (ValueError, FileNotFoundError) as exc:
    st.error(f"Strategy matrix problem: {exc}")
    st.stop()
if report["uncovered"]:
    st.error(f"Strategy matrix has uncovered cells: {report['uncovered']}")
    st.stop()

try:
    with st.spinner("Scoring retention priorities…"):
        table = build_retention_table(
            scores,
            features,
            matrix,
            value_basis=value_basis,
            strategic_weights=float(weight),
            value_quantiles=int(quantiles),
        )
except (CustomerIntelligenceError, KeyError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

st.session_state["retention_results"] = {
    "table": table.to_dict(orient="records"),
    "summary": retention_summary(table),
}
summary = retention_summary(table)
st.info(matrix.disclaimer)

st.subheader("At a glance")
kpi_row(
    [
        ("Revenue at risk", f"₹{summary['total_revenue_at_risk']:,.0f}"),
        ("Customers scored", f"{summary['customers']:,}"),
        ("Top action", max(summary["strategy_mix"], key=summary["strategy_mix"].get)),
        ("Critical band", str(summary["band_mix"].get("critical", 0))),
    ]
)

ret_l, ret_r = st.columns([1, 1])
with ret_l:
    st.plotly_chart(value_risk_matrix(table), use_container_width=True)
with ret_r:
    st.plotly_chart(revenue_at_risk_bar(table), use_container_width=True)
strat_l, strat_r = st.columns([1, 1])
with strat_l:
    st.plotly_chart(strategy_distribution_bar(table), use_container_width=True)
with strat_r:
    st.plotly_chart(priority_histogram(table), use_container_width=True)

with controls_expander("View Filters"):
    regions = sorted(table["region"].dropna().unique()) if "region" in table.columns else []
    region = multiselect_filter("Region", [str(r) for r in regions], key="ret_region")
    segs = sorted(table["segment"].dropna().unique()) if "segment" in table.columns else []
    seg_pick = multiselect_filter("Segment", [str(s) for s in segs], key="ret_seg")
    bands = sorted(table["risk_band"].unique())
    band_pick = multiselect_filter("Risk band", [str(b) for b in bands], key="ret_band")
view = table.copy()
if regions:
    view = view[view["region"].isin(region)]
if segs:
    view = view[view["segment"].isin(seg_pick)]
view = view[view["risk_band"].isin(band_pick)]
with st.expander("Ranked customers (top 200 of filter)", expanded=False):
    st.dataframe(view.head(200), use_container_width=True)
    st.caption(
        f"Showing up to 200 of {len(view)} filtered customers. Every row names its "
        "strategy rule — click into action_reason for the full audit trail."
    )

with st.expander("Offer catalog", expanded=False):
    for offer in sorted(view["suggested_offer_type"].unique()):
        st.markdown(f"**{offer}** — {describe_offer(offer)}")

with controls_expander("Campaign Settings"):
    top_n = st.number_input(
        "Campaign size (top priority)", min_value=10, max_value=5000, value=200, step=10
    )
    min_priority = st.number_input("Minimum priority score", min_value=0.0, value=0.0, step=0.05)
campaign, channel_summary = export_campaign(
    view, top_n=int(top_n), min_priority=float(min_priority)
)
with st.expander("Campaign export", expanded=False):
    st.dataframe(channel_summary, use_container_width=True)
    st.dataframe(campaign.head(100), use_container_width=True)
st.download_button(
    "Download campaign (CSV)",
    data=to_csv_bytes(campaign),
    file_name="retention_campaign.csv",
    mime="text/csv",
)
st.download_button(
    "Download full retention table (CSV)",
    data=to_csv_bytes(table),
    file_name="retention_actions.csv",
    mime="text/csv",
)

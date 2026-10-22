"""Page 08 — Retention intelligence: value at risk, actions, campaign export."""

import streamlit as st

from components.cached import cached_build_retention_table, cached_export_campaign
from components.empty_state import chart_section, empty_state
from components.filters import controls_expander, multiselect_filter
from components.metric_card import Metric, metric_row
from components.page_header import page_header, section_header
from components.sidebar import render_sidebar_footer, render_sidebar_nav, sidebar_filters
from components.strategy_cards import render_strategy_playbook
from components.table import render_table
from retention_engine.offer_recommender import describe_offer
from services import state_service
from services.export_service import to_csv_bytes
from services.retention_service import (
    default_matrix_report,
    retention_summary,
)
from utils import formatting
from utils.exceptions import CustomerIntelligenceError
from utils.logging_config import configure_logging
from visualization.retention_charts import (
    priority_histogram,
    revenue_at_risk_bar,
    strategy_distribution_bar,
    value_risk_matrix,
)
from visualization.theme import inject_terminal_css, without_title

configure_logging()
state_service.initialize(st.session_state)
inject_terminal_css()
render_sidebar_nav("pages/08_Retention_Strategies.py")

page_header(
    "Retention Strategies",
    "Decision support, not promises: historical value × modeled churn risk becomes "
    "a ranked, fully reasoned action list. No causal retention impact is claimed.",
)

scores = st.session_state.get("churn_scores")
features = st.session_state.get("customer_features")
if scores is None or features is None:
    empty_state(
        "No scores yet",
        "Score customers on Churn Prediction first (needs churn scores + features).",
    )
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
        table = cached_build_retention_table(
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
st.info(matrix.disclaimer)

with sidebar_filters("View Filters"):
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
if view.empty:
    empty_state(
        "No customers in these filters",
        "Widen a region, segment, or risk band to see retention actions.",
    )
    st.stop()
st.caption(
    f"View filters select {len(view):,} of {len(table):,} scored customers; every "
    "card and chart below follows them."
)
summary = retention_summary(view)

section_header("At a glance")
critical_count = int(summary["band_mix"].get("critical", 0))
metric_row(
    [
        Metric(
            "Revenue at risk",
            formatting.fmt_money_compact(summary["total_revenue_at_risk"]),
        ),
        Metric("Customers scored", formatting.fmt_int(summary["customers"])),
        Metric(
            "Top action",
            max(summary["strategy_mix"], key=summary["strategy_mix"].get),
        ),
        Metric(
            "Critical band",
            str(critical_count),
            tone="negative" if critical_count else "neutral",
        ),
    ]
)

section_header(
    "Strategy playbook", "One card per strategy rule, with the customers it currently owns."
)
render_strategy_playbook(view, matrix)

ret_l, ret_r = st.columns([1, 1])
with ret_l:
    with chart_section("Value–risk matrix"):
        st.plotly_chart(without_title(value_risk_matrix(view)), width="stretch")
with ret_r:
    with chart_section("Revenue at risk per action"):
        st.plotly_chart(without_title(revenue_at_risk_bar(view)), width="stretch")
strat_l, strat_r = st.columns([1, 1])
with strat_l:
    with chart_section("Customers per recommended action"):
        st.plotly_chart(without_title(strategy_distribution_bar(view)), width="stretch")
with strat_r:
    with chart_section("Priority score distribution"):
        st.plotly_chart(without_title(priority_histogram(view)), width="stretch")

with st.expander("Ranked customers (top 200 of filter)", expanded=False):
    render_table(
        view.head(200),
        risk_cols=("risk_band",),
        money_cols=("revenue_at_risk", "expected_value_proxy_historical"),
    )
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
campaign, channel_summary = cached_export_campaign(
    view, top_n=int(top_n), min_priority=float(min_priority)
)
with st.expander("Campaign export", expanded=False):
    st.dataframe(channel_summary, width="stretch")
    st.dataframe(campaign.head(100), width="stretch")
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

render_sidebar_footer()

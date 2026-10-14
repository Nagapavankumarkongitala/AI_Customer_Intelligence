"""Landing page: executive dashboard over the shared session state."""

import pandas as pd
import streamlit as st

from components.cached import (
    cached_apply_filters,
    cached_build_master,
    cached_compute_kpis,
    cached_high_value_high_risk,
    cached_prepare_transactions,
    cached_segment_portfolio,
    cached_trend_tables,
)
from components.empty_state import chart_section, empty_state
from components.export_buttons import csv_download_button
from components.filters import date_range_filter, multiselect_filter
from components.insight_box import insight_box
from components.metric_card import Metric, metric_row
from components.page_header import page_header, section_header
from components.panels import download_panel
from components.sidebar import render_sidebar_footer, render_sidebar_nav, sidebar_filters
from components.table import render_table
from services import state_service
from services.dashboard_service import mask_customer_id
from utils import formatting
from utils.config import load_config
from utils.exceptions import ConfigurationError
from utils.logging_config import configure_logging
from visualization.churn_charts import risk_band_bar
from visualization.demographic_charts import age_band_bar, gender_bar, revenue_geo_bar
from visualization.executive_dashboard import (
    activity_trend_fig,
    new_returning_fig,
    portfolio_bar_fig,
    revenue_area_fig,
)
from visualization.retention_charts import (
    priority_histogram,
    revenue_at_risk_bar,
    strategy_distribution_bar,
    value_risk_matrix,
)
from visualization.rfm_charts import (
    revenue_by_segment_bar,
    segment_share_donut,
)
from visualization.theme import apply_chart_theme, inject_terminal_css

configure_logging()

try:
    load_config()
except ConfigurationError as exc:
    st.error(f"Startup blocked: {exc}")
    st.stop()

state_service.initialize(st.session_state)

st.set_page_config(page_title="AI Customer Intelligence", layout="wide")
inject_terminal_css()
with st.spinner("Loading data…"):
    try:
        from services import bootstrap_service

        default_manifest = bootstrap_service.load_default_into_state(st.session_state)
    except Exception as exc:  # missing cache or corrupt artifact: app stays usable
        default_manifest = None
        st.warning(f"Default dataset unavailable ({exc}). Upload a file to begin.")
render_sidebar_nav("app.py")


def _fallback_features(txns: pd.DataFrame) -> pd.DataFrame:
    """Minimal customer grain when page 03 has not run: IDs + first/last dates."""
    grouped = txns.groupby("customer_id")["_txn_date"].agg(["min", "max"])
    frame = pd.DataFrame(
        {
            "customer_id": grouped.index,
            "first_transaction_date": grouped["min"].dt.date.astype(str),
            "last_transaction_date": grouped["max"].dt.date.astype(str),
            "monetary_value": txns.groupby("customer_id")["_amount"].sum().values,
            "frequency": txns.groupby("customer_id").size().values,
            "recency_days": 0,
        }
    )
    return frame.reset_index(drop=True)


def _retention_frame() -> pd.DataFrame | None:
    """Retention table from state (records orient) or None when page 08 has not run."""
    stored = st.session_state.get("retention_results") or {}
    rows = stored.get("table") if isinstance(stored, dict) else None
    return pd.DataFrame(rows) if rows else None


def _compact(fig):
    """Reflow a factory figure to the compact overview height (idempotent)."""
    return apply_chart_theme(fig, compact=True)


page_header(
    "Executive Dashboard",
    "Filter-consistent performance, risk, and retention at a glance.",
)
if default_manifest:
    st.success(
        f"**{default_manifest['source']}** auto-loaded: "
        f"{default_manifest['rows']:,} transactions, "
        f"{default_manifest['customers']:,} customers "
        f"(built {default_manifest['built_at']}). All cards below show its results."
    )
st.markdown(
    "Every card below derives from the same sidebar Filters. "
    "Customer IDs are masked except in the gated drill-down."
)

clean_data = st.session_state.get("clean_data")
if clean_data is None:
    empty_state(
        "No clean data yet",
        "Upload a file on Data Upload & Mapping, or load the demo data there.",
    )
    st.stop()

features = st.session_state.get("customer_features")
txns = cached_prepare_transactions(clean_data)
master = cached_build_master(
    features if features is not None else _fallback_features(txns),
    rfm=st.session_state.get("rfm_results"),
    segmented=st.session_state.get("segmented_customers"),
    scores=st.session_state.get("churn_scores"),
    retention=_retention_frame(),
)

with sidebar_filters():
    date_start, date_end = date_range_filter(txns["_txn_date"])
    filters: dict = {"date_start": date_start, "date_end": date_end}
    for level in ("country", "region", "state", "city", "gender"):
        opts = (
            sorted(txns[level].dropna().unique())
            if level in txns.columns
            else sorted(master[level].dropna().unique()) if level in master.columns else []
        )
        filters[level] = multiselect_filter(
            level.capitalize(), [str(o) for o in opts], key=f"f_{level}"
        )
    for key, column in (
        ("rfm_segment", "rfm_segment"),
        ("ai_segment", "ai_segment"),
        ("risk_band", "risk_band"),
        ("value_tier", "customer_value_tier"),
    ):
        opts = sorted(master[column].dropna().unique()) if column in master.columns else []
        filters[key] = multiselect_filter(
            key.replace("_", " ").title(), [str(o) for o in opts], key=f"f_{key}"
        )

txns_f, master_f = cached_apply_filters(txns, master, filters)
if master_f.empty:
    empty_state(
        "No customers match the current filters",
        "Widen a filter to continue.",
    )
    st.stop()
kpis = cached_compute_kpis(txns_f, master_f)
trends = cached_trend_tables(txns_f, master_f)

retention = _retention_frame()
ret_f = (
    retention[retention["customer_id"].isin(set(master_f["customer_id"]))]
    if retention is not None and not retention.empty
    else None
)
portfolio = (
    cached_segment_portfolio(master_f, "rfm_segment") if "rfm_segment" in master_f.columns else None
)
hvhr = cached_high_value_high_risk(master_f)

overview_tab, trends_tab, segments_tab, churn_tab, retention_tab = st.tabs(
    ["Business Overview", "Customer Trends", "Segments", "Churn Risk", "Retention"]
)

with overview_tab:
    metric_row(
        [
            Metric("Customers", formatting.fmt_int(kpis["total_customers"])),
            Metric("Revenue", formatting.fmt_money(kpis["total_revenue"])),
            Metric("Avg order value", formatting.fmt_money(kpis["average_order_value"])),
            Metric(
                "Predicted churn",
                (
                    formatting.fmt_pct(kpis["predicted_churn_rate"])
                    if kpis["predicted_churn_rate"] is not None
                    else "Needs page 07"
                ),
                tone="negative" if kpis["predicted_churn_rate"] is not None else "neutral",
            ),
            Metric(
                "Revenue at risk",
                (
                    formatting.fmt_money(kpis["revenue_at_risk"])
                    if kpis["revenue_at_risk"] is not None
                    else "Needs page 08"
                ),
                tone="negative" if kpis["revenue_at_risk"] is not None else "neutral",
            ),
        ]
    )
    row1_a, row1_b, row1_c = st.columns(3)
    with row1_a:
        with chart_section("Revenue trend"):
            st.plotly_chart(_compact(revenue_area_fig(trends["revenue"])), width="stretch")
    with row1_b:
        with chart_section("Customers per segment"):
            if portfolio is not None:
                st.plotly_chart(
                    _compact(segment_share_donut(portfolio, "rfm_segment")),
                    width="stretch",
                )
            else:
                empty_state(
                    "No RFM segments yet",
                    "RFM segments appear here after page 03 scoring.",
                )
    with row1_c:
        with chart_section("Customers per risk band"):
            if "risk_band" in master_f.columns:
                st.plotly_chart(_compact(risk_band_bar(master_f)), width="stretch")
            else:
                empty_state(
                    "No churn risk yet",
                    "Churn risk appears here after page 07 scoring.",
                )

    row2_a, row2_b, row2_c = st.columns(3)
    with row2_a:
        with chart_section("Value vs churn risk"):
            matrix_cols = {
                "churn_probability",
                "expected_value_proxy_historical",
                "revenue_at_risk",
                "recommended_action",
            }
            if ret_f is not None and not ret_f.empty and matrix_cols.issubset(ret_f.columns):
                st.plotly_chart(_compact(value_risk_matrix(ret_f)), width="stretch")
            else:
                empty_state(
                    "No retention actions yet",
                    "The bubble chart appears here after page 08.",
                )
    with row2_b:
        with chart_section("Top regions by revenue"):
            if "region" in txns_f.columns:
                region_rev = (
                    txns_f.assign(region=txns_f["region"].fillna("Unknown"))
                    .groupby("region")["_amount"]
                    .sum()
                    .reset_index()
                    .rename(columns={"_amount": "revenue"})
                )
                st.plotly_chart(
                    _compact(revenue_geo_bar(region_rev, "region", horizontal=True)),
                    width="stretch",
                )
            else:
                empty_state("No regions yet", "No regional field in this dataset.")
    with row2_c:
        with chart_section("Customer demographics"):
            demo_age_tab, demo_gender_tab = st.tabs(["Age", "Gender"])
            with demo_age_tab:
                if "age_band" in master_f.columns:
                    bands = (
                        master_f["age_band"]
                        .value_counts()
                        .rename_axis("age_band")
                        .reset_index(name="customers")
                    )
                    st.plotly_chart(_compact(age_band_bar(bands)), width="stretch")
                else:
                    empty_state(
                        "No age bands yet",
                        "Age bands appear here after page 04 profiling.",
                    )
            with demo_gender_tab:
                if "gender" in master_f.columns:
                    gender_tbl = (
                        master_f["gender"]
                        .fillna("Unknown")
                        .value_counts()
                        .rename_axis("gender")
                        .reset_index(name="customers")
                    )
                    st.plotly_chart(_compact(gender_bar(gender_tbl)), width="stretch")
                else:
                    empty_state(
                        "No gender data yet",
                        "Gender appears here after page 04 profiling.",
                    )

    row3_a, row3_b = st.columns(2)
    with row3_a:
        with chart_section("Top retention opportunities"):
            if ret_f is not None and "retention_priority_score" in ret_f.columns:
                top_opps = ret_f.sort_values("retention_priority_score", ascending=False).head(10)
                top_opps = top_opps.copy()
                top_opps["customer_id"] = top_opps["customer_id"].map(mask_customer_id)
                render_table(
                    top_opps,
                    risk_cols=("risk_band",),
                    money_cols=("revenue_at_risk",),
                    height=320,
                )
            else:
                empty_state(
                    "No opportunities yet",
                    "Retention opportunities appear here after page 08.",
                )
    with row3_b:
        with chart_section("Key insights & recommendations"):
            insights = 0
            if portfolio is not None and not portfolio.empty:
                top = portfolio.loc[portfolio["revenue_share"].idxmax()]
                insight_box(
                    f"Revenue concentration: {top['rfm_segment']} holds "
                    f"{top['revenue_share']:.1f}% of filtered revenue.",
                    "Compare against its customer share before acting.",
                    tone="warning" if top["revenue_share"] > 50 else "info",
                )
                insights += 1
            if not hvhr.empty:
                insight_box(
                    f"{len(hvhr)} high-value customers sit in high or critical bands.",
                    "Prioritize VIP intervention for these rows.",
                    tone="critical",
                )
                insights += 1
            if ret_f is not None and not ret_f.empty:
                insight_box(
                    f"{len(ret_f):,} of {len(master_f):,} filtered customers carry a "
                    "recommended retention action.",
                    "Uncovered customers are invisible to campaign export.",
                    tone="info",
                )
                insights += 1
            if not insights:
                empty_state(
                    "No insights yet",
                    "Insights appear as RFM, churn, and retention results arrive.",
                )

with trends_tab:
    section_header("Customer trends", "Activity and cohort mix over time.")
    cohort = trends["cohort"]
    metric_row(
        [
            Metric("Transactions", formatting.fmt_int(len(txns_f))),
            Metric("Active rate", formatting.fmt_pct(kpis["active_customer_rate"])),
            Metric("New customers", formatting.fmt_int(cohort["new_customers"].sum())),
            Metric("Returning customers", formatting.fmt_int(cohort["returning_customers"].sum())),
        ]
    )
    trend_left, trend_right = st.columns(2)
    with trend_left:
        with chart_section("Customer activity trend"):
            st.plotly_chart(activity_trend_fig(trends["activity"]), width="stretch")
    with trend_right:
        with chart_section("New vs returning customers"):
            st.plotly_chart(new_returning_fig(trends["cohort"]), width="stretch")
    with st.expander("Monthly revenue table", expanded=False):
        render_table(trends["revenue"], money_cols=("revenue",))

with segments_tab:
    section_header("Segments", "Portfolio mix and revenue contribution.")
    if portfolio is None or portfolio.empty:
        empty_state(
            "No RFM segments yet",
            "RFM segments appear here after page 03 scoring.",
        )
    else:
        top_seg = portfolio.loc[portfolio["revenue_share"].idxmax()]
        metric_row(
            [
                Metric("Segments", str(len(portfolio))),
                Metric(
                    "Largest segment",
                    f"{top_seg['rfm_segment']} ({top_seg['customers']:,})",
                ),
                Metric("Top revenue share", f"{top_seg['revenue_share']:.1f}%"),
            ]
        )
        seg_left, seg_right = st.columns(2)
        with seg_left:
            with chart_section("Customer vs revenue share by segment"):
                st.plotly_chart(portfolio_bar_fig(portfolio, "rfm_segment"), width="stretch")
        with seg_right:
            with chart_section("Revenue per segment"):
                st.plotly_chart(revenue_by_segment_bar(portfolio, "rfm_segment"), width="stretch")
        with st.expander("Segment portfolio table", expanded=False):
            render_table(portfolio, money_cols=("revenue",))

with churn_tab:
    section_header("Churn risk", "Where predicted churn concentrates.")
    if "risk_band" not in master_f.columns:
        empty_state(
            "No churn risk yet",
            "Churn risk appears here after page 07 scoring.",
        )
    else:
        bands = master_f["risk_band"]
        metric_row(
            [
                Metric(
                    "Predicted churn",
                    (
                        formatting.fmt_pct(kpis["predicted_churn_rate"])
                        if kpis["predicted_churn_rate"] is not None
                        else "Needs page 07"
                    ),
                    tone="negative" if kpis["predicted_churn_rate"] is not None else "neutral",
                ),
                Metric("High-risk customers", formatting.fmt_int((bands == "high").sum())),
                Metric("Critical customers", formatting.fmt_int((bands == "critical").sum())),
            ]
        )
        with chart_section("Customers per risk band"):
            st.plotly_chart(risk_band_bar(master_f), width="stretch")
        with st.expander("High-value, high-risk (IDs masked)", expanded=False):
            if not hvhr.empty:
                masked_hvhr = hvhr.copy()
                masked_hvhr["customer_id"] = masked_hvhr["customer_id"].map(mask_customer_id)
                render_table(masked_hvhr, risk_cols=("risk_band",), money_cols=("revenue_at_risk",))
            else:
                empty_state(
                    "No high-value customers in risky bands",
                    "Nothing in the current filters needs VIP intervention.",
                )

with retention_tab:
    section_header("Retention", "Actions for the filtered customers.")
    if ret_f is None or ret_f.empty:
        empty_state(
            "No retention actions yet",
            "Retention actions appear here after page 08.",
        )
    else:
        metric_row(
            [
                Metric(
                    "Priority opportunities",
                    (
                        formatting.fmt_int(kpis["priority_opportunities"])
                        if kpis["priority_opportunities"] is not None
                        else "Needs page 08"
                    ),
                ),
                Metric("Campaign-ready customers", formatting.fmt_int(len(ret_f))),
                Metric(
                    "Avg priority score",
                    (
                        f"{ret_f['retention_priority_score'].mean():.3f}"
                        if "retention_priority_score" in ret_f.columns
                        else "–"
                    ),
                ),
            ]
        )
        ret_left, ret_right = st.columns(2)
        with ret_left:
            with chart_section("Customers per recommended action"):
                st.plotly_chart(strategy_distribution_bar(ret_f), width="stretch")
        with ret_right:
            with chart_section("Revenue at risk per action"):
                st.plotly_chart(revenue_at_risk_bar(ret_f), width="stretch")
        with chart_section("Priority score distribution"):
            st.plotly_chart(priority_histogram(ret_f), width="stretch")
        with st.expander("Ranked customers (top 200)", expanded=False):
            ranked = ret_f.sort_values("retention_priority_score", ascending=False).head(200)
            ranked = ranked.copy()
            ranked["customer_id"] = ranked["customer_id"].map(mask_customer_id)
            render_table(
                ranked,
                risk_cols=("risk_band",),
                money_cols=("revenue_at_risk", "expected_value_proxy_historical"),
            )
        download_panel(ret_f, "executive_retention_export", "dl_exec_ret")

section_header("Authorized drill-down")
st.warning(
    "Full customer identifiers are sensitive. Reveal them only for an authorized "
    "purpose (e.g. loading a campaign audience)."
)
if st.checkbox("I am authorized to view raw customer identifiers"):
    detail = master_f.copy()
    detail["customer_id"] = detail["customer_id"].astype(str)
    with st.expander("Customer detail (top 200)", expanded=False):
        st.dataframe(detail.head(200), width="stretch")
    csv_download_button(
        detail, "executive_customer_detail.csv", "Download detail (CSV)", key="dl_exec_detail"
    )
else:
    masked = master_f.copy()
    masked["customer_id"] = masked["customer_id"].map(mask_customer_id)
    with st.expander("Customer overview, IDs masked (top 200)", expanded=False):
        st.dataframe(masked.head(200), width="stretch")

render_sidebar_footer()

"""Page 09 — Executive dashboard: filter-consistent KPIs, trends, risk, actions."""

import pandas as pd
import streamlit as st

from analytics.demographics import GEO_LEVELS
from components.empty_state import chart_section, empty_state
from components.export_buttons import csv_download_button
from components.filters import (
    control_group,
    controls_expander,
    date_range_filter,
    multiselect_filter,
)
from components.insight_box import insight_box
from components.metric_card import Metric, metric_row
from components.nav_tabs import render_nav_tabs
from components.page_header import page_header, section_header
from components.panels import download_panel
from components.table import render_table
from services import state_service
from services.dashboard_service import (
    apply_filters,
    build_master,
    compute_kpis,
    high_value_high_risk,
    mask_customer_id,
    prepare_transactions,
    segment_portfolio,
    trend_tables,
)
from utils import formatting
from utils.logging_config import configure_logging
from visualization.churn_charts import risk_band_bar
from visualization.demographic_charts import age_band_bar, geo_treemap, top_geo_bar
from visualization.executive_dashboard import (
    activity_trend_fig,
    new_returning_fig,
    portfolio_bar_fig,
    revenue_trend_fig,
)
from visualization.retention_charts import (
    revenue_at_risk_bar,
    strategy_distribution_bar,
    value_risk_matrix,
)
from visualization.theme import inject_terminal_css

configure_logging()
state_service.initialize(st.session_state)
inject_terminal_css()
render_nav_tabs("pages/09_Executive_Dashboard.py")


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


page_header(
    "Executive Dashboard",
    "Filter-consistent performance, risk, and retention at a glance.",
    badge="Step 9 of 9",
)
st.markdown(
    "Every card below derives from the same Global Slicers. "
    "Customer IDs are masked except in the gated drill-down."
)

clean_data = st.session_state.get("clean_data")
if clean_data is None:
    empty_state(
        "No clean data yet",
        "Complete Data Quality & Cleaning first — the dashboard needs clean data.",
    )
    st.stop()

features = st.session_state.get("customer_features")
txns = prepare_transactions(clean_data)
master = build_master(
    features if features is not None else _fallback_features(txns),
    rfm=st.session_state.get("rfm_results"),
    segmented=st.session_state.get("segmented_customers"),
    scores=st.session_state.get("churn_scores"),
    retention=_retention_frame(),
)

with controls_expander("Global Slicers"):
    with control_group("Date & Geography"):
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

txns_f, master_f = apply_filters(txns, master, filters)
if master_f.empty:
    empty_state(
        "No customers match the current filters",
        "Widen a filter to continue.",
    )
    st.stop()
kpis = compute_kpis(txns_f, master_f)

retention = _retention_frame()
ret_f = (
    retention[retention["customer_id"].isin(set(master_f["customer_id"]))]
    if retention is not None and not retention.empty
    else None
)

section_header(
    "Key performance indicators",
    "Business health first, risk second — all from the same filters.",
)
metric_row(
    [
        Metric("Customers", formatting.fmt_int(kpis["total_customers"])),
        Metric("Revenue", formatting.fmt_money(kpis["total_revenue"])),
        Metric("Avg order value", formatting.fmt_money(kpis["average_order_value"])),
        Metric("Active rate", formatting.fmt_pct(kpis["active_customer_rate"])),
        Metric(
            "Predicted churn",
            (
                formatting.fmt_pct(kpis["predicted_churn_rate"])
                if kpis["predicted_churn_rate"] is not None
                else "Needs page 07"
            ),
            tone="negative" if kpis["predicted_churn_rate"] is not None else "neutral",
        ),
    ]
)
metric_row(
    [
        Metric(
            "Revenue at risk",
            (
                formatting.fmt_money(kpis["revenue_at_risk"])
                if kpis["revenue_at_risk"] is not None
                else "Needs page 08"
            ),
            tone="negative" if kpis["revenue_at_risk"] is not None else "neutral",
        ),
        Metric(
            "Priority opportunities",
            (
                formatting.fmt_int(kpis["priority_opportunities"])
                if kpis["priority_opportunities"] is not None
                else "Needs page 08"
            ),
        ),
    ]
)

section_header("Trends", "Monthly revenue, activity, and cohort mix.")
trends = trend_tables(txns_f, master_f)
trend_left, trend_right = st.columns(2)
with trend_left:
    with chart_section("Revenue trend"):
        st.plotly_chart(revenue_trend_fig(trends["revenue"]), use_container_width=True)
with trend_right:
    with chart_section("Customer activity trend"):
        st.plotly_chart(activity_trend_fig(trends["activity"]), use_container_width=True)
mix_left, mix_right = st.columns(2)
with mix_left:
    with chart_section("New vs returning customers"):
        st.plotly_chart(new_returning_fig(trends["cohort"]), use_container_width=True)
portfolio = (
    segment_portfolio(master_f, "rfm_segment") if "rfm_segment" in master_f.columns else None
)
with mix_right:
    with chart_section("Customer vs revenue share by segment"):
        if portfolio is not None:
            st.plotly_chart(portfolio_bar_fig(portfolio, "rfm_segment"), use_container_width=True)
        else:
            empty_state(
                "No RFM segments yet",
                "RFM segments appear here after page 03 scoring.",
            )

if portfolio is not None and not portfolio.empty:
    top = portfolio.loc[portfolio["revenue_share"].idxmax()]
    insight_box(
        f"Revenue concentration: {top['rfm_segment']} holds "
        f"{top['revenue_share']:.1f}% of filtered revenue "
        f"across {top['customers']:,} customers.",
        "Compare against its customer share in the portfolio table before acting.",
        tone="warning" if top["revenue_share"] > 50 else "info",
    )
    with st.expander("Segment portfolio table", expanded=False):
        render_table(portfolio, money_cols=("revenue",))

section_header(
    "Value versus risk",
    "Historical value against churn probability; bubble size is revenue at risk.",
)
matrix_cols = {
    "churn_probability",
    "expected_value_proxy_historical",
    "revenue_at_risk",
    "recommended_action",
}
if ret_f is not None and not ret_f.empty and matrix_cols.issubset(ret_f.columns):
    with chart_section("Value–risk matrix"):
        st.plotly_chart(value_risk_matrix(ret_f), use_container_width=True)
    insight_box(
        f"{len(ret_f):,} of {len(master_f):,} filtered customers carry a "
        "recommended retention action.",
        "Uncovered customers are invisible to the campaign export below.",
        tone="info",
    )
else:
    empty_state(
        "No retention actions yet",
        "The value–risk matrix appears here after page 08.",
    )

section_header("Risk", "Where predicted churn concentrates.")
if "risk_band" in master_f.columns:
    risk_left, risk_right = st.columns(2)
    with risk_left:
        with chart_section("Customers per risk band"):
            st.plotly_chart(risk_band_bar(master_f), use_container_width=True)
    with risk_right:
        with chart_section("High-value, high-risk (IDs masked)"):
            hvhr = high_value_high_risk(master_f)
            if not hvhr.empty:
                render_table(hvhr, risk_cols=("risk_band",), money_cols=("revenue_at_risk",))
                insight_box(
                    f"{len(hvhr)} high-value customers sit in high or critical bands.",
                    "These rows top the campaign export by revenue at risk.",
                    tone="critical",
                )
            else:
                empty_state(
                    "No high-value customers in risky bands",
                    "Nothing in the current filters needs VIP intervention.",
                )
else:
    empty_state(
        "No churn risk yet",
        "Churn risk appears here after page 07 scoring.",
    )

section_header("Demographics & geography", "Who and where the filtered base is.")
geo_levels = [lvl for lvl in GEO_LEVELS if lvl in txns_f.columns]
if geo_levels:
    with controls_expander("Geography View"):
        level = st.selectbox("Geography level", geo_levels)
    geo = (
        txns_f[level]
        .fillna("Unknown")
        .value_counts()
        .rename_axis(level)
        .reset_index(name="customers")
    )
    demo_left, demo_right = st.columns(2)
    with demo_left:
        with chart_section("Customers per age band"):
            if "age_band" in master_f.columns:
                bands = (
                    master_f["age_band"]
                    .value_counts()
                    .rename_axis("age_band")
                    .reset_index(name="customers")
                )
                st.plotly_chart(age_band_bar(bands), use_container_width=True)
            else:
                empty_state(
                    "No age bands yet",
                    "Age bands appear here after page 04 profiling.",
                )
    with demo_right:
        with chart_section(f"Top {level}s by customers"):
            st.plotly_chart(top_geo_bar(geo, level), use_container_width=True)
    with chart_section(f"Customer footprint ({level})"):
        st.plotly_chart(geo_treemap(geo.head(30), level), use_container_width=True)

section_header("Retention", "Actions for the filtered customers.")
if ret_f is not None and not ret_f.empty:
    ret_left, ret_right = st.columns(2)
    with ret_left:
        with chart_section("Customers per recommended action"):
            st.plotly_chart(strategy_distribution_bar(ret_f), use_container_width=True)
    with ret_right:
        with chart_section("Revenue at risk per action"):
            st.plotly_chart(revenue_at_risk_bar(ret_f), use_container_width=True)
    metric_row([Metric("Campaign-ready customers", formatting.fmt_int(len(ret_f)))])
    download_panel(ret_f, "executive_retention_export", "dl_exec_ret")
else:
    empty_state(
        "No retention actions yet",
        "Retention actions appear here after page 08.",
    )

section_header("Authorized drill-down")
st.warning(
    "Full customer identifiers are sensitive. Reveal them only for an authorized "
    "purpose (e.g. loading a campaign audience)."
)
if st.checkbox("I am authorized to view raw customer identifiers"):
    detail = master_f.copy()
    detail["customer_id"] = detail["customer_id"].astype(str)
    with st.expander("Customer detail (top 200)", expanded=False):
        st.dataframe(detail.head(200), use_container_width=True)
    csv_download_button(
        detail, "executive_customer_detail.csv", "Download detail (CSV)", key="dl_exec_detail"
    )
else:
    masked = master_f.copy()
    masked["customer_id"] = masked["customer_id"].map(mask_customer_id)
    with st.expander("Customer overview, IDs masked (top 200)", expanded=False):
        st.dataframe(masked.head(200), use_container_width=True)

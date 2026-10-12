"""Page 09 — Executive dashboard: filter-consistent KPIs, trends, risk, actions."""

import pandas as pd
import streamlit as st

from analytics.demographics import GEO_LEVELS
from components.export_buttons import csv_download_button
from components.filters import (
    control_group,
    controls_expander,
    date_range_filter,
    multiselect_filter,
)
from components.metric_cards import kpi_row
from components.nav_tabs import render_nav_tabs
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
from utils.logging_config import configure_logging
from visualization.churn_charts import risk_band_bar
from visualization.demographic_charts import age_band_bar, geo_treemap, top_geo_bar
from visualization.executive_dashboard import (
    activity_trend_fig,
    new_returning_fig,
    portfolio_bar_fig,
    revenue_trend_fig,
)
from visualization.retention_charts import revenue_at_risk_bar, strategy_distribution_bar
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


st.title("09 · Executive Dashboard")
st.markdown(
    "Every card below derives from the same sidebar filters. "
    "Customer IDs are masked except in the gated drill-down."
)

clean_data = st.session_state.get("clean_data")
if clean_data is None:
    st.info("Complete 02 · Data Quality & Cleaning first — the dashboard needs clean data.")
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
    st.warning("No customers match the current filters. Widen a filter to continue.")
    st.stop()
kpis = compute_kpis(txns_f, master_f)

st.subheader("Key performance indicators")
kpi_row(
    [
        ("Customers", f"{kpis['total_customers']:,}"),
        ("Revenue", f"₹{kpis['total_revenue']:,.0f}"),
        ("Avg order value", f"₹{kpis['average_order_value']:,.0f}"),
        ("Active rate", f"{kpis['active_customer_rate']:.1%}"),
    ]
)
second = [
    (
        "Predicted churn",
        (
            f"{kpis['predicted_churn_rate']:.1%}"
            if kpis["predicted_churn_rate"] is not None
            else "needs p07"
        ),
    ),
    (
        "Revenue at risk",
        f"₹{kpis['revenue_at_risk']:,.0f}" if kpis["revenue_at_risk"] is not None else "needs p08",
    ),
    (
        "Priority opportunities",
        (
            str(kpis["priority_opportunities"])
            if kpis["priority_opportunities"] is not None
            else "needs p08"
        ),
    ),
]
kpi_row(second)

st.subheader("Trends")
trends = trend_tables(txns_f, master_f)
trend_a, trend_b, trend_c = st.columns([1, 1, 1])
with trend_a:
    st.plotly_chart(revenue_trend_fig(trends["revenue"]), use_container_width=True)
with trend_b:
    st.plotly_chart(activity_trend_fig(trends["activity"]), use_container_width=True)
with trend_c:
    st.plotly_chart(new_returning_fig(trends["cohort"]), use_container_width=True)
port_l, port_r = st.columns([3, 2])
with port_l:
    if "rfm_segment" in master_f.columns:
        portfolio = segment_portfolio(master_f, "rfm_segment")
        st.plotly_chart(portfolio_bar_fig(portfolio, "rfm_segment"), use_container_width=True)
    else:
        st.info("RFM segments appear here after page 03 scoring.")
with port_r:
    if "age_band" in master_f.columns:
        bands = (
            master_f["age_band"]
            .value_counts()
            .rename_axis("age_band")
            .reset_index(name="customers")
        )
        st.plotly_chart(age_band_bar(bands), use_container_width=True)

with st.expander("Segment portfolio table", expanded=False):
    if "rfm_segment" in master_f.columns:
        st.dataframe(segment_portfolio(master_f, "rfm_segment"), use_container_width=True)

st.subheader("Risk")
if "risk_band" in master_f.columns:
    risk_l, risk_r = st.columns([1, 1])
    with risk_l:
        st.plotly_chart(risk_band_bar(master_f), use_container_width=True)
    with risk_r:
        hvhr = high_value_high_risk(master_f)
        if not hvhr.empty:
            with st.expander("High-value, high-risk (IDs masked)", expanded=False):
                st.dataframe(hvhr, use_container_width=True)
else:
    st.info("Churn risk appears here after page 07 scoring.")

st.subheader("Demographics & geography")
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
    geo_l, geo_r = st.columns([3, 2])
    with geo_l:
        st.plotly_chart(top_geo_bar(geo, level), use_container_width=True)
    with geo_r:
        st.plotly_chart(geo_treemap(geo.head(30), level), use_container_width=True)

st.subheader("Retention")
retention = _retention_frame()
if retention is not None and not retention.empty:
    ret_f = retention[retention["customer_id"].isin(set(master_f["customer_id"]))]
    ret_l, ret_r = st.columns([1, 1])
    with ret_l:
        st.plotly_chart(strategy_distribution_bar(ret_f), use_container_width=True)
    with ret_r:
        st.plotly_chart(revenue_at_risk_bar(ret_f), use_container_width=True)
    st.metric("Campaign-ready customers", f"{len(ret_f):,}")
    csv_download_button(
        ret_f,
        "executive_retention_export.csv",
        "Download retention export (CSV)",
        key="dl_exec_ret",
    )
else:
    st.info("Retention actions appear here after page 08.")

st.subheader("Authorized drill-down")
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

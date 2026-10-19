"""Page 03 — Customer features, RFM/LRFM scoring, segments, and downloads."""

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from components.cached import cached_run_segment_trends_with_labels, cached_segment_profile
from components.empty_state import chart_section, empty_state
from components.metric_card import Metric, metric_row
from components.page_header import page_header, section_header
from components.segment_trends import render_trend_pair, transition_frame
from components.sidebar import render_sidebar_footer, render_sidebar_nav, sidebar_filters
from components.table import render_table
from services import state_service
from services.export_service import to_csv_bytes
from services.feature_service import (
    compute_customer_features,
    run_label_history,
    run_lrfm,
    run_rfm,
)
from services.segmentation_service import labeling_series
from utils import formatting
from utils.exceptions import CustomerIntelligenceError
from utils.logging_config import configure_logging
from visualization.rfm_charts import (
    revenue_by_segment_bar,
    rfm_parallel_plot,
    rfm_scatter_3d,
    rfm_scatter_matrix,
    segment_distribution_bar,
    segment_trend_facets,
    segment_trend_heatmap,
    segment_trend_lines,
    segment_trend_ribbon,
    segment_trend_share,
)
from visualization.theme import apply_chart_theme, inject_terminal_css, without_title


def _compact(fig):
    """Reflow a factory figure to the compact tab height, without its title.

    The factory's own height is preserved, so the taller small-multiples grid
    keeps its rows readable while every single-chart view stays at the uniform
    compact height.
    """
    return without_title(
        apply_chart_theme(fig, height=fig.layout.height, compact=True), compact=True
    )


#: Trend views in reading order; the first one stays the default. The default
#: is the line view because the transition Sankey beside it stays readable only
#: when each segment's own path is visible.
TREND_VIEWS = {
    "Segment lines": segment_trend_lines,
    "Stacked area": segment_trend_ribbon,
    "Share of base": segment_trend_share,
    "Small multiples": segment_trend_facets,
    "Heatmap": segment_trend_heatmap,
}

TREND_SOURCE_HELP = (
    "Which segmentation to trend. RFM/LRFM are re-scored at every period end; an AI "
    "run keeps the segment each customer holds today, counted among the base that "
    "existed by each period end."
)
SCORED_CAPTION = (
    "Segments are quantile-relative within each period, so their definitions drift as "
    "the population changes."
)
AI_CAPTION = (
    "AI segments use the saved run's current assignment — labels are fixed, not "
    "re-scored per period — so the mix moves with who arrived."
)


def _saved_labelings() -> dict[str, pd.Series]:
    """Saved AI runs as ``AI:<model>`` → customer-level business-name series."""
    runs = [r for r in (st.session_state.get("segmentation_runs") or []) if "labels" in r]
    return {f"AI:{run['model']}": labeling_series(run) for run in runs}


def _trend_history(
    history: pd.DataFrame,
    source: str,
    labelings: dict[str, pd.Series],
    clean: pd.DataFrame,
    granularity: str,
) -> pd.DataFrame:
    """Period rows for one source: stored RFM/LRFM history or a fixed AI labeling."""
    if source in labelings:
        return run_label_history(clean, {source: labelings[source]}, granularity)
    return history[history["model"] == source]


configure_logging()
state_service.initialize(st.session_state)
inject_terminal_css()
render_sidebar_nav("pages/03_RFM_LRFM_Analysis.py")

page_header(
    "RFM & LRFM Analysis",
    "Customer-level features feed quantile-based RFM and LRFM scoring. "
    "Segments are named from score components and raw values — never from the "
    "combined score alone.",
)

clean_data = st.session_state.get("clean_data")
if clean_data is None:
    empty_state(
        "No clean data yet",
        "Complete Data Quality & Cleaning first (apply cleaning to get clean data).",
    )
    st.stop()

cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

max_txn = pd.to_datetime(clean_data["transaction_date"]).max().date()
with sidebar_filters("Scoring"):
    quantiles = st.selectbox(
        "Quantile levels",
        [3, 4, 5],
        index=[3, 4, 5].index(cfg["rfm"]["quantiles"]),
        help="Higher levels give finer segments; 3 is safest for small datasets.",
    )
    snapshot = st.date_input(
        "Snapshot date",
        value=max_txn + pd.Timedelta(days=cfg["rfm"]["snapshot_offset_days"]),
        min_value=max_txn,
        help="Defaults to one day after the last transaction.",
    )
    recent_days = st.number_input("Recent window (days)", min_value=1, value=90, step=7)
    trend_granularity = st.selectbox(
        "Trend granularity",
        ["Quarterly (last 8)", "Monthly (last 12)"],
        help="Segment history is scored per period end; monthly takes longer.",
    )
    build_clicked = st.button("Build features & score", type="primary")
    trend_clicked = st.button("Build segment trends")

if build_clicked:
    try:
        with st.status("Building features & scoring RFM/LRFM...", expanded=True) as status:
            status.update(state="running", label="Computing customer features…")
            result = compute_customer_features(
                clean_data,
                snapshot_date=pd.Timestamp(snapshot),
                recent_days=int(recent_days),
                snapshot_offset_days=cfg["rfm"]["snapshot_offset_days"],
            )
            status.update(state="running", label="Scoring RFM and LRFM…")
            st.session_state["customer_features"] = result.features
            st.session_state["rfm_results"] = run_rfm(result.features, quantiles=quantiles)
            st.session_state["lrfm_results"] = run_lrfm(result.features, quantiles=quantiles)
            status.update(state="complete", label="Scoring complete", expanded=False)
        if result.demographic_conflicts:
            st.warning(
                f"{len(result.demographic_conflicts)} customers have conflicting "
                "demographic values across transactions; the latest non-null value "
                "was used. Conflicts are listed in the drill-down table notes."
            )
        if result.dropped_invalid_dates:
            st.warning(
                f"{result.dropped_invalid_dates} rows had no valid transaction date "
                "and were excluded from features."
            )
        st.success(
            f"Scored {len(result.features)} customers "
            f"into {st.session_state['rfm_results']['segment'].nunique()} RFM segments."
        )
    except (CustomerIntelligenceError, KeyError, ValueError) as exc:
        st.error(str(exc))

if trend_clicked:
    try:
        granularity = "monthly" if trend_granularity.startswith("Monthly") else "quarterly"
        with st.status(f"Scoring segments per {granularity} period…", expanded=True) as status:
            status.update(state="running", label="Computing period snapshots…")
            history, transitions, labels = cached_run_segment_trends_with_labels(
                clean_data, granularity, int(quantiles)
            )
            status.update(state="complete", label="Segment trends built", expanded=False)
        st.session_state["segment_history"] = {
            "table": history.to_dict(orient="records"),
            "transitions": transitions.to_dict(orient="records"),
            "granularity": granularity,
            "quantiles": int(quantiles),
        }
        st.session_state["segment_history_labels"] = labels
        st.success(
            f"Scored {history['period'].nunique()} {granularity} periods "
            "into RFM and LRFM segments."
        )
    except (CustomerIntelligenceError, KeyError, ValueError) as exc:
        st.error(str(exc))

features = st.session_state.get("customer_features")
if features is None:
    empty_state("No features yet", "Configure and run scoring to see segments.")
    st.stop()

section_header("Customer base")
metric_row(
    [
        Metric("Customers", formatting.fmt_int(len(features))),
        Metric("Total revenue", formatting.fmt_money(features["monetary_value"].sum())),
        Metric(
            "Average order value",
            formatting.fmt_money(features["monetary_value"].sum() / features["frequency"].sum()),
        ),
        Metric("Avg recency (days)", f"{features['recency_days'].mean():,.0f}"),
    ]
)

with st.expander("Customer feature preview (top 20)", expanded=False):
    st.dataframe(features.head(20), width="stretch")

tab_rfm, tab_lrfm = st.tabs(["RFM", "LRFM"])
model_labels = {"rfm_results": "RFM", "lrfm_results": "LRFM"}
for tab, key in ((tab_rfm, "rfm_results"), (tab_lrfm, "lrfm_results")):
    scored = st.session_state.get(key)
    if scored is None:
        continue
    with tab:
        enriched = features.merge(scored, on="customer_id", suffixes=("", "_seg"))
        profile = cached_segment_profile(enriched, monetary_col="monetary_value")
        dist_tab, trends_tab, rel_tab = st.tabs(["Distribution", "Trends", "Relationships"])
        with dist_tab:
            seg_l, seg_r = st.columns([1, 1])
            with seg_l:
                with chart_section("Customers per segment"):
                    st.plotly_chart(
                        _compact(segment_distribution_bar(profile)),
                        width="stretch",
                        key=f"{key}_seg_dist",
                    )
            with seg_r:
                with chart_section("Revenue per segment"):
                    st.plotly_chart(
                        _compact(revenue_by_segment_bar(profile)),
                        width="stretch",
                        key=f"{key}_revenue",
                    )
        with trends_tab:
            with chart_section("Customer segments over time"):
                hist_state = st.session_state.get("segment_history") or {}
                rows = hist_state.get("table") if isinstance(hist_state, dict) else None
                history = pd.DataFrame(rows) if rows else None
                if history is None or history.empty:
                    empty_state(
                        "No segment history yet",
                        "Build segment trends from the sidebar.",
                    )
                else:
                    own = model_labels[key]
                    labelings = _saved_labelings()
                    sources = [
                        own,
                        *[m for m in model_labels.values() if m != own],
                        *labelings,
                    ]
                    source = st.selectbox(
                        "Segment source",
                        sources,
                        key=f"{key}_trendsource",
                        help=TREND_SOURCE_HELP,
                    )
                    model_hist = _trend_history(
                        history,
                        source,
                        labelings,
                        clean_data,
                        hist_state.get("granularity", "quarterly"),
                    )
                    if model_hist.empty:
                        empty_state(
                            "No trend for this source",
                            "That run has no customers in the current trend periods.",
                        )
                    else:
                        trend_view = st.radio(
                            "Trend view",
                            list(TREND_VIEWS),
                            horizontal=True,
                            key=f"{key}_trendview",
                        )
                        flows_model = source if source in ("RFM", "LRFM") else None
                        render_trend_pair(
                            _compact(TREND_VIEWS[trend_view](model_hist)),
                            transition_frame(hist_state),
                            model=flows_model,
                            key_prefix=f"{key}_trendpair",
                            empty_message=(
                                "Saved AI labels are fixed, so there are no re-scored "
                                "transitions to draw. Choose RFM or LRFM for the Sankey."
                                if flows_model is None
                                else "No customers were present at both ends of a "
                                "period pair in this history."
                            ),
                        )
                        st.caption(AI_CAPTION if source in labelings else SCORED_CAPTION)
        with rel_tab:
            rel_view = st.selectbox(
                "Relationships view",
                ["Scatter matrix", "3D scatter", "Parallel coordinates"],
                key=f"{key}_relview",
            )
            with chart_section(rel_view):
                if rel_view == "Scatter matrix":
                    st.plotly_chart(
                        rfm_scatter_matrix(enriched),
                        width="stretch",
                        key=f"{key}_relmatrix",
                    )
                    if len(enriched) > 2000:
                        st.caption("Scatter matrix shows a 2,000-customer sample.")
                elif rel_view == "3D scatter":
                    st.plotly_chart(
                        rfm_scatter_3d(enriched),
                        width="stretch",
                        key=f"{key}_rel3d",
                    )
                    if len(enriched) > 2000:
                        st.caption("3D scatter shows a 2,000-customer sample.")
                else:
                    st.plotly_chart(
                        _compact(rfm_parallel_plot(enriched)),
                        width="stretch",
                        key=f"{key}_parcoords",
                    )

        with st.expander("Segment profile", expanded=False):
            render_table(profile, money_cols=("revenue",))

        with st.expander("Customer drill-down", expanded=False):
            choice = st.selectbox(
                "Segment",
                sorted(scored["segment"].unique()),
                key=f"drill_{key}",
                help="Customer identifiers appear only in this authorized detail view.",
            )
            drill = enriched[enriched["segment"] == choice].head(200)
            st.dataframe(drill, width="stretch", key=f"{key}_drill")
            st.caption(
                f"Showing up to 200 of {(scored['segment'] == choice).sum()} customers "
                f"in '{choice}'."
            )

with sidebar_filters("Exports"):
    for tab_key, tab_label in (("rfm_results", "RFM"), ("lrfm_results", "LRFM")):
        scored_state = st.session_state.get(tab_key)
        if scored_state is not None:
            st.download_button(
                f"Download {tab_label} (CSV)",
                data=to_csv_bytes(scored_state),
                file_name=f"{tab_key}.csv",
                mime="text/csv",
                key=f"dl_{tab_key}",
            )
    st.download_button(
        "Download customer features (CSV)",
        data=to_csv_bytes(features),
        file_name="customer_features.csv",
        mime="text/csv",
        key="dl_features",
    )

render_sidebar_footer()

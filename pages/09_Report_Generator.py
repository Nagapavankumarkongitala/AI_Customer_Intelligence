"""Page 09 — Report Generator: a systematic, narrative report with HTML/Excel/PDF/Word exports.

The page mirrors the executive dashboard one tab at a time — business and
customer trends, segments and demographics, churn risk, and the retention plan
— then adds the two layers a dashboard cannot carry on a printed page: an
executive summary of what the whole dataset says, and prioritized
recommendations with the evidence behind each. The appendix holds the data
foundation (source profile and cleaning treatments) and the method notes.
"""

import pandas as pd
import streamlit as st

from components.empty_state import chart_section, empty_state
from components.insight_box import insight_box
from components.metric_card import Metric, metric_row
from components.page_header import page_header, section_header
from components.sidebar import render_sidebar_footer, render_sidebar_nav
from components.table import render_table
from services import report_narrative, state_service
from services.dashboard_service import mask_customer_id
from services.report_charts import render_report_charts
from services.report_service import (
    ANALYTICAL_SECTIONS,
    CHART_EXPLANATIONS,
    REPORT_SECTIONS,
    SECTION_LABELS,
    available_chart_engine,
    available_docx_engine,
    available_pdf_engine,
    build_docx_bytes,
    build_pdf_bytes,
    build_report_context,
    context_tables,
    context_to_excel_bytes,
    render_html_report,
    report_readiness_lines,
)
from utils import formatting
from utils.exceptions import CustomerIntelligenceError
from utils.logging_config import configure_logging
from visualization.churn_charts import risk_band_bar
from visualization.demographic_charts import age_band_bar, gender_bar, top_geo_bar
from visualization.executive_dashboard import (
    activity_trend_fig,
    new_returning_fig,
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
    segment_distribution_bar,
    segment_trend_ribbon,
)
from visualization.theme import inject_terminal_css, without_title

configure_logging()
state_service.initialize(st.session_state)
inject_terminal_css()
render_sidebar_nav("pages/09_Report_Generator.py")

page_header(
    "Report Generator",
    "A systematic report over every dashboard tab — key insights, interpretation, "
    "and prioritized next steps — downloadable as HTML, Excel, PDF, and Word, with "
    "the charts embedded in the PDF and Word files.",
)

clean_data = st.session_state.get("clean_data")
if clean_data is None:
    empty_state(
        "No clean data yet",
        "Complete Data Quality & Cleaning first, then score RFM before reporting.",
    )
    st.stop()

context = build_report_context(st.session_state)

with st.expander("Readiness", expanded=False):
    for label, ready, hint in report_readiness_lines(context):
        if ready:
            st.success(f"{label} — ready.")
        else:
            st.warning(f"{label} — pending. {hint}")
if not context["readiness"].get("features"):
    empty_state(
        "No features yet",
        "Run Build features & score on RFM & LRFM Analysis to unlock the report.",
    )
    st.stop()

with st.expander("Report options", expanded=True):
    selected = [
        section
        for section in REPORT_SECTIONS
        if st.checkbox(SECTION_LABELS[section], value=True, key=f"rep_{section}")
    ] or list(REPORT_SECTIONS)
    reveal_ids = st.checkbox(
        "I am authorized to show raw customer identifiers in previews",
        value=False,
        key="rep_reveal",
    )
    top_n = st.number_input("Top rows per table", min_value=20, max_value=1000, value=200, step=20)
    report_theme = st.selectbox(
        "HTML theme",
        ["Print light", "Dark console"],
        help="Print light suits paper/PDF hand-offs; dark console mirrors the app.",
    )
    html_mode = "dark" if report_theme == "Dark console" else "light"

kpis = context["kpis"]
features = context["features"]
portfolio = context["segment_portfolio"]
scores = context["scores"]
retention = context["retention"]
trends = context.get("trends") or {}

figures: dict[str, object] = {}
figures_html: dict[str, str] = {}


def _maybe_mask(frame: pd.DataFrame) -> pd.DataFrame:
    """Mask IDs in previews unless the gated checkbox is ticked."""
    if reveal_ids or "customer_id" not in frame.columns:
        return frame
    out = frame.copy()
    out["customer_id"] = out["customer_id"].map(mask_customer_id)
    return out


def _register(key: str, fig) -> object:
    """Keep the live figure and its standalone HTML snippet together."""
    figures[key] = fig
    try:
        figures_html[key] = fig.to_html(full_html=False, include_plotlyjs="cdn")
    except (AttributeError, ValueError):
        figures_html[key] = ""
    return fig


def _insights(section: str) -> None:
    """Render the narrative interpretation attached to a section."""
    for item in report_narrative.section_insights(context, section):
        insight_box(item.title, item.detail, tone=item.tone)


def _explain(key: str) -> None:
    """Render the plain-English reading guide for one embedded chart."""
    exp = CHART_EXPLANATIONS.get(key)
    if exp:
        insight_box(exp["what"], f"{exp['how_to_read']} {exp.get('action', '')}")


def _render_summary() -> None:
    section_header(
        "1 · Executive summary",
        "The whole dataset in one screen: scale, exposure, and momentum.",
    )
    metric_row(
        [
            Metric("Customers", formatting.fmt_int(kpis.get("total_customers", 0))),
            Metric("Revenue", formatting.fmt_money(kpis.get("total_revenue", 0.0))),
            Metric("Avg order value", formatting.fmt_money(kpis.get("average_order_value", 0.0))),
            Metric(
                "Predicted churn",
                (
                    formatting.fmt_pct(kpis["predicted_churn_rate"])
                    if kpis.get("predicted_churn_rate") is not None
                    else "Needs page 07"
                ),
                tone="negative" if kpis.get("predicted_churn_rate") is not None else "neutral",
            ),
            Metric(
                "Revenue at risk",
                (
                    formatting.fmt_money(kpis["revenue_at_risk"])
                    if kpis.get("revenue_at_risk") is not None
                    else "Needs page 08"
                ),
                tone="negative" if kpis.get("revenue_at_risk") is not None else "neutral",
            ),
        ]
    )
    for item in report_narrative.summary_insights(context):
        insight_box(item.title, item.detail, tone=item.tone)


def _render_overview() -> None:
    section_header(
        "2 · Business & customer trends",
        "How the base performed and moved over the transaction window.",
    )
    revenue = trends.get("revenue")
    if revenue is None or revenue.empty:
        empty_state("No trend data", "Trends need dated transactions and customer features.")
        return
    activity = trends.get("activity")
    cohort = trends.get("cohort")
    metric_row(
        [
            Metric("Transactions", formatting.fmt_int(context["n_transactions"])),
            Metric("Active rate", formatting.fmt_pct(kpis.get("active_customer_rate"))),
            Metric(
                "New customers",
                (
                    formatting.fmt_int(cohort["new_customers"].sum())
                    if isinstance(cohort, pd.DataFrame) and not cohort.empty
                    else "–"
                ),
            ),
            Metric(
                "Returning customers",
                (
                    formatting.fmt_int(cohort["returning_customers"].sum())
                    if isinstance(cohort, pd.DataFrame) and not cohort.empty
                    else "–"
                ),
            ),
        ]
    )
    with chart_section("Revenue trend"):
        st.plotly_chart(
            _register("revenue_trend", without_title(revenue_area_fig(revenue))),
            width="stretch",
            key="rep_revenue_trend",
        )
    _explain("revenue_trend")
    left, right = st.columns(2)
    with left:
        with chart_section("Customer activity trend"):
            if isinstance(activity, pd.DataFrame) and not activity.empty:
                st.plotly_chart(
                    _register("activity_trend", without_title(activity_trend_fig(activity))),
                    width="stretch",
                    key="rep_activity",
                )
            else:
                empty_state("No activity data", "Activity needs dated transactions.")
        _explain("activity_trend")
    with right:
        with chart_section("New vs returning customers"):
            if isinstance(cohort, pd.DataFrame) and not cohort.empty:
                st.plotly_chart(
                    _register("new_returning", without_title(new_returning_fig(cohort))),
                    width="stretch",
                    key="rep_cohort",
                )
            else:
                empty_state("No cohort data", "Cohorts need first-transaction dates.")
        _explain("new_returning")
    _insights("overview")
    with st.expander("Monthly revenue table", expanded=False):
        render_table(revenue, money_cols=("revenue",))


def _render_segment_charts() -> None:
    left, right = st.columns(2)
    with left:
        with chart_section("Customers per segment"):
            st.plotly_chart(
                _register(
                    "segment_distribution",
                    without_title(segment_distribution_bar(portfolio, "rfm_segment")),
                ),
                width="stretch",
                key="rep_seg_dist",
            )
        _explain("segment_distribution")
    with right:
        with chart_section("Revenue per segment"):
            st.plotly_chart(
                _register(
                    "revenue_by_segment",
                    without_title(revenue_by_segment_bar(portfolio, "rfm_segment")),
                ),
                width="stretch",
                key="rep_seg_rev",
            )
        _explain("revenue_by_segment")


def _render_segment_trends() -> None:
    history = context.get("history")
    if history is None or history.empty or "model" not in history.columns:
        return
    rfm_hist = history[history["model"] == "RFM"]
    if rfm_hist.empty:
        return
    with chart_section("Segment trends over time"):
        st.plotly_chart(
            _register("segment_trends", without_title(segment_trend_ribbon(rfm_hist))),
            width="stretch",
            key="rep_seg_trends",
        )
    st.caption(
        "Segments are quantile-relative per period; definitions drift as the population changes."
    )
    _explain("segment_trends")


def _render_demographics() -> None:
    demo = context.get("demographics") or {}
    age_bands, gender, region = demo.get("age_bands"), demo.get("gender"), demo.get("region")
    if age_bands is None and gender is None and region is None:
        return
    left, middle, right = st.columns(3)
    with left:
        with chart_section("Customers per age band"):
            if isinstance(age_bands, pd.DataFrame) and not age_bands.empty:
                st.plotly_chart(
                    _register("demographics_age", without_title(age_band_bar(age_bands))),
                    width="stretch",
                    key="rep_demo_age",
                )
            else:
                empty_state("No age data", "Age appears after profiling.")
        _explain("demographics_age")
    with middle:
        with chart_section("Customers by gender"):
            if isinstance(gender, pd.DataFrame) and not gender.empty:
                st.plotly_chart(
                    _register("demographics_gender", without_title(gender_bar(gender))),
                    width="stretch",
                    key="rep_demo_gender",
                )
            else:
                empty_state("No gender data", "Gender appears after profiling.")
        _explain("demographics_gender")
    with right:
        with chart_section("Top regions by customers"):
            if isinstance(region, pd.DataFrame) and not region.empty:
                st.plotly_chart(
                    _register("demographics_region", without_title(top_geo_bar(region, "region"))),
                    width="stretch",
                    key="rep_demo_region",
                )
            else:
                empty_state("No region data", "Region appears after profiling.")
        _explain("demographics_region")


def _render_segments() -> None:
    section_header(
        "3 · Segments & demographics",
        "Who sits where, what each cohort is worth, and how the base is composed.",
    )
    if portfolio is None or portfolio.empty:
        empty_state("No RFM segments yet", "Score RFM on RFM & LRFM Analysis first.")
        return
    top_seg = portfolio.loc[portfolio["revenue_share"].idxmax()]
    ai_portfolio = context.get("ai_portfolio")
    metric_row(
        [
            Metric("Segments", str(len(portfolio))),
            Metric(
                "Largest segment",
                f"{top_seg['rfm_segment']} ({formatting.fmt_int(top_seg['customers'])})",
            ),
            Metric("Top revenue share", f"{top_seg['revenue_share']:.1f}%"),
            Metric(
                "AI clusters",
                str(len(ai_portfolio)) if ai_portfolio is not None else "Not run",
            ),
        ]
    )
    _render_segment_charts()
    _render_segment_trends()
    _render_demographics()
    _insights("segments")
    with st.expander("Segment portfolio table", expanded=False):
        render_table(portfolio, money_cols=("revenue",))
    if ai_portfolio is not None and not ai_portfolio.empty:
        with st.expander("AI segment portfolio table", expanded=False):
            render_table(ai_portfolio, money_cols=("revenue",))


def _render_churn() -> None:
    section_header("4 · Churn risk", "Where the modeled at-risk pool concentrates.")
    if scores is None or scores.empty:
        empty_state("No churn scores yet", "Train and score on Churn Prediction first.")
        return
    bands = scores["risk_band"] if "risk_band" in scores.columns else None
    metric_row(
        [
            Metric(
                "Predicted churn",
                (
                    formatting.fmt_pct(kpis["predicted_churn_rate"])
                    if kpis.get("predicted_churn_rate") is not None
                    else "Needs page 07"
                ),
                tone="negative" if kpis.get("predicted_churn_rate") is not None else "neutral",
            ),
            Metric(
                "High-risk customers",
                formatting.fmt_int((bands == "high").sum()) if bands is not None else "–",
            ),
            Metric(
                "Critical customers",
                formatting.fmt_int((bands == "critical").sum()) if bands is not None else "–",
            ),
        ]
    )
    with chart_section("Customers per risk band"):
        st.plotly_chart(
            _register("risk_bands", without_title(risk_band_bar(scores))),
            width="stretch",
            key="rep_risk",
        )
    _explain("risk_bands")
    _insights("churn")
    with st.expander("Top-risk customers (IDs masked unless authorized)", expanded=False):
        top_risk = scores.sort_values("churn_probability", ascending=False).head(int(top_n))
        render_table(_maybe_mask(top_risk), risk_cols=("risk_band",))
    hvhr = context.get("hvhr")
    if hvhr is not None and not hvhr.empty:
        with st.expander("High-value, high-risk accounts", expanded=False):
            render_table(hvhr, risk_cols=("risk_band",), money_cols=("revenue_at_risk",))


def _render_retention() -> None:
    section_header("5 · Retention plan", "Ranked, traceable actions — no promises.")
    if retention is None or retention.empty:
        empty_state("No retention actions yet", "Build the table on Retention Strategies first.")
        return
    summary = context.get("retention_summary") or {}
    metric_row(
        [
            Metric(
                "Priority opportunities",
                (
                    formatting.fmt_int(kpis["priority_opportunities"])
                    if kpis.get("priority_opportunities") is not None
                    else "Needs page 08"
                ),
            ),
            Metric("Campaign-ready customers", formatting.fmt_int(len(retention))),
            Metric(
                "Revenue at risk",
                (
                    formatting.fmt_money(summary.get("total_revenue_at_risk"))
                    if summary.get("total_revenue_at_risk") is not None
                    else "–"
                ),
            ),
        ]
    )
    with chart_section("Value–risk matrix"):
        st.plotly_chart(
            _register("value_risk_matrix", without_title(value_risk_matrix(retention))),
            width="stretch",
            key="rep_matrix",
        )
    _explain("value_risk_matrix")
    left, right = st.columns(2)
    with left:
        with chart_section("Revenue at risk per action"):
            st.plotly_chart(
                _register("revenue_at_risk", without_title(revenue_at_risk_bar(retention))),
                width="stretch",
                key="rep_rar",
            )
        _explain("revenue_at_risk")
    with right:
        with chart_section("Customers per recommended action"):
            st.plotly_chart(
                _register(
                    "strategy_distribution", without_title(strategy_distribution_bar(retention))
                ),
                width="stretch",
                key="rep_strat",
            )
        _explain("strategy_distribution")
    with chart_section("Priority score distribution"):
        st.plotly_chart(
            _register("priority_histogram", without_title(priority_histogram(retention))),
            width="stretch",
            key="rep_priority",
        )
    _explain("priority_histogram")
    _insights("retention")
    with st.expander("Ranked retention actions (top rows)", expanded=False):
        render_table(
            _maybe_mask(retention.head(int(top_n))),
            risk_cols=("risk_band",),
            money_cols=("revenue_at_risk", "expected_value_proxy_historical"),
        )
    st.caption(
        "Historical value is a proxy, never CLV. Every row names its strategy rule; "
        "reasons never claim causal impact."
    )


def _render_recommendations() -> None:
    section_header(
        "7 · Recommendations & next steps",
        "Interpretations recapped, then prioritized actions tied to the evidence.",
    )
    recap = [
        item
        for section in ANALYTICAL_SECTIONS
        if section in selected
        for item in report_narrative.section_insights(context, section)
    ]
    if recap:
        st.markdown("**Interpretations at a glance**")
        for item in recap:
            insight_box(item.title, item.detail, tone=item.tone)
    recs = report_narrative.recommendations(context)
    if not recs:
        empty_state(
            "No recommendations yet",
            "Recommendations appear as retention results arrive.",
        )
    else:
        st.markdown("**Prioritized actions**")
        for priority in report_narrative.PRIORITIES:
            group = [rec for rec in recs if rec.priority == priority]
            if not group:
                continue
            st.markdown(f"**{priority}**")
            for rec in group:
                insight_box(rec.title, f"{rec.rationale}  ·  {rec.evidence}", tone=rec.tone)
    st.caption(
        "Priority order is decision support over the computed results; it does not claim "
        "causal impact."
    )


def _render_appendix() -> None:
    section_header(
        "6 · Appendix & data foundation",
        "Source profile, coverage, cleaning treatments, and method notes.",
    )
    profile = context.get("data_profile") or {}
    metric_row(
        [
            Metric("Source rows", formatting.fmt_int(profile.get("rows_in") or 0)),
            Metric("Clean rows", formatting.fmt_int(profile.get("rows", 0))),
            Metric("Rows removed", formatting.fmt_int(profile.get("rows_removed", 0))),
            Metric("Missing cells", f"{float(profile.get('missing_pct', 0)):.2f}%"),
        ]
    )
    span = profile.get("date_span")
    st.caption(
        f"Source: {profile.get('source_filename', '–')}"
        + (f" · window {span[0]} → {span[1]}" if span else "")
    )
    _insights("appendix")
    treatments = context.get("treatments") or []
    if treatments:
        with st.expander("Cleaning treatments applied", expanded=False):
            render_table(pd.DataFrame(treatments))
    dictionary = context.get("data_dictionary") or []
    if dictionary:
        with st.expander("Data source profile (per field)", expanded=False):
            render_table(pd.DataFrame(dictionary))
    with st.expander("Report tables (as exported)", expanded=False):
        for name, frame in context_tables(context, mask_ids=not reveal_ids).items():
            st.markdown(f"**{name}**")
            render_table(frame.head(int(top_n)))
    st.caption(
        "Churn is leakage-safe by construction (predictors ≤ cutoff, label from the outcome "
        "window; cohort-ordered splits). RFM = 100R+10F+M; cluster IDs are arbitrary until "
        "profiled against population medians."
    )


if "summary" in selected:
    _render_summary()
if "overview" in selected:
    _render_overview()
if "segments" in selected:
    _render_segments()
if "churn" in selected:
    _render_churn()
if "retention" in selected:
    _render_retention()
if "appendix" in selected:
    _render_appendix()
if "recommendations" in selected:
    _render_recommendations()

section_header("Downloads", "One pack, four formats. IDs masked unless authorized.")
mask_ids = not reveal_ids
pdf_guidance = available_pdf_engine()
word_guidance = available_docx_engine()
chart_guidance = available_chart_engine()
export_available = not pdf_guidance or not word_guidance
chart_images = render_report_charts(context) if export_available else {}
if export_available and not chart_guidance:
    st.caption(
        "PDF and Word embed the report charts as images; every figure keeps its "
        "how-to-read interpretation."
    )
html_bytes = render_html_report(
    context,
    figures_html=figures_html,
    sections=tuple(selected),
    mask_ids=mask_ids,
    mode=html_mode,
).encode("utf-8")
st.download_button(
    "Download report (HTML)",
    data=html_bytes,
    file_name="customer_intelligence_report.html",
    mime="text/html",
    key="dl_rep_html",
)
try:
    excel_bytes = context_to_excel_bytes(context, mask_ids=mask_ids, top_n=int(top_n))
except CustomerIntelligenceError as exc:
    st.error(str(exc))
else:
    st.download_button(
        "Download tables (Excel)",
        data=excel_bytes,
        file_name="customer_intelligence_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_rep_excel",
    )
if chart_guidance and export_available:
    st.info(f"{chart_guidance} PDF and Word will list the figures without images.")
if pdf_guidance:
    st.info(f"{pdf_guidance} HTML, Excel, and Word remain available.")
else:
    try:
        pdf_bytes = build_pdf_bytes(
            context,
            mask_ids=mask_ids,
            sections=tuple(selected),
            chart_images=chart_images,
        )
    except CustomerIntelligenceError as exc:
        st.error(str(exc))
    else:
        st.download_button(
            "Download report (PDF)",
            data=pdf_bytes,
            file_name="customer_intelligence_report.pdf",
            mime="application/pdf",
            key="dl_rep_pdf",
        )
if word_guidance:
    st.info(f"{word_guidance} HTML, Excel, and PDF remain available.")
else:
    try:
        word_bytes = build_docx_bytes(
            context,
            mask_ids=mask_ids,
            sections=tuple(selected),
            chart_images=chart_images,
        )
    except CustomerIntelligenceError as exc:
        st.error(str(exc))
    else:
        st.download_button(
            "Download report (Word)",
            data=word_bytes,
            file_name="customer_intelligence_report.docx",
            mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            key="dl_rep_word",
        )

render_sidebar_footer()

"""Page 04 — Demographic and geographic profiling with segment comparisons."""

import pandas as pd
import streamlit as st

from analytics.demographics import GEO_LEVELS
from components.cached import (
    cached_build_demographic_profile,
    cached_run_segment_tests,
    cached_segment_composition,
)
from components.empty_state import chart_section, empty_state
from components.metric_card import Metric, metric_row
from components.page_header import page_header, section_header
from components.sidebar import render_sidebar_footer, render_sidebar_nav, sidebar_filters
from services import state_service
from services.export_service import to_csv_bytes
from utils import formatting
from utils.exceptions import CustomerIntelligenceError
from utils.logging_config import configure_logging
from visualization.demographic_charts import (
    age_band_bar,
    age_histogram,
    gender_bar,
    geo_treemap,
    plural_geo_level,
    revenue_geo_bar,
    stacked_composition_bar,
    top_geo_bar,
)
from visualization.theme import inject_terminal_css, without_title

configure_logging()
state_service.initialize(st.session_state)
inject_terminal_css()
render_sidebar_nav("pages/04_Demographic_Profiling.py")

page_header(
    "Demographic Profiling",
    "Who the customers are, where the revenue concentrates, and how profiles differ "
    "across segments. Missing fields produce notices, never crashes; maps are omitted "
    "because uploaded place names cannot be reliably geocoded.",
)

features = st.session_state.get("customer_features")
if features is None:
    empty_state("No features yet", "Build customer features on RFM / LRFM Analysis first.")
    st.stop()

rfm = st.session_state.get("rfm_results")
with sidebar_filters("Profiling Slicers"):
    segment_source = st.selectbox(
        "Segment labels for per-segment views",
        [s for s in ("RFM", "LRFM") if st.session_state.get(f"{s.lower()}_results") is not None]
        or ["(none available)"],
    )
    bands_spec = st.text_input(
        "Age bands (lo-hi pairs, last may be open like 65+)",
        value="0-17,18-24,25-34,35-49,50-64,65+",
    )

segments = None
if segment_source in ("RFM", "LRFM"):
    scored = st.session_state[f"{segment_source.lower()}_results"]
    segments = scored[["customer_id", "segment"]]

try:
    profile = cached_build_demographic_profile(features, segments, age_band_spec=bands_spec or None)
except (CustomerIntelligenceError, KeyError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

st.session_state["demographic_profile"] = {
    name: table.to_dict(orient="records") for name, table in profile.tables.items()
}
for note in profile.notes:
    st.info(note)

has_age = "age_summary" in profile.tables
has_gender = "gender" in profile.tables
if has_age or has_gender:
    section_header("Age and gender")
    if has_age:
        summary = profile.tables["age_summary"].iloc[0]
        metric_row(
            [
                Metric(
                    "Customers with age",
                    formatting.fmt_int(summary["customers_with_age"]),
                ),
                Metric("Mean age", f"{summary['mean_age']}"),
                Metric("Median age", f"{summary['median_age']}"),
            ]
        )
    demo_a, demo_b, demo_c = st.columns(3)
    with demo_a:
        with chart_section("Age distribution"):
            if has_age:
                st.plotly_chart(without_title(age_histogram(features["age"])), width="stretch")
            else:
                empty_state("No age data yet", "Age appears here after profiling.")
    with demo_b:
        with chart_section("Customers per age band"):
            if has_age:
                st.plotly_chart(
                    without_title(age_band_bar(profile.tables["age_bands"])),
                    width="stretch",
                )
            else:
                empty_state("No age bands yet", "Age bands appear here after profiling.")
    with demo_c:
        with chart_section("Customers by gender"):
            if has_gender:
                st.plotly_chart(
                    without_title(gender_bar(profile.tables["gender"])), width="stretch"
                )
            else:
                empty_state("No gender data yet", "Gender appears here after profiling.")

section_header("Geography")
geo_levels = [lvl for lvl in GEO_LEVELS if f"geo_{lvl}" in profile.tables]
if not geo_levels:
    empty_state(
        "No geography available",
        "No geographic fields (city/state/country/region) are present in this dataset.",
    )
geo_tab = st.tabs([lvl.capitalize() for lvl in geo_levels]) if geo_levels else []
for tab, level in zip(geo_tab, geo_levels, strict=True):
    with tab:
        table = profile.tables[f"geo_{level}"]
        geo_l, geo_r = st.columns([1, 1])
        with geo_l:
            with chart_section(f"Top {plural_geo_level(level)} by customers"):
                st.plotly_chart(
                    without_title(top_geo_bar(table, level)),
                    width="stretch",
                    key=f"geo_top_{level}",
                )
        with geo_r:
            if "revenue" in table.columns:
                with chart_section(f"Top {plural_geo_level(level)} by revenue"):
                    st.plotly_chart(
                        without_title(revenue_geo_bar(table, level)),
                        width="stretch",
                        key=f"geo_rev_{level}",
                    )
        with chart_section(f"Customer share by {level}"):
            st.plotly_chart(
                without_title(geo_treemap(table.head(30), level)),
                width="stretch",
                key=f"geo_tree_{level}",
            )
        with st.expander(f"{level.capitalize()} table (top 50)", expanded=False):
            st.dataframe(table.head(50), width="stretch", key=f"geo_table_{level}")

if segments is not None:
    comp_charts = [
        (column, cached_segment_composition(features, segments, column))
        for column in ("gender", "region")
        if column in features.columns
    ]
    for i in range(0, len(comp_charts), 2):
        pair = comp_charts[i : i + 2]
        cols = st.columns([1] * len(pair))
        for col, (column, comp) in zip(cols, pair, strict=True):
            with col:
                with chart_section(f"{column} mix by segment"):
                    st.plotly_chart(
                        without_title(stacked_composition_bar(comp, column)),
                        width="stretch",
                        key=f"comp_{column}",
                    )

if "concentration" in profile.tables:
    with st.expander("Geographic concentration", expanded=False):
        st.dataframe(profile.tables["concentration"], width="stretch")
        st.caption(
            "Top-3 share = customers in the 3 largest values. HHI near 1 means one value "
            "dominates; near 0 means evenly spread."
        )

if segments is not None and "segment_profiles" in profile.tables:
    with st.expander("Profiles by segment", expanded=False):
        st.dataframe(profile.tables["segment_profiles"], width="stretch")
    with st.expander("Population vs segment", expanded=False):
        st.dataframe(profile.tables["segment_comparison"], width="stretch")

    section_header("Do segments differ significantly?")
    st.markdown(
        "Continuous variables use ANOVA, Welch ANOVA, or Kruskal-Wallis (chosen by "
        "normality and variance checks); categorical use chi-square or Fisher exact. "
        "Significance is decided after Holm correction across all tests below."
    )
    continuous = [
        c
        for c in (
            "age",
            "monetary_value",
            "frequency",
            "recency_days",
            "relationship_length_days",
            "average_order_value",
        )
        if c in features.columns
    ]
    categorical = [c for c in ("gender", "region", "city", "country") if c in features.columns]
    seg_series = (
        features[["customer_id"]]
        .merge(segments, on="customer_id", how="left")["segment"]
        .fillna("Unsegmented")
    )
    with st.spinner("Running statistical tests…"):
        test_results = cached_run_segment_tests(features, seg_series, continuous, categorical)
    with st.expander("Segment significance tests", expanded=False):
        st.dataframe(test_results, width="stretch")

if "gender" in profile.tables:
    with st.expander("Gender table", expanded=False):
        st.dataframe(profile.tables["gender"], width="stretch")

with st.expander("Downloads", expanded=False):
    for name, table in profile.tables.items():
        st.download_button(
            f"Download {name} (CSV)",
            data=to_csv_bytes(pd.DataFrame(table)),
            file_name=f"demographic_{name}.csv",
            mime="text/csv",
            key=f"dl_demo_{name}",
        )

render_sidebar_footer()

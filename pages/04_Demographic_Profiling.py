"""Page 04 — Demographic and geographic profiling with segment comparisons."""

import pandas as pd
import streamlit as st

from analytics.demographics import (
    GEO_LEVELS,
    build_demographic_profile,
    segment_composition,
)
from analytics.statistical_tests import run_segment_tests
from components.empty_state import chart_section, empty_state
from components.filters import control_group, controls_expander
from components.metric_card import Metric, metric_row
from components.page_header import page_header, section_header
from components.sidebar import render_sidebar
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
    revenue_geo_bar,
    stacked_composition_bar,
    top_geo_bar,
)
from visualization.theme import inject_terminal_css

configure_logging()
state_service.initialize(st.session_state)
inject_terminal_css()
render_sidebar("pages/04_Demographic_Profiling.py")

page_header(
    "Demographic Profiling",
    "Who the customers are, where the revenue concentrates, and how profiles differ "
    "across segments. Missing fields produce notices, never crashes; maps are omitted "
    "because uploaded place names cannot be reliably geocoded.",
    badge="Step 4 of 9",
)

features = st.session_state.get("customer_features")
if features is None:
    empty_state("No features yet", "Build customer features on RFM / LRFM Analysis first.")
    st.stop()

rfm = st.session_state.get("rfm_results")
with controls_expander("Profiling Controls"):
    with control_group("Profiling Slicers"):
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
    profile = build_demographic_profile(features, segments, age_band_spec=bands_spec or None)
except (CustomerIntelligenceError, KeyError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

st.session_state["demographic_profile"] = {
    name: table.to_dict(orient="records") for name, table in profile.tables.items()
}
for note in profile.notes:
    st.info(note)

if profile.missing_pct:
    section_header("Missing demographic indicators")
    metric_row(
        [Metric(f"{field} missing", f"{pct}%") for field, pct in profile.missing_pct.items()]
    )

if "age_summary" in profile.tables:
    summary = profile.tables["age_summary"].iloc[0]
    section_header("Age")
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
    age_l, age_r = st.columns([1, 1])
    with age_l:
        with chart_section("Age distribution"):
            st.plotly_chart(age_histogram(features["age"]), use_container_width=True)
    with age_r:
        with chart_section("Customers per age band"):
            st.plotly_chart(age_band_bar(profile.tables["age_bands"]), use_container_width=True)

if "gender" in profile.tables:
    section_header("Gender")
    gender_l, gender_r = st.columns([3, 2])
    with gender_l:
        with chart_section("Customers by gender"):
            st.plotly_chart(gender_bar(profile.tables["gender"]), use_container_width=True)
    with gender_r:
        with st.expander("Gender table", expanded=False):
            st.dataframe(profile.tables["gender"], use_container_width=True)

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
            with chart_section(f"Top {level}s by customers"):
                st.plotly_chart(
                    top_geo_bar(table, level), use_container_width=True, key=f"geo_top_{level}"
                )
        with geo_r:
            if "revenue" in table.columns:
                with chart_section(f"Top {level}s by revenue"):
                    st.plotly_chart(
                        revenue_geo_bar(table, level),
                        use_container_width=True,
                        key=f"geo_rev_{level}",
                    )
        with chart_section(f"Customer share by {level}"):
            st.plotly_chart(
                geo_treemap(table.head(30), level),
                use_container_width=True,
                key=f"geo_tree_{level}",
            )
        with st.expander(f"{level.capitalize()} table (top 50)", expanded=False):
            st.dataframe(table.head(50), use_container_width=True, key=f"geo_table_{level}")

if "concentration" in profile.tables:
    with st.expander("Geographic concentration", expanded=False):
        st.dataframe(profile.tables["concentration"], use_container_width=True)
        st.caption(
            "Top-3 share = customers in the 3 largest values. HHI near 1 means one value "
            "dominates; near 0 means evenly spread."
        )

if segments is not None and "segment_profiles" in profile.tables:
    with st.expander("Profiles by segment", expanded=False):
        st.dataframe(profile.tables["segment_profiles"], use_container_width=True)
    with st.expander("Population vs segment", expanded=False):
        st.dataframe(profile.tables["segment_comparison"], use_container_width=True)
    comp_charts = [
        (column, segment_composition(features, segments, column))
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
                        stacked_composition_bar(comp, column),
                        use_container_width=True,
                        key=f"comp_{column}",
                    )

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
        test_results = run_segment_tests(features, seg_series, continuous, categorical)
    with st.expander("Segment significance tests", expanded=False):
        st.dataframe(test_results, use_container_width=True)

with st.expander("Downloads", expanded=False):
    for name, table in profile.tables.items():
        st.download_button(
            f"Download {name} (CSV)",
            data=to_csv_bytes(pd.DataFrame(table)),
            file_name=f"demographic_{name}.csv",
            mime="text/csv",
            key=f"dl_demo_{name}",
        )

"""Page 06 — Traditional (RFM/LRFM) vs AI segmentation comparison and recommendation."""

import pandas as pd
import streamlit as st

from analytics.segmentation_comparison import (
    agreement_ari,
    churn_rate_by_segment,
    crosstab_counts,
)
from components.empty_state import chart_section, empty_state
from components.filters import controls_expander
from components.insight_box import insight_box
from components.metric_card import Metric, metric_row
from components.metric_guides import render_metric_guides
from components.page_header import page_header, section_header
from components.segment_cards import render_segment_cards
from components.sidebar import render_sidebar_footer, render_sidebar_nav, sidebar_filters
from services import state_service
from services.comparison_service import build_comparison, scorecard_for_display
from services.export_service import to_csv_bytes, to_json_bytes
from services.segment_explain_service import build_segment_explanations
from utils import formatting
from utils.logging_config import configure_logging
from visualization.segmentation_charts import metric_comparison_bar, sankey_mapping
from visualization.theme import inject_terminal_css, without_title

configure_logging()
state_service.initialize(st.session_state)
inject_terminal_css()
render_sidebar_nav("pages/06_Segmentation_Comparison.py")

page_header(
    "Segmentation Comparison",
    "Fair comparison across statistical separation, stability, interpretability, and "
    "revenue differentiation. Silhouette is comparable only among AI runs sharing a "
    "feature space — the operational pick never comes from one metric alone.",
)

rfm = st.session_state.get("rfm_results")
lrfm = st.session_state.get("lrfm_results")
ai_runs = [r for r in (st.session_state.get("segmentation_runs") or []) if "labels" in r]
if st.session_state.get("customer_features") is None or (rfm is None and lrfm is None):
    empty_state("No scores yet", "Score RFM/LRFM on page 03 first.")
    st.stop()
if not ai_runs:
    empty_state(
        "No AI runs yet",
        "Save at least one AI run on page 05 first (re-save older runs to attach labels).",
    )
    st.stop()

# The scorecard is computed once in services.comparison_service so this page and
# the report's Model comparison section can never disagree about the same run.
comparison = build_comparison(st.session_state)
if comparison is None:
    empty_state("Nothing to compare", "Score RFM/LRFM and save an AI run first.")
    st.stop()
table = comparison["table"]
labelings = comparison["labelings"]
churn = comparison["churn_labels"]

with st.expander("Model scorecard", expanded=False):
    st.dataframe(scorecard_for_display(table), width="stretch")
    st.caption(
        "RFM/LRFM silhouette is blank by design: different feature space, not comparable. "
        "Stability 1.0 for RFM/LRFM: deterministic rules on fixed features."
    )

# --- Scorecard glossary: reading guide for the columns in the table above ------
section_header(
    "How to read the scorecard",
    "Every column the scorecard emits, with its direction, its range, and the gate it "
    "carries in the recommendation below.",
)
render_metric_guides()

metrics = [
    ("silhouette", "Silhouette (AI runs, shared space)"),
    ("stability_ari", "Seed stability (ARI)"),
    ("value_eta2", "Revenue differentiation (monetary eta²)"),
    ("evenness", "Size balance (evenness)"),
]
for i in range(0, len(metrics), 2):
    pair = metrics[i : i + 2]
    cols = st.columns([1] * len(pair))
    for col, (metric, title) in zip(cols, pair, strict=True):
        with col:
            with chart_section(title):
                st.plotly_chart(
                    without_title(metric_comparison_bar(table, metric, title)),
                    width="stretch",
                    key=f"cmp_{metric}",
                )

# --- Pairwise agreement: traditional vs selected AI ---------------------------
ai_names = [r["model"] for r in table.to_dict("records") if r["kind"] == "AI"]
traditional = [r["model"] for r in table.to_dict("records") if r["kind"] == "Traditional"]
# Default the pair to the model the comparison actually recommends (or the human
# pick already recorded). Defaulting to the first saved run made the chart read
# "kmeans" while the recommendation said "gmm" — the evidence contradicted itself.
pair_defaults = comparison.get("pair") or {}
default_ref = pair_defaults.get("reference", traditional[0])
default_target = pair_defaults.get("target", ai_names[0])
with sidebar_filters("Comparison Pair"):
    ref = st.selectbox(
        "Traditional reference",
        traditional,
        index=traditional.index(default_ref) if default_ref in traditional else 0,
    )
    target = st.selectbox(
        "AI model",
        ai_names,
        index=ai_names.index(default_target) if default_target in ai_names else 0,
    )
ct = crosstab_counts(labelings[ref], labelings[target])
ari = agreement_ari(labelings[ref], labelings[target])
agree_l, agree_r = st.columns([1, 2])
with agree_l:
    with chart_section(
        "Pair agreement",
        "1.0 = identical groupings, 0 = chance-level agreement.",
    ):
        metric_row(
            [Metric(f"Adjusted Rand Index ({ref} vs {target})", formatting.fmt_float(ari, 3))]
        )
with agree_r:
    with chart_section(f"Customer flow ({ref} → {target})"):
        st.plotly_chart(without_title(sankey_mapping(ct, ref, target)), width="stretch")
with st.expander("Cross-tabulation (counts)", expanded=False):
    st.dataframe(ct, width="stretch")

# --- What the AI groups mean ---------------------------------------------------
# The flow says where customers moved; these cards say what the destination
# groups actually are. They read from the same run labels the pair chart uses.
features = st.session_state.get("customer_features")
if features is not None and "customer_id" in features.columns:
    labeled = features.set_index("customer_id").reindex(labelings[target].index)
    labeled["ai_segment"] = labelings[target].to_numpy()
    segment_notes = build_segment_explanations(labeled)
else:
    segment_notes = []
if segment_notes:
    section_header(
        "What the AI segment groups mean",
        "Value and activity tiers are relative thirds within this AI run; A/B suffixes "
        "separate clusters sharing a grid cell. Cards are ordered by revenue contribution.",
    )
    render_segment_cards(segment_notes)

# --- Per-model checklists ------------------------------------------------------
with st.expander("Interpretability checklists"):
    for row in table.to_dict("records"):
        st.markdown(f"**{row['model']}** — {row['interpretability_passes']} checks pass")
        st.dataframe(
            pd.DataFrame(row["checklist"]),
            width="stretch",
            key=f"checklist_{row['model']}",
        )

# --- Churn association (Phase 6 unlocks real data) ------------------------------
churn_table = churn_rate_by_segment(
    labelings[target], churn.reindex(labelings[target].index) if churn is not None else None
)
if churn_table is None:
    st.info("Churn-rate association unlocks in Phase 6 once churn labels exist.")
else:
    with st.expander(f"Churn rate by {target} segment", expanded=False):
        st.dataframe(churn_table, width="stretch")

# --- Recommendation ------------------------------------------------------------
recommendation = comparison["recommendation"]
insight_box(
    "Model recommendation",
    recommendation["rationale"],
    tone="info",
)
with controls_expander("Operational Pick"):
    models = table["model"].tolist()
    # Honour a previously recorded override, so the pick, the pair summary, and the
    # report stay on the same model across reruns instead of snapping back.
    recorded_pick = (st.session_state.get("comparison_report") or {}).get("human_pick")
    default_pick = recorded_pick if recorded_pick in models else recommendation["operational_pick"]
    choice = st.selectbox(
        "Operational model (evidence above — disagree if it says otherwise)",
        models,
        index=models.index(default_pick) if default_pick in models else 0,
    )
st.session_state["comparison_report"] = {**recommendation, "human_pick": choice}
st.success(
    f"Recorded operational pick: {choice} "
    f"({'agrees' if choice == recommendation['operational_pick'] else 'overrides'} "
    "the recommendation)."
)

st.download_button(
    "Download scorecard (CSV)",
    data=to_csv_bytes(scorecard_for_display(table)),
    file_name="segmentation_comparison.csv",
    mime="text/csv",
)
st.download_button(
    "Download recommendation (JSON)",
    data=to_json_bytes(st.session_state["comparison_report"]),
    file_name="segmentation_recommendation.json",
    mime="application/json",
)

render_sidebar_footer()

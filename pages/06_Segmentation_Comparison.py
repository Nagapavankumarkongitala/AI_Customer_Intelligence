"""Page 06 — Traditional (RFM/LRFM) vs AI segmentation comparison and recommendation."""

import pandas as pd
import streamlit as st

from analytics.segmentation_comparison import (
    agreement_ari,
    churn_rate_by_segment,
    crosstab_counts,
    interpretability_checklist,
    recommend_model,
    revenue_concentration,
    size_balance,
    value_separation,
)
from components.filters import control_group, controls_expander
from components.nav_tabs import render_nav_tabs
from services import state_service
from services.export_service import to_csv_bytes, to_json_bytes
from utils.logging_config import configure_logging
from visualization.segmentation_charts import metric_comparison_bar, sankey_mapping
from visualization.theme import inject_terminal_css

configure_logging()
state_service.initialize(st.session_state)
inject_terminal_css()
render_nav_tabs("pages/06_Segmentation_Comparison.py")

st.markdown(
    "Fair comparison across statistical separation, stability, interpretability, and "
    "revenue differentiation. Silhouette is comparable only among AI runs sharing a "
    "feature space — the operational pick never comes from one metric alone."
)

features = st.session_state.get("customer_features")
rfm = st.session_state.get("rfm_results")
lrfm = st.session_state.get("lrfm_results")
ai_runs = [r for r in (st.session_state.get("segmentation_runs") or []) if "labels" in r]
if features is None or (rfm is None and lrfm is None):
    st.info("Score RFM/LRFM on page 03 first.")
    st.stop()
if not ai_runs:
    st.info("Save at least one AI run on page 05 first (re-save older runs to attach labels).")
    st.stop()

money = features.set_index("customer_id")["monetary_value"]
churn_scores = st.session_state.get("churn_scores")
churn = None
if isinstance(churn_scores, pd.DataFrame):
    if "predicted_class" in churn_scores.columns:
        churn = churn_scores.set_index("customer_id")["predicted_class"]
    elif "churn_probability" in churn_scores.columns:
        churn = churn_scores.set_index("customer_id")["churn_probability"] >= 0.5

# --- Candidate rows: AI runs + traditional labelings -------------------------
rows: list[dict] = []
labelings: dict[str, pd.Series] = {}
for run in ai_runs:
    name = f"AI:{run['model']}"
    ids = run["customer_ids"]
    labels = pd.Series(run["labels"], index=ids, name="cluster")
    names = {int(k): v for k, v in run.get("names", {}).items()}
    display = pd.Series(
        [names.get(int(v), str(v)) for v in labels], index=labels.index, name="segment"
    )
    labelings[name] = display
    metrics, stability = run["metrics"], run["stability"]
    sizes = labels.value_counts().to_dict()
    balance = size_balance({int(k): int(v) for k, v in sizes.items()})
    revenues = money.reindex(ids).groupby(display.values).sum().to_dict()
    checklist = interpretability_checklist(
        metrics["n_clusters"],
        bool(names),
        metrics["noise_pct"],
        stability.get("mean_pairwise_ari"),
        balance["min_share"],
    )
    rows.append(
        {
            "model": name,
            "kind": "AI",
            "n_segments": metrics["n_clusters"],
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "stability_ari": stability.get("mean_pairwise_ari"),
            "value_eta2": value_separation(money.reindex(ids), display),
            "min_share": balance["min_share"],
            "evenness": balance["evenness"],
            "top_revenue_share": revenue_concentration(
                {str(k): float(v) for k, v in revenues.items()}
            )["top_segment_share"],
            "noise_pct": metrics["noise_pct"],
            "interpretability_passes": sum(1 for i in checklist if i["pass"]),
            "checklist": checklist,
        }
    )

for label, scored in (("RFM", rfm), ("LRFM", lrfm)):
    if scored is None:
        continue
    display = scored.set_index("customer_id")["segment"]
    labelings[label] = display
    ids = display.index.tolist()
    sizes = display.value_counts().to_dict()
    balance = size_balance({str(k): int(v) for k, v in sizes.items()})
    revenues = money.reindex(ids).groupby(display.values).sum().to_dict()
    checklist = interpretability_checklist(
        display.nunique(),
        True,
        0.0,
        1.0,
        balance["min_share"],
    )
    rows.append(
        {
            "model": label,
            "kind": "Traditional",
            "n_segments": display.nunique(),
            "silhouette": None,
            "davies_bouldin": None,
            "stability_ari": 1.0,
            "value_eta2": value_separation(money.reindex(ids), display),
            "min_share": balance["min_share"],
            "evenness": balance["evenness"],
            "top_revenue_share": revenue_concentration(
                {str(k): float(v) for k, v in revenues.items()}
            )["top_segment_share"],
            "noise_pct": 0.0,
            "interpretability_passes": sum(1 for i in checklist if i["pass"]),
            "checklist": checklist,
        }
    )

table = pd.DataFrame(rows)
with st.expander("Model scorecard", expanded=False):
    st.dataframe(table.drop(columns=["checklist"]), use_container_width=True)
    st.caption(
        "RFM/LRFM silhouette is blank by design: different feature space, not comparable. "
        "Stability 1.0 for RFM/LRFM: deterministic rules on fixed features."
    )

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
            st.plotly_chart(
                metric_comparison_bar(table, metric, title),
                use_container_width=True,
                key=f"cmp_{metric}",
            )

# --- Pairwise agreement: traditional vs selected AI ---------------------------
ai_names = [r["model"] for r in rows if r["kind"] == "AI"]
traditional = [r["model"] for r in rows if r["kind"] == "Traditional"]
with controls_expander("Comparison Controls"):
    with control_group("Comparison Pair"):
        ref = st.selectbox("Traditional reference", traditional)
        target = st.selectbox("AI model", ai_names)
ct = crosstab_counts(labelings[ref], labelings[target])
ari = agreement_ari(labelings[ref], labelings[target])
agree_l, agree_r = st.columns([1, 2])
with agree_l:
    st.metric(f"Adjusted Rand Index ({ref} vs {target})", ari)
with agree_r:
    st.plotly_chart(sankey_mapping(ct, ref, target), use_container_width=True)
with st.expander("Cross-tabulation (counts)", expanded=False):
    st.dataframe(ct, use_container_width=True)

# --- Per-model checklists ------------------------------------------------------
with st.expander("Interpretability checklists"):
    for row in rows:
        st.markdown(f"**{row['model']}** — {row['interpretability_passes']} checks pass")
        st.dataframe(
            pd.DataFrame(row["checklist"]),
            use_container_width=True,
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
        st.dataframe(churn_table, use_container_width=True)

# --- Recommendation ------------------------------------------------------------
recommendation = recommend_model(table.rename(columns={"model": "name"}).to_dict(orient="records"))
st.info(recommendation["rationale"])
with controls_expander("Operational Pick"):
    choice = st.selectbox(
        "Operational model (evidence above — disagree if it says otherwise)",
        table["model"].tolist(),
        index=table["model"].tolist().index(recommendation["operational_pick"]),
    )
st.session_state["comparison_report"] = {**recommendation, "human_pick": choice}
st.success(
    f"Recorded operational pick: {choice} "
    f"({'agrees' if choice == recommendation['operational_pick'] else 'overrides'} "
    "the recommendation)."
)

st.download_button(
    "Download scorecard (CSV)",
    data=to_csv_bytes(table.drop(columns=["checklist"])),
    file_name="segmentation_comparison.csv",
    mime="text/csv",
)
st.download_button(
    "Download recommendation (JSON)",
    data=to_json_bytes(st.session_state["comparison_report"]),
    file_name="segmentation_recommendation.json",
    mime="application/json",
)

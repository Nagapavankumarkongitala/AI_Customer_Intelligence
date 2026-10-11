"""Page 05 — Advanced segmentation: pipeline, six models, evaluation, naming."""

from pathlib import Path

import streamlit as st
import yaml

from components.metric_cards import kpi_row
from segmentation.common import labels_to_series
from segmentation.evaluation import evaluate_clustering
from segmentation.segment_naming import propose_names
from segmentation.segment_profiling import build_cluster_profile
from services import state_service
from services.export_service import to_csv_bytes
from services.segmentation_service import (
    assess_stability,
    model_availability,
    prepare_segmentation_features,
    run_model,
    save_segmentation_run,
    sweep_clusters,
)
from utils.exceptions import CustomerIntelligenceError
from utils.logging_config import configure_logging
from visualization.segmentation_charts import (
    cluster_size_bar,
    pca_scatter,
    profile_heatmap,
    umap_scatter,
)

configure_logging()
state_service.initialize(st.session_state)

st.title("05 · Advanced Segmentation")
st.markdown(
    "Unsupervised segmentation over the customer feature table. Prepare features, "
    "compare models and cluster counts, then name clusters from their profiles — "
    "IDs stay numeric; business names are assigned only after profiling."
)

features = st.session_state.get("customer_features")
if features is None:
    st.info("Build customer features on 03 · RFM / LRFM Analysis first.")
    st.stop()

cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
numeric_cols = features.select_dtypes(include="number").columns.tolist()
defaults = [f for f in cfg["segmentation"]["default_features"] if f in numeric_cols]

st.subheader("1 · Feature pipeline")
f1, f2, f3 = st.columns(3)
with f1:
    selected = st.multiselect("Clustering features", numeric_cols, default=defaults or numeric_cols)
with f2:
    scaler = st.selectbox(
        "Scaler",
        ["robust", "standard"],
        help="Robust (median/IQR) resists outliers; standard uses mean/std.",
    )
with f3:
    use_log1p = st.checkbox(
        "log1p skewed features",
        value=True,
        help="Applies to nonnegative columns with skew above 1.",
    )

if st.button("Prepare features"):
    try:
        prepared = prepare_segmentation_features(
            features,
            selected,
            scaler=scaler,
            log1p=use_log1p,
            minimum_customers=cfg["segmentation"]["minimum_customers"],
        )
        st.session_state["seg_prepared"] = {
            "matrix": prepared.matrix,
            "feature_names": prepared.feature_names,
            "pipeline": prepared.pipeline,
            "report": {
                "dropped_zero_variance": prepared.dropped_zero_variance,
                "log1p_applied": prepared.log1p_applied,
                "high_correlations": prepared.high_correlations,
            },
        }
        st.success(
            f"Prepared {prepared.matrix.shape[0]} customers × "
            f"{prepared.matrix.shape[1]} features."
        )
    except (CustomerIntelligenceError, KeyError, ValueError) as exc:
        st.error(str(exc))

prepared_state = st.session_state.get("seg_prepared")
if prepared_state is None:
    st.info("Select features and prepare them to continue.")
    st.stop()

report = prepared_state["report"]
with st.expander("Pipeline review", expanded=True):
    st.write(f"Kept features: {', '.join(prepared_state['feature_names'])}")
    if report["dropped_zero_variance"]:
        st.write(f"Dropped (no variance): {', '.join(report['dropped_zero_variance'])}")
    if report["log1p_applied"]:
        st.write(f"log1p applied: {', '.join(report['log1p_applied'])}")
    if report["high_correlations"]:
        st.warning("Highly correlated pairs (review before trusting splits):")
        st.write(report["high_correlations"])

st.subheader("2 · Model & cluster count")
availability = model_availability()
model_names = ["kmeans", "gmm", "hierarchical", "hdbscan", "autoencoder", "ensemble"]
model = st.selectbox("Model", model_names)
if availability.get(model):
    st.warning(availability[model])
m1, m2, m3 = st.columns(3)
with m1:
    n_clusters = st.slider("Clusters (k)", 2, 10, 5, disabled=(model == "hdbscan"))
with m2:
    seed = st.number_input("Random seed", value=42, step=1)
with m3:
    min_cluster_size = st.number_input("HDBSCAN min cluster size", min_value=5, value=15, step=5)
with st.expander("Autoencoder settings"):
    latent_dim = st.number_input("Latent dimensions", min_value=2, max_value=32, value=8)
    epochs = st.number_input("Max epochs (early stopping)", min_value=5, max_value=500, value=100)

k_min, k_max = cfg["segmentation"]["cluster_range"]
if model != "hdbscan" and st.button(f"Sweep k = {k_min}…{k_max}"):
    try:
        with st.spinner("Sweeping cluster counts..."):
            sweep = sweep_clusters(
                model, prepared_state["matrix"], k_min=k_min, k_max=k_max, random_state=int(seed)
            )
        st.dataframe(sweep, use_container_width=True)
        st.caption(
            "Higher silhouette / Calinski-Harabasz and lower Davies-Bouldin are "
            "better — but never choose on one metric alone (see page 06)."
        )
    except (CustomerIntelligenceError, ValueError) as exc:
        st.error(str(exc))

labels = None
if st.button("Run segmentation", type="primary"):
    try:
        with st.spinner(f"Fitting {model}... (autoencoder trains a neural net)"):
            extra = {}
            if model == "hdbscan":
                extra["min_cluster_size"] = int(min_cluster_size)
            if model == "autoencoder":
                extra.update({"latent_dim": int(latent_dim), "epochs": int(epochs)})
            labels, fitted = run_model(
                model,
                prepared_state["matrix"],
                n_clusters=int(n_clusters),
                random_state=int(seed),
                **extra,
            )
        metrics = evaluate_clustering(prepared_state["matrix"], labels)
        stability = assess_stability(model, prepared_state["matrix"], random_state=int(seed))
        st.session_state["seg_run"] = {
            "model": model,
            "labels": labels,
            "fitted": fitted,
            "metrics": metrics,
            "stability": stability,
            "feature_names": prepared_state["feature_names"],
            "params": {"n_clusters": int(n_clusters), "seed": int(seed), **extra},
        }
        st.success(
            f"{model} produced {metrics['n_clusters']} clusters "
            f"({metrics['noise_pct']}% noise)."
        )
    except CustomerIntelligenceError as exc:
        st.error(str(exc))
    except (ValueError, KeyError) as exc:
        st.error(str(exc))

run = st.session_state.get("seg_run")
if run is None:
    st.stop()

st.subheader("3 · Evaluation")
metrics = run["metrics"]
kpi_row(
    [
        ("Clusters", str(metrics["n_clusters"])),
        ("Silhouette (higher better)", str(metrics["silhouette"])),
        ("Davies-Bouldin (lower better)", str(metrics["davies_bouldin"])),
        ("Noise", f"{metrics['noise_pct']}%"),
        ("Min cluster size", str(metrics["min_cluster_size"])),
        ("Seed stability (ARI)", str(run["stability"]["mean_pairwise_ari"])),
    ]
)
if run["stability"].get("bootstrap") is not None:
    st.caption(f"Bootstrap stability: {run['stability']['bootstrap']}")

st.plotly_chart(pca_scatter(prepared_state["matrix"], run["labels"]), use_container_width=True)
try:
    st.plotly_chart(umap_scatter(prepared_state["matrix"], run["labels"]), use_container_width=True)
except ImportError as exc:
    st.info(str(exc))
if run["model"] == "autoencoder":
    from deep_learning.latent_space_visualization import latent_scatter

    st.plotly_chart(
        latent_scatter(run["fitted"]["latent"], run["labels"]), use_container_width=True
    )
    with st.expander("Training history"):
        st.write({k: v[-5:] for k, v in run["fitted"]["history"].items()})

st.subheader("4 · Profile & name clusters")
profile = build_cluster_profile(
    features,
    run["labels"],
    run["feature_names"],
    id_col="customer_id",
)
name_table = propose_names(profile)
profile = profile.merge(name_table[["cluster_id", "business_name"]], on="cluster_id", how="left")
profile["business_name"] = profile["business_name"].fillna("Noise")
st.plotly_chart(cluster_size_bar(profile), use_container_width=True)
st.plotly_chart(profile_heatmap(profile), use_container_width=True)
st.dataframe(profile.drop(columns=["member_ids"], errors="ignore"), use_container_width=True)

st.markdown("Adjust business names if the profiles suggest better ones:")
overrides: dict[int, str] = {}
default_names = dict(zip(name_table["cluster_id"], name_table["business_name"], strict=True))
for cluster_id in sorted(profile["cluster_id"].unique()):
    if cluster_id == -1:
        overrides[cluster_id] = "Noise"
        continue
    overrides[cluster_id] = st.text_input(
        f"Cluster {cluster_id}",
        value=str(default_names.get(cluster_id, f"Cluster {cluster_id}")),
        key=f"segname_{cluster_id}",
    )

if st.button("Save this run"):
    labeled = features[["customer_id"]].copy()
    labeled["cluster_id"] = labels_to_series(run["labels"], features.index).tolist()
    labeled["business_name"] = labeled["cluster_id"].map(overrides)
    record = {
        "model": run["model"],
        "params": run["params"],
        "metrics": metrics,
        "stability": run["stability"],
        "names": {int(k): v for k, v in overrides.items()},
        "feature_names": run["feature_names"],
    }
    runs = st.session_state.get("segmentation_runs") or []
    runs.append(record)
    st.session_state["segmentation_runs"] = runs
    st.session_state["selected_segmentation_model"] = run["model"]
    st.session_state["segmented_customers"] = labeled
    path = save_segmentation_run(
        (
            run["fitted"]
            if not isinstance(run["fitted"], dict) or run["model"] != "autoencoder"
            else run["fitted"]["kmeans"]
        ),
        prepared_state["pipeline"],
        {**record, "metrics": metrics},
        run_name=f"{run['model']}_k{metrics['n_clusters']}",
    )
    st.success(f"Run saved to session state and model persisted at {path}.")
    st.download_button(
        "Download labeled customers (CSV)",
        data=to_csv_bytes(labeled),
        file_name="segmented_customers.csv",
        mime="text/csv",
    )

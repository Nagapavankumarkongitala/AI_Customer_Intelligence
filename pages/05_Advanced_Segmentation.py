"""Page 05 — Advanced segmentation: pipeline, six models, evaluation, naming."""

from pathlib import Path

import streamlit as st
import yaml

from components.cached import cached_build_cluster_profile
from components.control_guides import (
    INTRO,
    RUN_HELP,
    SWEEP_HELP,
    TRAIN_ALL_HELP,
    USAGE_CAPTION,
    control_help,
    control_usage_rows,
)
from components.empty_state import chart_section, empty_state
from components.filters import control_group, controls_expander
from components.metric_card import Metric, metric_row
from components.page_header import page_header, section_header
from components.sidebar import render_sidebar_footer, render_sidebar_nav
from segmentation.common import labels_to_series
from segmentation.evaluation import evaluate_clustering
from segmentation.segment_naming import propose_names
from services import state_service
from services.export_service import to_csv_bytes
from services.segmentation_service import (
    assess_stability,
    model_availability,
    prepare_segmentation_features,
    run_model,
    save_segmentation_run,
    sweep_clusters,
    train_all_models,
)
from utils.exceptions import CustomerIntelligenceError
from utils.logging_config import configure_logging
from visualization.segmentation_charts import (
    cluster_size_bar,
    pca_scatter,
    profile_heatmap,
    umap_scatter,
)
from visualization.theme import inject_terminal_css, without_title

configure_logging()
state_service.initialize(st.session_state)
inject_terminal_css()
render_sidebar_nav("pages/05_Advanced_Segmentation.py")

page_header(
    "Advanced Segmentation",
    "Unsupervised segmentation over the customer feature table. Prepare features, "
    "compare models and cluster counts, then name clusters from their profiles — "
    "IDs stay numeric; business names are assigned only after profiling.",
)

features = st.session_state.get("customer_features")
if features is None:
    empty_state("No features yet", "Build customer features on RFM / LRFM Analysis first.")
    st.stop()

cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
numeric_cols = features.select_dtypes(include="number").columns.tolist()
defaults = [f for f in cfg["segmentation"]["default_features"] if f in numeric_cols]

with controls_expander("Feature Controls"):
    with control_group("Feature Set"):
        selected = st.multiselect(
            "Clustering features", numeric_cols, default=defaults or numeric_cols
        )
        use_log1p = st.checkbox(
            "log1p skewed features",
            value=True,
            help="Applies to nonnegative columns with skew above 1.",
        )
    with control_group("Scaling"):
        scaler = st.selectbox(
            "Scaler",
            ["robust", "standard"],
            help="Robust (median/IQR) resists outliers; standard uses mean/std.",
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
    empty_state("No prepared features", "Select features and prepare them to continue.")
    st.stop()

report = prepared_state["report"]
with st.expander("Pipeline review", expanded=False):
    st.write(f"Kept features: {', '.join(prepared_state['feature_names'])}")
    if report["dropped_zero_variance"]:
        st.write(f"Dropped (no variance): {', '.join(report['dropped_zero_variance'])}")
    if report["log1p_applied"]:
        st.write(f"log1p applied: {', '.join(report['log1p_applied'])}")
    if report["high_correlations"]:
        st.warning("Highly correlated pairs (review before trusting splits):")
        st.write(report["high_correlations"])

with controls_expander("Model Controls"):
    with control_group("Model Setup"):
        st.caption(INTRO)
        availability = model_availability()
        model_names = ["kmeans", "gmm", "hierarchical", "hdbscan", "autoencoder", "ensemble"]
        model = st.selectbox("Model", model_names, help=control_help("Model"))
        n_clusters = st.slider(
            "Clusters (k)",
            2,
            10,
            5,
            disabled=(model == "hdbscan"),
            help=control_help("Clusters (k)"),
        )
        seed = st.number_input("Random seed", value=42, step=1, help=control_help("Random seed"))
        min_cluster_size = st.number_input(
            "HDBSCAN min cluster size",
            min_value=5,
            value=15,
            step=5,
            help=control_help("HDBSCAN min cluster size"),
        )
        latent_dim = st.number_input(
            "AE latent dims",
            min_value=2,
            max_value=32,
            value=8,
            help=control_help("AE latent dims"),
        )
        epochs = st.number_input(
            "AE max epochs",
            min_value=5,
            max_value=500,
            value=100,
            help=control_help("AE max epochs"),
        )
    with control_group("Which controls each model reads"):
        st.dataframe(control_usage_rows(), width="stretch")
        st.caption(USAGE_CAPTION)
if availability.get(model):
    st.warning(availability[model])

k_min, k_max = cfg["segmentation"]["cluster_range"]
if model != "hdbscan" and st.button(f"Sweep k = {k_min}…{k_max}", help=SWEEP_HELP):
    try:
        with st.spinner("Sweeping cluster counts…"):
            sweep = sweep_clusters(
                model, prepared_state["matrix"], k_min=k_min, k_max=k_max, random_state=int(seed)
            )
        with st.expander(f"k sweep {k_min}…{k_max}", expanded=False):
            st.dataframe(sweep, width="stretch")
            st.caption(
                "Higher silhouette / Calinski-Harabasz and lower Davies-Bouldin are "
                "better — but never choose on one metric alone (see page 06)."
            )
    except (CustomerIntelligenceError, ValueError) as exc:
        st.error(str(exc))

labels = None
run_col, batch_col = st.columns(2)
with run_col:
    run_clicked = st.button("Run segmentation", type="primary", help=RUN_HELP)
with batch_col:
    batch_clicked = st.button("Train all models", help=TRAIN_ALL_HELP)
if batch_clicked:
    progress = st.progress(0.0, text="Training all models…")
    with st.spinner("Training every model (autoencoder trains a neural net)…"):
        batch = train_all_models(
            prepared_state["matrix"],
            features["customer_id"].tolist(),
            prepared_state["feature_names"],
            original_features=features,
            n_clusters=int(n_clusters),
            random_state=int(seed),
            min_cluster_size=int(min_cluster_size),
            latent_dim=int(latent_dim),
            epochs=int(epochs),
        )
    progress.progress(1.0)
    ok_runs = [r for r in batch if r.get("status") == "ok"]
    skipped = [r for r in batch if r.get("status") != "ok"]
    runs = st.session_state.get("segmentation_runs") or []
    runs.extend(ok_runs)
    st.session_state["segmentation_runs"] = runs
    st.success(f"Trained {len(ok_runs)} model(s); {len(skipped)} skipped.")
    with st.expander("Batch summary", expanded=False):
        st.dataframe(
            [
                {
                    "model": r["model"],
                    "status": r.get("status"),
                    "silhouette": (r.get("metrics") or {}).get("silhouette"),
                    "noise_pct": (r.get("metrics") or {}).get("noise_pct"),
                    "stability": (r.get("stability") or {}).get("mean_pairwise_ari"),
                    "reason": r.get("reason", ""),
                }
                for r in batch
            ],
            width="stretch",
        )
    if skipped:
        for r in skipped:
            st.info(f"{r['model']} skipped: {r.get('reason', 'unavailable')}")
if run_clicked:
    try:
        with st.spinner(f"Fitting {model}… (autoencoder trains a neural net)"):
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

section_header("Evaluation")
metrics = run["metrics"]
metric_row(
    [
        Metric("Clusters", str(metrics["n_clusters"])),
        Metric("Silhouette ↑", str(metrics["silhouette"])),
        Metric("Noise", f"{metrics['noise_pct']}%"),
        Metric("Seed stability (ARI)", str(run["stability"]["mean_pairwise_ari"])),
    ]
)
metric_row(
    [
        Metric("Davies-Bouldin ↓", str(metrics["davies_bouldin"])),
        Metric("Min cluster size", str(metrics["min_cluster_size"])),
        Metric("Calinski-Harabasz ↑", str(metrics["calinski_harabasz"])),
        Metric(
            "Bootstrap",
            str((run["stability"].get("bootstrap") or {}).get("mean_ari", "n/a")),
        ),
    ]
)

emb_l, emb_r = st.columns([1, 1])
with emb_l:
    with chart_section("Clusters (PCA 2D)"):
        st.plotly_chart(
            without_title(pca_scatter(prepared_state["matrix"], run["labels"])),
            width="stretch",
        )
with emb_r:
    with chart_section("Clusters (UMAP 2D)"):
        try:
            st.plotly_chart(
                without_title(umap_scatter(prepared_state["matrix"], run["labels"])),
                width="stretch",
            )
        except ImportError as exc:
            st.info(str(exc))
if run["model"] == "autoencoder":
    from deep_learning.latent_space_visualization import latent_scatter

    with chart_section("Latent space (autoencoder)"):
        st.plotly_chart(
            without_title(latent_scatter(run["fitted"]["latent"], run["labels"])),
            width="stretch",
        )
    with st.expander("Training history", expanded=False):
        st.write({k: v[-5:] for k, v in run["fitted"]["history"].items()})

section_header("Profile & name clusters")
profile = cached_build_cluster_profile(
    features,
    run["labels"],
    run["feature_names"],
    id_col="customer_id",
)
name_table = propose_names(profile)
profile = profile.merge(name_table[["cluster_id", "business_name"]], on="cluster_id", how="left")
profile["business_name"] = profile["business_name"].fillna("Noise")
prof_l, prof_r = st.columns([1, 1])
with prof_l:
    with chart_section("Customers per cluster"):
        st.plotly_chart(without_title(cluster_size_bar(profile)), width="stretch")
with prof_r:
    with chart_section("Cluster profile (median, z-scored per feature)"):
        st.plotly_chart(without_title(profile_heatmap(profile)), width="stretch")
with st.expander("Cluster profile table", expanded=False):
    st.dataframe(profile.drop(columns=["member_ids"], errors="ignore"), width="stretch")

with st.expander("Name clusters", expanded=False):
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
        "customer_ids": labeled["customer_id"].tolist(),
        "labels": [int(v) for v in run["labels"]],
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

render_sidebar_footer()

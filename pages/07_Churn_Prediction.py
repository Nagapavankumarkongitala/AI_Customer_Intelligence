"""Page 07 — Leakage-safe churn prediction, evaluation, and risk scoring."""

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from analytics.customer_features import build_customer_features
from churn.dataset_builder import build_churn_dataset
from churn.evaluation import (
    calibration_table,
    classification_metrics,
    lift_gain_table,
)
from churn.explainability import logreg_effects, shap_values
from components.empty_state import chart_section, empty_state
from components.filters import control_group, controls_expander
from components.metric_card import Metric, metric_row
from components.page_header import page_header, section_header
from components.sidebar import render_sidebar_footer, render_sidebar_nav
from components.table import render_table
from services import state_service
from services.churn_service import (
    available_models,
    save_churn_run,
    score_all_customers,
    train_churn_model,
)
from services.export_service import to_csv_bytes, to_json_bytes
from utils.exceptions import CustomerIntelligenceError
from utils.logging_config import configure_logging
from visualization.churn_charts import (
    calibration_curve_fig,
    confusion_fig,
    gain_chart_fig,
    importance_bar_fig,
    risk_band_bar,
    roc_curve_fig,
)
from visualization.theme import inject_terminal_css

configure_logging()
state_service.initialize(st.session_state)
inject_terminal_css()
render_sidebar_nav("pages/07_Churn_Prediction.py")

page_header(
    "Churn Prediction",
    "Future inactivity predicted from historical behavior only: predictors use "
    "transactions on or before the cutoff; the label uses activity after it. "
    "Any post-cutoff predictor is a leakage bug — the builder proves otherwise below.",
    badge="Step 7 of 9",
)

clean_data = st.session_state.get("clean_data")
if clean_data is None:
    empty_state("No clean data yet", "Complete Data Quality & Cleaning first.")
    st.stop()

cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
txn_dates = pd.to_datetime(clean_data["transaction_date"])
min_date, max_date = txn_dates.min().date(), txn_dates.max().date()

with controls_expander("Dataset Controls"):
    with control_group("Dataset Window"):
        obs_start = st.date_input(
            "Observation start", value=min_date, min_value=min_date, max_value=max_date
        )
        default_cutoff = max_date - pd.Timedelta(days=cfg["churn"]["default_outcome_horizon_days"])
        cutoff = st.date_input(
            "Cutoff date",
            value=max(default_cutoff, min_date),
            min_value=min_date,
            max_value=max_date,
        )
        horizon = st.number_input(
            "Outcome horizon (days)",
            min_value=7,
            max_value=365,
            value=cfg["churn"]["default_outcome_horizon_days"],
        )
        min_history = st.number_input(
            "Min. customer history (days)", min_value=0, max_value=365, value=30
        )

if st.button("Build churn dataset", type="primary"):
    try:
        with st.spinner("Labeling from post-cutoff activity…"):
            dataset = build_churn_dataset(
                clean_data,
                pd.Timestamp(cutoff),
                horizon_days=int(horizon),
                min_history_days=int(min_history),
                observation_start=pd.Timestamp(obs_start),
            )
        st.session_state["churn_dataset_obj"] = dataset
        st.session_state["churn_dataset"] = {
            "cutoff": str(dataset.labeling.cutoff.date()),
            "horizon_days": dataset.labeling.horizon_days,
            "churn_rate": round(float(dataset.frame["churn"].mean()), 4),
            "customers": len(dataset.frame),
            "excluded_short_history": dataset.labeling.excluded_short_history,
        }
        rate = dataset.frame["churn"].mean()
        metric_row(
            [
                Metric("Labeled customers", f"{len(dataset.frame):,}"),
                Metric("Churn rate", f"{rate:.1%}"),
                Metric(
                    "Excluded (short history)",
                    str(dataset.labeling.excluded_short_history),
                ),
                Metric(
                    "Predictors ≤ cutoff",
                    "proven" if dataset.leakage_ok else "LEAKED",
                    tone="positive" if dataset.leakage_ok else "negative",
                ),
            ]
        )
        if dataset.labeling.partial_window:
            st.warning(
                f"Outcome window only partially observed "
                f"({dataset.labeling.observed_outcome_days} of {horizon} days): "
                "labels near the window end undercount churn. Extend the data "
                "or shorten the horizon."
            )
        else:
            st.success(
                f"Predictors peak at {dataset.predictor_max_date.date()} "
                f"(cutoff {dataset.labeling.cutoff.date()}); labels from activity "
                f"through {dataset.labeling.outcome_end.date()}."
            )
    except (CustomerIntelligenceError, ValueError, KeyError) as exc:
        st.error(str(exc))

dataset = st.session_state.get("churn_dataset_obj")
if dataset is None:
    empty_state(
        "No churn dataset yet",
        "Configure the observation/cutoff/horizon and build the dataset.",
    )
    st.stop()

with controls_expander("Training Controls"):
    with control_group("Training"):
        availability = available_models()
        options = list(availability)
        train_choice = st.multiselect(
            "Models to train",
            options,
            default=["logreg", "random_forest"],
            help="LightGBM/XGBoost train only when installed.",
        )
        oversample = st.checkbox("Oversample minority (train only)", value=False)
        calibrate = st.checkbox("Calibrate probabilities (sigmoid on validation)", value=True)
for name in train_choice:
    if availability.get(name):
        st.warning(availability[name])

if st.button("Train selected models"):
    bundles: dict = {}
    progress = st.progress(0.0)
    for i, name in enumerate(train_choice):
        try:
            with st.spinner(f"Training {name}…"):
                bundles[name] = train_churn_model(
                    dataset, name, oversample=oversample, calibrate=calibrate
                )
        except (CustomerIntelligenceError, ValueError) as exc:
            st.error(f"{name}: {exc}")
        progress.progress((i + 1) / max(len(train_choice), 1))
    if bundles:
        st.session_state["churn_bundles"] = bundles
        best = max(bundles, key=lambda m: bundles[m]["test_metrics"]["roc_auc"] or -1)
        st.session_state["churn_model"] = bundles[best]["pipeline"]
        st.session_state["churn_model_metadata"] = {
            "model": best,
            "threshold": bundles[best]["suggested_threshold"],
            "metrics": bundles[best]["test_metrics"],
            "calibration": bundles[best]["calibration"],
            "cutoff": bundles[best]["cutoff"],
            "horizon_days": bundles[best]["horizon_days"],
        }
        st.success(f"Trained {len(bundles)} model(s). Best test ROC-AUC: {best}.")

bundles = st.session_state.get("churn_bundles")
if bundles:
    with st.expander("Test metrics (validation-max-F1 threshold)", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "model": name,
                        **{k: v for k, v in b["test_metrics"].items() if k != "confusion"},
                    }
                    for name, b in bundles.items()
                ]
            ),
            width="stretch",
        )
    with controls_expander("Model Detail"):
        detail = st.selectbox("Model detail", list(bundles))
        threshold = st.slider(
            "Decision threshold",
            0.05,
            0.95,
            float(bundles[detail]["suggested_threshold"]),
            0.05,
            help="Lower catches more churners with more false alarms; higher is stricter.",
        )
    bundle = bundles[detail]
    eval_l, eval_r = st.columns([1, 1])
    with eval_l:
        with chart_section("ROC curve"):
            st.plotly_chart(
                roc_curve_fig(bundle["test_y"], bundle["test_proba"]),
                width="stretch",
            )
    with eval_r:
        with chart_section("Calibration curve"):
            st.plotly_chart(
                calibration_curve_fig(calibration_table(bundle["test_y"], bundle["test_proba"])),
                width="stretch",
            )
    perf_l, perf_r = st.columns([1, 1])
    with perf_l:
        with chart_section("Confusion matrix"):
            st.plotly_chart(confusion_fig(bundle["test_metrics"]["confusion"]), width="stretch")
    with perf_r:
        with chart_section("Cumulative gain"):
            st.plotly_chart(
                gain_chart_fig(lift_gain_table(bundle["test_y"], bundle["test_proba"])),
                width="stretch",
            )
    with st.expander("Threshold metrics", expanded=False):
        st.write(
            classification_metrics(bundle["test_y"], bundle["test_proba"], threshold=threshold)
        )

    section_header("Explainability")
    with chart_section("Global feature importance"):
        st.plotly_chart(importance_bar_fig(bundle["importance"]), width="stretch")
    if detail == "logreg":
        with st.expander("Coefficients & odds ratios", expanded=False):
            st.dataframe(logreg_effects(bundle["pipeline"]), width="stretch")
    try:
        shap_frame = shap_values(bundle["pipeline"], dataset.frame[bundle["feature_cols"]])
        st.markdown("Mean |SHAP| (global, sample):")
        st.bar_chart(shap_frame.abs().mean().sort_values(ascending=False).head(15))
    except ImportError as exc:
        st.info(str(exc))

    section_header("Score all customers")
    st.markdown(
        "Scores current customers (features as of the latest transaction) with the "
        "trained pipeline. Bands follow `config.yaml` risk bands."
    )
    if st.button("Score customers", type="primary"):
        with st.spinner("Scoring…"):
            latest = txn_dates.max() + pd.Timedelta(days=1)
            scoring_features = build_customer_features(clean_data, snapshot_date=latest).features
            scores = score_all_customers(
                bundle,
                scoring_features,
                cfg["churn"]["risk_bands"],
                threshold,
                detail,
            )
        st.session_state["churn_scores"] = scores
        st.session_state["churn_model_metadata"] = {
            **st.session_state.get("churn_model_metadata", {}),
            "threshold": threshold,
            "model": detail,
        }
        path = save_churn_run(bundle, f"{detail}_{bundle['cutoff']}")
        st.success(f"Scored {len(scores)} customers. Model saved at {path}.")

scores = st.session_state.get("churn_scores")
if scores is not None:
    risk_l, risk_r = st.columns([1, 2])
    with risk_l:
        with chart_section("Customers per risk band"):
            st.plotly_chart(risk_band_bar(scores), width="stretch")
    with risk_r:
        with st.expander("Scored customers (top 200)", expanded=False):
            render_table(scores.head(200), risk_cols=("risk_band",))
    st.download_button(
        "Download churn scores (CSV)",
        data=to_csv_bytes(scores),
        file_name="churn_scores.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download model metadata (JSON)",
        data=to_json_bytes(st.session_state.get("churn_model_metadata", {})),
        file_name="churn_model_metadata.json",
        mime="application/json",
    )

render_sidebar_footer()

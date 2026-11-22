"""Report service: context, narratives, and multi-format exports."""

import io
import zipfile

import pandas as pd

from services.report_charts import render_report_charts
from services.report_service import (
    CHART_EXPLANATIONS,
    REPORT_SECTIONS,
    SECTION_CHARTS,
    build_docx_bytes,
    build_pdf_bytes,
    build_report_context,
    churn_details_table,
    churn_selection_table,
    context_tables,
    context_to_excel_bytes,
    render_html_report,
    report_readiness_lines,
)


def _sample_state() -> dict:
    ids = ["C00001", "C00002", "C00003"]
    features = pd.DataFrame(
        {
            "customer_id": ids,
            "monetary_value": [100.0, 400.0, 900.0],
            "frequency": [1, 3, 8],
            "recency_days": [90, 10, 5],
            "first_transaction_date": ["2024-01-01"] * 3,
        }
    )
    clean = pd.DataFrame(
        {
            "customer_id": ids,
            "transaction_date": ["2024-06-01", "2024-06-02", "2024-06-03"],
            "amount_spent": [100.0, 400.0, 900.0],
        }
    )
    rfm = pd.DataFrame({"customer_id": ids, "segment": ["Lost", "Loyal", "Champions"]})
    scores = pd.DataFrame(
        {
            "customer_id": ids,
            "churn_probability": [0.9, 0.2, 0.75],
            "risk_band": ["critical", "low", "high"],
        }
    )
    retention = pd.DataFrame(
        {
            "customer_id": ids,
            "churn_probability": [0.9, 0.2, 0.75],
            "risk_band": ["critical", "low", "high"],
            "customer_value_tier": ["High", "Low", "High"],
            "expected_value_proxy_historical": [100.0, 400.0, 900.0],
            "revenue_at_risk": [90.0, 80.0, 675.0],
            "retention_priority_score": [0.9, 0.1, 0.7],
            "recommended_action": ["VIP intervention", "Routine nurture", "VIP intervention"],
        }
    )
    return {
        "clean_data": clean,
        "customer_features": features,
        "rfm_results": rfm,
        "lrfm_results": None,
        "segmented_customers": None,
        "churn_scores": scores,
        "retention_results": {"table": retention.to_dict(orient="records"), "summary": {}},
        "churn_model_metadata": {
            "model": "rf",
            "threshold": 0.5,
            "cutoff": "2024-06-01",
            "candidates": [
                {
                    "model": "rf",
                    "roc_auc": 0.71,
                    "pr_auc": 0.62,
                    "f1": 0.55,
                    "recall": 0.4,
                    "threshold": 0.5,
                },
                {
                    "model": "logreg",
                    "roc_auc": 0.66,
                    "pr_auc": 0.58,
                    "f1": 0.52,
                    "recall": 0.35,
                    "threshold": 0.45,
                },
            ],
        },
        "cleaning_audit": {},
        "column_mapping": {},
    }


def test_build_report_context_flags_and_kpis() -> None:
    """Readiness flags and KPI/segment summaries derive from state."""
    context = build_report_context(_sample_state())
    assert context["readiness"]["rfm"] is True
    assert context["readiness"]["churn"] is True
    assert context["readiness"]["retention"] is True
    assert context["kpis"]["total_customers"] == 3
    assert context["segment_portfolio"] is not None
    assert context["risk_mix"]["critical"] == 1
    assert context["churn_selection"]["selected"] == "rf"
    assert context["churn_selection"]["n_candidates"] == 2


def test_churn_model_selection_reaches_every_export() -> None:
    """The scored model and its candidate comparison are reportable, not implicit."""
    context = build_report_context(_sample_state())
    selection = context["churn_selection"]
    assert selection["table"]["selected"].sum() == 1
    assert selection["table"].iloc[0]["model"] == "rf"
    assert "highest test ROC-AUC" in selection["rationale"]

    tables = context_tables(context, mask_ids=True)
    assert "Churn_Model_Selection" in tables
    assert set(tables["Churn_Model_Selection"]["Model"]) == {"rf", "logreg"}

    html_report = render_html_report(context, figures_html={}, mask_ids=True)
    assert "Model selection" in html_report
    assert "Candidate metrics are from the held-out test split" in html_report

    pdf = build_pdf_bytes(context, chart_images={})
    assert pdf[:4] == b"%PDF"

    docx = build_docx_bytes(context, chart_images={})
    archive = zipfile.ZipFile(io.BytesIO(docx))
    assert "Model selection" in archive.read("word/document.xml").decode("utf-8")


def test_churn_selection_table_flags_operational_pick() -> None:
    """The candidate table marks exactly the model used for scoring."""
    state = _sample_state()
    table = churn_selection_table(state["churn_model_metadata"])
    assert list(table["Model"]) == ["rf", "logreg"]  # best score first
    assert table["Operational pick"].eq("yes").sum() == 1
    assert table.iloc[0]["Operational pick"] == "yes"
    assert table.iloc[1]["Operational pick"] == ""


def test_near_chance_churn_selection_is_flagged() -> None:
    """A top ROC-AUC close to 0.5 is called out as prioritization-only signal."""
    state = _sample_state()
    state["churn_model_metadata"]["candidates"][0]["roc_auc"] = 0.51
    context = build_report_context(state)
    selection = context["churn_selection"]
    assert selection["near_chance"] is True
    assert "chance line" in selection["rationale"]


def test_empty_state_reports_pending() -> None:
    """Missing upstream results report pending instead of raising."""
    context = build_report_context({})
    assert context["readiness"]["features"] is False
    assert context["kpis"] == {}


def test_chart_explanations_have_no_causal_claims() -> None:
    """Narratives explain and direct effort without promising impact."""
    assert set(REPORT_SECTIONS) == {
        "summary",
        "overview",
        "segments",
        "comparison",
        "churn",
        "retention",
        "recommendations",
        "appendix",
    }
    for key, exp in CHART_EXPLANATIONS.items():
        assert {"title", "what", "how_to_read", "action"} <= set(exp), key
        text = " ".join(exp.values()).lower()
        for banned in ("will save", "guaranteed uplift", "will retain", "proven to lift"):
            assert banned not in text, key


def test_html_report_masks_ids_and_labels_proxy() -> None:
    """HTML export masks IDs by default and labels value as a proxy."""
    context = build_report_context(_sample_state())
    html_report = render_html_report(context, figures_html={}, mask_ids=True)
    assert "C•••" in html_report
    assert "never CLV" in html_report
    assert "Executive summary" in html_report
    assert "Segments produced" in html_report
    assert "Churn risk calculated" in html_report
    assert "Retention strategies covered" in html_report
    assert "Recommendations &amp; next steps" in html_report


def test_html_report_dark_skin_mirrors_console() -> None:
    """Dark HTML export uses the charcoal-and-lime skin; unknown modes stay light."""
    context = build_report_context(_sample_state())
    dark = render_html_report(context, figures_html={}, mask_ids=True, mode="dark")
    assert "#0c0f14" in dark and "#bef264" in dark
    assert "#1a2333" not in dark
    fallback = render_html_report(context, figures_html={}, mask_ids=True, mode="nope")
    assert "#1a2333" in fallback and "#bef264" not in fallback


def test_excel_tables_mask_ids() -> None:
    """Excel workbook masks raw identifiers when requested."""
    context = build_report_context(_sample_state())
    tables = context_tables(context, mask_ids=True)
    assert "Retention_Top" in tables
    assert all("•••" in str(v) for v in tables["Retention_Top"]["customer_id"])
    assert not any(v == "C00001" for v in tables["Retention_Top"]["customer_id"])
    data = context_to_excel_bytes(context, mask_ids=True)
    assert data[:2] == b"PK"


def test_chart_registry_covers_every_explained_chart() -> None:
    """Every chart the sections embed (and explain) has a static renderer."""
    explained = set(CHART_EXPLANATIONS)
    embedded = {key for keys in SECTION_CHARTS.values() for key in keys}
    assert embedded <= explained
    from services.report_charts import CHART_BUILDERS

    assert embedded <= set(CHART_BUILDERS)


def test_report_charts_render_png_bytes() -> None:
    """The static renderer returns PNGs for the charts the context can support."""
    context = build_report_context(_sample_state())
    charts = render_report_charts(context)
    assert {"revenue_trend", "risk_bands", "value_risk_matrix"} <= set(charts)
    for png in charts.values():
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_pdf_embeds_chart_images() -> None:
    """The PDF export carries the chart PNGs, not only text tables."""
    context = build_report_context(_sample_state())
    charts = render_report_charts(context)
    data = build_pdf_bytes(context, chart_images=charts)
    assert data[:4] == b"%PDF"
    assert b"/Subtype /Image" in data


def test_word_export_embeds_charts_and_closes_with_recommendations() -> None:
    """The Word export is a real .docx with chart media and the closing recap."""
    context = build_report_context(_sample_state())
    data = build_docx_bytes(context, chart_images=render_report_charts(context))
    assert data[:2] == b"PK"
    archive = zipfile.ZipFile(io.BytesIO(data))
    assert any(name.startswith("word/media/") for name in archive.namelist())
    document = archive.read("word/document.xml").decode("utf-8")
    assert "Interpretations at a glance" in document
    assert "Prioritized actions" in document
    assert document.index("Appendix — data &amp; method") < document.index(
        "Recommendations &amp; next steps"
    )


def test_html_closes_with_interpretations_and_recommendations() -> None:
    """The HTML report ends with the recap and actions, after the appendix."""
    context = build_report_context(_sample_state())
    html_report = render_html_report(context, figures_html={}, mask_ids=True)
    appendix_at = html_report.index("Appendix — data &amp; method")
    closing_at = html_report.index("Recommendations &amp; next steps")
    assert appendix_at < closing_at
    assert closing_at < html_report.index("Interpretations at a glance")
    assert closing_at < html_report.index("Prioritized actions")


def test_context_carries_foundation_and_narrative_inputs() -> None:
    """The context exposes the extra inputs the narrative and appendix need."""
    context = build_report_context(_sample_state())
    assert context["segment_portfolio"] is not None
    assert context["data_profile"]["rows"] == 3
    assert context["data_profile"]["customers"] == 3
    assert not context["trends"]["revenue"].empty
    assert context["retention_summary"]["customers"] == 3
    assert context["treatments"] == []
    assert "Monthly_Trends" in context_tables(context)


def _state_with_ai_runs() -> dict:
    """Sample state plus two saved AI runs, enough to build the comparison."""
    state = _sample_state()
    state["segmentation_runs"] = [
        {
            "model": "kmeans",
            "labels": [0, 0, 1],
            "customer_ids": ["C00001", "C00002", "C00003"],
            "names": {"0": "Core", "1": "Light"},
            "metrics": {
                "n_clusters": 2,
                "silhouette": 0.61,
                "davies_bouldin": 0.9,
                "noise_pct": 0.0,
            },
            "stability": {"mean_pairwise_ari": 0.95},
        },
        {
            "model": "gmm",
            "labels": [0, 1, 1],
            "customer_ids": ["C00001", "C00002", "C00003"],
            "names": {},
            "metrics": {
                "n_clusters": 2,
                "silhouette": 0.42,
                "davies_bouldin": 1.2,
                "noise_pct": 0.0,
            },
            "stability": {"mean_pairwise_ari": 0.88},
        },
    ]
    return state


def test_comparison_section_builds_from_ai_runs() -> None:
    """The report context carries the same scorecard page 06 shows."""
    context = build_report_context(_state_with_ai_runs())
    comparison = context["comparison"]
    assert comparison is not None
    models = set(comparison["table"]["model"])
    assert {"AI:kmeans", "AI:gmm", "RFM"} <= models
    assert comparison["recommendation"]["operational_pick"] in models
    assert comparison["pair"] is not None
    assert "Model_Scorecard" in context_tables(context)


def test_comparison_without_ai_runs_has_no_pairwise() -> None:
    """Traditional scoring alone still lists, but pairwise agreement needs an AI run."""
    context = build_report_context(_sample_state())
    comparison = context["comparison"]
    assert comparison is not None
    assert set(comparison["table"]["model"]) == {"RFM"}
    assert comparison["pair"] is None
    html_report = render_html_report(context, figures_html={}, mask_ids=True)
    assert "Model comparison" in html_report


def test_comparison_absent_with_nothing_to_compare() -> None:
    """With no features at all there is no comparison, and the report says so."""
    context = build_report_context({})
    assert context["comparison"] is None
    html_report = render_html_report(context, figures_html={}, mask_ids=True)
    assert "Model comparison" in html_report
    assert "No models to compare yet" in html_report


def test_comparison_scorecard_renders_in_every_export() -> None:
    """Page, HTML, PDF, Word, and Excel all carry the model comparison section."""
    context = build_report_context(_state_with_ai_runs())
    html_report = render_html_report(context, figures_html={}, mask_ids=True)
    assert "Model comparison" in html_report
    assert "Model scorecard" in html_report

    pdf = build_pdf_bytes(context, chart_images={})
    assert pdf[:4] == b"%PDF"
    docx = build_docx_bytes(context, chart_images={})
    archive = zipfile.ZipFile(io.BytesIO(docx))
    document = archive.read("word/document.xml").decode("utf-8")
    assert "Model comparison" in document
    assert "Model scorecard" in document

    excel = context_to_excel_bytes(context, mask_ids=True)
    workbook = pd.read_excel(io.BytesIO(excel), sheet_name=None)
    assert "Model_Scorecard" in workbook
    # The pairwise flow travels with the pack when a pair exists.
    assert "Model_Flow" in workbook


def test_comparison_scorecard_text_blanks_missing_metrics() -> None:
    """A blank silhouette becomes an empty cell, not 'nan', in text exports."""
    from services.comparison_service import build_comparison, scorecard_as_text

    comparison = build_comparison(_state_with_ai_runs())
    text = scorecard_as_text(comparison["table"])
    rfm_row = text.loc[text["model"] == "RFM"].iloc[0]
    assert rfm_row["silhouette"] == ""
    assert "nan" not in text.to_string().lower()


def _churn_bundle() -> dict:
    """A small trained-bundle stand-in with the evaluation artifacts reports read."""
    import numpy as np

    from churn.evaluation import classification_metrics

    rng = np.random.default_rng(42)
    y_true = np.array([0] * 20 + [1] * 20)
    proba = np.clip(
        np.concatenate([rng.normal(0.3, 0.1, 20), rng.normal(0.7, 0.1, 20)]), 0.01, 0.99
    )
    return {
        "model_name": "rf",
        "suggested_threshold": 0.45,
        "calibration": "sigmoid on validation",
        "cutoff": "2024-06-01",
        "horizon_days": 90,
        "splits": {"train": 60, "val": 15, "test": 40},
        "test_metrics": classification_metrics(y_true, proba, threshold=0.45),
        "val_metrics": classification_metrics(y_true, proba, threshold=0.45),
        "test_proba": proba,
        "test_y": y_true,
        "importance": pd.DataFrame(
            {
                "feature": ["recency_days", "frequency", "monetary"],
                "importance": [0.12, 0.05, 0.01],
            }
        ),
    }


def test_churn_evaluation_detail_reaches_context_and_exports() -> None:
    """Curves, confusion, importance, and model detail travel to every export."""
    state = _sample_state()
    bundle = _churn_bundle()
    state["churn_bundles"] = {"rf": bundle}
    state["churn_model_metadata"]["evaluation"] = {
        "model": "rf",
        "threshold": 0.45,
        "test_metrics": bundle["test_metrics"],
        "val_metrics": bundle["val_metrics"],
        "splits": bundle["splits"],
        "calibration": bundle["calibration"],
        "cutoff": bundle["cutoff"],
        "horizon_days": bundle["horizon_days"],
        "roc": [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 0.2, "tpr": 0.8}, {"fpr": 1.0, "tpr": 1.0}],
        "calibration_curve": [{"mean_predicted": 0.2, "observed_rate": 0.15, "count": 10}],
        "gain": [{"decile": 1, "cum_gain": 0.5}, {"decile": 2, "cum_gain": 1.0}],
        "importance": [
            {"feature": "recency_days", "importance": 0.12},
            {"feature": "frequency", "importance": 0.05},
        ],
    }
    context = build_report_context(state)
    evaluation = context["churn_evaluation"]
    assert evaluation["model"] == "rf"
    assert evaluation["roc"] is not None and not evaluation["roc"].empty
    assert evaluation["calibration_curve"] is not None
    assert evaluation["gain"] is not None
    assert evaluation["importance"] is not None
    assert evaluation["confusion"]["tp"] >= 0
    assert context["input_fingerprint"]

    details = context_tables(context)["Churn_Model_Details"]
    assert "Test ROC-AUC" in set(details["metric"])
    assert "Churn_Feature_Importance" in context_tables(context)

    html_report = render_html_report(context, figures_html={}, mask_ids=True)
    for title in (
        "ROC curve — churn model",
        "Calibration curve — churn model",
        "Confusion matrix — churn model",
        "Cumulative gain — churn model",
        "Global feature importance — churn model",
        "Scored model detail",
    ):
        assert title in html_report, title

    charts = render_report_charts(context)
    assert {
        "churn_roc",
        "churn_calibration",
        "churn_confusion",
        "churn_gain",
        "churn_importance",
    } <= set(charts)
    assert build_pdf_bytes(context, chart_images=charts)[:4] == b"%PDF"
    docx = build_docx_bytes(context, chart_images=charts)
    document = zipfile.ZipFile(io.BytesIO(docx)).read("word/document.xml").decode("utf-8")
    assert "Scored model detail" in document


def test_churn_evaluation_falls_back_to_cached_payload() -> None:
    """A session restored from cache keeps its churn charts without bundles."""
    from services.churn_service import evaluation_payload

    state = _sample_state()
    payload = evaluation_payload(_churn_bundle())
    state["churn_model_metadata"].update(
        {"metrics": payload["test_metrics"], "evaluation": payload, "horizon_days": 90}
    )
    context = build_report_context(state)
    evaluation = context["churn_evaluation"]
    assert evaluation["roc"] is not None
    assert "churn_roc" in render_report_charts(context)
    # Flat cached metrics still yield a detail block even without curve payloads.
    state["churn_model_metadata"].pop("evaluation")
    fallback = build_report_context(state)
    assert fallback["churn_evaluation"]["roc"] is None
    assert not churn_details_table(fallback).empty


def test_churn_details_flag_a_saturated_threshold() -> None:
    """A threshold that flags every test customer is called out beside the confusion."""
    state = _sample_state()
    state["churn_model_metadata"]["evaluation"] = {
        "model": "rf",
        "threshold": 0.05,
        "test_metrics": {
            "roc_auc": 0.51,
            "pr_auc": 0.86,
            "precision": 0.86,
            "recall": 1.0,
            "f1": 0.93,
            "balanced_accuracy": 0.5,
            "n": 40,
            "positive_rate": 0.875,
            "confusion": {"tn": 0, "fp": 5, "fn": 0, "tp": 35},
        },
    }
    details = churn_details_table(build_report_context(state))
    values = details.set_index("metric")["value"]
    assert "flags every test customer" in values["Threshold behaviour"]
    assert "TN 0" in values["Confusion (test)"]

    # Non-saturated thresholds keep the table free of the caveat row.
    state["churn_model_metadata"]["evaluation"]["test_metrics"]["confusion"] = {
        "tn": 4,
        "fp": 1,
        "fn": 2,
        "tp": 33,
    }
    clean = churn_details_table(build_report_context(state))
    assert "Threshold behaviour" not in set(clean["metric"])


def test_readiness_requires_churn_curves_not_just_metadata() -> None:
    """Flat metrics alone must not report the evaluation checklist as ready."""
    state = _sample_state()
    fresh = build_report_context(state)
    lines = {label: ready for label, ready, _ in report_readiness_lines(fresh)}
    assert lines["Churn model evaluation"] is False

    state["churn_model_metadata"]["evaluation"] = {
        "model": "rf",
        "threshold": 0.45,
        "test_metrics": {"roc_auc": 0.71, "precision": 0.6, "recall": 0.5, "n": 40},
        "roc": [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}],
    }
    lines = {
        label: ready for label, ready, _ in report_readiness_lines(build_report_context(state))
    }
    assert lines["Churn model evaluation"] is True


def test_report_fingerprint_tracks_input_changes() -> None:
    """The source revision changes when data, scores, or metadata change."""
    state = _sample_state()
    first = build_report_context(state)["input_fingerprint"]
    assert build_report_context(state)["input_fingerprint"] == first
    changed = dict(state)
    changed["churn_scores"] = state["churn_scores"].assign(churn_probability=[0.1, 0.2, 0.3])
    assert build_report_context(changed)["input_fingerprint"] != first
    repriced = dict(state)
    repriced["churn_model_metadata"] = {**state["churn_model_metadata"], "threshold": 0.61}
    assert build_report_context(repriced)["input_fingerprint"] != first

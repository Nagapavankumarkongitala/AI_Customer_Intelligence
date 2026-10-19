"""Report service: context, narratives, and multi-format exports."""

import pandas as pd

from services.report_service import (
    CHART_EXPLANATIONS,
    REPORT_SECTIONS,
    build_report_context,
    context_tables,
    context_to_excel_bytes,
    render_html_report,
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
        "churn_model_metadata": {"model": "rf", "threshold": 0.5, "cutoff": "2024-06-01"},
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

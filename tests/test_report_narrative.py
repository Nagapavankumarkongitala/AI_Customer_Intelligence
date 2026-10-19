"""Report narrative engine: insights, interpretation, and recommendations."""

import pandas as pd

from services.report_narrative import (
    PRIORITIES,
    TONES,
    recommendations,
    section_insights,
    summary_insights,
)

BANNED_CAUSAL = ("will save", "guaranteed", "proven to", "causes", "will reduce")


def _state() -> dict:
    ids = ["C00001", "C00002", "C00003", "C00004"]
    features = pd.DataFrame(
        {
            "customer_id": ids,
            "monetary_value": [100.0, 400.0, 900.0, 50.0],
            "frequency": [1, 3, 8, 1],
            "recency_days": [90, 10, 5, 200],
            "first_transaction_date": ["2024-01-01"] * 4,
            "gender": ["F", "M", "F", None],
            "age": [25, 40, 55, 30],
            "region": ["North", "South", "North", "East"],
        }
    )
    clean = pd.DataFrame(
        {
            "customer_id": ids * 2,
            "transaction_date": [
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
                "2024-05-01",
                "2024-06-01",
                "2024-07-01",
                "2024-08-01",
            ],
            "amount_spent": [100.0, 400.0, 900.0, 50.0, 20.0, 55.0, 60.0, 70.0],
            "region": ["North", "South", "North", "East"] * 2,
        }
    )
    rfm = pd.DataFrame({"customer_id": ids, "segment": ["Lost", "Loyal", "Champions", "Lost"]})
    scores = pd.DataFrame(
        {
            "customer_id": ids,
            "churn_probability": [0.9, 0.2, 0.75, 0.85],
            "risk_band": ["critical", "low", "high", "critical"],
        }
    )
    retention = pd.DataFrame(
        {
            "customer_id": ids,
            "churn_probability": [0.9, 0.2, 0.75, 0.85],
            "risk_band": ["critical", "low", "high", "critical"],
            "customer_value_tier": ["High", "Low", "High", "Low"],
            "expected_value_proxy_historical": [100.0, 400.0, 900.0, 50.0],
            "revenue_at_risk": [90.0, 80.0, 675.0, 42.5],
            "retention_priority_score": [0.9, 0.1, 0.7, 0.6],
            "recommended_action": [
                "VIP intervention",
                "Routine nurture",
                "VIP intervention",
                "Discount offer",
            ],
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
        "churn_model_metadata": {"model": "rf", "threshold": 0.5, "cutoff": "2024-04-01"},
        "cleaning_audit": {
            "input_rows": 10,
            "output_rows": 8,
            "rows_removed": 2,
            "rules": {"duplicates": {"action": "drop", "records_affected": 2}},
            "source_filename": "sample.csv",
        },
        "column_mapping": {},
        "data_dictionary": [],
    }


def _context() -> dict:
    from services.report_service import build_report_context

    return build_report_context(_state())


def test_summary_insights_are_actionable_and_tone_tagged() -> None:
    """The executive summary states observations with valid tones."""
    insights = summary_insights(_context())
    assert insights
    for item in insights:
        assert item.title and item.detail
        assert item.tone in TONES
    text = " ".join(f"{i.title} {i.detail}" for i in insights).lower()
    assert "concentrates" in text
    for banned in BANNED_CAUSAL:
        assert banned not in text


def test_section_insights_cover_every_labelled_section() -> None:
    """Each narrative section explains itself; unknown sections stay empty."""
    context = _context()
    for section in ("overview", "segments", "churn", "retention", "appendix"):
        items = section_insights(context, section)
        assert items, section
        for item in items:
            assert item.tone in TONES
    assert section_insights(context, "nope") == ()


def test_recommendations_are_ordered_and_evidence_linked() -> None:
    """Recommendations run Now -> Next -> Later and always carry evidence."""
    recs = recommendations(_context())
    assert recs
    order = [PRIORITIES.index(rec.priority) for rec in recs]
    assert order == sorted(order)
    for rec in recs:
        assert rec.priority in PRIORITIES
        assert rec.title and rec.rationale and rec.evidence
        assert rec.tone in TONES
    text = " ".join(f"{r.title} {r.rationale} {r.evidence}" for r in recs).lower()
    for banned in BANNED_CAUSAL:
        assert banned not in text
    assert any(rec.priority == "Now" for rec in recs)
    assert any(rec.priority == "Later" for rec in recs)


def test_narrative_survives_an_empty_context() -> None:
    """Missing results degrade to empty output, never an exception."""
    empty: dict = {}
    assert summary_insights(empty) == ()
    assert recommendations(empty) == ()
    for section in ("overview", "segments", "churn", "retention", "appendix"):
        assert section_insights(empty, section) == ()

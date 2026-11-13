"""AI segment explainers: tier parsing, profile maths, and card rendering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from components.segment_cards import segment_card_html
from services.segment_explain_service import (
    SegmentExplanation,
    build_segment_explanations,
    describe_tiers,
    parse_tiers,
)
from visualization import tokens


def test_compare_page_and_dashboard_render_the_shared_cards() -> None:
    """The explainers stay wired to both the comparison flow and Customer Trends."""
    page = Path("pages/06_Segmentation_Comparison.py").read_text(encoding="utf-8")
    app = Path("app.py").read_text(encoding="utf-8")
    assert "render_segment_cards(segment_notes)" in page
    assert "render_segment_cards(ai_segment_notes)" in app
    assert "Customer Trends" in app
    assert "build_segment_explanations" in page and "build_segment_explanations" in app


def _frame() -> pd.DataFrame:
    """Six named customers across three groups plus noise and a custom name."""
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
            "ai_segment": [
                "Premium Active A",
                "Premium Active A",
                "Premium Active A",
                "Core Dormant",
                "Core Dormant",
                "VIP Crew",
                "Noise",
                None,
            ],
            "monetary_value": [100.0, 200.0, 300.0, 50.0, 150.0, 400.0, 999.0, 500.0],
            "recency_days": [1.0, 2.0, 3.0, 100.0, 200.0, 10.0, 5.0, 5.0],
            "frequency": [4.0, 5.0, 6.0, 1.0, 2.0, 8.0, 3.0, 3.0],
        }
    )


def test_tiers_parse_collision_suffixes_and_degrade_for_custom_names() -> None:
    """Generated names keep their tiers; anything else is honestly unknown."""
    assert parse_tiers("Premium Active A") == ("Premium", "Active")
    assert parse_tiers("Core Dormant") == ("Core", "Dormant")
    assert parse_tiers("VIP Crew") == (None, None)
    assert parse_tiers(None) == (None, None)
    assert describe_tiers(None, None)[0].startswith("AI-defined")


def test_profile_excludes_noise_and_computes_shares_and_medians() -> None:
    """Shares use named groups only; medians are over each group's customers."""
    notes = build_segment_explanations(_frame())
    by_name = {note.name: note for note in notes}

    assert set(by_name) == {"Premium Active A", "Core Dormant", "VIP Crew"}
    premium = by_name["Premium Active A"]
    assert premium.customers == 3
    assert premium.customer_pct == 50.0
    assert premium.revenue == 600.0
    assert premium.revenue_pct == 50.0
    assert premium.median_recency_days == 2.0
    assert premium.median_frequency == 5.0
    assert premium.median_monetary_value == 200.0
    assert premium.tier_label == "Premium value · Active activity"

    dormant = by_name["Core Dormant"]
    assert dormant.customers == 2
    assert dormant.median_recency_days == 150.0
    assert dormant.median_frequency == 1.5
    assert dormant.median_monetary_value == 100.0

    custom = by_name["VIP Crew"]
    assert custom.tier_label == "AI-defined group"
    assert custom.summary.startswith("AI-defined customer group")


def test_cards_order_by_revenue_and_follow_the_contract() -> None:
    """The highest-revenue group is first, and percentages sum to 100."""
    notes = build_segment_explanations(_frame())
    assert [note.name for note in notes] == ["Premium Active A", "VIP Crew", "Core Dormant"]
    assert sum(note.customer_pct for note in notes) == 100.0
    assert sum(note.revenue_pct for note in notes) == 100.0


def test_empty_and_missing_frames_return_no_cards() -> None:
    """Degrade quietly when AI segmentation has not produced labels."""
    assert build_segment_explanations(None) == []
    assert build_segment_explanations(pd.DataFrame()) == []
    assert build_segment_explanations(pd.DataFrame({"customer_id": ["C1"]})) == []


def test_card_html_escapes_names_and_uses_the_group_and_mode_tokens() -> None:
    """User-supplied names cannot inject markup; accents follow the segment."""
    note = SegmentExplanation(
        name="<script>alert(1)</script>",
        value_tier=None,
        activity_tier=None,
        summary="A group.",
        action="Read it.",
        customers=2,
        customer_pct=25.0,
        revenue=100.0,
        revenue_pct=10.0,
        median_recency_days=12.0,
        median_frequency=3.0,
        median_monetary_value=50.0,
    )
    out = segment_card_html(note)

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
    assert "<script>" not in out
    assert "AI-defined group" in out and "A group." in out and "Read it." in out
    assert tokens.segment_color("<script>alert(1)</script>") in out
    assert tokens.color("text") in out and tokens.RADIUS["lg"] in out
    assert tokens.shadow("card") in out

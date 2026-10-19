"""Explainer coverage: the glossaries must cover what the pages actually render.

Page 06 charts eight scorecard columns and page 05 exposes six model controls;
each set is documented in a component (``components.metric_guides`` and
``components.control_guides``). These tests pin the copy to the pages' own
column and label lists, so a rename cannot silently leave a metric or a control
undocumented, and they check the card markup stays token-driven and escaped.
"""

from __future__ import annotations

import re
from pathlib import Path

from components.control_guides import (
    CONTROL_HELP,
    MODEL_CONTROL_USAGE,
    RUN_HELP,
    SWEEP_HELP,
    TRAIN_ALL_HELP,
    control_help,
    control_usage_rows,
)
from components.metric_guides import (
    DEFAULT_COLUMNS,
    METRIC_GUIDES,
    MetricGuide,
    metric_guide_html,
)
from visualization import tokens

COMPARE_PAGE = Path("pages/06_Segmentation_Comparison.py")
SEGMENT_PAGE = Path("pages/05_Advanced_Segmentation.py")

#: The eight evidence columns the page-06 scorecard emits.
SCORECARD_COLUMNS = (
    "silhouette",
    "davies_bouldin",
    "stability_ari",
    "value_eta2",
    "min_share",
    "evenness",
    "noise_pct",
    "interpretability_passes",
)


def _page_model_names() -> list[str]:
    """The model registry names page 05 offers, read from the page source."""
    block = re.search(r"model_names = \[([^\]]+)\]", SEGMENT_PAGE.read_text(encoding="utf-8"))
    assert block is not None, "page 05 no longer builds a model_names list"
    return [name.strip().strip('"') for name in block.group(1).split(",")]


# --------------------------------------------------------------------------- #
# Scorecard glossary (page 06).
# --------------------------------------------------------------------------- #


def test_glossary_covers_every_scorecard_column() -> None:
    """The cards explain exactly the columns the scorecard emits, in order."""
    assert tuple(guide.column for guide in METRIC_GUIDES) == SCORECARD_COLUMNS
    assert len(METRIC_GUIDES) % DEFAULT_COLUMNS == 0


def test_glossary_cards_are_directional_and_documented() -> None:
    """Every card names a direction, a scale, and a real use for the metric."""
    for guide in METRIC_GUIDES:
        assert guide.direction in {"higher", "lower"}
        assert guide.title and guide.scale
        assert len(guide.what) >= 80, guide.column
        assert len(guide.how) >= 80, guide.column


def test_compare_page_emits_and_explains_every_documented_column() -> None:
    """The page still scores each column and renders the glossary for it."""
    source = COMPARE_PAGE.read_text(encoding="utf-8")
    assert "render_metric_guides()" in source
    assert "How to read the scorecard" in source
    for guide in METRIC_GUIDES:
        assert f'"{guide.column}"' in source, f"{guide.column} is no longer scored on page 06"


def test_metric_card_html_escapes_and_paints_from_tokens() -> None:
    """Card markup escapes user content and draws every colour from tokens."""
    guide = MetricGuide(
        column="<b>silhouette</b>",
        title="A & B",
        direction="lower",
        scale="0 … 1",
        what="What it measures.",
        how="How it is used.",
    )
    out = metric_guide_html(guide)

    assert "&lt;b&gt;silhouette&lt;/b&gt;" in out and "A &amp; B" in out
    assert "<b>" not in out
    assert tokens.color("primary") in out and tokens.color("text") in out
    badge_bg, badge_fg = tokens.tone_badge("info")
    assert badge_bg in out and badge_fg in out
    assert tokens.RADIUS["lg"] in out and tokens.shadow("card") in out


def test_metric_cards_follow_the_active_mode() -> None:
    """Card ink resolves per mode, so a dark re-skin cannot leave light ink."""
    original = tokens.active_mode()
    try:
        tokens.set_active_mode(tokens.MODE_DARK)
        dark = metric_guide_html(METRIC_GUIDES[0])
        assert tokens.DARK["text"] in dark
        assert tokens.LIGHT["text"] not in dark
        assert tokens.DARK["surface"] in dark
    finally:
        tokens.set_active_mode(original)
    assert tokens.LIGHT["text"] in metric_guide_html(METRIC_GUIDES[0])


def test_unknown_direction_degrades_to_the_info_pill() -> None:
    """A direction typo shows through as text instead of raising mid-page."""
    guide = MetricGuide(
        column="mystery", title="Mystery", direction="sideways", scale="–", what="w", how="h"
    )
    out = metric_guide_html(guide)
    assert tokens.tone_badge("info")[1] in out and "sideways" in out


# --------------------------------------------------------------------------- #
# Model-control explainers (page 05).
# --------------------------------------------------------------------------- #


def test_every_model_control_is_explained() -> None:
    """Page 05 asks for help exactly for the controls the module documents."""
    source = SEGMENT_PAGE.read_text(encoding="utf-8")
    used = re.findall(r'control_help\("([^"]+)"\)', source)
    assert used, "page 05 no longer wires control_help into its widgets"
    assert set(used) == set(CONTROL_HELP)
    assert len(used) == len(CONTROL_HELP)  # each control carries its own tooltip
    for label in CONTROL_HELP:
        assert f'"{label}"' in source


def test_control_copy_is_actionable() -> None:
    """Each explanation is a sentence, not a placeholder."""
    for text in (*CONTROL_HELP.values(), SWEEP_HELP, RUN_HELP, TRAIN_ALL_HELP):
        assert len(text) >= 80
        assert text.endswith(".")


def test_control_help_lookup_degrades_quietly() -> None:
    """An unknown label yields no tooltip rather than a KeyError mid-page."""
    assert control_help("Model") == CONTROL_HELP["Model"]
    assert control_help("not-a-control") == ""


def test_usage_matrix_matches_the_models_the_page_can_fit() -> None:
    """The matrix covers every model the page offers, in the same order."""
    offered = _page_model_names()
    assert [row["model"] for row in MODEL_CONTROL_USAGE] == offered
    assert all(row["cluster count"] and row["extra controls"] for row in MODEL_CONTROL_USAGE)


def test_usage_rows_are_defensive_copies() -> None:
    """Handing the matrix to Streamlit cannot mutate the module constant."""
    rows = control_usage_rows()
    assert rows == [dict(row) for row in MODEL_CONTROL_USAGE]
    rows[0]["model"] = "mutated"
    assert MODEL_CONTROL_USAGE[0]["model"] == "kmeans"


def test_segment_page_renders_the_intro_and_matrix() -> None:
    """The explanations are visible in the controls, not only in tooltips."""
    source = SEGMENT_PAGE.read_text(encoding="utf-8")
    assert "st.caption(INTRO)" in source
    assert "control_usage_rows()" in source and "st.caption(USAGE_CAPTION)" in source

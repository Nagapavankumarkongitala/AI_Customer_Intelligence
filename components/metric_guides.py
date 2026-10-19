"""Scorecard glossary cards: how to read each column of the model comparison.

Page 06 ranks segmentations on eight evidence columns, and every one of them
carries its own direction, range, and gate. These cards render that reading
guide next to the comparison charts so nobody has to reverse-engineer the
scorecard: what a column measures, which way is better, and exactly how the
recommendation uses it — including the thresholds that qualify a run for the
gated pool.

The copy mirrors ``analytics/segmentation_comparison.py`` (gate constants,
interpretability checklist, and ranking order). If a threshold changes there,
change it here too; ``tests/test_metric_guides.py`` pins the coverage of the
eight columns the scorecard emits.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st

from visualization import tokens

#: Cards per row: four lays the glossary out as two tidy rows of eight.
DEFAULT_COLUMNS = 4

#: Direction wording and pill tone — the only colour claim a card makes.
DIRECTION_TONES = {"higher": "positive", "lower": "info"}
DIRECTION_LABELS = {"higher": "Higher is better", "lower": "Lower is better"}


@dataclass(frozen=True)
class MetricGuide:
    """One scorecard column: what it measures and how the recommendation uses it."""

    column: str
    title: str
    direction: str
    scale: str
    what: str
    how: str


METRIC_GUIDES: tuple[MetricGuide, ...] = (
    MetricGuide(
        column="silhouette",
        title="Cluster separation",
        direction="higher",
        scale="−1 … 1 (0 ≈ chance)",
        what=(
            "How much closer each customer sits to its own cluster than to the nearest "
            "other one, averaged over the run. Points flagged as noise are excluded."
        ),
        how=(
            "Comparable only among AI runs sharing one feature space, which is why "
            "RFM/LRFM are blank by design. It crowns the best statistical separation "
            "line and breaks ties inside the gated pool, but never picks the model alone."
        ),
    ),
    MetricGuide(
        column="davies_bouldin",
        title="Cluster compactness ratio",
        direction="lower",
        scale="0 … unbounded (0 = clean)",
        what=(
            "Average ratio of within-cluster spread to the distance between clusters. "
            "Lower means tighter, better-separated clusters; there is no fixed good "
            "value, so read it against the other candidates."
        ),
        how=(
            "A second statistical opinion plotted beside silhouette. It gates nothing "
            "and crowns nothing, so a run can lead on one of the two and trail on the "
            "other without being wrong."
        ),
    ),
    MetricGuide(
        column="stability_ari",
        title="Seed stability",
        direction="higher",
        scale="0 … 1 (1 = identical)",
        what=(
            "Mean pairwise Adjusted Rand Index between refits that differ only in random "
            "seed: 1.0 means the same partition every time, 0 means the groupings are "
            "unrelated between runs."
        ),
        how=(
            "Gate for the operational pick — a run must reach 0.5 to enter the gated "
            "pool — and the interpretability checklist asks for 0.7 or better. RFM/LRFM "
            "report 1.0 by construction because their rules are deterministic."
        ),
    ),
    MetricGuide(
        column="value_eta2",
        title="Revenue differentiation",
        direction="higher",
        scale="0 … 1",
        what=(
            "Share of monetary-value variance explained by segment membership (eta "
            "squared): how much of the spend spread the segmentation accounts for."
        ),
        how=(
            "Crowns the best revenue differentiation line in the recommendation. "
            "Historical monetary value is a value proxy, not CLV, and a high score does "
            "not mean the segments cause the spending."
        ),
    ),
    MetricGuide(
        column="min_share",
        title="Smallest segment share",
        direction="higher",
        scale="0 … 1 (5% gate)",
        what=(
            "Customers in the smallest segment as a share of the whole base — the guard "
            "against an orphan cluster too small to act on."
        ),
        how=(
            "Gate for the operational pick: the smallest segment must stay at 5% of the "
            "base or more to enter the gated pool. The interpretability checklist fails "
            "on the same threshold."
        ),
    ),
    MetricGuide(
        column="evenness",
        title="Size balance",
        direction="higher",
        scale="0 … 1 (1 = all equal)",
        what=(
            "Normalised entropy of the segment sizes: 1.0 when every segment holds the "
            "same number of customers, 0 when one segment holds everyone."
        ),
        how=(
            "Read-only diagnostic — it gates nothing and crowns nothing. Use it to "
            "interpret the size and revenue charts: an even split is not automatically "
            "the better one for the business."
        ),
    ),
    MetricGuide(
        column="noise_pct",
        title="Unassigned (noise) share",
        direction="lower",
        scale="0 … 100% (< 15% gate)",
        what=(
            "Customers the fit left unlabelled as noise. Only the density-based HDBSCAN "
            "leaves points unassigned; every other model labels the whole base."
        ),
        how=(
            "Gate for the operational pick: noise must stay below 15% to enter the gated "
            "pool. The interpretability checklist is stricter still, at below 10%."
        ),
    ),
    MetricGuide(
        column="interpretability_passes",
        title="Interpretability checks",
        direction="higher",
        scale="0 … 5 checks",
        what=(
            "How many of the five human-facing checks the run clears: at most 8 "
            "segments, business names assigned, noise below 10%, stable across seeds "
            "(ARI ≥ 0.7), and a smallest segment of at least 5%."
        ),
        how=(
            "Scores the operational pick first, with silhouette breaking ties, inside "
            "the gated pool — evidence and business readability count before raw "
            "separation. The Interpretability checklists section shows the per-run detail."
        ),
    ),
)


def metric_guide_html(guide: MetricGuide) -> str:
    """One glossary card: column, plain name, direction pill, range, and its use."""
    accent = tokens.color("primary")
    text = tokens.color("text")
    muted = tokens.color("muted")
    border = tokens.color("border")
    surface = tokens.color("surface")
    tone = DIRECTION_TONES.get(guide.direction, "info")
    badge_bg, badge_fg = tokens.tone_badge(tone)
    label = DIRECTION_LABELS.get(guide.direction, guide.direction)
    return (
        "<div style='"
        f"background:{surface};border:1px solid {border};border-top:3px solid {accent};"
        f"border-radius:{tokens.RADIUS['lg']};box-shadow:{tokens.shadow('card')};"
        "padding:0.85rem 0.95rem;display:flex;flex-direction:column;gap:0.35rem;height:100%;'>"
        f"<div style='font-size:0.72rem;font-weight:600;letter-spacing:0.03em;"
        f"color:{accent};overflow-wrap:break-word;'>{html.escape(guide.column)}</div>"
        f"<div style='font-weight:700;font-size:0.96rem;color:{text};"
        f"overflow-wrap:break-word;'>{html.escape(guide.title)}</div>"
        "<div style='display:flex;align-items:center;gap:0.45rem;flex-wrap:wrap;'>"
        f"<span style='background:{badge_bg};color:{badge_fg};"
        f"border-radius:{tokens.RADIUS['pill']};padding:0.1rem 0.5rem;font-size:0.68rem;"
        f"font-weight:600;white-space:nowrap;'>{html.escape(label)}</span>"
        f"<span style='font-size:0.7rem;color:{muted};'>{html.escape(guide.scale)}</span></div>"
        f"<div style='font-size:0.8rem;color:{muted};"
        f"overflow-wrap:break-word;'>{html.escape(guide.what)}</div>"
        f"<div style='margin-top:auto;border-top:1px solid {border};padding-top:0.4rem;"
        f"font-size:0.78rem;color:{muted};overflow-wrap:break-word;'>"
        f"<strong style='color:{text};'>How it is used</strong> {html.escape(guide.how)}</div>"
        "</div>"
    )


def render_metric_guides(
    guides: tuple[MetricGuide, ...] = METRIC_GUIDES,
    width: int = DEFAULT_COLUMNS,
) -> None:
    """Render the glossary as rows of cards (Streamlit sink)."""
    for start in range(0, len(guides), width):
        row = guides[start : start + width]
        columns = st.columns(len(row))
        for column, guide in zip(columns, row, strict=True):
            with column:
                st.markdown(metric_guide_html(guide), unsafe_allow_html=True)

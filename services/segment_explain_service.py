"""Plain-language explanations for AI-generated segment groups.

The clustering run names every non-noise cluster on the value × activity grid in
``segmentation.segment_naming.GRID`` (Premium/Core/Light × Active/Steady/Dormant,
with a letter suffix when two clusters collide). Those names are business labels,
not descriptions: a reader still has to know that "Premium" means relative
monetary value, "Active" means recent/regular buying, and what to do with each
combination.

This module turns a labelled customer frame into one explanation per group, with
the live size, revenue contribution, and median behaviour alongside the copy. It
is shared by page 06 (under the traditional → AI customer flow) and the
dashboard's Customer Trends tab so a group reads the same in both places.

Tiers come from the name because the saved run metadata does not carry them. A
custom name therefore degrades to a neutral explanation instead of inventing a
tier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

#: Collision suffix appended by ``segmentation.segment_naming.propose_names``.
_COLLISION_SUFFIX = re.compile(r" [A-Z]$")

#: Labels that are not actionable business groups.
_NON_GROUPS = {"noise", "unsegmented", "nan", "none", ""}

#: (value tier, activity tier) -> (what the group is, what to do next).
_GRID_EXPLANATIONS: dict[tuple[str, str], tuple[str, str]] = {
    ("Premium", "Active"): (
        "High-value customers who bought recently and often.",
        "Protect the relationship with recognition, early access, and loyalty rewards.",
    ),
    ("Premium", "Steady"): (
        "High-value customers with moderate recency and frequency.",
        "Keep them engaged before they drift: targeted cross-sell and service check-ins.",
    ),
    ("Premium", "Dormant"): (
        "High-value customers who have gone quiet.",
        "Prioritise personal win-back; their historical value justifies the outreach.",
    ),
    ("Core", "Active"): (
        "Mid-value customers who bought recently and often.",
        "Grow frequency and basket size with relevant cross-sell.",
    ),
    ("Core", "Steady"): (
        "Mid-value customers with moderate recency and frequency.",
        "Nurture with relevant offers and watch for signs of drift.",
    ),
    ("Core", "Dormant"): (
        "Mid-value customers who have gone quiet.",
        "Use low-cost reactivation before they lapse further.",
    ),
    ("Light", "Active"): (
        "Lower-value customers who bought recently and often.",
        "Build the habit with low-cost engagement; avoid heavy incentives.",
    ),
    ("Light", "Steady"): (
        "Lower-value customers with moderate recency and frequency.",
        "Automate nurture and keep acquisition costs low.",
    ),
    ("Light", "Dormant"): (
        "Lower-value customers who have gone quiet.",
        "Use low-cost re-engagement or accept the lapse.",
    ),
}

_DEFAULT_EXPLANATION = (
    "AI-defined customer group with a custom or unrecognised name.",
    "Read its size, revenue share, and median behaviour before choosing an action.",
)


@dataclass(frozen=True)
class SegmentExplanation:
    """One AI segment with its profile and a plain-language reading."""

    name: str
    value_tier: str | None
    activity_tier: str | None
    summary: str
    action: str
    customers: int
    customer_pct: float
    revenue: float
    revenue_pct: float
    median_recency_days: float | None
    median_frequency: float | None
    median_monetary_value: float | None

    @property
    def tier_label(self) -> str:
        """Human tier line, or a neutral label when the name is custom."""
        if self.value_tier and self.activity_tier:
            return f"{self.value_tier} value · {self.activity_tier} activity"
        return "AI-defined group"


def parse_tiers(name: object) -> tuple[str | None, str | None]:
    """Read the value/activity tiers from a generated segment name.

    Collision suffixes ("Premium Active A") are ignored. Anything that is not a
    grid name — including a user's custom name — returns ``(None, None)``.
    """
    base = _COLLISION_SUFFIX.sub("", str(name).strip())
    for value_tier in ("Premium", "Core", "Light"):
        for activity_tier in ("Active", "Steady", "Dormant"):
            if base == f"{value_tier} {activity_tier}":
                return value_tier, activity_tier
    return None, None


def describe_tiers(value_tier: str | None, activity_tier: str | None) -> tuple[str, str]:
    """(summary, action) for a tier pair, falling back to neutral copy."""
    if value_tier and activity_tier:
        return _GRID_EXPLANATIONS[(value_tier, activity_tier)]
    return _DEFAULT_EXPLANATION


def _median(group: pd.DataFrame, column: str) -> float | None:
    """Median of a numeric column, or None when the column is absent/blank."""
    if column not in group.columns:
        return None
    numeric = pd.to_numeric(group[column], errors="coerce")
    value = numeric.median()
    return None if pd.isna(value) else float(value)


def build_segment_explanations(
    frame: pd.DataFrame,
    segment_col: str = "ai_segment",
    monetary_col: str = "monetary_value",
    recency_col: str = "recency_days",
    frequency_col: str = "frequency",
) -> list[SegmentExplanation]:
    """Profile each named AI group and attach its explanation.

    Noise and blank labels are excluded: they are not actionable groups. Shares
    use the named groups as the base, so customer and revenue percentages sum to
    100% across the cards. Groups are ordered by revenue, largest first.
    """
    empty: list[SegmentExplanation] = []
    if frame is None or frame.empty or segment_col not in frame.columns:
        return empty

    work = frame.copy()
    work[segment_col] = work[segment_col].astype("string").str.strip()
    work = work[work[segment_col].notna()]
    work = work[~work[segment_col].str.lower().isin(_NON_GROUPS)]
    if work.empty:
        return empty

    work["_revenue"] = (
        pd.to_numeric(work[monetary_col], errors="coerce").fillna(0.0)
        if monetary_col in work.columns
        else 0.0
    )
    total_customers = len(work)
    total_revenue = float(work["_revenue"].sum())

    explanations: list[SegmentExplanation] = []
    for name, group in work.groupby(segment_col, sort=False):
        label = str(name)
        value_tier, activity_tier = parse_tiers(label)
        summary, action = describe_tiers(value_tier, activity_tier)
        revenue = float(group["_revenue"].sum())
        explanations.append(
            SegmentExplanation(
                name=label,
                value_tier=value_tier,
                activity_tier=activity_tier,
                summary=summary,
                action=action,
                customers=int(len(group)),
                customer_pct=(len(group) / total_customers * 100.0 if total_customers else 0.0),
                revenue=revenue,
                revenue_pct=(revenue / total_revenue * 100.0 if total_revenue else 0.0),
                median_recency_days=_median(group, recency_col),
                median_frequency=_median(group, frequency_col),
                median_monetary_value=_median(group, monetary_col),
            )
        )

    explanations.sort(key=lambda item: (-item.revenue, -item.customers, item.name))
    return explanations


__all__ = [
    "SegmentExplanation",
    "build_segment_explanations",
    "describe_tiers",
    "parse_tiers",
]

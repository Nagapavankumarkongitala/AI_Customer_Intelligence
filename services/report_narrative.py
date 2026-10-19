"""Report narrative engine: the meaning layer over the computed results.

``services.report_service.build_report_context`` collects numbers and tables;
this module turns that context into the three things a report reader actually
needs on top of the numbers:

* :func:`summary_insights` — the executive hook: what the whole dataset says.
* :func:`section_insights` — the interpretation attached to each report
  section, so a chart is never left to speak for itself.
* :func:`recommendations` — prioritized, evidence-linked next actions.

Every function is pure and derives only from already-computed context values.
The language rules are strict and mirror the rest of the app: historical
monetary value is a *value proxy*, never CLV, and nothing here claims that an
action or a model *causes* an outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from utils import formatting

#: Tone vocabulary shared with ``components.insight_box``.
TONES = ("info", "positive", "warning", "critical")

#: Recommendation horizons, in display order.
PRIORITIES = ("Now", "Next", "Later")


@dataclass(frozen=True)
class Insight:
    """One observation plus its business consequence in plain words."""

    title: str
    detail: str
    tone: str = "info"


@dataclass(frozen=True)
class Recommendation:
    """A prioritized next action with the evidence that produced it."""

    priority: str
    title: str
    rationale: str
    evidence: str
    tone: str = "info"


def _tone(tone: str) -> str:
    """Coerce an unknown tone to the safe default."""
    return tone if tone in TONES else "info"


def _frame(context: dict[str, Any], key: str) -> pd.DataFrame | None:
    """Return a non-empty DataFrame from the context, else None."""
    value = context.get(key)
    return value if isinstance(value, pd.DataFrame) and not value.empty else None


def _top_row(frame: pd.DataFrame | None, by: str, ascending: bool = False) -> pd.Series | None:
    """Top row of ``frame`` by ``by``, or None when the column is absent."""
    if frame is None or by not in frame.columns:
        return None
    return frame.sort_values(by, ascending=ascending).iloc[0]


def summary_insights(context: dict[str, Any]) -> tuple[Insight, ...]:
    """Executive-summary observations, ordered by business importance."""
    insights: list[Insight] = []
    kpis = context.get("kpis") or {}
    trends = context.get("trends") or {}
    portfolio = _frame(context, "segment_portfolio")
    ai_portfolio = _frame(context, "ai_portfolio")
    hvhr = _frame(context, "hvhr")
    summary = context.get("retention_summary") or {}
    profile = context.get("data_profile") or {}

    customers = kpis.get("total_customers")
    if customers:
        insights.append(
            Insight(
                f"{formatting.fmt_int(customers)} customers · "
                f"{formatting.fmt_money(kpis.get('total_revenue'))} revenue",
                "Average order value "
                f"{formatting.fmt_money(kpis.get('average_order_value'))} across "
                f"{formatting.fmt_int(context.get('n_transactions'))} transactions — the historical "
                "baseline every later section qualifies.",
            )
        )

    top_segment = _top_row(portfolio, "revenue_share")
    if top_segment is not None:
        share = float(top_segment["revenue_share"])
        insights.append(
            Insight(
                f"{top_segment['rfm_segment']} concentrates {share:.1f}% of historical revenue",
                f"It holds {formatting.fmt_int(top_segment['customers'])} customers. When revenue "
                "share outruns customer share, value sits in comparatively few accounts.",
                tone=_tone("warning" if share >= 50 else "info"),
            )
        )

    rate = kpis.get("predicted_churn_rate")
    rar = kpis.get("revenue_at_risk")
    if rate is not None or rar is not None:
        bits = []
        if rate is not None:
            bits.append(f"{formatting.fmt_pct(rate)} predicted churn")
        if rar is not None:
            bits.append(f"{formatting.fmt_money(rar)} revenue at risk")
        insights.append(
            Insight(
                "Modeled churn exposure — " + " · ".join(bits),
                "Revenue at risk is modeled probability × historical value: exposure to plan for, "
                "not a confirmed loss.",
                tone=_tone("critical" if (rar or 0) > 0 else "info"),
            )
        )

    if hvhr is not None:
        insights.append(
            Insight(
                f"{len(hvhr)} high-value customers sit in high or critical risk bands",
                "These accounts carry the most historical value at modeled risk and are the first "
                "candidates for VIP outreach.",
                tone="critical",
            )
        )

    revenue_trend = trends.get("revenue")
    if isinstance(revenue_trend, pd.DataFrame) and len(revenue_trend) >= 2:
        first = float(revenue_trend["revenue"].iloc[0])
        last = float(revenue_trend["revenue"].iloc[-1])
        pct = (last - first) / first * 100 if first else 0.0
        up = last >= first
        insights.append(
            Insight(
                f"Revenue is {'up' if up else 'down'} {abs(pct):.1f}% from "
                f"{revenue_trend['month'].iloc[0]} to {revenue_trend['month'].iloc[-1]}",
                "Read momentum alongside the cohort and segment mix; a single month is not a trend.",
                tone=_tone("positive" if up else "warning"),
            )
        )

    covered = summary.get("customers")
    if covered and customers:
        pct = covered / customers * 100
        insights.append(
            Insight(
                f"{formatting.fmt_int(covered)} of {formatting.fmt_int(customers)} customers carry a "
                "recommended retention action",
                f"{pct:.0f}% coverage. Customers outside every strategy rule are invisible to a "
                "campaign export, so the gap is a reach decision.",
                tone=_tone("info" if pct >= 80 else "warning"),
            )
        )

    biggest_ai = _top_row(ai_portfolio, "customers")
    if biggest_ai is not None:
        insights.append(
            Insight(
                f"AI clustering splits the base into {len(ai_portfolio)} groups; largest is "
                f"{biggest_ai['ai_segment']}",
                "Cluster IDs are arbitrary — read each group by its profile against population "
                "medians, never by its number.",
            )
        )

    if profile.get("rows_in"):
        removed = int(profile.get("rows_removed") or 0)
        pct = removed / int(profile["rows_in"]) * 100
        insights.append(
            Insight(
                f"Cleaning removed {formatting.fmt_int(removed)} of "
                f"{formatting.fmt_int(profile['rows_in'])} source rows ({pct:.1f}%)",
                "No row is dropped silently: every treatment is recorded in the audit, and unmapped "
                "columns are listed there too.",
                tone=_tone("info" if pct < 5 else "warning"),
            )
        )
    return tuple(insights)


def _overview_insights(context: dict[str, Any]) -> tuple[Insight, ...]:
    """Business-performance interpretation: engagement and cohort mix."""
    out: list[Insight] = []
    kpis = context.get("kpis") or {}
    trends = context.get("trends") or {}
    active = kpis.get("active_customer_rate")
    if active is not None:
        out.append(
            Insight(
                f"{formatting.fmt_pct(active)} of customers were active in the last 90 days of "
                "the window",
                "Active rate is the share with at least one transaction in the trailing window; it "
                "reads as engagement, not loyalty.",
                tone=_tone("positive" if active >= 0.5 else "warning"),
            )
        )
    cohort = trends.get("cohort")
    if isinstance(cohort, pd.DataFrame) and not cohort.empty:
        new = int(cohort["new_customers"].sum())
        returning = int(cohort["returning_customers"].sum())
        total = new + returning
        new_pct = new / total * 100 if total else 0.0
        out.append(
            Insight(
                f"{new_pct:.0f}% of monthly active customers were new "
                f"({formatting.fmt_int(new)} of {formatting.fmt_int(total)})",
                "A high new share signals acquisition-led growth; a high returning share signals "
                "the existing base is doing the repeat work.",
            )
        )
    return tuple(out)


def _segment_insights(context: dict[str, Any]) -> tuple[Insight, ...]:
    """Segment interpretation: concentration, volume-vs-value, and demographics."""
    out: list[Insight] = []
    portfolio = _frame(context, "segment_portfolio")
    if portfolio is not None and {"customers", "revenue_share"} <= set(portfolio.columns):
        biggest = _top_row(portfolio, "customers")
        if biggest is not None:
            head_share = float(biggest["customers"]) / float(portfolio["customers"].sum()) * 100
            rev_share = float(biggest["revenue_share"])
            gap = head_share - rev_share
            tail = (
                "A wide customer-minus-revenue gap is a volume-without-value pattern: grow it with "
                "low-cost nurture rather than expensive outreach."
                if gap >= 5
                else "Its revenue share tracks its headcount, so the cohort spends roughly average."
            )
            out.append(
                Insight(
                    f"{biggest['rfm_segment']} is the largest cohort at {head_share:.0f}% of customers",
                    f"It contributes {rev_share:.1f}% of revenue. {tail}",
                    tone=_tone("warning" if gap >= 10 else "info"),
                )
            )
        richest = _top_row(portfolio, "revenue")
        if richest is not None:
            out.append(
                Insight(
                    f"{richest['rfm_segment']} carries the most historical value",
                    f"{formatting.fmt_money(richest['revenue'])} from "
                    f"{formatting.fmt_int(richest['customers'])} customers — protect it first when "
                    "capacity is limited.",
                )
            )
    ai = _frame(context, "ai_portfolio")
    if ai is not None:
        out.append(
            Insight(
                f"{len(ai)} AI clusters were saved for comparison",
                "They are an alternative lens, not a replacement: use the comparison scorecard to "
                "choose between them, and profile clusters before naming them.",
            )
        )
    region = (context.get("demographics") or {}).get("region")
    if isinstance(region, pd.DataFrame) and not region.empty and "pct" in region.columns:
        top = region.sort_values("customers", ascending=False).iloc[0]
        out.append(
            Insight(
                f"{top['region']} is the largest region at {float(top['pct']):.0f}% of customers",
                "Geography is an optional attribute: it profiles the base and does not change "
                "segmentation or scoring.",
            )
        )
    return tuple(out)


def _churn_insights(context: dict[str, Any]) -> tuple[Insight, ...]:
    """Churn interpretation: band mix and the leakage-safe model note."""
    out: list[Insight] = []
    risk_mix = context.get("risk_mix") or {}
    total = sum(int(v) for v in risk_mix.values())
    if total:
        risky = int(risk_mix.get("high", 0)) + int(risk_mix.get("critical", 0))
        pct = risky / total * 100
        out.append(
            Insight(
                f"{formatting.fmt_int(risky)} of {formatting.fmt_int(total)} scored customers "
                f"({pct:.1f}%) fall in high or critical bands",
                "Bands come from predicted probability against the configured threshold: this is "
                "the modeled at-risk pool, not confirmed leavers.",
                tone=_tone("critical" if pct >= 20 else "warning" if pct >= 10 else "info"),
            )
        )
    meta = context.get("churn_metadata") or {}
    if meta:
        out.append(
            Insight(
                f"Model {meta.get('model', '–')} · threshold {meta.get('threshold', '–')} · "
                f"cutoff {meta.get('cutoff', '–')}",
                "Predictors use transactions on or before the cutoff only, and the label is "
                "activity after it — leakage-safe by construction.",
            )
        )
    return tuple(out)


def _retention_insights(context: dict[str, Any]) -> tuple[Insight, ...]:
    """Retention interpretation: exposure, the top playbook, and the priority tail."""
    out: list[Insight] = []
    summary = context.get("retention_summary") or {}
    if summary.get("total_revenue_at_risk") is not None:
        out.append(
            Insight(
                f"{formatting.fmt_money(summary['total_revenue_at_risk'])} of historical value "
                "sits under modeled churn risk",
                "It is probability × historical value summed across customers — exposure, not a "
                "forecast of loss.",
                tone="critical",
            )
        )
    mix = summary.get("strategy_mix") or {}
    if mix:
        top = max(mix, key=mix.get)
        out.append(
            Insight(
                f"'{top}' is the most-assigned action ({formatting.fmt_int(mix[top])} customers)",
                "Every row names its strategy rule, so the assignment is traceable back to the "
                "matrix and never invented.",
            )
        )
    retention = _frame(context, "retention")
    if retention is not None and "retention_priority_score" in retention.columns:
        scores = retention["retention_priority_score"].dropna()
        if len(scores):
            q = float(scores.quantile(0.9))
            top10 = int((scores >= q).sum())
            out.append(
                Insight(
                    f"The top decile holds {formatting.fmt_int(top10)} customers (score ≥ {q:.3f})",
                    "Priority score ranks effort order; cutting the campaign at the right tail "
                    "spends capacity where the modeled value is highest.",
                )
            )
    return tuple(out)


def _appendix_insights(context: dict[str, Any]) -> tuple[Insight, ...]:
    """Data-foundation interpretation: window and completeness."""
    out: list[Insight] = []
    profile = context.get("data_profile") or {}
    span = profile.get("date_span")
    if span:
        out.append(
            Insight(
                f"Transaction window {span[0]} → {span[1]}",
                "Recency, active rate, and churn cutoffs are all measured against the end of this "
                "window.",
            )
        )
    if profile.get("missing_pct") is not None:
        out.append(
            Insight(
                f"{float(profile['missing_pct']):.2f}% of cells are missing in the clean data",
                "Optional demographic and geographic fields are the common source; required fields "
                "are guaranteed populated.",
            )
        )
    return tuple(out)


_SECTION_BUILDERS = {
    "overview": _overview_insights,
    "segments": _segment_insights,
    "churn": _churn_insights,
    "retention": _retention_insights,
    "appendix": _appendix_insights,
}


def section_insights(context: dict[str, Any], section: str) -> tuple[Insight, ...]:
    """Interpretation lines for one report section (empty for unknown sections)."""
    builder = _SECTION_BUILDERS.get(section)
    return builder(context) if builder else ()


def recommendations(context: dict[str, Any]) -> tuple[Recommendation, ...]:
    """Prioritized next actions, each tied to the evidence that produced it.

    Ordered Now → Next → Later. Nothing here promises an outcome; every item is
    decision support over the computed results.
    """
    recs: list[Recommendation] = []
    kpis = context.get("kpis") or {}
    summary = context.get("retention_summary") or {}
    portfolio = _frame(context, "segment_portfolio")
    hvhr = _frame(context, "hvhr")
    meta = context.get("churn_metadata") or {}
    profile = context.get("data_profile") or {}
    readiness = context.get("readiness") or {}
    customers = int(kpis.get("total_customers") or 0)

    mix = summary.get("strategy_mix") or {}
    if summary.get("customers"):
        top = max(mix, key=mix.get) if mix else "the top-ranked playbook"
        recs.append(
            Recommendation(
                "Now",
                f"Work the ranked list, starting with '{top}'",
                "Priority = normalized historical value × modeled churn probability × strategic "
                "weight; it ranks effort order and does not promise a save.",
                f"{formatting.fmt_int(summary['customers'])} customers · "
                f"{formatting.fmt_money(summary.get('total_revenue_at_risk'))} revenue at risk",
                tone="critical",
            )
        )

    if hvhr is not None:
        recs.append(
            Recommendation(
                "Now",
                "Assign VIP intervention to high-value, high-risk accounts",
                "These customers combine a high value tier with a high or critical risk band, so "
                "they concentrate the modeled exposure.",
                f"{len(hvhr)} accounts in the current results",
                tone="critical",
            )
        )

    top_segment = _top_row(portfolio, "revenue_share")
    if top_segment is not None and float(top_segment["revenue_share"]) >= 25:
        recs.append(
            Recommendation(
                "Now",
                f"Protect {top_segment['rfm_segment']} revenue",
                "It carries the largest share of historical revenue, so any retention gain spreads "
                "over the most value.",
                f"{float(top_segment['revenue_share']):.1f}% of revenue from "
                f"{formatting.fmt_int(top_segment['customers'])} customers",
                tone="warning",
            )
        )

    if portfolio is not None and {"customers", "revenue_share"} <= set(portfolio.columns):
        biggest = _top_row(portfolio, "customers")
        if biggest is not None:
            head = float(biggest["customers"]) / float(portfolio["customers"].sum()) * 100
            if head - float(biggest["revenue_share"]) >= 5:
                recs.append(
                    Recommendation(
                        "Next",
                        f"Grow {biggest['rfm_segment']} with low-cost nurture",
                        "It is the largest cohort but contributes less revenue per customer than its "
                        "headcount implies — a volume-without-value pattern.",
                        f"{head:.0f}% of customers · "
                        f"{float(biggest['revenue_share']):.1f}% of revenue",
                    )
                )

    if readiness.get("ai_segments"):
        recs.append(
            Recommendation(
                "Next",
                "Lock the most interpretable segmentation for campaigns",
                "The comparison scorecard gates runs on seed stability, smallest-segment share, and "
                "noise before ranking; business readability breaks ties.",
                "Re-run Segmentation Comparison after any data refresh.",
            )
        )

    if profile.get("rows_removed"):
        recs.append(
            Recommendation(
                "Next",
                "Close the top data-quality gaps at the source",
                "Cleaning records every treatment, so the same gaps can be traced back to the "
                "extraction or input process.",
                f"{formatting.fmt_int(profile['rows_removed'])} source rows removed",
                tone="warning",
            )
        )

    if meta.get("cutoff"):
        recs.append(
            Recommendation(
                "Later",
                "Revalidate the churn model on a fresh outcome window",
                "Labels are only as current as the cutoff and behavior drifts, so the ranking "
                "should be re-fit before it drives a new campaign wave.",
                f"Cutoff {meta['cutoff']} · model {meta.get('model', '–')}",
            )
        )

    covered = int(summary.get("customers") or 0)
    if customers and covered < customers:
        recs.append(
            Recommendation(
                "Later",
                "Extend retention coverage to uncovered customers",
                "Customers outside every strategy rule receive no recommended action, so they are "
                "invisible to a campaign export.",
                f"{formatting.fmt_int(customers - covered)} of {formatting.fmt_int(customers)} "
                "customers uncovered",
            )
        )
    return tuple(recs)


__all__ = [
    "PRIORITIES",
    "TONES",
    "Insight",
    "Recommendation",
    "recommendations",
    "section_insights",
    "summary_insights",
]

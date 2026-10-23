"""Report generator backend: context, narratives, and multi-format exports.

Pure functions over session-state contents (presentation lives in
``pages/09_Report_Generator.py``). Every narrative is factual and
decision-support only: historical monetary value is labeled a historical
proxy, never CLV, and no causal retention impact is claimed.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from datetime import UTC, datetime
from io import BytesIO
from typing import Any, Final

import pandas as pd

from analytics.demographics import DEFAULT_AGE_BANDS, assign_age_band
from services import report_charts, report_narrative
from services.comparison_service import (
    build_comparison,
    scorecard_as_text,
    scorecard_for_display,
)
from services.dashboard_service import (
    build_master,
    compute_kpis,
    high_value_high_risk,
    mask_customer_id,
    prepare_transactions,
    trend_tables,
)
from services.data_dictionary_service import normalize_dictionary_rows
from utils import formatting
from utils.exceptions import CustomerIntelligenceError

#: Report sections in display order. Pages toggle these per export. The
#: recommendations section closes the report — interpretations first, then the
#: prioritized actions — so every export ends with the "so what".
REPORT_SECTIONS: tuple[str, ...] = (
    "summary",
    "overview",
    "segments",
    "comparison",
    "churn",
    "retention",
    "appendix",
    "recommendations",
)

#: Human-readable section labels, shared by the page toggle and the HTML headings.
SECTION_LABELS: dict[str, str] = {
    "summary": "Executive summary",
    "overview": "Business & customer trends",
    "segments": "Segments & demographics",
    "comparison": "Model comparison",
    "churn": "Churn risk",
    "retention": "Retention plan",
    "appendix": "Appendix & data foundation",
    "recommendations": "Recommendations & next steps",
}

#: Sections whose interpretation is recapped in the closing section.
ANALYTICAL_SECTIONS: Final[tuple[str, ...]] = ("overview", "segments", "churn", "retention")

#: Sections rendered from :data:`SECTION_CHARTS` by the PDF and Word exporters, in
#: report order. Superset of :data:`ANALYTICAL_SECTIONS` — the comparison section
#: carries computed evidence (a table and charts) but no narrative insights, so it
#: renders like an analytical section without being recapped at the close.
CHARTED_SECTIONS: Final[tuple[str, ...]] = (
    "overview",
    "segments",
    "comparison",
    "churn",
    "retention",
)

#: Chart keys per section, in reading order. Shared by HTML, PDF, and Word so
#: the three exports can never drift apart.
SECTION_CHARTS: Final[dict[str, tuple[str, ...]]] = {
    "overview": ("revenue_trend", "activity_trend", "new_returning"),
    "segments": (
        "segment_distribution",
        "revenue_by_segment",
        "segment_trends",
        "demographics_age",
        "demographics_gender",
        "demographics_region",
    ),
    "comparison": (
        "comparison_separation",
        "comparison_stability",
        "comparison_value",
        "comparison_evenness",
    ),
    "churn": ("risk_bands",),
    "retention": (
        "value_risk_matrix",
        "revenue_at_risk",
        "strategy_distribution",
        "priority_histogram",
    ),
}

#: One plain-English explainer per embedded chart. ``action`` never promises
#: causal impact; value is always historical, never CLV.
CHART_EXPLANATIONS: dict[str, dict[str, str]] = {
    "segment_distribution": {
        "title": "Customers per segment",
        "what": "How many customers fall into each RFM segment.",
        "how_to_read": "Taller bars are bigger cohorts. Compare against the "
        "revenue chart before acting: a large bar with a short revenue bar "
        "is volume without value.",
        "action": "Size follow-ups to the cohort, not the headline: small "
        "high-value bars suit personal outreach, large low-value bars suit "
        "low-cost nurture.",
    },
    "revenue_by_segment": {
        "title": "Revenue per segment",
        "what": "Historical revenue contributed by each RFM segment.",
        "how_to_read": "Taller bars hold more past revenue. A segment that is "
        "small in customers but tall here concentrates value.",
        "action": "Protect tall bars first when planning retention capacity; "
        "they carry the most historical value at stake.",
    },
    "segment_trends": {
        "title": "Segment trends over time",
        "what": "Customer counts per RFM segment across period ends (quarterly/monthly).",
        "how_to_read": "Stacked area shows how cohorts shift over time; period definitions are quantile-relative.",
        "action": "Watch surges/declines by segment to time targeted plays before peaks shift.",
    },
    "risk_bands": {
        "title": "Customers per risk band",
        "what": "Scored customers grouped into low / medium / high / critical "
        "churn-risk bands from the trained churn model.",
        "how_to_read": "Bands come from predicted churn probability and the "
        "configured threshold. High and critical bars are the modeled "
        "at-risk pool, not confirmed leavers.",
        "action": "Work the high and critical bars top-down by priority "
        "score; low bars stay on routine monitoring.",
    },
    "value_risk_matrix": {
        "title": "Value–risk matrix",
        "what": "Each customer plotted by churn probability (x) against "
        "historical value proxy (y); bubble size is revenue at risk.",
        "how_to_read": "Top-right bubbles are high historical value with high "
        "modeled risk. Value here is past spend, not predicted future CLV.",
        "action": "Start with large top-right bubbles for VIP-style "
        "intervention; bottom-left bubbles stay on automated nurture.",
    },
    "revenue_at_risk": {
        "title": "Revenue at risk per action",
        "what": "Historical value × modeled churn probability, summed per "
        "recommended retention action.",
        "how_to_read": "Taller bars name where the most modeled value sits, "
        "per traceable strategy rule. It is exposure under the model, not a "
        "guaranteed loss.",
        "action": "Staff and budget the tallest bars first; re-check after " "each campaign wave.",
    },
    "strategy_distribution": {
        "title": "Customers per recommended action",
        "what": "How many customers map to each traceable retention action.",
        "how_to_read": "Tall bars are the most-assigned playbooks. Every row "
        "names its strategy rule in action_reason.",
        "action": "Operationalize the tallest bars (templates, channels, "
        "owners) before the long tail.",
    },
    "priority_histogram": {
        "title": "Priority score distribution",
        "what": "Spread of retention priority scores (normalized historical "
        "value × churn probability × strategic weight).",
        "how_to_read": "Right-tail customers outrank the middle. The score "
        "ranks effort order; it does not promise a save.",
        "action": "Cut the campaign at the right tail (e.g. top 200 or a "
        "minimum score) rather than working the whole list.",
    },
    "revenue_trend": {
        "title": "Revenue trend",
        "what": "Monthly revenue over the filtered transaction window.",
        "how_to_read": "Slope is momentum; dips invite a drill into segments "
        "and cohorts for that month.",
        "action": "Read alongside segment and cohort mix before concluding "
        "anything about a single month.",
    },
    "activity_trend": {
        "title": "Customer activity trend",
        "what": "Distinct customers transacting in each month of the window.",
        "how_to_read": "It is engagement over time, not loyalty: the same "
        "customer can count in several months.",
        "action": "A falling line with flat revenue means fewer, larger "
        "buyers — check the cohort mix before reacting.",
    },
    "new_returning": {
        "title": "New vs returning customers",
        "what": "Each month's active customers split by whether the month is "
        "their first transaction.",
        "how_to_read": "The new share shows acquisition intensity; the "
        "returning share shows repeat activity in the existing base.",
        "action": "Acquisition-led months need onboarding; repeating-led "
        "months reward retention capacity instead.",
    },
    "demographics_age": {
        "title": "Customers per age band",
        "what": "Customer counts per age band, using the configured bands.",
        "how_to_read": "Bands are counts, not value: an older cohort is not "
        "automatically more valuable.",
        "action": "Read beside the revenue-by-segment charts before designing "
        "an age-targeted message.",
    },
    "demographics_gender": {
        "title": "Customers by gender",
        "what": "Customer counts per gender value, with blanks folded into " "'Unknown'.",
        "how_to_read": "It is a composition view only; it does not imply a "
        "difference in behavior.",
        "action": "Use it for representativeness checks, never for pricing or "
        "targeting decisions on its own.",
    },
    "demographics_region": {
        "title": "Top regions by customers",
        "what": "The largest customer regions in the base.",
        "how_to_read": "Geography is an optional attribute; missing values stay "
        "visible as 'Unknown'.",
        "action": "Compare regional headcount against regional revenue before " "shifting spend.",
    },
    "comparison_separation": {
        "title": "Silhouette by model (AI runs)",
        "what": "Cluster separation for each AI run, measured in the shared feature "
        "space. Blank for RFM/LRFM by design — a different space is not comparable.",
        "how_to_read": "Higher is tighter, better-separated clusters. It is a "
        "geometry score, not a business verdict, and it is one voice among four.",
        "action": "Never pick on silhouette alone: a high score with an unstable or "
        "ill-sized split is a trap the recommendation gates are there to catch.",
    },
    "comparison_stability": {
        "title": "Seed stability by model (ARI)",
        "what": "Agreement between runs of the same model under different random "
        "seeds. RFM/LRFM are 1.0 by construction — deterministic rules.",
        "how_to_read": "1.0 means the grouping barely moves when the seed changes. "
        "Low values mean any segmentation you read here could reshuffle tomorrow.",
        "action": "Treat a model below the stability gate as a research result, not "
        "a production segmentation, however strong its other metrics look.",
    },
    "comparison_value": {
        "title": "Revenue differentiation by model (eta²)",
        "what": "Share of monetary-value variance explained by segment membership.",
        "how_to_read": "Higher means segments actually separate customers by historical "
        "value. Value is a past-spend proxy, never a predicted CLV.",
        "action": "Prefer the model with real value separation when the spend "
        "differences are what the campaign will act on.",
    },
    "comparison_evenness": {
        "title": "Size balance by model (evenness)",
        "what": "How evenly customers are spread across segments (1 = perfectly even).",
        "how_to_read": "Low evenness means one giant segment and a long tail, which "
        "usually means an under-specified or too-fine split.",
        "action": "Check the smallest-segment share against the size gate before "
        "committing outreach capacity to a lopsided split.",
    },
}


def _as_frame(value: Any) -> pd.DataFrame | None:
    """Return value when it is a non-empty DataFrame, else None."""
    if isinstance(value, pd.DataFrame) and not value.empty:
        return value
    return None


def _kpi_value(key: str, value: Any) -> str:
    """Human-readable KPI value: ratios as percent, money with the symbol, counts whole."""
    if value is None:
        return "–"
    label = str(key).lower()
    if "rate" in label:
        return formatting.fmt_pct(value)
    if any(word in label for word in ("revenue", "value", "amount", "spend")):
        return formatting.fmt_money_compact(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return formatting.fmt_int(value)
    return str(value)


def _title_sentence(title: str) -> str:
    """Close an insight title with a period so it reads before the detail sentence."""
    text = title.strip()
    return text if text.endswith((".", ":", "!", "?")) else f"{text}."


def _retention_frame(stored: Any) -> pd.DataFrame | None:
    """Retention table from state (records orient) or None."""
    if isinstance(stored, dict):
        rows = stored.get("table")
        if rows:
            return pd.DataFrame(rows)
    return _as_frame(stored)


def _mask_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Mask customer_id for exports/previews shown without authorization."""
    out = frame.copy()
    if "customer_id" in out.columns:
        out["customer_id"] = out["customer_id"].map(mask_customer_id)
    return out


def _label_portfolio(
    features: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    label_col: str,
    out_col: str,
) -> pd.DataFrame | None:
    """Customers + historical revenue per segment label (RFM/LRFM/AI shared path)."""
    if features is None or labels is None or label_col not in labels.columns:
        return None
    if "monetary_value" not in features.columns:
        return None
    merged = features.merge(labels[["customer_id", label_col]], on="customer_id", how="inner")
    table = (
        merged.groupby(label_col)
        .agg(customers=("customer_id", "size"), revenue=("monetary_value", "sum"))
        .reset_index()
        .rename(columns={label_col: out_col})
    )
    total = float(table["revenue"].sum())
    table["customer_share"] = (table["customers"] / table["customers"].sum() * 100).round(2)
    table["revenue_share"] = (table["revenue"] / total * 100 if total else 0.0).round(2)
    return table.sort_values("customers", ascending=False).reset_index(drop=True)


def _demographics(features: pd.DataFrame | None) -> dict[str, pd.DataFrame | None]:
    """Gender, age-band, and region composition over the customer feature table."""
    out: dict[str, pd.DataFrame | None] = {"gender": None, "age_bands": None, "region": None}
    if features is None:
        return out
    if "gender" in features.columns:
        gender = features["gender"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        table = gender.value_counts().rename_axis("gender").reset_index(name="customers")
        table["pct"] = (table["customers"] / table["customers"].sum() * 100).round(2)
        out["gender"] = table
    if "age" in features.columns:
        banded = assign_age_band(features["age"], DEFAULT_AGE_BANDS)
        table = banded.value_counts().rename_axis("age_band").reset_index(name="customers")
        table["pct"] = (table["customers"] / table["customers"].sum() * 100).round(2)
        out["age_bands"] = table
    if "region" in features.columns:
        region = features["region"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        table = region.value_counts().rename_axis("region").reset_index(name="customers")
        table["pct"] = (table["customers"] / table["customers"].sum() * 100).round(2)
        out["region"] = table.head(10)
    return out


def _data_profile(
    clean: pd.DataFrame | None, features: pd.DataFrame | None, audit: Mapping[str, Any]
) -> dict[str, Any]:
    """Source-level foundation numbers for the appendix (rows, span, completeness)."""
    rows = int(len(clean)) if clean is not None else 0
    columns = int(len(clean.columns)) if clean is not None else 0
    cells = rows * columns
    missing = int(clean.isna().sum().sum()) if clean is not None else 0
    span: tuple[str, str] | None = None
    if clean is not None and "transaction_date" in clean.columns:
        dates = pd.to_datetime(clean["transaction_date"], errors="coerce").dropna()
        if len(dates):
            span = (dates.min().date().isoformat(), dates.max().date().isoformat())
    return {
        "rows": rows,
        "customers": int(len(features)) if features is not None else 0,
        "columns": columns,
        "missing_pct": round(missing / cells * 100, 2) if cells else 0.0,
        "date_span": span,
        "rows_in": int(audit.get("input_rows") or 0) or None,
        "rows_removed": int(audit.get("rows_removed") or 0),
        "source_filename": str(audit.get("source_filename") or "(in-session dataset)"),
    }


def _treatments(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Cleaning treatments applied, in the recorded order."""
    rules = audit.get("rules") or {}
    out: list[dict[str, Any]] = []
    if not isinstance(rules, Mapping):
        return out
    for name, detail in rules.items():
        if not isinstance(detail, Mapping):
            continue
        out.append(
            {
                "treatment": str(name).replace("_", " ").capitalize(),
                "action": str(detail.get("action", "")),
                "records_affected": int(detail.get("records_affected", 0) or 0),
            }
        )
    return out


def build_report_context(state: Mapping[str, Any]) -> dict[str, Any]:
    """Collect every report input from session state into one dict.

    Never raises on missing upstream results; readiness flags tell the page
    which sections have data and which need earlier pages first.
    """
    clean = _as_frame(state.get("clean_data"))
    features = _as_frame(state.get("customer_features"))
    rfm = _as_frame(state.get("rfm_results"))
    lrfm = _as_frame(state.get("lrfm_results"))
    segmented = _as_frame(state.get("segmented_customers"))
    scores = _as_frame(state.get("churn_scores"))
    retention = _retention_frame(state.get("retention_results"))
    metadata = state.get("churn_model_metadata") or {}
    audit = state.get("cleaning_audit") or {}
    mapping = state.get("column_mapping") or {}

    readiness = {
        "clean_data": clean is not None,
        "features": features is not None,
        "rfm": rfm is not None,
        "lrfm": lrfm is not None,
        "ai_segments": segmented is not None,
        "churn": scores is not None,
        "retention": retention is not None,
    }

    txns = None
    if clean is not None:
        try:
            txns = prepare_transactions(clean)
        except (KeyError, ValueError):
            txns = None

    kpis: dict[str, Any] = {}
    trends: dict[str, pd.DataFrame] = {}
    master = None
    if txns is not None and features is not None:
        try:
            master = build_master(
                features, rfm=rfm, segmented=segmented, scores=scores, retention=retention
            )
            kpis = compute_kpis(txns, master)
        except (KeyError, ValueError):
            master = None
        if master is not None:
            try:
                trends = trend_tables(txns, master)
            except (KeyError, ValueError):
                trends = {}

    ai_col = None
    if segmented is not None:
        if "business_name" in segmented.columns:
            ai_col = "business_name"
        elif "cluster_id" in segmented.columns:
            ai_col = "cluster_id"

    segment_portfolio = _label_portfolio(features, rfm, "segment", "rfm_segment")
    lrfm_portfolio = _label_portfolio(features, lrfm, "segment", "lrfm_segment")
    ai_portfolio = _label_portfolio(features, segmented, ai_col, "ai_segment") if ai_col else None

    risk_mix: dict[str, int] = {}
    if scores is not None and "risk_band" in scores.columns:
        risk_mix = {str(k): int(v) for k, v in scores["risk_band"].value_counts().items()}

    hvhr = None
    if master is not None:
        try:
            hvhr = high_value_high_risk(master)
        except (KeyError, ValueError):
            hvhr = None

    retention_summary: dict[str, Any] = {}
    if retention is not None:
        stored = state.get("retention_results")
        if (
            isinstance(stored, dict)
            and isinstance(stored.get("summary"), dict)
            and stored["summary"]
        ):
            retention_summary = dict(stored["summary"])
        else:
            retention_summary = {
                "customers": len(retention),
                "total_revenue_at_risk": (
                    round(float(retention["revenue_at_risk"].sum()), 2)
                    if "revenue_at_risk" in retention.columns
                    else 0.0
                ),
                "strategy_mix": (
                    retention["recommended_action"].value_counts().to_dict()
                    if "recommended_action" in retention.columns
                    else {}
                ),
                "band_mix": (
                    retention["risk_band"].value_counts().to_dict()
                    if "risk_band" in retention.columns
                    else {}
                ),
            }

    history = None
    hist_state = state.get("segment_history") or {}
    if isinstance(hist_state, dict) and hist_state.get("table"):
        history = pd.DataFrame(hist_state["table"])

    # Model comparison reuses the same computation page 06 shows, so the report
    # and the page can never disagree about which model won and why.
    try:
        comparison = build_comparison(state)
    except (KeyError, ValueError, TypeError):
        comparison = None

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "readiness": readiness,
        "kpis": kpis,
        "trends": trends,
        "features": features,
        "rfm": rfm,
        "lrfm": lrfm,
        "segmented": segmented,
        "scores": scores,
        "retention": retention,
        "comparison": comparison,
        "segment_portfolio": segment_portfolio,
        "lrfm_portfolio": lrfm_portfolio,
        "ai_portfolio": ai_portfolio,
        "demographics": _demographics(features),
        "risk_mix": risk_mix,
        "hvhr": hvhr,
        "history": history,
        "retention_summary": retention_summary,
        "churn_metadata": dict(metadata) if isinstance(metadata, dict) else {},
        "cleaning_audit": dict(audit) if isinstance(audit, dict) else {},
        "column_mapping": dict(mapping) if isinstance(mapping, dict) else {},
        "treatments": _treatments(audit if isinstance(audit, Mapping) else {}),
        "data_dictionary": normalize_dictionary_rows(state.get("data_dictionary")),
        "data_profile": _data_profile(clean, features, audit if isinstance(audit, Mapping) else {}),
        "n_customers": len(features) if features is not None else 0,
        "n_transactions": len(clean) if clean is not None else 0,
    }


def context_tables(
    context: dict[str, Any], mask_ids: bool = True, top_n: int = 200
) -> dict[str, pd.DataFrame]:
    """Named tables for the Excel workbook and the HTML appendix."""
    tables: dict[str, pd.DataFrame] = {}
    kpis = context.get("kpis") or {}
    if kpis:
        tables["KPIs"] = pd.DataFrame(
            [
                {"metric": str(k).replace("_", " "), "value": _kpi_value(k, v)}
                for k, v in kpis.items()
            ]
        )
    portfolio = context.get("segment_portfolio")
    if portfolio is not None and not portfolio.empty:
        p = portfolio.copy()
        if "revenue" in p.columns:
            p["revenue"] = [formatting.fmt_money_compact(v) for v in p["revenue"]]
        tables["RFM_Portfolio"] = p
    lrfm_portfolio = context.get("lrfm_portfolio")
    if lrfm_portfolio is not None and not lrfm_portfolio.empty:
        p = lrfm_portfolio.copy()
        if "revenue" in p.columns:
            p["revenue"] = [formatting.fmt_money_compact(v) for v in p["revenue"]]
        tables["LRFM_Portfolio"] = p
    ai_portfolio = context.get("ai_portfolio")
    if ai_portfolio is not None and not ai_portfolio.empty:
        p = ai_portfolio.copy()
        if "revenue" in p.columns:
            p["revenue"] = [formatting.fmt_money_compact(v) for v in p["revenue"]]
        tables["AI_Portfolio"] = p
    comparison = context.get("comparison") or {}
    scorecard = comparison.get("table")
    if isinstance(scorecard, pd.DataFrame) and not scorecard.empty:
        tables["Model_Scorecard"] = scorecard_for_display(scorecard)
    pair = comparison.get("pair") or {}
    cross = pair.get("cross_tab")
    if isinstance(cross, pd.DataFrame) and not cross.empty:
        tables["Model_Flow"] = cross.reset_index()
    trends = context.get("trends") or {}
    revenue_trend = trends.get("revenue")
    if isinstance(revenue_trend, pd.DataFrame) and not revenue_trend.empty:
        monthly = revenue_trend.copy()
        cohort = trends.get("cohort")
        if isinstance(cohort, pd.DataFrame) and not cohort.empty:
            monthly = monthly.merge(cohort, on="month", how="left")
        activity = trends.get("activity")
        if isinstance(activity, pd.DataFrame) and not activity.empty:
            monthly = monthly.merge(activity, on="month", how="left")
        if "revenue" in monthly.columns:
            monthly["revenue"] = [formatting.fmt_money_compact(v) for v in monthly["revenue"]]
        tables["Monthly_Trends"] = monthly
    demographics = context.get("demographics") or {}
    for name, key in (("Demographics_Gender", "gender"), ("Demographics_Age", "age_bands")):
        frame = demographics.get(key)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            tables[name] = frame.copy()
    risk_mix = context.get("risk_mix") or {}
    if risk_mix:
        tables["Risk_Bands"] = pd.DataFrame(
            [{"risk_band": band, "customers": formatting.fmt_int(count)} for band, count in risk_mix.items()]
        )
    retention = context.get("retention")
    if retention is not None and not retention.empty:
        ranked = (
            retention.sort_values("retention_priority_score", ascending=False).head(top_n)
            if "retention_priority_score" in retention.columns
            else retention.head(top_n)
        )
        r = _mask_frame(ranked) if mask_ids else ranked.copy()
        for col in ("revenue_at_risk", "expected_value_proxy_historical", "monetary_value"):
            if col in r.columns:
                r[col] = [formatting.fmt_money_compact(v) for v in r[col]]
        if "retention_priority_score" in r.columns:
            r["retention_priority_score"] = [formatting.fmt_float(v, 3) for v in r["retention_priority_score"]]
        if "churn_probability" in r.columns:
            r["churn_probability"] = [formatting.fmt_pct(v, 1) for v in r["churn_probability"]]
        tables["Retention_Top"] = r
    scores = context.get("scores")
    if scores is not None and not scores.empty:
        cols = [c for c in ("customer_id", "churn_probability", "risk_band") if c in scores.columns]
        top_risk = scores.sort_values("churn_probability", ascending=False).head(top_n)[cols].copy()
        if "churn_probability" in top_risk.columns:
            top_risk["churn_probability"] = [formatting.fmt_pct(v, 1) for v in top_risk["churn_probability"]]
        tables["Churn_Top_Risk"] = _mask_frame(top_risk) if mask_ids else top_risk
    summary = context.get("retention_summary") or {}
    if summary.get("strategy_mix"):
        tables["Strategy_Mix"] = pd.DataFrame(
            [
                {"action": action, "customers": formatting.fmt_int(count)}
                for action, count in summary["strategy_mix"].items()
            ]
        )
    dictionary = context.get("data_dictionary") or []
    if dictionary:
        keep = [
            "canonical_field",
            "source_column",
            "data_type",
            "non_null_count",
            "missing_count",
            "missing_pct",
            "unique_count",
            "examples",
        ]
        frame = pd.DataFrame(dictionary)
        tables["Data_Source"] = frame[[c for c in keep if c in frame.columns]]
    treatments = context.get("treatments") or []
    if treatments:
        tables["Cleaning_Treatments"] = pd.DataFrame(treatments)
    return tables


def context_to_excel_bytes(
    context: dict[str, Any], mask_ids: bool = True, top_n: int = 200
) -> bytes:
    """Multi-sheet Excel workbook for the report tables."""
    tables = context_tables(context, mask_ids=mask_ids, top_n=top_n)
    if not tables:
        raise CustomerIntelligenceError(
            "Nothing to export yet. Build features on RFM & LRFM Analysis first."
        )
    buffer = BytesIO()
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise CustomerIntelligenceError(
            "Excel export needs the 'openpyxl' package. Install it with "
            "'pip install openpyxl' and try again."
        ) from exc
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet, frame in tables.items():
            frame.to_excel(writer, index=False, sheet_name=sheet[:31])
    return buffer.getvalue()


#: Standalone-report skins. The light skin is the print default; the dark skin
#: mirrors the charcoal-and-lime console for on-screen reading. PDF stays
#: print-light by decision (paper, not screen).
_REPORT_SKINS: Final = {
    "light": {
        "page_bg": "#ffffff",
        "body_ink": "#1a2333",
        "rule": "#e2e8f0",
        "muted": "#64748b",
        "kpi_border": "#e2e8f0",
        "explain_bg": "#f1f5f9",
        "explain_edge": "#2563eb",
        "th_bg": "#f8fafc",
        "cell_border": "#e2e8f0",
    },
    "dark": {
        "page_bg": "#0c0f14",
        "body_ink": "#f2f5f9",
        "rule": "#2a3342",
        "muted": "#9aa7b8",
        "kpi_border": "#2a3342",
        "explain_bg": "#1b222d",
        "explain_edge": "#bef264",
        "th_bg": "#1b222d",
        "cell_border": "#2a3342",
    },
}


def _section_heading(section: str, label: str) -> str:
    """Numbered section heading that follows REPORT_SECTIONS order."""
    number = REPORT_SECTIONS.index(section) + 1 if section in REPORT_SECTIONS else 0
    return f"{number} · {label}"


def _closing_insights(
    context: dict[str, Any], sections: tuple[str, ...]
) -> tuple[report_narrative.Insight, ...]:
    """Recap interpretation across the selected analytical sections (deduplicated)."""
    seen: set[str] = set()
    recap: list[report_narrative.Insight] = []
    for section in ANALYTICAL_SECTIONS:
        if section not in sections:
            continue
        for item in report_narrative.section_insights(context, section):
            if item.title in seen:
                continue
            seen.add(item.title)
            recap.append(item)
    return tuple(recap)


def render_html_report(
    context: dict[str, Any],
    figures_html: dict[str, str] | None = None,
    sections: tuple[str, ...] = REPORT_SECTIONS,
    mask_ids: bool = True,
    mode: str = "light",
) -> str:
    """Self-contained HTML report with narratives, charts, and tables.

    ``mode`` selects the print-light skin (default) or the dark console skin;
    unknown values fall back to light.
    """
    skin = _REPORT_SKINS.get(mode, _REPORT_SKINS["light"])
    figures_html = figures_html or {}
    kpis = context.get("kpis") or {}
    readiness = context.get("readiness") or {}
    metadata = context.get("churn_metadata") or {}
    masked_note = (
        "Customer IDs are masked in this report."
        if mask_ids
        else ("Customer IDs are shown in full (authorized export).")
    )

    def _fig(key: str) -> str:
        snippet = figures_html.get(key, "")
        if snippet:
            return f"<div class='fig'>{snippet}</div>"
        return "<p class='muted'>Chart unavailable — complete the source page first.</p>"

    def _explain(key: str) -> str:
        exp = CHART_EXPLANATIONS.get(key, {})
        if not exp:
            return ""
        return (
            "<div class='explain'><strong>How to read this chart. </strong>"
            f"{html.escape(exp.get('how_to_read', ''))} "
            f"<strong>What to do. </strong>{html.escape(exp.get('action', ''))}</div>"
        )

    kpi_cards = (
        "".join(
            f"<div class='kpi'><div class='kpi-label'>{html.escape(str(k).replace('_', ' '))}</div>"
            f"<div class='kpi-value'>{html.escape(_kpi_value(k, v))}</div></div>"
            for k, v in kpis.items()
        )
        or "<p class='muted'>KPIs appear after RFM scoring.</p>"
    )

    tables = context_tables(context, mask_ids=mask_ids)
    tables_html = (
        "".join(
            f"<h3>{html.escape(sheet)}</h3>{frame.head(50).to_html(index=False, border=0, classes='tbl')}"
            for sheet, frame in tables.items()
        )
        or "<p class='muted'>No tables yet.</p>"
    )

    def _insights(section: str) -> str:
        items = report_narrative.section_insights(context, section)
        return "".join(
            "<div class='insight'><strong>"
            f"{html.escape(_title_sentence(item.title))}</strong> "
            f"<span>{html.escape(item.detail)}</span></div>"
            for item in items
        )

    built: list[str] = []
    if "summary" in sections:
        insights_html = "".join(
            "<div class='insight'><strong>"
            f"{html.escape(_title_sentence(item.title))}</strong> "
            f"<span>{html.escape(item.detail)}</span></div>"
            for item in report_narrative.summary_insights(context)
        )
        built.append(
            f"<section><h2>{_section_heading('summary', 'Executive summary')}</h2>"
            f"<div class='kpi-row'>{kpi_cards}</div>"
            f"{insights_html}</section>"
        )
    if "overview" in sections:
        built.append(
            f"<section><h2>{_section_heading('overview', 'Business &amp; customer trends')}</h2>"
            f"{_insights('overview')}"
            f"{_fig('revenue_trend')}{_explain('revenue_trend')}"
            f"{_fig('activity_trend')}{_explain('activity_trend')}"
            f"{_fig('new_returning')}{_explain('new_returning')}</section>"
        )
    if "segments" in sections:
        built.append(
            f"<section><h2>{_section_heading('segments', 'Segments produced')}</h2>"
            f"{_insights('segments')}"
            f"{_fig('segment_distribution')}{_explain('segment_distribution')}"
            f"{_fig('revenue_by_segment')}{_explain('revenue_by_segment')}"
            f"{_fig('segment_trends')}{_explain('segment_trends')}"
            f"{_fig('demographics_age')}{_explain('demographics_age')}"
            f"{_fig('demographics_gender')}{_explain('demographics_gender')}"
            f"{_fig('demographics_region')}{_explain('demographics_region')}</section>"
        )
    if "comparison" in sections:
        comparison = context.get("comparison") or {}
        if comparison:
            rec = comparison.get("recommendation") or {}
            pair = comparison.get("pair") or {}
            scorecard = scorecard_as_text(comparison.get("table"))
            scorecard_html = (
                scorecard.to_html(index=False, border=0, classes="tbl")
                if not scorecard.empty
                else "<p class='muted'>No comparable models yet.</p>"
            )
            rec_html = (
                f"<div class='explain'><strong>Recommended model. </strong>"
                f"{html.escape(str(rec.get('rationale', '')))}</div>"
                if rec.get("rationale")
                else ""
            )
            pair_html = ""
            if pair:
                cross = pair.get("cross_tab")
                cross_html = ""
                if isinstance(cross, pd.DataFrame) and not cross.empty:
                    cross_html = (
                        "<details><summary>Customer flow (counts)</summary>"
                        f"{cross.to_html(border=0, classes='tbl')}</details>"
                    )
                pair_html = (
                    f"<p>Pair agreement: <strong>{html.escape(str(pair.get('reference')))}</strong> "
                    f"vs <strong>{html.escape(str(pair.get('target')))}</strong> · "
                    f"Adjusted Rand Index {html.escape(formatting.fmt_float(pair.get('ari'), 3))} "
                    "(1.0 = identical groupings, 0 = chance-level).</p>"
                    f"{cross_html}"
                )
            built.append(
                f"<section><h2>{_section_heading('comparison', 'Model comparison')}</h2>"
                f"{pair_html}{rec_html}"
                f"<h3>Model scorecard</h3>{scorecard_html}"
                f"<p class='muted'>Silhouette is comparable only among AI runs sharing a "
                "feature space; RFM/LRFM leave it blank by design. Stability of 1.0 for "
                "RFM/LRFM is deterministic, not evidence of superiority.</p>"
                f"{_fig('comparison_separation')}{_explain('comparison_separation')}"
                f"{_fig('comparison_stability')}{_explain('comparison_stability')}"
                f"{_fig('comparison_value')}{_explain('comparison_value')}"
                f"{_fig('comparison_evenness')}{_explain('comparison_evenness')}</section>"
            )
        else:
            built.append(
                f"<section><h2>{_section_heading('comparison', 'Model comparison')}</h2>"
                "<p class='muted'>No models to compare yet — score RFM/LRFM and save at "
                "least one AI run on the Segmentation Comparison page first."
                "</p></section>"
            )
    if "churn" in sections:
        model_line = ""
        if metadata:
            model_line = (
                f"<p>Model: <strong>{html.escape(str(metadata.get('model', '–')))}</strong> · "
                f"threshold {html.escape(str(metadata.get('threshold', '–')))} · "
                f"cutoff {html.escape(str(metadata.get('cutoff', '–')))}. Predictors use "
                "transactions on or before the cutoff only; the label is activity after the "
                "cutoff (leakage-safe by construction).</p>"
            )
        built.append(
            f"<section><h2>{_section_heading('churn', 'Churn risk calculated')}</h2>"
            f"{model_line}{_insights('churn')}"
            f"{_fig('risk_bands')}{_explain('risk_bands')}</section>"
        )
    if "retention" in sections:
        built.append(
            f"<section><h2>{_section_heading('retention', 'Retention strategies covered')}</h2>"
            f"{_insights('retention')}"
            f"{_fig('value_risk_matrix')}{_explain('value_risk_matrix')}"
            f"{_fig('revenue_at_risk')}{_explain('revenue_at_risk')}"
            f"{_fig('strategy_distribution')}{_explain('strategy_distribution')}"
            f"{_fig('priority_histogram')}{_explain('priority_histogram')}</section>"
        )
    if "appendix" in sections:
        ready = ", ".join(f"{k}: {'yes' if v else 'pending'}" for k, v in readiness.items())
        profile = context.get("data_profile") or {}
        treatments = context.get("treatments") or []
        treat_html = (
            "".join(
                f"<li>{html.escape(str(t['treatment']))}: {html.escape(str(t['action']))} "
                f"({int(t['records_affected']):,} records)</li>"
                for t in treatments
            )
            or "<li>No cleaning treatments recorded for this dataset.</li>"
        )
        built.append(
            f"<section><h2>{_section_heading('appendix', 'Appendix — data &amp; method')}</h2>"
            f"{_insights('appendix')}"
            f"<p>Source: <strong>{html.escape(str(profile.get('source_filename', '–')))}</strong> · "
            f"{int(profile.get('rows', 0)):,} clean rows × {int(profile.get('columns', 0))} columns · "
            f"{int(profile.get('customers', 0)):,} customers · "
            f"{float(profile.get('missing_pct', 0)):.2f}% cells missing.</p>"
            "<h3>Cleaning treatments</h3><ul>" + treat_html + "</ul>"
            "<p class='muted'>Method: churn is leakage-safe by construction (predictors ≤ cutoff, "
            "label from the outcome window; cohort-ordered splits). RFM = 100R+10F+M; cluster IDs "
            "are arbitrary until profiled against population medians.</p>"
            f"<p class='muted'>Coverage — {html.escape(ready)}. {html.escape(masked_note)}</p>"
            f"{tables_html}</section>"
        )
    if "recommendations" in sections:
        recs = report_narrative.recommendations(context)
        recap = _closing_insights(context, sections)
        recap_html = (
            "".join(
                "<div class='insight'><strong>"
                f"{html.escape(_title_sentence(item.title))}</strong> "
                f"<span>{html.escape(item.detail)}</span></div>"
                for item in recap
            )
            or "<p class='muted'>Section interpretations appear as the upstream results arrive.</p>"
        )
        rec_html = (
            "".join(
                f"<div class='rec'><span class='pill'>{html.escape(r.priority)}</span> "
                f"<strong>{html.escape(_title_sentence(r.title))}</strong>"
                f"<div class='muted'>{html.escape(r.rationale)}</div>"
                f"<div class='evidence'>{html.escape(r.evidence)}</div></div>"
                for r in recs
            )
            or "<p class='muted'>Recommendations appear once retention results exist.</p>"
        )
        built.append(
            f"<section><h2>{_section_heading('recommendations', 'Recommendations &amp; next steps')}</h2>"
            "<h3>Interpretations at a glance</h3>"
            f"{recap_html}"
            "<h3>Prioritized actions</h3>"
            f"{rec_html}"
            "<p class='muted'>Priority order is decision support over the computed results; it "
            "does not claim causal impact.</p></section>"
        )
    body = "\n".join(built)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Customer Intelligence Report — {html.escape(str(context.get('generated_at', '')))}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;max-width:1000px;margin:0 auto;padding:24px;background:{skin['page_bg']};color:{skin['body_ink']};}}
h1{{font-size:1.6rem;margin-bottom:0;}}h2{{font-size:1.2rem;margin-top:2rem;border-bottom:2px solid {skin['rule']};padding-bottom:4px;}}
.muted{{color:{skin['muted']};}}.kpi-row{{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0;}}
.kpi{{border:1px solid {skin['kpi_border']};border-radius:8px;padding:10px 14px;min-width:150px;}}
.kpi-label{{font-size:.75rem;color:{skin['muted']};text-transform:uppercase;letter-spacing:.05em;}}
.kpi-value{{font-size:1.15rem;font-weight:700;}}
.explain{{background:{skin['explain_bg']};border-left:4px solid {skin['explain_edge']};padding:8px 12px;margin:8px 0 20px;font-size:.9rem;}}
.insight{{background:{skin['explain_bg']};border-left:4px solid {skin['explain_edge']};padding:8px 12px;margin:8px 0;font-size:.9rem;}}
.rec{{border:1px solid {skin['kpi_border']};border-radius:8px;padding:10px 14px;margin:8px 0;}}
.rec .pill{{display:inline-block;border-radius:999px;padding:2px 10px;font-size:.72rem;font-weight:700;background:{skin['explain_bg']};color:{skin['explain_edge']};}}
.rec .muted{{font-size:.88rem;margin-top:4px;}}.rec .evidence{{font-size:.82rem;margin-top:4px;font-weight:600;}}
.fig{{margin:12px 0;}}table.tbl{{border-collapse:collapse;width:100%;font-size:.82rem;margin:8px 0 20px;}}
table.tbl th,table.tbl td{{border:1px solid {skin['cell_border']};padding:6px 8px;text-align:left;}}
table.tbl th{{background:{skin['th_bg']};}}
</style></head>
<body>
<h1>Customer Intelligence Report</h1>
<p class="muted">Generated {html.escape(str(context.get('generated_at', '')))} · {html.escape(masked_note)} ·
Historical monetary value is a value proxy, never CLV. Retention reasons never claim causal impact.</p>
{body}
</body></html>"""


#: Registered (body, bold) font names for the PDF story, cached per process.
_PDF_FONTS: dict[str, tuple[str, str]] = {}


def _pdf_fonts() -> tuple[str, str]:
    """(body, bold) font names for PDF paragraphs.

    DejaVu (shipped with matplotlib) carries the currency symbol and arrows the
    report uses; Helvetica is the fallback when the font files are unavailable.
    """
    if "fonts" in _PDF_FONTS:
        return _PDF_FONTS["fonts"]
    body, bold = "Helvetica", "Helvetica-Bold"
    try:
        from pathlib import Path

        import matplotlib
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        regular, strong = font_dir / "DejaVuSans.ttf", font_dir / "DejaVuSans-Bold.ttf"
        if regular.exists() and strong.exists():
            registered = set(pdfmetrics.getRegisteredFontNames())
            if "ReportBody" not in registered:
                pdfmetrics.registerFont(TTFont("ReportBody", str(regular)))
            if "ReportBold" not in registered:
                pdfmetrics.registerFont(TTFont("ReportBold", str(strong)))
            body, bold = "ReportBody", "ReportBold"
    except Exception:  # noqa: BLE001 - fonts are cosmetic; exports must never fail on them
        body, bold = "Helvetica", "Helvetica-Bold"
    _PDF_FONTS["fonts"] = (body, bold)
    return body, bold


def _pdf_styles() -> Any:
    """Sample stylesheet rewired to the report fonts and print sizes."""
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()
    body, bold = _pdf_fonts()
    styles["Title"].fontName = bold
    styles["Title"].fontSize = 20
    styles["Title"].leading = 24
    for name, size, before, after in (
        ("Heading2", 13, 14, 6),
        ("Heading3", 10, 8, 3),
    ):
        styles[name].fontName = bold
        styles[name].fontSize = size
        styles[name].spaceBefore = before
        styles[name].spaceAfter = after
    for name in ("Normal", "Italic"):
        styles[name].fontName = body
        styles[name].fontSize = 8.8
        styles[name].leading = 11.8
    return styles


def build_pdf_bytes(
    context: dict[str, Any],
    mask_ids: bool = True,
    sections: tuple[str, ...] = REPORT_SECTIONS,
    chart_images: Mapping[str, bytes] | None = None,
) -> bytes:
    """Print-ready PDF with KPIs, narratives, embedded charts, and tables.

    Charts are rendered to PNG from the same context tables the live page uses
    and placed next to their reading guide; a missing engine or a sparse table
    leaves that figure as a short note instead of failing the export. The
    document closes with the interpretation recap and prioritized actions.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import Image as RLImage
        from reportlab.platypus import (
            KeepTogether,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise CustomerIntelligenceError(
            "PDF export needs the 'reportlab' package. Install it with "
            "'pip install reportlab' and try again."
        ) from exc

    charts = (
        dict(chart_images)
        if chart_images is not None
        else report_charts.render_report_charts(context)
    )
    chart_note = report_charts.available_chart_engine()
    buffer = BytesIO()
    margin = 1.5 * cm
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=margin,
        bottomMargin=margin,
        leftMargin=margin,
        rightMargin=margin,
    )
    content_width = A4[0] - 2 * margin
    styles = _pdf_styles()
    cell_style = ParagraphStyle("ReportCell", parent=styles["Normal"], fontSize=6.5, leading=8)
    head_style = ParagraphStyle(
        "ReportHead", parent=cell_style, fontName=_pdf_fonts()[1], fontSize=6.8, leading=8.4
    )
    story: list[Any] = [
        Paragraph("Customer Intelligence Report", styles["Title"]),
        Paragraph(
            f"Generated {html.escape(str(context.get('generated_at', '')))}. "
            "Historical value is a proxy, never CLV. Actions are decision support only.",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]

    def _insight(item: report_narrative.Insight) -> None:
        story.append(
            Paragraph(
                f"<b>{html.escape(_title_sentence(item.title))}</b> " f"{html.escape(item.detail)}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 3))

    def _table(frame: pd.DataFrame, title: str, max_rows: int = 25, level: int = 2) -> None:
        story.append(Paragraph(title, styles["Heading2" if level == 2 else "Heading3"]))
        if frame.empty:
            story.append(Paragraph("No data for this section yet.", styles["Normal"]))
            return
        view = frame.head(max_rows).astype(str)
        width = content_width / max(len(view.columns), 1)
        data = [
            [Paragraph(html.escape(str(name)), head_style) for name in view.columns],
            *[
                [Paragraph(html.escape(str(value)), cell_style) for value in row]
                for row in view.itertuples(index=False)
            ],
        ]
        table = Table(data, colWidths=[width] * len(view.columns), repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

    def _chart(key: str) -> None:
        exp = CHART_EXPLANATIONS.get(key, {})
        block: list[Any] = [Paragraph(html.escape(exp.get("title", key)), styles["Heading3"])]
        png = charts.get(key)
        if png:
            reader = ImageReader(BytesIO(png))
            pixel_width, pixel_height = reader.getSize()
            block.append(
                RLImage(
                    BytesIO(png),
                    width=content_width,
                    height=content_width * pixel_height / max(pixel_width, 1),
                )
            )
        else:
            block.append(
                Paragraph(
                    chart_note
                    or "Chart image unavailable — the source table had no rows " "for this figure.",
                    styles["Italic"],
                )
            )
        if exp:
            block.append(
                Paragraph(
                    f"<b>What it shows.</b> {html.escape(exp.get('what', ''))} "
                    f"<b>How to read it.</b> {html.escape(exp.get('how_to_read', ''))} "
                    f"<b>What to do.</b> {html.escape(exp.get('action', ''))}",
                    styles["Normal"],
                )
            )
        block.append(Spacer(1, 8))
        story.append(KeepTogether(block))

    kpis = context.get("kpis") or {}
    if "summary" in sections:
        story.append(
            Paragraph(_section_heading("summary", "Executive summary"), styles["Heading2"])
        )
        if kpis:
            _table(
                pd.DataFrame(
                    [
                        {"Metric": key, "Value": _kpi_value(key, value)}
                        for key, value in kpis.items()
                    ]
                ),
                "Key metrics",
                level=3,
            )
        for item in report_narrative.summary_insights(context):
            _insight(item)

    for section in CHARTED_SECTIONS:
        if section not in sections:
            continue
        story.append(
            Paragraph(_section_heading(section, SECTION_LABELS[section]), styles["Heading2"])
        )
        metadata = context.get("churn_metadata") or {}
        if section == "churn" and metadata:
            story.append(
                Paragraph(
                    f"Model <b>{html.escape(str(metadata.get('model', '–')))}</b> · threshold "
                    f"{html.escape(str(metadata.get('threshold', '–')))} · cutoff "
                    f"{html.escape(str(metadata.get('cutoff', '–')))}. Predictors use transactions "
                    "on or before the cutoff only; the label is activity after it — leakage-safe "
                    "by construction.",
                    styles["Normal"],
                )
            )
            story.append(Spacer(1, 4))
        if section == "comparison":
            comparison = context.get("comparison") or {}
            if comparison:
                rec = comparison.get("recommendation") or {}
                pair = comparison.get("pair") or {}
                if pair:
                    story.append(
                        Paragraph(
                            f"Pair agreement: <b>{html.escape(str(pair.get('reference')))}</b> vs "
                            f"<b>{html.escape(str(pair.get('target')))}</b> · Adjusted Rand Index "
                            f"{html.escape(formatting.fmt_float(pair.get('ari'), 3))} "
                            "(1.0 = identical, 0 = chance-level).",
                            styles["Normal"],
                        )
                    )
                if rec.get("rationale"):
                    story.append(
                        Paragraph(html.escape(str(rec.get("rationale", ""))), styles["Normal"])
                    )
                scorecard = scorecard_as_text(comparison.get("table"))
                if not scorecard.empty:
                    _table(scorecard, "Model scorecard", max_rows=20, level=3)
                cross = pair.get("cross_tab") if pair else None
                if isinstance(cross, pd.DataFrame) and not cross.empty:
                    _table(cross.astype(str), "Customer flow (counts)", max_rows=12, level=3)
        for item in report_narrative.section_insights(context, section):
            _insight(item)
        for key in SECTION_CHARTS[section]:
            _chart(key)

    if "appendix" in sections:
        profile = context.get("data_profile") or {}
        treatments = context.get("treatments") or []
        story.append(
            Paragraph(_section_heading("appendix", "Appendix — data & method"), styles["Heading2"])
        )
        for item in report_narrative.section_insights(context, "appendix"):
            _insight(item)
        story.append(
            Paragraph(
                f"Source <b>{html.escape(str(profile.get('source_filename', '–')))}</b> · "
                f"{int(profile.get('rows', 0)):,} clean rows × {int(profile.get('columns', 0))} "
                f"columns · {int(profile.get('customers', 0)):,} customers · "
                f"{float(profile.get('missing_pct', 0)):.2f}% cells missing.",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 4))
        story.append(Paragraph("Cleaning treatments", styles["Heading3"]))
        if treatments:
            for treatment in treatments:
                story.append(
                    Paragraph(
                        f"• {html.escape(str(treatment['treatment']))}: "
                        f"{html.escape(str(treatment['action']))} "
                        f"({int(treatment['records_affected']):,} records)",
                        styles["Normal"],
                    )
                )
        else:
            story.append(
                Paragraph("No cleaning treatments recorded for this dataset.", styles["Normal"])
            )
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                "Method: churn is leakage-safe by construction (predictors ≤ cutoff, label from "
                "the outcome window; cohort-ordered splits). RFM = 100R+10F+M; cluster IDs are "
                "arbitrary until profiled against population medians.",
                styles["Normal"],
            )
        )
        story.append(
            Paragraph(
                (
                    "Customer IDs are masked in this report."
                    if mask_ids
                    else "Customer IDs are shown in full (authorized export)."
                ),
                styles["Normal"],
            )
        )
        for sheet, frame in context_tables(context, mask_ids=mask_ids).items():
            _table(frame, f"Table — {sheet}", level=3)

    if "recommendations" in sections:
        recap = _closing_insights(context, sections)
        recs = report_narrative.recommendations(context)
        story.append(
            Paragraph(
                _section_heading("recommendations", "Recommendations & next steps"),
                styles["Heading2"],
            )
        )
        story.append(Paragraph("Interpretations at a glance", styles["Heading3"]))
        if recap:
            for item in recap:
                _insight(item)
        else:
            story.append(
                Paragraph(
                    "Section interpretations appear as the upstream results arrive.",
                    styles["Normal"],
                )
            )
        story.append(Paragraph("Prioritized actions", styles["Heading3"]))
        if recs:
            for rec in recs:
                story.append(
                    Paragraph(
                        f"<b>{html.escape(rec.priority)} — "
                        f"{html.escape(_title_sentence(rec.title))}</b> "
                        f"{html.escape(rec.rationale)} <i>{html.escape(rec.evidence)}</i>",
                        styles["Normal"],
                    )
                )
                story.append(Spacer(1, 4))
        else:
            story.append(
                Paragraph("Recommendations appear once retention results exist.", styles["Normal"])
            )
        story.append(
            Paragraph(
                "Priority order is decision support over the computed results; it does not claim "
                "causal impact.",
                styles["Italic"],
            )
        )

    doc.build(story)
    return buffer.getvalue()


def build_docx_bytes(
    context: dict[str, Any],
    mask_ids: bool = True,
    sections: tuple[str, ...] = REPORT_SECTIONS,
    chart_images: Mapping[str, bytes] | None = None,
) -> bytes:
    """Editable Word report with the same sections, charts, and narrative as PDF.

    Charts are embedded as PNG images so the .docx opens with graphs on any
    machine; ``python-docx`` is optional and raises install guidance when absent.
    """
    try:
        from docx import Document
        from docx.shared import Inches, Pt
    except ImportError as exc:
        raise CustomerIntelligenceError(
            "Word export needs the 'python-docx' package. Install it with "
            "'pip install python-docx' and try again."
        ) from exc

    charts = (
        dict(chart_images)
        if chart_images is not None
        else report_charts.render_report_charts(context)
    )
    chart_note = report_charts.available_chart_engine()
    document = Document()
    document.styles["Normal"].font.size = Pt(9.5)
    document.styles["Normal"].font.name = "Calibri"
    document.add_heading("Customer Intelligence Report", level=0)
    meta = document.add_paragraph()
    meta.add_run(
        f"Generated {context.get('generated_at', '')}. Historical value is a proxy, never CLV. "
        "Actions are decision support only."
    ).italic = True
    meta.add_run(
        " Customer IDs are masked in this report."
        if mask_ids
        else " Customer IDs are shown in full (authorized export)."
    ).italic = True

    def _insight(item: report_narrative.Insight) -> None:
        paragraph = document.add_paragraph()
        paragraph.add_run(_title_sentence(item.title)).bold = True
        paragraph.add_run(f" {item.detail}")

    def _table(frame: pd.DataFrame, title: str, max_rows: int = 25, level: int = 2) -> None:
        document.add_heading(title, level=level)
        if frame.empty:
            document.add_paragraph("No data for this section yet.")
            return
        view = frame.head(max_rows).astype(str)
        table = document.add_table(rows=1, cols=len(view.columns))
        table.style = "Table Grid"
        header = table.rows[0].cells
        for cell, name in zip(header, view.columns, strict=True):
            cell.text = str(name)
            for run in cell.paragraphs[0].runs:
                run.bold = True
                run.font.size = Pt(7.5)
        for record in view.itertuples(index=False):
            cells = table.add_row().cells
            for cell, value in zip(cells, record, strict=True):
                cell.text = str(value)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(7.5)
        document.add_paragraph()

    def _chart(key: str) -> None:
        exp = CHART_EXPLANATIONS.get(key, {})
        document.add_heading(exp.get("title", key), level=2)
        png = charts.get(key)
        if png:
            document.add_picture(BytesIO(png), width=Inches(6.1))
        else:
            document.add_paragraph(
                chart_note
                or "Chart image unavailable — the source table had no rows for this figure."
            ).italic = True
        if exp:
            caption = document.add_paragraph()
            caption.add_run("What it shows. ").bold = True
            caption.add_run(exp.get("what", ""))
            caption.add_run(" How to read it. ").bold = True
            caption.add_run(exp.get("how_to_read", ""))
            caption.add_run(" What to do. ").bold = True
            caption.add_run(exp.get("action", ""))
            for run in caption.runs:
                run.italic = True
                run.font.size = Pt(8.5)

    if "summary" in sections:
        document.add_heading(_section_heading("summary", "Executive summary"), level=1)
        kpis = context.get("kpis") or {}
        if kpis:
            _table(
                pd.DataFrame(
                    [
                        {"Metric": key, "Value": _kpi_value(key, value)}
                        for key, value in kpis.items()
                    ]
                ),
                "Key metrics",
                level=3,
            )
        for item in report_narrative.summary_insights(context):
            _insight(item)

    for section in CHARTED_SECTIONS:
        if section not in sections:
            continue
        document.add_heading(_section_heading(section, SECTION_LABELS[section]), level=1)
        metadata = context.get("churn_metadata") or {}
        if section == "churn" and metadata:
            document.add_paragraph(
                f"Model {metadata.get('model', '–')} · threshold {metadata.get('threshold', '–')} · "
                f"cutoff {metadata.get('cutoff', '–')}. Predictors use transactions on or before "
                "the cutoff only; the label is activity after it — leakage-safe by construction."
            ).italic = True
        if section == "comparison":
            comparison = context.get("comparison") or {}
            if comparison:
                rec = comparison.get("recommendation") or {}
                pair = comparison.get("pair") or {}
                if pair:
                    document.add_paragraph(
                        f"Pair agreement: {pair.get('reference')} vs {pair.get('target')} · "
                        f"Adjusted Rand Index {formatting.fmt_float(pair.get('ari'), 3)} "
                        "(1.0 = identical, 0 = chance-level)."
                    )
                if rec.get("rationale"):
                    document.add_paragraph(str(rec.get("rationale", "")))
                scorecard = scorecard_as_text(comparison.get("table"))
                if not scorecard.empty:
                    _table(scorecard, "Model scorecard", level=3)
                cross = pair.get("cross_tab") if pair else None
                if isinstance(cross, pd.DataFrame) and not cross.empty:
                    _table(cross.astype(str), "Customer flow (counts)", level=3)
        for item in report_narrative.section_insights(context, section):
            _insight(item)
        for key in SECTION_CHARTS[section]:
            _chart(key)

    if "appendix" in sections:
        profile = context.get("data_profile") or {}
        treatments = context.get("treatments") or []
        document.add_heading(_section_heading("appendix", "Appendix — data & method"), level=1)
        for item in report_narrative.section_insights(context, "appendix"):
            _insight(item)
        document.add_paragraph(
            f"Source {profile.get('source_filename', '–')} · "
            f"{int(profile.get('rows', 0)):,} clean rows × {int(profile.get('columns', 0))} "
            f"columns · {int(profile.get('customers', 0)):,} customers · "
            f"{float(profile.get('missing_pct', 0)):.2f}% cells missing."
        )
        document.add_heading("Cleaning treatments", level=2)
        if treatments:
            for treatment in treatments:
                document.add_paragraph(
                    f"{treatment['treatment']}: {treatment['action']} "
                    f"({int(treatment['records_affected']):,} records)",
                    style="List Bullet",
                )
        else:
            document.add_paragraph("No cleaning treatments recorded for this dataset.")
        document.add_paragraph(
            "Method: churn is leakage-safe by construction (predictors ≤ cutoff, label from the "
            "outcome window; cohort-ordered splits). RFM = 100R+10F+M; cluster IDs are arbitrary "
            "until profiled against population medians."
        ).italic = True
        for sheet, frame in context_tables(context, mask_ids=mask_ids).items():
            _table(frame, f"Table — {sheet}", level=3)

    if "recommendations" in sections:
        recap = _closing_insights(context, sections)
        recs = report_narrative.recommendations(context)
        document.add_heading(
            _section_heading("recommendations", "Recommendations & next steps"), level=1
        )
        document.add_heading("Interpretations at a glance", level=2)
        if recap:
            for item in recap:
                _insight(item)
        else:
            document.add_paragraph("Section interpretations appear as the upstream results arrive.")
        document.add_heading("Prioritized actions", level=2)
        if recs:
            for rec in recs:
                paragraph = document.add_paragraph()
                paragraph.add_run(f"{rec.priority} — {_title_sentence(rec.title)}").bold = True
                paragraph.add_run(f" {rec.rationale} ")
                paragraph.add_run(rec.evidence).italic = True
        else:
            document.add_paragraph("Recommendations appear once retention results exist.")
        document.add_paragraph(
            "Priority order is decision support over the computed results; it does not claim "
            "causal impact."
        ).italic = True

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def available_pdf_engine() -> str | None:
    """None when reportlab is installed, else an install-guidance message."""
    try:
        import reportlab  # noqa: F401
    except ImportError:
        return "PDF export needs the 'reportlab' package. Install it with 'pip install reportlab'."
    return None


def available_docx_engine() -> str | None:
    """None when python-docx is installed, else an install-guidance message."""
    try:
        import docx  # noqa: F401
    except ImportError:
        return (
            "Word export needs the 'python-docx' package. Install it with "
            "'pip install python-docx'."
        )
    return None


def available_chart_engine() -> str | None:
    """None when chart images can be rendered, else an install-guidance message."""
    return report_charts.available_chart_engine()


def report_readiness_lines(context: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Checklist rows of (label, ready, hint) for the page."""
    readiness = context.get("readiness") or {}
    return [
        ("Clean data", bool(readiness.get("clean_data")), "Finish Data Quality & Cleaning."),
        (
            "Customer features + RFM",
            bool(readiness.get("features") and readiness.get("rfm")),
            "Run scoring on RFM & LRFM Analysis.",
        ),
        (
            "AI segments (optional)",
            bool(readiness.get("ai_segments")),
            "Run Advanced Segmentation.",
        ),
        ("Churn scores", bool(readiness.get("churn")), "Train and score on Churn Prediction."),
        (
            "Retention actions",
            bool(readiness.get("retention")),
            "Build the table on Retention Strategies.",
        ),
    ]


__all__ = [
    "ANALYTICAL_SECTIONS",
    "CHARTED_SECTIONS",
    "CHART_EXPLANATIONS",
    "REPORT_SECTIONS",
    "SECTION_CHARTS",
    "SECTION_LABELS",
    "available_chart_engine",
    "available_docx_engine",
    "available_pdf_engine",
    "build_docx_bytes",
    "build_pdf_bytes",
    "build_report_context",
    "context_tables",
    "context_to_excel_bytes",
    "render_html_report",
    "report_readiness_lines",
]

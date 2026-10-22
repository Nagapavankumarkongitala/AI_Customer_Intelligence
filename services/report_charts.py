"""Static chart images for the PDF and Word report exports.

The interactive HTML export embeds the live Plotly figures. PDF and Word cannot
carry Plotly markup, so this module re-draws the report charts from the same
context tables and returns PNG bytes keyed by the ``CHART_EXPLANATIONS`` keys.
Matplotlib is optional by design: when it is missing the exporters still build,
with a plain note in place of the image (never a hard failure).

Every chart is cosmetic over numbers computed upstream — no chart recalculates
segments, risk, or retention, and historical value is always a proxy, never CLV.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

from utils.formatting import currency_symbol, fmt_compact, fmt_int
from visualization import tokens

logger = logging.getLogger(__name__)

try:  # optional dependency: exports degrade to text when absent
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator
except ImportError:  # pragma: no cover - exercised only without matplotlib
    plt = None  # type: ignore[assignment]
    Line2D = None  # type: ignore[assignment]
    FuncFormatter = None  # type: ignore[assignment]
    MaxNLocator = None  # type: ignore[assignment]

_LIGHT = tokens.LIGHT
_MUTED = _LIGHT["muted"]
_BORDER = _LIGHT["border"]
_GRID = _LIGHT["grid"]
_PRIMARY = _LIGHT["primary"]
_SECONDARY = _LIGHT["secondary"]
_POSITIVE = _LIGHT["positive"]
_DISCRETE = tokens.discrete_colors("light")

#: Series order used by the risk-band chart (unknown bands follow, sorted).
_RISK_ORDER = ("low", "medium", "high", "critical")

#: Figure geometry in inches; the exporters scale the PNG to the page width.
_FIGSIZE = (7.4, 3.0)
_DPI = 150

#: Up to this many x labels are drawn before ticks are thinned.
_MAX_TICKS = 9

#: Segment-trend charts stay readable with a capped legend.
_MAX_LEGEND = 10

#: Scatter charts sample deterministically past this many points.
_MAX_POINTS = 1200


def available_chart_engine() -> str | None:
    """None when matplotlib is installed, else an install-guidance message."""
    if plt is None:
        return (
            "Chart images need the 'matplotlib' package. Install it with "
            "'pip install matplotlib' to embed charts in PDF and Word exports."
        )
    return None


def _as_frame(value: Any) -> pd.DataFrame | None:
    """Return value when it is a non-empty DataFrame, else None."""
    if isinstance(value, pd.DataFrame) and not value.empty:
        return value
    return None


def _short(text: Any, limit: int = 30) -> str:
    """Trim a long category label so it fits a printed axis."""
    label = str(text)
    return label if len(label) <= limit else f"{label[: limit - 1]}…"


def _compact_money(value: float) -> str:
    """Compact currency label for ticks and bar ends (₹1.2M)."""
    return f"{currency_symbol()}{fmt_compact(value)}"


def _axes(figsize: tuple[float, float] = _FIGSIZE) -> tuple[Any, Any]:
    """Styled light-surface figure/axes pair shared by every chart."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BORDER)
    ax.tick_params(colors=_MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=_GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig: Any) -> bytes:
    """Close a figure and return its PNG bytes."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def _thin_ticks(ax: Any, positions: Iterable[int], labels: list[str]) -> None:
    """Draw at most ``_MAX_TICKS`` evenly spaced labels over categorical x."""
    positions = list(positions)
    if not positions:
        return
    step = max(1, int(np.ceil(len(positions) / _MAX_TICKS)))
    ax.set_xticks(positions[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")


def _horizontal_bars(
    labels: list[str],
    values: list[float],
    colors: list[str],
    money: bool = False,
) -> bytes:
    """Horizontal bar chart with value labels (long category names stay legible).

    ``money`` picks the axis and label format: compact currency when the values
    are revenue, whole integers when they are counts. The figure grows with the
    category count so ten segments stay legible instead of crushing the labels.
    """
    height = min(5.2, max(_FIGSIZE[1], 0.32 * len(values) + 1.0))
    fig, ax = _axes((_FIGSIZE[0], height))
    positions = np.arange(len(values))
    bars = ax.barh(positions, values, color=colors, height=0.68)
    ax.set_yticks(positions)
    ax.set_yticklabels([_short(label) for label in labels], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=_GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    if money:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: _compact_money(value)))
        end_labels = [_compact_money(value) for value in values]
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        end_labels = [fmt_int(value) for value in values]
    ax.bar_label(bars, labels=end_labels, padding=3, fontsize=7)
    top = max(values, default=0.0) * 1.18
    ax.set_xlim(0, top if top > 0 else 1)
    return _save(fig)


def _revenue_trend(context: dict[str, Any]) -> bytes | None:
    """Monthly revenue line with a light fill."""
    frame = _as_frame((context.get("trends") or {}).get("revenue"))
    if frame is None or not {"month", "revenue"} <= set(frame.columns):
        return None
    data = frame[["month", "revenue"]].copy()
    data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce").fillna(0.0)
    fig, ax = _axes()
    x = np.arange(len(data))
    ax.plot(x, data["revenue"], color=_PRIMARY, linewidth=2, marker="o", markersize=4)
    ax.fill_between(x, data["revenue"], color=_PRIMARY, alpha=0.12)
    _thin_ticks(ax, x, [str(month) for month in data["month"]])
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: _compact_money(value)))
    return _save(fig)


def _activity_trend(context: dict[str, Any]) -> bytes | None:
    """Monthly distinct active customers."""
    frame = _as_frame((context.get("trends") or {}).get("activity"))
    if frame is None or not {"month", "active_customers"} <= set(frame.columns):
        return None
    data = frame[["month", "active_customers"]].copy()
    data["active_customers"] = pd.to_numeric(data["active_customers"], errors="coerce").fillna(0.0)
    fig, ax = _axes()
    x = np.arange(len(data))
    ax.plot(x, data["active_customers"], color=_SECONDARY, linewidth=2, marker="o", markersize=4)
    _thin_ticks(ax, x, [str(month) for month in data["month"]])
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: fmt_int(value)))
    return _save(fig)


def _new_returning(context: dict[str, Any]) -> bytes | None:
    """Monthly new vs returning customers as a stacked bar."""
    frame = _as_frame((context.get("trends") or {}).get("cohort"))
    needed = {"month", "new_customers", "returning_customers"}
    if frame is None or not needed <= set(frame.columns):
        return None
    data = frame[["month", "new_customers", "returning_customers"]].copy()
    for column in ("new_customers", "returning_customers"):
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    fig, ax = _axes()
    x = np.arange(len(data))
    ax.bar(x, data["returning_customers"], color=_PRIMARY, label="Returning", width=0.68)
    ax.bar(
        x,
        data["new_customers"],
        bottom=data["returning_customers"],
        color=_POSITIVE,
        label="New",
        width=0.68,
    )
    _thin_ticks(ax, x, [str(month) for month in data["month"]])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: fmt_int(value)))
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
    )
    return _save(fig)


def _segment_distribution(context: dict[str, Any]) -> bytes | None:
    """Customer counts per RFM segment."""
    frame = _as_frame(context.get("segment_portfolio"))
    if frame is None or not {"rfm_segment", "customers"} <= set(frame.columns):
        return None
    data = frame.sort_values("customers", ascending=True)
    return _horizontal_bars(
        [str(value) for value in data["rfm_segment"]],
        [float(value) for value in data["customers"]],
        [tokens.segment_color(value, "light") for value in data["rfm_segment"]],
    )


def _revenue_by_segment(context: dict[str, Any]) -> bytes | None:
    """Historical revenue per RFM segment."""
    frame = _as_frame(context.get("segment_portfolio"))
    if frame is None or not {"rfm_segment", "revenue"} <= set(frame.columns):
        return None
    data = frame.sort_values("revenue", ascending=True)
    return _horizontal_bars(
        [str(value) for value in data["rfm_segment"]],
        [float(value) for value in data["revenue"]],
        [tokens.segment_color(value, "light") for value in data["rfm_segment"]],
        money=True,
    )


def _segment_trends(context: dict[str, Any]) -> bytes | None:
    """Stacked customer counts per RFM segment across saved periods."""
    history = _as_frame(context.get("history"))
    if history is None or not {"period", "segment", "customers"} <= set(history.columns):
        return None
    data = history[history["model"] == "RFM"] if "model" in history.columns else history
    if data.empty:
        return None
    pivot = data.pivot_table(
        index="period", columns="segment", values="customers", aggfunc="sum", fill_value=0
    )
    if pivot.empty:
        return None
    pivot = pivot.tail(12)
    if len(pivot.columns) > _MAX_LEGEND:
        totals = pivot.sum(axis=0).sort_values(ascending=False)
        keep = list(totals.index[: _MAX_LEGEND - 1])
        other = pivot.drop(columns=keep).sum(axis=1)
        pivot = pivot[keep].copy()
        pivot["Other segments"] = other
    periods = [str(period) for period in pivot.index]
    x = np.arange(len(periods))
    colors = [tokens.segment_color(column, "light") for column in pivot.columns]
    fig, ax = _axes()
    ax.stackplot(
        x, [pivot[column].to_numpy() for column in pivot.columns], colors=colors, alpha=0.9
    )
    _thin_ticks(ax, x, periods)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: fmt_int(value)))
    ax.grid(axis="x", visible=False)
    ax.legend(
        [str(column) for column in pivot.columns],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=3,
        fontsize=7,
        frameon=False,
    )
    return _save(fig)


def _demographics_age(context: dict[str, Any]) -> bytes | None:
    """Customer counts per age band."""
    frame = _as_frame((context.get("demographics") or {}).get("age_bands"))
    if frame is None or not {"age_band", "customers"} <= set(frame.columns):
        return None
    data = frame.copy()
    fig, ax = _axes()
    x = np.arange(len(data))
    bars = ax.bar(x, data["customers"], color=_DISCRETE[: len(data)], width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in data["age_band"]], fontsize=7.5)
    ax.bar_label(
        bars, labels=[fmt_int(value) for value in data["customers"]], fontsize=7, padding=3
    )
    ax.set_ylim(0, float(data["customers"].max() or 0) * 1.16)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return _save(fig)


def _demographics_gender(context: dict[str, Any]) -> bytes | None:
    """Customer counts per gender value."""
    frame = _as_frame((context.get("demographics") or {}).get("gender"))
    if frame is None or not {"gender", "customers"} <= set(frame.columns):
        return None
    data = frame.copy()
    fig, ax = _axes()
    x = np.arange(len(data))
    bars = ax.bar(x, data["customers"], color=_DISCRETE[: len(data)], width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(label, 14) for label in data["gender"]], fontsize=7.5)
    ax.bar_label(
        bars, labels=[fmt_int(value) for value in data["customers"]], fontsize=7, padding=3
    )
    ax.set_ylim(0, float(data["customers"].max() or 0) * 1.16)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return _save(fig)


def _demographics_region(context: dict[str, Any]) -> bytes | None:
    """Top regions by customer count."""
    frame = _as_frame((context.get("demographics") or {}).get("region"))
    if frame is None or not {"region", "customers"} <= set(frame.columns):
        return None
    data = frame.sort_values("customers", ascending=True)
    return _horizontal_bars(
        [str(value) for value in data["region"]],
        [float(value) for value in data["customers"]],
        [_DISCRETE[index % len(_DISCRETE)] for index in range(len(data))],
    )


def _risk_bands(context: dict[str, Any]) -> bytes | None:
    """Scored customers per churn risk band."""
    mix = context.get("risk_mix") or {}
    if not isinstance(mix, dict) or not mix:
        return None
    ordered = [band for band in _RISK_ORDER if band in mix]
    ordered += sorted(band for band in mix if band not in _RISK_ORDER)
    values = [int(mix[band]) for band in ordered]
    labels = [band.capitalize() for band in ordered]
    fig, ax = _axes()
    x = np.arange(len(ordered))
    bars = ax.bar(
        x, values, color=[tokens.risk_color(band, "light") for band in ordered], width=0.62
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.bar_label(bars, labels=[fmt_int(value) for value in values], fontsize=8, padding=3)
    ax.set_ylim(0, max(values, default=0) * 1.16)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: fmt_int(value)))
    return _save(fig)


def _value_risk_matrix(context: dict[str, Any]) -> bytes | None:
    """Churn probability vs historical value, bubble-sized by revenue at risk."""
    frame = _as_frame(context.get("retention"))
    needed = {"churn_probability", "expected_value_proxy_historical"}
    if frame is None or not needed <= set(frame.columns):
        return None
    columns = [column for column in needed if column in frame.columns]
    if "revenue_at_risk" in frame.columns:
        columns.append("revenue_at_risk")
    if "risk_band" in frame.columns:
        columns.append("risk_band")
    data = frame[columns].copy()
    data["churn_probability"] = pd.to_numeric(data["churn_probability"], errors="coerce")
    data["expected_value_proxy_historical"] = pd.to_numeric(
        data["expected_value_proxy_historical"], errors="coerce"
    )
    data = data.dropna(subset=["churn_probability", "expected_value_proxy_historical"])
    if data.empty:
        return None
    if len(data) > _MAX_POINTS:
        data = data.sample(n=_MAX_POINTS, random_state=42)
    if "revenue_at_risk" in data.columns:
        at_risk = pd.to_numeric(data["revenue_at_risk"], errors="coerce").fillna(0.0)
    else:
        at_risk = pd.Series(0.0, index=data.index)
    peak = float(at_risk.max()) or 1.0
    sizes = 18 + 300 * np.sqrt(at_risk / peak)

    fig, ax = _axes()
    groups = (
        list(data["risk_band"].astype(str)) if "risk_band" in data.columns else ["all"] * len(data)
    )
    data = data.assign(_band=groups)
    handles: list[Any] = []
    for band in dict.fromkeys(groups):
        subset = data[data["_band"] == band]
        color = tokens.risk_color(band, "light") if band != "all" else _PRIMARY
        ax.scatter(
            subset["churn_probability"],
            subset["expected_value_proxy_historical"],
            s=sizes.loc[subset.index],
            color=color,
            alpha=0.55,
            edgecolors="white",
            linewidths=0.4,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=color,
                label=band.capitalize(),
                markersize=6,
            )
        )
    ax.axvline(0.6, color=_MUTED, linestyle="--", linewidth=0.9, alpha=0.8)
    median_value = float(data["expected_value_proxy_historical"].median())
    ax.axhline(median_value, color=_MUTED, linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set_xlabel("Churn probability", color=_MUTED, fontsize=8)
    ax.set_ylabel("Historical value (proxy, not CLV)", color=_MUTED, fontsize=8)
    ax.set_xlim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: _compact_money(value)))
    ax.grid(axis="x", color=_GRID, linewidth=0.9)
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left")
    ax.text(
        1.0,
        1.0,
        "Act first",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=_MUTED,
        fontsize=8,
    )
    return _save(fig)


def _action_totals(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Revenue at risk and customer counts per recommended action."""
    if "recommended_action" not in frame.columns:
        return None
    grouped = frame.groupby("recommended_action").size().rename("customers").reset_index()
    if "revenue_at_risk" in frame.columns:
        revenue = (
            frame.groupby("recommended_action")["revenue_at_risk"]
            .sum()
            .rename("revenue_at_risk")
            .reset_index()
        )
        grouped = grouped.merge(revenue, on="recommended_action", how="left")
    else:
        grouped["revenue_at_risk"] = 0.0
    return grouped


def _revenue_at_risk(context: dict[str, Any]) -> bytes | None:
    """Historical value under modeled risk per recommended action."""
    frame = _as_frame(context.get("retention"))
    if frame is None:
        return None
    grouped = _action_totals(frame)
    if grouped is None:
        return None
    data = grouped.sort_values("revenue_at_risk", ascending=True)
    return _horizontal_bars(
        [str(value) for value in data["recommended_action"]],
        [float(value) for value in data["revenue_at_risk"]],
        [tokens.action_color(value, "light") for value in data["recommended_action"]],
        money=True,
    )


def _strategy_distribution(context: dict[str, Any]) -> bytes | None:
    """Customers per recommended retention action."""
    frame = _as_frame(context.get("retention"))
    if frame is None:
        return None
    grouped = _action_totals(frame)
    if grouped is None:
        return None
    data = grouped.sort_values("customers", ascending=True)
    return _horizontal_bars(
        [str(value) for value in data["recommended_action"]],
        [float(value) for value in data["customers"]],
        [tokens.action_color(value, "light") for value in data["recommended_action"]],
    )


def _priority_histogram(context: dict[str, Any]) -> bytes | None:
    """Priority-score distribution with its cumulative share."""
    frame = _as_frame(context.get("retention"))
    if frame is None or "retention_priority_score" not in frame.columns:
        return None
    scores = pd.to_numeric(frame["retention_priority_score"], errors="coerce").dropna()
    if scores.empty:
        return None
    counts, edges = np.histogram(scores, bins=20)
    centers = (edges[:-1] + edges[1:]) / 2
    cumulative = np.cumsum(counts) / max(int(counts.sum()), 1) * 100
    fig, ax = _axes()
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=_PRIMARY, alpha=0.85)
    ax.set_xlabel("Retention priority score", color=_MUTED, fontsize=8)
    ax.set_ylabel("Customers", color=_MUTED, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: fmt_int(value)))
    twin = ax.twinx()
    twin.plot(centers, cumulative, color=_SECONDARY, linewidth=1.8, marker="o", markersize=3)
    twin.set_ylim(0, 105)
    twin.set_ylabel("Cumulative %", color=_MUTED, fontsize=8)
    twin.tick_params(colors=_MUTED, labelsize=7.5)
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_color(_BORDER)
    return _save(fig)


def _comparison_metric(
    context: dict[str, Any], metric: str, title: str, percent: bool = False
) -> bytes | None:
    """One scorecard metric across the candidate models, as a vertical bar chart.

    Bars are coloured by the model's kind (AI vs traditional) so the RFM/LRFM
    entries — which leave silhouette blank by design — read as a different class
    rather than as missing data. ``percent`` scales a 0–1 metric to a percentage
    axis; the value labels stay in the metric's own units.
    """
    comparison = context.get("comparison") or {}
    table = _as_frame(comparison.get("table")) if isinstance(comparison, dict) else None
    if table is None or metric not in table.columns:
        return None
    data = table.dropna(subset=[metric])
    if data.empty:
        return None
    labels = [_short(str(value), 16) for value in data["model"]]
    values = pd.to_numeric(data[metric], errors="coerce").astype(float).to_numpy()
    colors = [
        _SECONDARY if str(kind) == "Traditional" else _PRIMARY
        for kind in data.get("kind", pd.Series(["AI"] * len(data)))
    ]
    fig, ax = _axes()
    positions = np.arange(len(values))
    bars = ax.bar(positions, values, color=colors, width=0.62)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=_GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
        end_labels = [f"{value:.0%}" for value in values]
    else:
        end_labels = [f"{value:.3f}" for value in values]
    ax.bar_label(bars, labels=end_labels, padding=3, fontsize=7)
    top = float(values.max()) if len(values) else 1.0
    ax.set_ylim(0, top * 1.2 if top > 0 else 1)
    return _save(fig)


def _comparison_separation(context: dict[str, Any]) -> bytes | None:
    """Silhouette per AI run (RFM/LRFM blank by design)."""
    return _comparison_metric(context, "silhouette", "Silhouette by model")


def _comparison_stability(context: dict[str, Any]) -> bytes | None:
    """Seed stability (ARI) per model."""
    return _comparison_metric(context, "stability_ari", "Seed stability by model")


def _comparison_value(context: dict[str, Any]) -> bytes | None:
    """Revenue differentiation (eta²) per model."""
    return _comparison_metric(context, "value_eta2", "Revenue differentiation by model")


def _comparison_evenness(context: dict[str, Any]) -> bytes | None:
    """Size balance (evenness) per model, as a percentage of even."""
    return _comparison_metric(context, "evenness", "Size balance by model", percent=True)


#: Registry consumed by the PDF and Word exporters, keyed like CHART_EXPLANATIONS.
CHART_BUILDERS: dict[str, Callable[[dict[str, Any]], bytes | None]] = {
    "revenue_trend": _revenue_trend,
    "activity_trend": _activity_trend,
    "new_returning": _new_returning,
    "segment_distribution": _segment_distribution,
    "revenue_by_segment": _revenue_by_segment,
    "segment_trends": _segment_trends,
    "demographics_age": _demographics_age,
    "demographics_gender": _demographics_gender,
    "demographics_region": _demographics_region,
    "comparison_separation": _comparison_separation,
    "comparison_stability": _comparison_stability,
    "comparison_value": _comparison_value,
    "comparison_evenness": _comparison_evenness,
    "risk_bands": _risk_bands,
    "value_risk_matrix": _value_risk_matrix,
    "revenue_at_risk": _revenue_at_risk,
    "strategy_distribution": _strategy_distribution,
    "priority_histogram": _priority_histogram,
}


def render_report_charts(
    context: dict[str, Any], keys: Iterable[str] | None = None
) -> dict[str, bytes]:
    """PNG bytes per chart key; missing data or a missing engine yields no entry.

    Individual chart failures are logged and skipped so one sparse table can
    never break a whole export.
    """
    if plt is None:
        return {}
    out: dict[str, bytes] = {}
    for key in keys or tuple(CHART_BUILDERS):
        builder = CHART_BUILDERS.get(key)
        if builder is None:
            continue
        try:
            png = builder(context)
        except (KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
            logger.warning("Report chart %s could not be rendered: %s", key, exc)
            continue
        if png:
            out[key] = png
    return out


__all__ = ["CHART_BUILDERS", "available_chart_engine", "render_report_charts"]

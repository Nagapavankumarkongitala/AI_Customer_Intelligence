"""Segmentation model comparison: one computation shared by page 06 and the report.

Page 06 (Segmentation Comparison) and the report's "Model comparison" section must
show the *same* scorecard. Duplicating the metric assembly in two places is how
two answers to one question drift apart, so it lives here: pages and exporters
both call :func:`build_comparison` over session state and render what it returns.

The comparison is deliberately multi-dimensional. Each metric crowns its own
winner and the operational pick clears stability/size/noise gates first, exactly
as :func:`analytics.segmentation_comparison.recommend_model` documents — this
module only assembles the candidate rows that function scores.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from analytics.segmentation_comparison import (
    agreement_ari,
    crosstab_counts,
    interpretability_checklist,
    recommend_model,
    revenue_concentration,
    size_balance,
    value_separation,
)

#: Scorecard metric columns, in display order, with the label the charts use.
SCORECARD_METRICS: tuple[tuple[str, str], ...] = (
    ("silhouette", "Silhouette (AI runs)"),
    ("stability_ari", "Seed stability (ARI)"),
    ("value_eta2", "Revenue differentiation (eta²)"),
    ("evenness", "Size balance (evenness)"),
)

#: Columns the scorecard table always presents (checklists stay out of the grid).
SCORECARD_COLUMNS: tuple[str, ...] = (
    "model",
    "kind",
    "n_segments",
    "silhouette",
    "stability_ari",
    "value_eta2",
    "min_share",
    "evenness",
    "top_revenue_share",
    "noise_pct",
    "interpretability_passes",
)


def _as_frame(value: Any) -> pd.DataFrame | None:
    """Return value when it is a non-empty frame, else None."""
    return value if isinstance(value, pd.DataFrame) and not value.empty else None


def _churn_labels(state: Mapping[str, Any]) -> pd.Series | None:
    """Predicted churn as 0/1 labels when scores exist, else None."""
    scores = _as_frame(state.get("churn_scores"))
    if scores is None or "customer_id" not in scores.columns:
        return None
    indexed = scores.set_index("customer_id")
    if "predicted_class" in indexed.columns:
        return indexed["predicted_class"]
    if "churn_probability" in indexed.columns:
        return indexed["churn_probability"] >= 0.5
    return None


def build_comparison(state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Assemble the model scorecard, recommendation, and cross-tabs from state.

    Returns ``None`` when there is not enough to compare (no features, or no
    traditional scores *and* no AI runs), so callers can show their own empty
    state. The returned dict carries:

    * ``table`` — one row per candidate model, the columns in
      :data:`SCORECARD_COLUMNS`, plus a ``checklist`` list per row;
    * ``recommendation`` — the output of ``recommend_model``;
    * ``labelings`` — model name to per-customer segment labels;
    * ``cross_tab`` / ``pair`` / ``pair_ari`` — the pairwise agreement between
      the operational traditional reference and the selected AI model.
    """
    features = _as_frame(state.get("customer_features"))
    if features is None or "customer_id" not in features.columns:
        return None
    rfm = _as_frame(state.get("rfm_results"))
    lrfm = _as_frame(state.get("lrfm_results"))
    ai_runs = [r for r in (state.get("segmentation_runs") or []) if "labels" in r]
    if rfm is None and lrfm is None and not ai_runs:
        return None

    money = features.set_index("customer_id")["monetary_value"]
    rows: list[dict[str, Any]] = []
    labelings: dict[str, pd.Series] = {}

    for run in ai_runs:
        name = f"AI:{run['model']}"
        ids = run["customer_ids"]
        labels = pd.Series(run["labels"], index=ids, name="cluster")
        names = {int(k): v for k, v in run.get("names", {}).items()}
        display = pd.Series(
            [names.get(int(v), str(v)) for v in labels], index=labels.index, name="segment"
        )
        labelings[name] = display
        metrics, stability = run["metrics"], run["stability"]
        sizes = labels.value_counts().to_dict()
        balance = size_balance({int(k): int(v) for k, v in sizes.items()})
        revenues = money.reindex(ids).groupby(display.values).sum().to_dict()
        checklist = interpretability_checklist(
            metrics["n_clusters"],
            bool(names),
            metrics["noise_pct"],
            stability.get("mean_pairwise_ari"),
            balance["min_share"],
        )
        rows.append(
            {
                "model": name,
                "kind": "AI",
                "n_segments": metrics["n_clusters"],
                "silhouette": metrics["silhouette"],
                "davies_bouldin": metrics["davies_bouldin"],
                "stability_ari": stability.get("mean_pairwise_ari"),
                "value_eta2": value_separation(money.reindex(ids), display),
                "min_share": balance["min_share"],
                "evenness": balance["evenness"],
                "top_revenue_share": revenue_concentration(
                    {str(k): float(v) for k, v in revenues.items()}
                )["top_segment_share"],
                "noise_pct": metrics["noise_pct"],
                "interpretability_passes": sum(1 for i in checklist if i["pass"]),
                "checklist": checklist,
            }
        )

    # Traditional labelings are deterministic rules on fixed features, so
    # stability is 1.0 by construction and silhouette stays blank: a different
    # feature space makes it non-comparable with the AI runs.
    for label, scored in (("RFM", rfm), ("LRFM", lrfm)):
        if scored is None or "customer_id" not in scored.columns:
            continue
        display = scored.set_index("customer_id")["segment"]
        labelings[label] = display
        ids = display.index.tolist()
        sizes = display.value_counts().to_dict()
        balance = size_balance({str(k): int(v) for k, v in sizes.items()})
        revenues = money.reindex(ids).groupby(display.values).sum().to_dict()
        checklist = interpretability_checklist(
            display.nunique(), True, 0.0, 1.0, balance["min_share"]
        )
        rows.append(
            {
                "model": label,
                "kind": "Traditional",
                "n_segments": display.nunique(),
                "silhouette": None,
                "davies_bouldin": None,
                "stability_ari": 1.0,
                "value_eta2": value_separation(money.reindex(ids), display),
                "min_share": balance["min_share"],
                "evenness": balance["evenness"],
                "top_revenue_share": revenue_concentration(
                    {str(k): float(v) for k, v in revenues.items()}
                )["top_segment_share"],
                "noise_pct": 0.0,
                "interpretability_passes": sum(1 for i in checklist if i["pass"]),
                "checklist": checklist,
            }
        )

    if not rows:
        return None

    table = pd.DataFrame(rows)
    recommendation = recommend_model(
        table.rename(columns={"model": "name"}).to_dict(orient="records")
    )

    # Pairwise agreement: the recorded human pick when it is an AI model, else the
    # operational pick. The traditional reference is the first available one.
    traditional = [r["model"] for r in rows if r["kind"] == "Traditional"]
    ai_names = [r["model"] for r in rows if r["kind"] == "AI"]
    recorded = (state.get("comparison_report") or {}).get("human_pick")
    target = recorded if recorded in ai_names else (ai_names[0] if ai_names else None)
    reference = traditional[0] if traditional else None

    pair: dict[str, Any] | None = None
    if reference and target:
        cross = crosstab_counts(labelings[reference], labelings[target])
        pair = {
            "reference": reference,
            "target": target,
            "ari": agreement_ari(labelings[reference], labelings[target]),
            "cross_tab": cross,
        }

    return {
        "table": table,
        "recommendation": recommendation,
        "labelings": labelings,
        "pair": pair,
        "churn_labels": _churn_labels(state),
    }


def scorecard_for_display(table: pd.DataFrame) -> pd.DataFrame:
    """The scorecard grid without the per-row checklist payload (pure).

    Exports and previews never want the nested checklist column; it renders as a
    wall of Python in every table format. Callers that want the checklist read it
    straight from the untrimmed table.
    """
    if table is None or table.empty:
        return pd.DataFrame(columns=list(SCORECARD_COLUMNS))
    return table[[c for c in SCORECARD_COLUMNS if c in table.columns]].copy()


#: Decimal places per scorecard metric for text export (None = integer fields).
_SCORECARD_DECIMALS: dict[str, int | None] = {
    "n_segments": None,
    "interpretability_passes": None,
    "silhouette": 4,
    "stability_ari": 4,
    "value_eta2": 4,
    "min_share": 4,
    "evenness": 4,
    "top_revenue_share": 4,
    "noise_pct": 2,
}


def scorecard_as_text(table: pd.DataFrame) -> pd.DataFrame:
    """Scorecard as all-string cells with blanks for missing metrics (pure).

    PDF and Word table builders run every cell through HTML escaping, which
    needs strings — a raw ``NaN`` breaks them, and a literal ``"nan"`` reads as
    data. Missing metrics become an empty cell, which is honest for the
    RFM/LRFM silhouette that is blank by design.
    """
    grid = scorecard_for_display(table)
    if grid.empty:
        return grid
    out = pd.DataFrame(index=grid.index)
    for column in grid.columns:
        if column not in _SCORECARD_DECIMALS:
            out[column] = grid[column].astype("string").fillna("").astype(str)
            continue
        decimals = _SCORECARD_DECIMALS[column]
        numeric = pd.to_numeric(grid[column], errors="coerce")
        if decimals is None:
            out[column] = numeric.map(lambda value: "" if pd.isna(value) else f"{int(value):,}")
        else:
            out[column] = numeric.map(
                lambda value, places=decimals: ("" if pd.isna(value) else f"{value:.{places}f}")
            )
    return out


__all__ = [
    "SCORECARD_COLUMNS",
    "SCORECARD_METRICS",
    "build_comparison",
    "scorecard_as_text",
    "scorecard_for_display",
]

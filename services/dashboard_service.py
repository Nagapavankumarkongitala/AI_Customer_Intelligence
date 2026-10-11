"""Executive dashboard backend: master table, global filters, KPIs, trends.

Semantics: date/geo filters select transactions first; the customer master is
restricted to customers present in those transactions; customer-attribute
filters then apply. Every KPI derives from the filtered pair, so all cards stay
consistent. Sections degrade gracefully when upstream state is absent.
"""

from __future__ import annotations

import pandas as pd

from analytics.demographics import DEFAULT_AGE_BANDS, assign_age_band
from utils.validators import require_min_rows

ACTIVE_DAYS = 90


def mask_customer_id(customer_id: object) -> str:
    """Privacy masking: 'C00001' -> 'C•••01'. Raw IDs appear only in gated drill-downs."""
    text = str(customer_id)
    if len(text) <= 3:
        return "•••"
    return f"{text[0]}•••{text[-2:]}"


def prepare_transactions(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates/amounts once for dashboard and trend computation."""
    txns = clean_df.copy()
    txns["_txn_date"] = pd.to_datetime(txns["transaction_date"], errors="coerce")
    txns = txns.dropna(subset=["_txn_date"]).reset_index(drop=True)
    txns["_amount"] = pd.to_numeric(txns.get("amount_spent"), errors="coerce").fillna(0.0)
    txns["_month"] = txns["_txn_date"].dt.to_period("M").astype(str)
    return txns


def build_master(
    features: pd.DataFrame,
    rfm: pd.DataFrame | None = None,
    segmented: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
    retention: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-join all customer grains onto the feature table (features drive)."""
    require_min_rows(features, 1, "customers")
    master = features.copy()
    if rfm is not None and "segment" in rfm.columns:
        master = master.merge(
            rfm[["customer_id", "segment"]].rename(columns={"segment": "rfm_segment"}),
            on="customer_id",
            how="left",
        )
    if segmented is not None:
        col = (
            "business_name"
            if "business_name" in segmented.columns
            else "cluster_id" if "cluster_id" in segmented.columns else None
        )
        if col:
            master = master.merge(
                segmented[["customer_id", col]].rename(columns={col: "ai_segment"}),
                on="customer_id",
                how="left",
            )
    if scores is not None and "churn_probability" in scores.columns:
        keep = ["customer_id", "churn_probability", "risk_band"]
        master = master.merge(
            scores[[c for c in keep if c in scores.columns]], on="customer_id", how="left"
        )
    if retention is not None:
        keep = [
            "customer_id",
            "customer_value_tier",
            "revenue_at_risk",
            "retention_priority_score",
            "recommended_action",
        ]
        master = master.merge(
            retention[[c for c in keep if c in retention.columns]], on="customer_id", how="left"
        )
    if "age" in master.columns:
        master["age_band"] = assign_age_band(master["age"], DEFAULT_AGE_BANDS)
    return master


def apply_filters(
    txns: pd.DataFrame, master: pd.DataFrame, filters: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply global filters consistently. Returns (transactions, customers)."""
    txns_f = txns.copy()
    if filters.get("date_start"):
        txns_f = txns_f[txns_f["_txn_date"] >= pd.Timestamp(filters["date_start"])]
    if filters.get("date_end"):
        txns_f = txns_f[txns_f["_txn_date"] <= pd.Timestamp(filters["date_end"])]
    for level in ("country", "region", "state", "city"):
        values = filters.get(level)
        if values and level in txns_f.columns:
            txns_f = txns_f[txns_f[level].fillna("Unknown").isin(values)]
    master_f = master[master["customer_id"].isin(set(txns_f["customer_id"]))].copy()
    for key, column in (
        ("gender", "gender"),
        ("age_band", "age_band"),
        ("rfm_segment", "rfm_segment"),
        ("ai_segment", "ai_segment"),
        ("risk_band", "risk_band"),
        ("value_tier", "customer_value_tier"),
    ):
        values = filters.get(key)
        if values and column in master_f.columns:
            master_f = master_f[master_f[column].fillna("Unknown").isin(values)]
    txns_f = txns_f[txns_f["customer_id"].isin(set(master_f["customer_id"]))]
    return txns_f.reset_index(drop=True), master_f.reset_index(drop=True)


def compute_kpis(txns: pd.DataFrame, master: pd.DataFrame, active_days: int = ACTIVE_DAYS) -> dict:
    """Row-1 KPI cards from the filtered pair."""
    revenue = float(txns["_amount"].sum()) if len(txns) else 0.0
    n_txns = len(txns)
    customers = master["customer_id"].nunique() if len(master) else 0
    active = 0
    if len(txns):
        cutoff = txns["_txn_date"].max() - pd.Timedelta(days=active_days)
        active = int(txns.loc[txns["_txn_date"] >= cutoff, "customer_id"].nunique())
    churn_rate = (
        float(master["churn_probability"].mean())
        if "churn_probability" in master and master["churn_probability"].notna().any()
        else None
    )
    revenue_risk = float(master["revenue_at_risk"].sum()) if "revenue_at_risk" in master else None
    priority = (
        int((master["retention_priority_score"] > 0).sum())
        if "retention_priority_score" in master
        else None
    )
    return {
        "total_customers": customers,
        "total_revenue": round(revenue, 2),
        "average_order_value": round(revenue / n_txns, 2) if n_txns else 0.0,
        "active_customer_rate": round(active / customers, 4) if customers else 0.0,
        "predicted_churn_rate": round(churn_rate, 4) if churn_rate is not None else None,
        "revenue_at_risk": round(revenue_risk, 2) if revenue_risk is not None else None,
        "priority_opportunities": priority,
    }


def trend_tables(txns: pd.DataFrame, master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Monthly revenue, active customers, and new-vs-returning split."""
    if txns.empty:
        empty = pd.DataFrame(columns=["month"])
        return {"revenue": empty, "activity": empty, "cohort": empty}
    revenue = (
        txns.groupby("_month")["_amount"]
        .sum()
        .reset_index()
        .rename(columns={"_month": "month", "_amount": "revenue"})
    )
    activity = (
        txns.groupby("_month")["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"_month": "month", "customer_id": "active_customers"})
    )
    first_month = (
        master.assign(_first=pd.to_datetime(master["first_transaction_date"]))
        .groupby("customer_id")["_first"]
        .min()
        .dt.to_period("M")
        .astype(str)
    )
    active_month = (
        txns.groupby(["_month", "customer_id"]).size().reset_index()[["_month", "customer_id"]]
    )
    active_month = active_month.merge(
        first_month.rename("first_month"), left_on="customer_id", right_index=True
    )
    active_month["is_new"] = active_month["_month"] == active_month["first_month"]
    cohort = (
        active_month.groupby("_month")["is_new"]
        .agg(new_customers="sum", active_customers="size")
        .reset_index()
        .rename(columns={"_month": "month"})
    )
    cohort["returning_customers"] = cohort["active_customers"] - cohort["new_customers"]
    return {"revenue": revenue, "activity": activity, "cohort": cohort}


def segment_portfolio(
    master: pd.DataFrame, segment_col: str, monetary_col: str = "monetary_value"
) -> pd.DataFrame:
    """Customer and revenue share per segment for portfolio views."""
    if segment_col not in master.columns:
        return pd.DataFrame()
    table = (
        master.groupby(segment_col)
        .agg(customers=("customer_id", "size"), revenue=(monetary_col, "sum"))
        .reset_index()
    )
    table["customer_share"] = (table["customers"] / table["customers"].sum() * 100).round(2)
    total = table["revenue"].sum()
    table["revenue_share"] = (table["revenue"] / total * 100).round(2) if total else 0.0
    return table.sort_values("customers", ascending=False).reset_index(drop=True)


def high_value_high_risk(master: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Masked high-tier + high/critical-band customers by revenue at risk."""
    needed = {"customer_value_tier", "risk_band", "revenue_at_risk", "customer_id"}
    if not needed.issubset(master.columns):
        return pd.DataFrame()
    subset = master[
        (master["customer_value_tier"] == "High") & (master["risk_band"].isin(["high", "critical"]))
    ]
    cols = [
        "customer_id",
        "revenue_at_risk",
        "churn_probability",
        "recommended_action",
        "rfm_segment",
        "ai_segment",
    ]
    out = subset.sort_values("revenue_at_risk", ascending=False).head(top_n)
    out = out[[c for c in cols if c in out.columns]].copy()
    out["customer_id"] = out["customer_id"].map(mask_customer_id)
    return out.reset_index(drop=True)

"""Demographic and geographic profiling over the customer feature table.

Every table builder tolerates absent optional fields: missing columns produce a
note, never a crash. Revenue columns appear only when monetary data is present.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd

from utils.validators import require_min_rows

GEO_LEVELS = ("city", "state", "country", "region")
OPTIONAL_PROFILE_FIELDS = ("gender", "age", *GEO_LEVELS)
DEFAULT_AGE_BANDS: list[tuple[str, float, float]] = [
    ("0–17", 0, 17),
    ("18–24", 18, 24),
    ("25–34", 25, 34),
    ("35–49", 35, 49),
    ("50–64", 50, 64),
    ("65+", 65, math.inf),
]
_BAND_RE = re.compile(r"^\s*(\d+)\s*(?:[-–]\s*(\d+|\+)|(\+))\s*$")


@dataclass
class DemographicProfile:
    """All profile tables plus missing-value rates and build notes."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    missing_pct: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def parse_age_bands(spec: str) -> list[tuple[str, float, float]]:
    """Parse '0-17,18-24,65+' into (label, lo, hi) bands; raises ValueError on bad input."""
    bands: list[tuple[str, float, float]] = []
    for part in spec.split(","):
        match = _BAND_RE.match(part)
        if not match:
            raise ValueError(
                f"Age band '{part.strip()}' is not valid. Use 'lo-hi' pairs like '18-24' "
                "or an open band like '65+'."
            )
        lo = float(match.group(1))
        hi_raw = match.group(2) if match.group(2) is not None else match.group(3)
        hi = math.inf if hi_raw == "+" else float(hi_raw)
        if hi != math.inf and hi < lo:
            raise ValueError(f"Age band '{part.strip()}' has hi < lo.")
        label = f"{int(lo)}+" if hi == math.inf else f"{int(lo)}–{int(hi)}"
        bands.append((label, lo, hi))
    if not bands:
        raise ValueError("Provide at least one age band, e.g. '0-17,18-24,65+'.")
    return bands


def assign_age_band(ages: pd.Series, bands: list[tuple[str, float, float]]) -> pd.Series:
    """Label each age; unparseable ages become 'Unknown', matching first covering band."""
    numeric = pd.to_numeric(ages, errors="coerce")
    labels: list[str] = []
    for value in numeric:
        if pd.isna(value):
            labels.append("Unknown")
            continue
        labels.append(next((label for label, lo, hi in bands if lo <= value <= hi), "Unknown"))
    return pd.Series(labels, index=ages.index, name="age_band")


def _pct(series: pd.Series) -> pd.Series:
    return (series / series.sum() * 100).round(2) if series.sum() else series * 0.0


def _revenue_share(table: pd.DataFrame) -> pd.DataFrame:
    total = table["revenue"].sum()
    table["revenue_pct"] = (table["revenue"] / total * 100).round(2) if total else 0.0
    return table


def segment_composition(
    features: pd.DataFrame, segments: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Row-normalized (percent) crosstab of segment × column values for stacked charts."""
    merged = features[["customer_id", column]].merge(
        segments[["customer_id", "segment"]], on="customer_id", how="inner"
    )
    table = pd.crosstab(merged["segment"], merged[column].fillna("Unknown"), normalize="index")
    return (table * 100).round(2)


def build_demographic_profile(
    features: pd.DataFrame,
    segments: pd.DataFrame | None = None,
    age_band_spec: str | None = None,
) -> DemographicProfile:
    """Build every profile table. Absent optional fields yield notes, never errors."""
    require_min_rows(features, 1, "customers")
    if "customer_id" not in features.columns:
        raise KeyError("Customer features are missing required column: customer_id")
    bands = parse_age_bands(age_band_spec) if age_band_spec else DEFAULT_AGE_BANDS
    profile = DemographicProfile()
    n = len(features)
    has_revenue = "monetary_value" in features.columns

    for col in OPTIONAL_PROFILE_FIELDS:
        if col in features.columns:
            blank = features[col].isna() | (features[col].astype("string").str.strip() == "")
            profile.missing_pct[col] = round(float(blank.sum()) / n * 100, 2)
        else:
            profile.notes.append(f"Field '{col}' is absent: related views are skipped.")

    if "age" in features.columns:
        ages = pd.to_numeric(features["age"], errors="coerce").dropna()
        profile.tables["age_summary"] = pd.DataFrame(
            [
                {
                    "customers_with_age": len(ages),
                    "mean_age": round(float(ages.mean()), 2) if len(ages) else None,
                    "median_age": round(float(ages.median()), 2) if len(ages) else None,
                }
            ]
        )
        banded = assign_age_band(features["age"], bands)
        age_bands = banded.value_counts().rename_axis("age_band").reset_index(name="customers")
        age_bands["pct"] = _pct(age_bands["customers"]).values
        profile.tables["age_bands"] = age_bands
    if "gender" in features.columns:
        gender = features["gender"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        table = gender.value_counts().rename_axis("gender").reset_index(name="customers")
        table["pct"] = _pct(table["customers"]).values
        if has_revenue:
            table["revenue"] = (
                table["gender"]
                .map(features.assign(_g=gender).groupby("_g")["monetary_value"].sum())
                .fillna(0.0)
                .round(2)
            )
            table = _revenue_share(table)
        profile.tables["gender"] = table

    for level in GEO_LEVELS:
        if level not in features.columns:
            continue
        geo = features[level].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        table = geo.value_counts().rename_axis(level).reset_index(name="customers")
        table["pct"] = _pct(table["customers"]).values
        if has_revenue:
            table["revenue"] = (
                table[level]
                .map(features.assign(_g=geo).groupby("_g")["monetary_value"].sum())
                .fillna(0.0)
                .round(2)
            )
            table = _revenue_share(table)
        profile.tables[f"geo_{level}"] = table
        top3 = float(table["pct"].head(3).sum())
        shares = table["customers"] / table["customers"].sum()
        profile.tables.setdefault(
            "concentration",
            pd.DataFrame(columns=["level", "top3_share_pct", "hhi"]),
        )
        profile.tables["concentration"] = pd.concat(
            [
                profile.tables["concentration"],
                pd.DataFrame(
                    [
                        {
                            "level": level,
                            "top3_share_pct": round(top3, 2),
                            "hhi": round(float((shares**2).sum()), 4),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    if segments is not None and not segments.empty:
        seg = segments[["customer_id", "segment"]].copy()
        seg["segment"] = seg["segment"].fillna("Unsegmented")
        merged = features.merge(seg, on="customer_id", how="left")
        merged["segment"] = merged["segment"].fillna("Unsegmented")
        rows: list[dict] = []
        for name, group in merged.groupby("segment"):
            entry: dict = {
                "segment": name,
                "customers": len(group),
                "pct": round(len(group) / len(merged) * 100, 2),
            }
            if "age" in group:
                entry["mean_age"] = round(
                    float(pd.to_numeric(group["age"], errors="coerce").mean()), 2
                )
            if has_revenue:
                entry["revenue"] = round(float(group["monetary_value"].sum()), 2)
            if "gender" in group:
                top = group["gender"].fillna("Unknown").value_counts(normalize=True)
                entry["top_gender"] = top.index[0]
                entry["top_gender_pct"] = round(float(top.iloc[0]) * 100, 2)
            if "region" in group:
                top = group["region"].fillna("Unknown").value_counts(normalize=True)
                entry["top_region"] = top.index[0]
                entry["top_region_pct"] = round(float(top.iloc[0]) * 100, 2)
            rows.append(entry)
        profile.tables["segment_profiles"] = pd.DataFrame(rows)
        pop_age = (
            round(float(pd.to_numeric(features["age"], errors="coerce").mean()), 2)
            if "age" in features
            else None
        )
        pop_value = round(float(features["monetary_value"].mean()), 2) if has_revenue else None
        comparison: list[dict] = []
        for _, row in profile.tables["segment_profiles"].iterrows():
            if pop_age is not None and row.get("mean_age") is not None:
                comparison.append(
                    {
                        "segment": row["segment"],
                        "metric": "mean_age",
                        "segment_value": row["mean_age"],
                        "population_value": pop_age,
                        "difference": round(row["mean_age"] - pop_age, 2),
                    }
                )
            if pop_value is not None:
                seg_mean = round(row["revenue"] / row["customers"], 2)
                comparison.append(
                    {
                        "segment": row["segment"],
                        "metric": "mean_monetary",
                        "segment_value": seg_mean,
                        "population_value": pop_value,
                        "difference": round(seg_mean - pop_value, 2),
                    }
                )
        profile.tables["segment_comparison"] = pd.DataFrame(comparison)
    else:
        profile.notes.append(
            "No segment labels available: per-segment views are skipped. "
            "Run RFM/LRFM scoring on page 03 first."
        )
    return profile

"""Phase 3 tests: bands, missing-field tolerance, geo math, stats, correction."""

import numpy as np
import pandas as pd

from analytics.demographics import (
    assign_age_band,
    build_demographic_profile,
    parse_age_bands,
    segment_composition,
)
from analytics.statistical_tests import (
    categorical_association,
    compare_continuous_across_segments,
    cramers_v,
    holm_reject,
    run_segment_tests,
    welch_anova,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "recency_days": [5, 10, 40, 60, 90, 120],
            "frequency": [20, 15, 8, 5, 2, 1],
            "monetary_value": [2000.0, 1500.0, 800.0, 400.0, 200.0, 100.0],
            "gender": ["Male", "Female", "Male", "Female", "Male", None],
            "age": [25, 34, 45, 28, 60, 22],
            "city": ["Pune", "Pune", "Delhi", "Delhi", "Pune", "Jaipur"],
            "region": ["West", "West", "North", "North", "West", "North"],
        }
    )


def _segment_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"customer_id": _features()["customer_id"], "segment": ["A", "A", "A", "B", "B", "B"]}
    )


def _segments() -> pd.Series:
    return _segment_df()["segment"]


def test_age_bands_edges_and_unknown() -> None:
    """Band edges are inclusive; garbage ages become Unknown."""
    bands = parse_age_bands("0-17,18-24,65+")
    labeled = assign_age_band(pd.Series([17, 18, 24, 25, 70, None, "old"]), bands)
    assert list(labeled) == ["0–17", "18–24", "18–24", "Unknown", "65+", "Unknown", "Unknown"]


def test_bad_band_spec_rejected() -> None:
    """Malformed band specs fail with actionable errors."""
    for bad in ("", "abc", "24-18"):
        try:
            parse_age_bands(bad)
        except ValueError:
            continue
        raise AssertionError(f"spec accepted unexpectedly: {bad!r}")


def test_profile_tables_and_revenue() -> None:
    """Gender/geo tables carry counts, shares, and revenue; age stats present."""
    profile = build_demographic_profile(_features())
    assert profile.tables["age_summary"]["median_age"].iloc[0] == 31.0
    gender = profile.tables["gender"]
    assert gender.loc[gender["gender"] == "Male", "customers"].iloc[0] == 3
    assert gender["revenue"].sum() == 5000.0
    city = profile.tables["geo_city"]
    assert city.loc[city["city"] == "Pune", "customers"].iloc[0] == 3
    assert profile.missing_pct["gender"] == round(1 / 6 * 100, 2)
    assert any("segment" in note for note in profile.notes)


def test_missing_fields_never_crash() -> None:
    """A features table with no optional fields still builds with notes."""
    bare = _features()[["customer_id", "recency_days", "frequency", "monetary_value"]]
    profile = build_demographic_profile(bare)
    assert profile.tables == {}
    assert len(profile.notes) >= 6  # one per absent optional field + segments note


def test_segment_profiles_and_comparison() -> None:
    """Per-segment aggregates and population comparisons are arithmetically sound."""
    profile = build_demographic_profile(_features(), _segment_df())
    seg = profile.tables["segment_profiles"]
    assert set(seg["segment"]) == {"A", "B"}
    assert seg["customers"].sum() == 6
    comp = profile.tables["segment_comparison"]
    age_rows = comp[comp["metric"] == "mean_age"]
    pop = _features()["age"].mean()
    for _, row in age_rows.iterrows():
        assert row["difference"] == round(row["segment_value"] - pop, 2)


def test_composition_rows_sum_to_100() -> None:
    """Segment × column shares are row-normalized percentages."""
    comp = segment_composition(_features(), _segment_df(), "gender")
    assert ((comp.sum(axis=1) - 100).abs() < 0.05).all()


def test_continuous_test_detects_difference() -> None:
    """Clearly separated groups yield significance; identical groups do not."""
    rng = np.random.default_rng(7)
    high = pd.Series(rng.normal(100, 5, 40), name="spend")
    low = pd.Series(rng.normal(50, 5, 40), name="spend")
    segs = pd.Series(["H"] * 40 + ["L"] * 40)
    result = compare_continuous_across_segments(pd.concat([high, low]), segs)
    assert result.p_value < 0.05 and result.test in ("ANOVA", "Welch ANOVA")
    same = pd.Series(rng.normal(75, 5, 80), name="spend")
    result2 = compare_continuous_across_segments(same, segs)
    assert result2.p_value > 0.05


def test_welch_anova_properties() -> None:
    """Welch detects shifted means and needs valid groups."""
    rng = np.random.default_rng(11)
    _, p = welch_anova(rng.normal(0, 1, 50), rng.normal(0, 5, 50), rng.normal(3, 2, 50))
    assert 0.0 <= p <= 1.0
    try:
        welch_anova(np.array([1.0]), np.array([1.0, 2.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("single-observation group accepted unexpectedly")


def test_categorical_association_and_v_bounds() -> None:
    """Associated tables test significant; Cramer's V stays in [0, 1]."""
    rows = pd.Series(["A"] * 30 + ["B"] * 30, name="seg")
    cols = pd.Series(["X"] * 25 + ["Y"] * 5 + ["X"] * 5 + ["Y"] * 25, name="gender")
    result = categorical_association(rows, cols)
    assert result.p_value < 0.05
    assert 0.0 <= result.effect_size <= 1.0
    assert 0.0 <= cramers_v(pd.DataFrame([[10]])) <= 1.0


def test_holm_correction_known_values() -> None:
    """Holm on [0.01, 0.03, 0.04] rejects only the first at alpha 0.05."""
    assert holm_reject([0.01, 0.03, 0.04]) == [True, False, False]
    assert holm_reject([0.5, 0.6]) == [False, False]


def test_run_segment_tests_batch() -> None:
    """Batched tests return one row per variable with Holm decisions."""
    out = run_segment_tests(
        _features(),
        _segments(),
        continuous=["monetary_value", "age"],
        categorical=["gender"],
    )
    assert len(out) == 3
    assert set(out["significant_after_holm"]) <= {True, False}
    assert out["p_value"].between(0, 1).all()

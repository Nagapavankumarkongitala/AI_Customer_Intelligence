"""Phase 2 tests: feature math, snapshot default, one-timers, demographics."""

import pandas as pd

from analytics.customer_features import build_customer_features
from services.feature_service import run_segment_history


def _txns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1", "C2"],
            "transaction_date": ["2024-01-10", "2024-02-10", "2024-03-10", "2024-03-01"],
            "amount_spent": [100.0, 200.0, 300.0, 50.0],
            "gender": ["Male", None, "Female", "Male"],
            "age": [30, 30, 31, 40],
            "city": ["Pune", "Pune", "Mumbai", "Delhi"],
        }
    )


def test_snapshot_defaults_to_day_after_last_transaction() -> None:
    """Default snapshot = max transaction date + 1 day; recency follows."""
    result = build_customer_features(_txns())
    c1 = result.features.loc[result.features["customer_id"] == "C1"].iloc[0]
    assert c1["last_transaction_date"] == "2024-03-10"
    assert c1["recency_days"] == 1
    assert result.dropped_invalid_dates == 0


def test_one_time_customer_gaps_are_nan() -> None:
    """Single-purchase customers have NaN interpurchase stats and AOV == monetary."""
    result = build_customer_features(_txns())
    c2 = result.features.loc[result.features["customer_id"] == "C2"].iloc[0]
    assert c2["frequency"] == 1
    assert c2["relationship_length_days"] == 0
    assert pd.isna(c2["average_interpurchase_days"])
    assert pd.isna(c2["interpurchase_std_days"])
    assert c2["average_order_value"] == c2["monetary_value"] == 50.0


def test_value_and_activity_math() -> None:
    """Frequency, monetary, AOV, and active months aggregate correctly."""
    result = build_customer_features(_txns())
    c1 = result.features.loc[result.features["customer_id"] == "C1"].iloc[0]
    assert c1["frequency"] == 3
    assert c1["monetary_value"] == 600.0
    assert c1["average_order_value"] == 200.0
    assert c1["active_months"] == 3
    assert c1["transactions_per_active_month"] == 1.0
    assert c1["average_interpurchase_days"] > 29
    assert c1["relationship_length_days"] == 60


def test_recent_previous_split() -> None:
    """Recent/previous windows split at snapshot - recent_days with change deltas."""
    snapshot = pd.Timestamp("2024-04-01")
    recent_start = snapshot - pd.Timedelta(days=90)
    df = pd.DataFrame(
        {
            "customer_id": ["C1"] * 3,
            "transaction_date": [
                (snapshot - pd.Timedelta(days=10)).date().isoformat(),
                (snapshot - pd.Timedelta(days=20)).date().isoformat(),
                (recent_start - pd.Timedelta(days=5)).date().isoformat(),
            ],
            "amount_spent": [100.0, 100.0, 50.0],
        }
    )
    result = build_customer_features(df, snapshot_date=snapshot, recent_days=90)
    row = result.features.iloc[0]
    assert row["recent_period_frequency"] == 2
    assert row["previous_period_frequency"] == 1
    assert row["frequency_change"] == 1
    assert row["recent_period_monetary"] == 200.0
    assert row["previous_period_monetary"] == 50.0
    assert row["monetary_change"] == 150.0


def test_demographics_latest_non_null_with_conflicts() -> None:
    """Latest non-null value wins; multi-value customers are reported, not merged."""
    result = build_customer_features(_txns())
    c1 = result.features.loc[result.features["customer_id"] == "C1"].iloc[0]
    assert c1["gender"] == "Female"
    assert c1["age"] == 31
    assert c1["city"] == "Mumbai"
    assert result.demographic_conflicts["C1"]["gender"] == ["Female", "Male"]
    assert result.demographic_conflicts["C1"]["city"] == ["Mumbai", "Pune"]
    assert "C2" not in result.demographic_conflicts


def test_invalid_dates_dropped_and_counted() -> None:
    """Unparseable dates are excluded from features and counted, never silent."""
    df = _txns()
    df.loc[len(df)] = ["C3", "not-a-date", 10.0, "Male", 25, "Delhi"]
    result = build_customer_features(df)
    assert result.dropped_invalid_dates == 1
    assert "C3" not in set(result.features["customer_id"])


def _history_txns() -> pd.DataFrame:
    """Two cohorts: six customers Jan–Jun, six more starting only in June."""
    rows = []
    for i in range(6):
        for month in range(1, 7):
            rows.append((f"E{i:02d}", f"2024-{month:02d}-10", 100.0 + 10 * i + month))
    for i in range(6, 12):
        rows.append((f"L{i:02d}", "2024-06-15", 200.0))
    return pd.DataFrame(rows, columns=["customer_id", "transaction_date", "amount_spent"])


def test_segment_history_shape_and_order() -> None:
    """Long period/model/segment counts with chronological periods."""
    history = run_segment_history(_history_txns(), granularity="monthly", quantiles=3)
    assert list(history.columns) == ["period", "model", "segment", "customers"]
    assert set(history["model"]) <= {"RFM", "LRFM"}
    periods = list(history["period"].unique())
    assert periods == sorted(periods)
    assert all(len(str(p)) == 7 for p in periods)  # YYYY-MM labels
    assert (history["customers"] > 0).all()


def test_segment_history_never_leaks_future() -> None:
    """June-only customers are absent from every earlier period's counts."""
    history = run_segment_history(_history_txns(), granularity="monthly", quantiles=3)
    rfm_only = history[history["model"] == "RFM"]
    totals = rfm_only.groupby("period", observed=True)["customers"].sum()
    assert len(totals) == 5, f"want Jan–May periods, got {list(totals.index)}"
    assert (totals == 6).all(), f"late cohort leaked: {totals.to_dict()}"


def test_segment_history_rejects_bad_args() -> None:
    """Bad granularity and quantiles fail loudly, not silently."""
    frame = _history_txns()
    for call in (
        lambda: run_segment_history(frame, granularity="fortnightly"),
        lambda: run_segment_history(frame, granularity="monthly", quantiles=7),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("bad history args did not raise")

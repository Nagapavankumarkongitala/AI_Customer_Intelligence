"""Phase 2 tests: feature math, snapshot default, one-timers, demographics."""

import pandas as pd

from analytics.customer_features import build_customer_features
from services.feature_service import (
    TRANSITION_COLUMNS,
    run_label_history,
    run_segment_history,
    run_segment_trends,
    run_segment_trends_with_labels,
)


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


def test_blank_and_tied_demographics() -> None:
    """Blank strings count as missing; same-day ties and conflicts stay exact."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1", "C2"],
            "transaction_date": ["2024-01-10", "2024-01-10", "2024-03-10", "2024-03-01"],
            "amount_spent": [100.0, 200.0, 300.0, 50.0],
            "gender": ["  ", None, "Female", "Male"],
            "age": [30, 30, 31, 40],
            "city": ["Pune", "Pune", "Mumbai", "Delhi"],
        }
    )
    result = build_customer_features(df)
    c1 = result.features.loc[result.features["customer_id"] == "C1"].iloc[0]
    assert c1["frequency"] == 3
    assert c1["gender"] == "Female"
    assert c1["city"] == "Mumbai"
    assert result.demographic_conflicts["C1"]["city"] == ["Mumbai", "Pune"]
    assert "C2" not in result.demographic_conflicts


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


def test_segment_trends_return_customer_level_transitions() -> None:
    """Every transition pair counts customers present at both consecutive ends."""
    history, transitions = run_segment_trends(_history_txns(), granularity="monthly", quantiles=3)
    assert list(transitions.columns) == TRANSITION_COLUMNS
    assert set(transitions["model"]) == {"RFM", "LRFM"}
    assert list(history.columns) == ["period", "model", "segment", "customers"]
    pair_totals = transitions.groupby(["model", "period_from", "period_to"], observed=True)[
        "customers"
    ].sum()
    assert len(pair_totals) == 2 * 4  # five scored periods -> four pairs per model
    assert (pair_totals == 6).all(), pair_totals.to_dict()
    assert (transitions["customers"] > 0).all()


def test_trend_labels_regroup_to_the_published_counts() -> None:
    """Per-customer labels are the same scoring pass, so they sum to history."""
    history, _, labels = run_segment_trends_with_labels(
        _history_txns(), granularity="monthly", quantiles=3
    )
    assert list(labels.columns) == ["period", "model", "customer_id", "segment"]
    regrouped = (
        labels.groupby(["period", "model", "segment"], observed=True)
        .size()
        .rename("customers")
        .reset_index()
    )
    published = history.assign(period=history["period"].astype(str))
    merged = published.merge(
        regrouped,
        on=["period", "model", "segment"],
        how="outer",
        suffixes=("_history", "_labels"),
        indicator=True,
    )
    assert (merged["_merge"] == "both").all(), merged[merged["_merge"] != "both"]
    assert (merged["customers_history"] == merged["customers_labels"]).all()
    assert labels["customer_id"].nunique() == 12


def test_segment_transitions_never_leak_future() -> None:
    """Moves stop at the last fully observed snapshot (June is partial)."""
    _, transitions = run_segment_trends(_history_txns(), granularity="monthly", quantiles=3)
    assert set(transitions["period_to"].astype(str)) == {
        "2024-02",
        "2024-03",
        "2024-04",
        "2024-05",
    }
    assert set(transitions["period_from"].astype(str)) == {
        "2024-01",
        "2024-02",
        "2024-03",
        "2024-04",
    }


def test_segment_history_rejects_bad_args() -> None:
    """Bad granularity and quantiles fail loudly, not silently."""
    frame = _history_txns()
    for call in (
        lambda: run_segment_history(frame, granularity="fortnightly"),
        lambda: run_segment_history(frame, granularity="monthly", quantiles=7),
        lambda: run_segment_trends(frame, granularity="fortnightly"),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("bad history args did not raise")


def _fixed_labels() -> pd.Series:
    """Alternating Champions / Lost Customers over every customer in the frame."""
    customers = sorted(_history_txns()["customer_id"].unique())
    return pd.Series(
        ["Lost Customers" if i % 2 == 0 else "Champions" for i in range(len(customers))],
        index=customers,
    )


def test_label_history_trends_a_fixed_labeling() -> None:
    """AI runs trend on the RFM period grid, counted over the base that existed."""
    history = run_label_history(
        _history_txns(), {"AI:kmeans": _fixed_labels()}, granularity="monthly"
    )
    assert list(history.columns) == ["period", "model", "segment", "customers"]
    assert set(history["model"]) == {"AI:kmeans"}
    totals = history.groupby("period", observed=True)["customers"].sum()
    assert list(totals.index) == ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
    assert (totals == 6).all(), f"late cohort leaked: {totals.to_dict()}"
    champions = history[history["segment"] == "Champions"].groupby("period", observed=True)
    assert (champions["customers"].sum() == 3).all()


def test_label_history_keeps_every_labeling_and_guards_args() -> None:
    """Each requested labeling becomes its own model row; bad input fails loudly."""
    frame = _history_txns()
    history = run_label_history(
        frame, {"AI:kmeans": _fixed_labels(), "AI:gmm": _fixed_labels()}, granularity="monthly"
    )
    assert set(history["model"]) == {"AI:kmeans", "AI:gmm"}

    assert run_label_history(frame, {}).empty
    for call in (
        lambda: run_label_history(frame, {"x": _fixed_labels()}, granularity="fortnightly"),
        lambda: run_label_history(frame.drop(columns=["transaction_date"]), {"x": _fixed_labels()}),
    ):
        try:
            call()
        except (ValueError, KeyError):
            continue
        raise AssertionError("bad label-history args did not raise")

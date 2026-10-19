"""Dashboard filter semantics: history re-sliced by customer and date window."""

import pandas as pd

from services.dashboard_service import filter_segment_trends
from services.feature_service import period_bounds


def _labels() -> pd.DataFrame:
    """Three customers over three quarters; C first appears in Q2."""
    rows = [
        ("2024-Q1", "RFM", "A", "Champions"),
        ("2024-Q1", "RFM", "B", "Loyal"),
        ("2024-Q1", "LRFM", "A", "Cannot Lose Them"),
        ("2024-Q2", "RFM", "A", "Loyal"),
        ("2024-Q2", "RFM", "B", "Loyal"),
        ("2024-Q2", "RFM", "C", "New"),
        ("2024-Q3", "RFM", "A", "Loyal"),
        ("2024-Q3", "RFM", "B", "At Risk"),
        ("2024-Q3", "RFM", "C", "New"),
    ]
    return pd.DataFrame(rows, columns=["period", "model", "customer_id", "segment"])


def _history() -> pd.DataFrame:
    """Aggregate counts matching ``_labels`` for the RFM model."""
    rows = [
        ("2024-Q1", "RFM", "Champions", 1),
        ("2024-Q1", "RFM", "Loyal", 1),
        ("2024-Q2", "RFM", "Loyal", 2),
        ("2024-Q2", "RFM", "New", 1),
        ("2024-Q3", "RFM", "At Risk", 1),
        ("2024-Q3", "RFM", "Loyal", 1),
        ("2024-Q3", "RFM", "New", 1),
    ]
    return pd.DataFrame(rows, columns=["period", "model", "segment", "customers"])


def _moves() -> pd.DataFrame:
    """Stored transitions matching ``_labels``, including an LRFM row to drop."""
    rows = [
        ("RFM", "2024-Q1", "2024-Q2", "Champions", "Loyal", 1),
        ("RFM", "2024-Q1", "2024-Q2", "Loyal", "Loyal", 1),
        ("RFM", "2024-Q2", "2024-Q3", "Loyal", "Loyal", 1),
        ("RFM", "2024-Q2", "2024-Q3", "Loyal", "At Risk", 1),
        ("RFM", "2024-Q2", "2024-Q3", "New", "New", 1),
        ("LRFM", "2024-Q2", "2024-Q3", "X", "Y", 9),
    ]
    return pd.DataFrame(
        rows,
        columns=["model", "period_from", "period_to", "segment_from", "segment_to", "customers"],
    )


def test_period_bounds_are_the_inverse_of_period_labels() -> None:
    """Quarterly and monthly labels map back to their first and last day."""
    assert period_bounds("2024-Q1") == (
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-03-31"),
    )
    assert period_bounds("2024-Q4") == (
        pd.Timestamp("2024-10-01"),
        pd.Timestamp("2024-12-31"),
    )
    assert period_bounds("2024-02") == (
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-02-29"),
    )


def test_no_filters_reproduce_the_stored_history() -> None:
    """All customers and dates selected leaves the aggregate counts untouched."""
    view = filter_segment_trends(
        _labels(),
        history=_history(),
        transitions=_moves(),
        model="RFM",
        customer_ids={"A", "B", "C"},
        date_start=pd.Timestamp("2024-01-01").date(),
        date_end=pd.Timestamp("2024-12-31").date(),
    )
    assert view.customer_filtered is True
    got = view.history.sort_values(["period", "segment"]).reset_index(drop=True)
    want = _history().sort_values(["period", "segment"]).reset_index(drop=True)
    assert got["customers"].tolist() == want["customers"].tolist()
    assert len(view.transitions) == 5


def test_customer_filter_reslices_counts_and_moves() -> None:
    """A customer subset changes counts, and only shared customers become moves."""
    view = filter_segment_trends(
        _labels(),
        history=_history(),
        transitions=_moves(),
        model="RFM",
        customer_ids={"A", "C"},
        date_start=pd.Timestamp("2024-01-01").date(),
        date_end=pd.Timestamp("2024-12-31").date(),
    )
    totals = view.history.groupby("period")["customers"].sum()
    assert totals.to_dict() == {"2024-Q1": 1, "2024-Q2": 2, "2024-Q3": 2}
    moves = view.transitions.sort_values(["period_from", "segment_from"]).reset_index(drop=True)
    assert list(moves["segment_from"]) == ["Champions", "Loyal", "New"]
    assert list(moves["segment_to"]) == ["Loyal", "Loyal", "New"]
    assert (moves["customers"] == 1).all()


def test_date_window_drops_periods_and_bridges_nothing() -> None:
    """Only overlapping periods survive; no client-invented moves across gaps."""
    view = filter_segment_trends(
        _labels(),
        history=_history(),
        transitions=_moves(),
        model="RFM",
        customer_ids={"A", "B", "C"},
        date_start=pd.Timestamp("2024-04-01").date(),
        date_end=pd.Timestamp("2024-09-30").date(),
    )
    assert sorted(view.history["period"].unique()) == ["2024-Q2", "2024-Q3"]
    assert list(view.transitions["period_from"]) == ["2024-Q2"]
    assert list(view.transitions["period_to"]) == ["2024-Q3"]


def test_without_labels_only_the_date_window_applies() -> None:
    """Legacy cached runs still narrow by date and say customer filters are out."""
    view = filter_segment_trends(
        None,
        history=_history(),
        transitions=_moves(),
        model="RFM",
        customer_ids={"A"},
        date_start=pd.Timestamp("2024-04-01").date(),
        date_end=pd.Timestamp("2024-09-30").date(),
    )
    assert view.customer_filtered is False
    assert sorted(view.history["period"].unique()) == ["2024-Q2", "2024-Q3"]
    assert set(view.history["model"]) == {"RFM"}
    assert len(view.transitions) == 3
    assert set(view.transitions["period_from"]) == {"2024-Q2"}


def test_fallback_keeps_only_consecutive_stored_pairs() -> None:
    """A dropped middle period must not leave a jump labelled as one move."""
    labels = _labels()
    view = filter_segment_trends(
        None,
        history=_history(),
        transitions=_moves(),
        model="RFM",
        date_start=pd.Timestamp("2024-01-01").date(),
        date_end=pd.Timestamp("2024-03-31").date(),
    )
    assert labels is not None  # the fallback is exercised, not the labels path
    assert sorted(view.history["period"].unique()) == ["2024-Q1"]
    assert view.transitions.empty


def test_empty_selection_returns_empty_frames() -> None:
    """No matching customers is an empty view, not an error."""
    view = filter_segment_trends(
        _labels(),
        history=_history(),
        transitions=_moves(),
        model="RFM",
        customer_ids={"missing"},
    )
    assert view.history.empty
    assert view.transitions.empty
    assert list(view.history.columns) == ["period", "model", "segment", "customers"]

"""Phase 1 tests: exports, dictionary, state contract, sample-data end to end."""

import pandas as pd

from analytics.customer_features import build_customer_features
from analytics.lrfm import score_lrfm
from analytics.rfm import score_rfm
from services import state_service
from services.cleaning_service import CleaningConfig, clean
from services.column_detection import auto_map
from services.data_dictionary_service import build_dictionary
from services.export_service import to_csv_bytes, to_excel_bytes, to_json_bytes
from utils.constants import SESSION_KEYS


def test_exports_round_trip() -> None:
    """CSV and Excel bytes deserialize back to the same frame."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert pd.read_csv(pd.io.common.BytesIO(to_csv_bytes(df))).equals(df)
    assert pd.read_excel(pd.io.common.BytesIO(to_excel_bytes(df))).equals(df)
    assert to_json_bytes({"mapping": {"a": "b"}}).startswith(b"{")


def test_dictionary_row_per_field_with_stats() -> None:
    """Each mapped field gets a documented row; numerics carry min/max/mean/median."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "amount_spent": [10.0, 20.0],
            "gender": ["Male", "Female"],
        }
    )
    mapping = {"customer_id": "cid", "amount_spent": "amt", "gender": "g"}
    dictionary = build_dictionary(df, mapping)
    assert list(dictionary["canonical_field"]) == ["customer_id", "amount_spent", "gender"]
    assert dictionary.loc[1, "mean"] == 15.0
    assert dictionary.loc[0, "required"]
    assert not dictionary.loc[2, "required"]


def test_state_contract_lifecycle() -> None:
    """Initialize fills defaults, validate passes, reset restores, report describes."""
    state: dict = {}
    state_service.initialize(state)
    assert state_service.validate(state) == []
    assert set(state) == set(SESSION_KEYS)
    state["raw_data"] = pd.DataFrame({"a": [1]})
    report = state_service.status_report(state)
    assert report["raw_data"] == "1 rows × 1 cols"
    assert report["clean_data"] == "empty"
    state_service.reset(state, keys=["raw_data"])
    assert state["raw_data"] is None
    state_service.reset(state)
    assert all(v is None or v == {} or v == [] for v in state.values())


def test_sample_data_end_to_end() -> None:
    """Sample CSV auto-maps, cleans, audits, and scores RFM/LRFM consistently."""
    df = pd.read_csv("data/sample/sample_customer_transactions.csv")
    proposal = auto_map(list(df.columns))
    assert proposal.missing_required == []
    frame, audit = clean(df, proposal.mapping, CleaningConfig())
    assert len(frame) > 1000
    assert audit["output_rows"] == audit["input_rows"] - audit["rows_removed"]
    dictionary = build_dictionary(frame, proposal.mapping)
    assert len(dictionary) == len(proposal.mapping)
    result = build_customer_features(frame)
    assert len(result.features) > 1000
    rfm = score_rfm(result.features)
    lrfm = score_lrfm(result.features)
    assert len(rfm) == len(lrfm) == len(result.features)
    assert rfm["segment"].notna().all() and lrfm["segment"].notna().all()

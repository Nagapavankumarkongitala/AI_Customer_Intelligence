"""Default-data bootstrap: cached data.csv run loads into session state."""

import json

import pandas as pd
import pytest

from services import bootstrap_service, state_service

_MANIFEST = bootstrap_service.CACHE_DIR / "manifest.json"

needs_cache = pytest.mark.skipif(
    not _MANIFEST.exists(), reason="default run cache not built (run scripts/build_default_run.py)"
)


@needs_cache
def test_load_populates_contract_keys() -> None:
    """Empty state gains the full pipeline outputs without overwriting nothing."""
    state: dict = {}
    state_service.initialize(state)
    manifest = bootstrap_service.load_default_into_state(state)
    assert manifest is not None
    assert manifest["source"] == "data.csv"
    assert state_service.validate(state) == []
    for key in (
        "raw_data",
        "clean_data",
        "customer_features",
        "rfm_results",
        "lrfm_results",
        "segmented_customers",
        "churn_scores",
    ):
        assert state[key] is not None, key
    assert state["retention_results"]["summary"]
    assert state["churn_model_metadata"]["model"] == manifest["selected_churn_model"]
    # Page 07 gates on the live dataset object, not the summary dict: without
    # this restore the churn page alone reports "No churn dataset yet".
    from churn.dataset_builder import ChurnDataset

    dataset = state["churn_dataset_obj"]
    assert isinstance(dataset, ChurnDataset)
    assert not dataset.frame.empty
    assert len(dataset.frame) == state["churn_dataset"]["customers"]
    assert dataset.feature_cols and dataset.categorical_cols
    assert dataset.labeling is not None
    assert dataset.labeling.cutoff == pd.Timestamp(state["churn_dataset"]["cutoff"])
    assert dataset.leakage_ok is True
    candidates = state["churn_model_metadata"]["candidates"]
    assert candidates, "default run preloads the churn candidate comparison"
    from services.churn_service import available_models

    # Every estimator available in this environment must be in the cached run
    # (lightgbm included once its optional dependency is installed).
    expected_models = {name for name, warning in available_models().items() if warning is None}
    assert set(manifest["churn_models"]) == expected_models
    assert {row["model"] for row in candidates} == expected_models
    assert any(
        row["model"] == manifest["selected_churn_model"] for row in candidates
    ), "the selected churn model is in the candidate set"
    history_state = state["segment_history"]
    assert history_state["table"], "default run preloads quarterly segment history"
    assert history_state["granularity"] == "quarterly"
    assert isinstance(history_state["transitions"], list)
    assert history_state["transitions"], "default run preloads customer transitions"


@needs_cache
def test_load_preserves_user_data() -> None:
    """A session that already has data keeps it; the manifest still reports."""
    state: dict = {}
    state_service.initialize(state)
    state["raw_data"] = "user-frame"
    manifest = bootstrap_service.load_default_into_state(state)
    assert manifest is not None
    assert state["raw_data"] == "user-frame"


@needs_cache
def test_cached_selection_follows_gated_recommendation() -> None:
    """The rail's Segments value is the gated pick, not the silhouette leader."""
    index = json.loads(
        (bootstrap_service.CACHE_DIR / "segmentation_index.json").read_text(encoding="utf-8")
    )
    if "recommendation" not in index:
        pytest.skip("cache predates the gated selection; rebuild to refresh")
    operational = index["recommendation"]["operational_pick"]
    if not operational.startswith("AI:"):
        pytest.skip("operational pick is traditional; the cache keeps an AI run")
    assert index["selected"] == operational.removeprefix("AI:")


def test_manifest_shape() -> None:
    """The manifest carries every field the landing page renders."""
    if not _MANIFEST.exists():
        pytest.skip("default run cache not built (run scripts/build_default_run.py)")
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    for key in (
        "source",
        "rows",
        "customers",
        "quality_overall",
        "rfm_segments",
        "lrfm_segments",
        "segmentation_models",
        "selected_segmentation_model",
        "churn_models",
        "selected_churn_model",
        "churn_auc",
        "churn_rate",
        "retention_customers",
        "cache_version",
        "segment_history_granularity",
        "segment_history_periods",
    ):
        assert key in manifest, key

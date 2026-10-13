"""Default-data bootstrap: cached data_edited.csv run loads into session state."""

import json

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
    assert manifest["source"] == "data_edited.csv"
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


@needs_cache
def test_load_preserves_user_data() -> None:
    """A session that already has data keeps it; the manifest still reports."""
    state: dict = {}
    state_service.initialize(state)
    state["raw_data"] = "user-frame"
    manifest = bootstrap_service.load_default_into_state(state)
    assert manifest is not None
    assert state["raw_data"] == "user-frame"


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
    ):
        assert key in manifest, key

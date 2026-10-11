"""Model persistence helpers: joblib save/load plus JSON metadata sidecars."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    """Persist a fitted model (pipeline, estimator, clusterer) and optional metadata sidecar."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    meta = dict(metadata or {})
    meta.setdefault("saved_at", datetime.now(UTC).isoformat())
    meta_path = dest.with_suffix(dest.suffix + ".metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return dest


def load_model(path: str | Path) -> Any:
    """Load a model persisted with save_model."""
    return joblib.load(path)


def load_metadata(model_path: str | Path) -> dict[str, Any]:
    """Load the metadata sidecar for a persisted model (empty dict if absent)."""
    meta_path = Path(str(model_path) + ".metadata.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))

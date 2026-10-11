"""Phase 0 smoke tests: config, constants, and sample data are coherent."""

from pathlib import Path

import pandas as pd
import yaml

from utils.constants import (
    CANONICAL_FIELDS,
    COLUMN_ALIASES,
    RANDOM_SEED,
    REQUIRED_FIELDS,
    SESSION_KEYS,
)


def test_config_has_required_sections() -> None:
    """config.yaml must contain every section the spec §7 requires."""
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    for section in ("app", "quality", "rfm", "segmentation", "churn", "retention"):
        assert section in cfg, f"config.yaml missing section: {section}"
    assert cfg["app"]["random_seed"] == RANDOM_SEED
    assert cfg["quality"]["age_min"] == 0 and cfg["quality"]["age_max"] == 120
    low, high = cfg["churn"]["risk_bands"]["low"], cfg["churn"]["risk_bands"]["critical"]
    assert low[0] == 0.0 and high[1] == 1.0


def test_canonical_schema_and_aliases() -> None:
    """Required fields are canonical and every canonical field has alias coverage."""
    assert set(REQUIRED_FIELDS) <= set(CANONICAL_FIELDS)
    for field in CANONICAL_FIELDS:
        assert field in COLUMN_ALIASES and field in COLUMN_ALIASES[field]
    assert len(SESSION_KEYS) == 19


def test_sample_data_present_and_shaped() -> None:
    """Generator output exists with required columns and >= 1000 customers."""
    path = Path("data/sample/sample_customer_transactions.csv")
    assert path.exists(), "run: python utils/sample_data.py"
    df = pd.read_csv(path)
    for field in REQUIRED_FIELDS:
        assert field in df.columns, f"sample data missing column: {field}"
    assert df["customer_id"].nunique() >= 1000
    assert len(df) > df["customer_id"].nunique()  # repeated transactions per customer

"""Startup configuration validation with business-readable errors."""

from __future__ import annotations

import yaml

from utils.exceptions import ConfigurationError


def load_config(path: str = "config.yaml") -> dict:
    """Load and validate config.yaml. Raises ConfigurationError on any problem."""
    try:
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(
            "config.yaml is missing. Restore it from version control before starting."
        ) from exc
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Check every business section; fail fast with the exact offending value."""
    problems: list[str] = []
    for section in ("app", "quality", "rfm", "segmentation", "churn", "retention"):
        if section not in cfg:
            problems.append(f"missing section '{section}'")
    if problems:
        raise ConfigurationError("Invalid config.yaml: " + "; ".join(problems) + ".")
    quality = cfg["quality"]
    if not quality["age_min"] <= quality["age_max"]:
        problems.append("quality.age_min must be <= age_max")
    if quality["outlier_iqr_multiplier"] <= 0:
        problems.append("quality.outlier_iqr_multiplier must be positive")
    if cfg["rfm"]["quantiles"] not in (3, 4, 5):
        problems.append("rfm.quantiles must be 3, 4, or 5")
    seg = cfg["segmentation"]
    if seg["minimum_customers"] < 2:
        problems.append("segmentation.minimum_customers must be >= 2")
    bounds = sorted(edge for band in cfg["churn"]["risk_bands"].values() for edge in band)
    if [round(b, 2) for b in bounds] != [0.0, 0.30, 0.30, 0.60, 0.60, 0.80, 0.80, 1.0]:
        problems.append("churn.risk_bands must partition [0, 1] without gaps/overlaps")
    if cfg["retention"]["value_quantiles"] not in (2, 3):
        problems.append("retention.value_quantiles must be 2 or 3")
    if problems:
        raise ConfigurationError("Invalid config.yaml: " + "; ".join(problems) + ".")

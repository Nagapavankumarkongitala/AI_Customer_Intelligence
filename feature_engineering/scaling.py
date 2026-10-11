"""Scaler factory shared by the segmentation and churn pipelines."""

from __future__ import annotations

from sklearn.preprocessing import RobustScaler, StandardScaler

SCALERS = ("robust", "standard")


def make_scaler(name: str = "robust"):
    """Return an unfitted scaler. Robust (median/IQR) is the default: outlier-proof."""
    if name == "robust":
        return RobustScaler()
    if name == "standard":
        return StandardScaler()
    raise ValueError(f"Unknown scaler '{name}'. Choose from {SCALERS}.")

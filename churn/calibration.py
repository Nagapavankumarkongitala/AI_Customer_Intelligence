"""Probability calibration fitted on validation data only.

Platt scaling (logistic sigmoid over the base model's validation probabilities)
implemented directly: it works with DataFrame-column pipelines on every sklearn
version, unlike CalibratedClassifierCV's cross-validation machinery, and it never
revisits training data.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Sigmoid calibrator wrapping any fitted estimator with predict_proba."""

    def __init__(self, estimator) -> None:
        self.estimator = estimator
        self.platt = LogisticRegression(max_iter=1000)

    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> PlattCalibrator:
        """Learn the sigmoid on validation probabilities only."""
        p = np.asarray(self.estimator.predict_proba(X_val)[:, 1]).reshape(-1, 1)
        self.platt.fit(p, np.asarray(y_val))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated P(churn) for new rows."""
        p = np.asarray(self.estimator.predict_proba(X)[:, 1]).reshape(-1, 1)
        return self.platt.predict_proba(p)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Class labels at 0.5 on calibrated probabilities."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibrate_on_validation(estimator, X_val: np.ndarray, y_val: np.ndarray):
    """Wrap a fitted estimator with Platt scaling learned on the validation split."""
    return PlattCalibrator(estimator).fit(X_val, y_val)

"""Clustering stability: mean pairwise Adjusted Rand Index across seeds and bootstraps.

ARI is permutation-invariant, so no label alignment is needed. Bootstrap compares
full-data labels against a resample-fitted model's predictions on the full data
(models with .predict); seed comparison works for every model.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample

FitFn = Callable[..., tuple[np.ndarray, object]]


def _mean_pairwise_ari(labelings: list[np.ndarray]) -> float:
    scores: list[float] = []
    for i in range(len(labelings)):
        for j in range(i + 1, len(labelings)):
            scores.append(adjusted_rand_score(labelings[i], labelings[j]))
    return round(float(np.mean(scores)), 4) if scores else 1.0


def seed_stability(
    fit_fn: FitFn, X: np.ndarray, seeds: tuple[int, ...] = (42, 7, 123), **kwargs
) -> dict:
    """Fit once per seed on the full data; report mean pairwise ARI."""
    labelings = [np.asarray(fit_fn(X, random_state=s, **kwargs)[0]) for s in seeds]
    return {
        "method": "seeds",
        "seeds": list(seeds),
        "mean_pairwise_ari": _mean_pairwise_ari(labelings),
    }


def bootstrap_stability(
    fit_fn: FitFn,
    X: np.ndarray,
    n_boot: int = 3,
    frac: float = 0.8,
    seed: int = 42,
    **kwargs,
) -> dict:
    """Resample-fit models must expose .predict; ARI vs full-data labels on full X."""
    X = np.asarray(X, dtype=float)
    full_labels = np.asarray(fit_fn(X, random_state=seed, **kwargs)[0])
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _b in range(n_boot):
        sample = resample(X, n_samples=int(len(X) * frac), random_state=rng.integers(1e9))
        _, model = fit_fn(sample, random_state=seed, **kwargs)
        if not hasattr(model, "predict"):
            raise ValueError("Bootstrap stability needs a model with .predict().")
        scores.append(adjusted_rand_score(full_labels, np.asarray(model.predict(X))))
    return {
        "method": "bootstrap",
        "n_boot": n_boot,
        "frac": frac,
        "mean_ari": round(float(np.mean(scores)), 4),
    }

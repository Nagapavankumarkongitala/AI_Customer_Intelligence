"""Model-control explainers for the Advanced Segmentation page.

Page 05 exposes six settings that drive every run, and each one belongs to a
different model or workflow step: today only HDBSCAN reads the minimum cluster
size, only the autoencoder reads the latent dims and epochs, and the seed does
double duty as the stability probe. This module owns that copy — the tooltip
beside each control and the matrix of which controls a model actually reads —
so the page stays orchestration and the explanations stay reviewable in one
place (``tests/test_control_guides.py`` pins both to the page's own labels).
"""

from __future__ import annotations

#: Orientation line printed above the model controls.
INTRO = (
    "These settings drive every run you start below and are saved with the run for "
    "page 06's comparison."
)

#: Tooltip copy keyed by the exact widget label rendered on page 05.
CONTROL_HELP: dict[str, str] = {
    "Model": (
        "Which algorithm to fit. k-means, GMM and hierarchical split a fixed k; HDBSCAN "
        "finds its own cluster count and may leave customers as noise; the autoencoder "
        "clusters its compressed latent space; the ensemble consensus-combines several "
        "models. Optional models need their package installed."
    ),
    "Clusters (k)": (
        "How many segments this model should produce. Read by k-means, GMM, hierarchical, "
        "the autoencoder (on its latent space) and the ensemble; HDBSCAN derives its own "
        "count, so the control is disabled while it is selected."
    ),
    "Random seed": (
        "Seeds every stochastic step (cluster init, subsampling, autoencoder weights) so a "
        "run can be reproduced. The stability score refits with other seeds to test whether "
        "the partition itself survives the change."
    ),
    "HDBSCAN min cluster size": (
        "Smallest group HDBSCAN will call a cluster. Lower values split more aggressively "
        "and push more customers into noise. Used only by HDBSCAN."
    ),
    "AE latent dims": (
        "Width of the autoencoder's compressed bottleneck; k-means then runs on that space. "
        "More dimensions keep more detail, fewer force a coarser grouping. Used only by the "
        "autoencoder."
    ),
    "AE max epochs": (
        "Cap on autoencoder training passes. Early stopping can finish sooner; more epochs "
        "cost time and can overfit. Used only by the autoencoder."
    ),
}

#: Tooltip copy for the three run actions (their labels are built at runtime).
SWEEP_HELP = (
    "Fits the selected k-based model for every k in the configured range and reports the "
    "internal metrics, so you can pick a k before committing a run. Not available for "
    "HDBSCAN, which finds its own count."
)
RUN_HELP = (
    "Fits the selected model with the settings above, then evaluates it and charts the "
    "clusters. Save the run to compare it against RFM/LRFM on page 06."
)
TRAIN_ALL_HELP = (
    "Fits all six models with the settings above in one pass; a model whose optional package "
    "is missing records as skipped instead of failing the batch. The autoencoder makes this "
    "take minutes."
)

#: Which controls each model reads (the seed is shared by all of them).
MODEL_CONTROL_USAGE: tuple[dict[str, str], ...] = (
    {"model": "kmeans", "cluster count": "Clusters (k)", "extra controls": "—"},
    {"model": "gmm", "cluster count": "Clusters (k)", "extra controls": "—"},
    {"model": "hierarchical", "cluster count": "Clusters (k)", "extra controls": "—"},
    {
        "model": "hdbscan",
        "cluster count": "found in the data",
        "extra controls": "HDBSCAN min cluster size",
    },
    {
        "model": "autoencoder",
        "cluster count": "Clusters (k) on the latent space",
        "extra controls": "AE latent dims, AE max epochs",
    },
    {"model": "ensemble", "cluster count": "Clusters (k)", "extra controls": "—"},
)

#: Sentence printed under the usage matrix.
USAGE_CAPTION = (
    "The random seed is passed to every model and drives the stochastic ones (k-means, "
    "GMM, autoencoder, ensemble). A batch run records each model's status, so a missing "
    "optional dependency shows as skipped rather than silently absent."
)


def control_help(label: str) -> str:
    """Tooltip for a model control; an unknown label simply explains nothing."""
    return CONTROL_HELP.get(label, "")


def control_usage_rows() -> list[dict[str, str]]:
    """Copy of the usage matrix, ready for ``st.dataframe``."""
    return [dict(row) for row in MODEL_CONTROL_USAGE]

# Model Card — ensemble_k5

Date: 2026-10-11

## Purpose
Unsupervised customer segmentation with ensemble.

## Training data
recorded in run audit

## Input features
recency_days, frequency, monetary_value, relationship_length_days, average_order_value

## Preprocessing
Median imputation, skew-triggered log1p, robust scaling.

## Hyperparameters
{'n_clusters': 5, 'seed': 42}

## Evaluation metrics
{'silhouette': 0.1773, 'davies_bouldin': 1.3808, 'calinski_harabasz': 331.2554, 'n_clusters': 5, 'noise_pct': 0.0, 'min_cluster_size': 129}

## Limitations
Clusters describe past behavior; cluster IDs are arbitrary until profiled; small/noisy clusters need review.

## Fairness considerations
Demographics used for profiling only, never as clustering inputs unless explicitly selected; review segment composition for disparate impact before targeted action.

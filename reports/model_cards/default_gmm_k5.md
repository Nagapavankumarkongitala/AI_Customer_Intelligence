# Model Card — default_gmm_k5

Date: 2026-11-16

## Purpose
Unsupervised customer segmentation with gmm.

## Training data
recorded in run audit

## Input features
recency_days, frequency, monetary_value, relationship_length_days, average_order_value

## Preprocessing
Median imputation, skew-triggered log1p, robust scaling.

## Hyperparameters
{'n_clusters': 5, 'seed': 42}

## Evaluation metrics
{'silhouette': 0.048, 'davies_bouldin': 2.4293, 'calinski_harabasz': 5755.2722, 'n_clusters': 5, 'noise_pct': 0.0, 'min_cluster_size': 2115}

## Limitations
Clusters describe past behavior; cluster IDs are arbitrary until profiled; small/noisy clusters need review.

## Fairness considerations
Demographics used for profiling only, never as clustering inputs unless explicitly selected; review segment composition for disparate impact before targeted action.

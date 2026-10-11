# Model Card — kmeans_v1

Date: 2026-10-11

## Purpose
Unsupervised customer segmentation with kmeans.

## Training data
2024

## Input features
recency_days

## Preprocessing
Median imputation, skew-triggered log1p, robust scaling.

## Hyperparameters
{'k': 3}

## Evaluation metrics
{'silhouette': 0.5, 'davies_bouldin': None, 'calinski_harabasz': None, 'n_clusters': None, 'noise_pct': None, 'min_cluster_size': None}

## Limitations
Clusters describe past behavior; cluster IDs are arbitrary until profiled; small/noisy clusters need review.

## Fairness considerations
Demographics used for profiling only, never as clustering inputs unless explicitly selected; review segment composition for disparate impact before targeted action.

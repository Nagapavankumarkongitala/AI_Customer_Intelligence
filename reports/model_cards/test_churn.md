# Model Card — logreg_v1

Date: 2026-10-20

## Purpose
Leakage-safe churn classification with logreg.

## Training data
Predictors ≤ 2024-06-01; outcome window 90d after cutoff.

## Input features
recency_days

## Preprocessing
Median/most-frequent imputation, one-hot encoding, scaling for logreg; balanced class weights.

## Hyperparameters
{'model': 'logreg', 'threshold': 0.5}

## Evaluation metrics
{'roc_auc': 0.9, 'pr_auc': None, 'precision': None, 'recall': None, 'f1': None, 'balanced_accuracy': None, 'log_loss': None, 'brier': None}

## Limitations
Predicts inactivity, not intent; performance degrades as behavior drifts — revalidate on fresh outcome windows.

## Fairness considerations
Demographics are optional predictors; check score calibration across gender/region slices before operational use.

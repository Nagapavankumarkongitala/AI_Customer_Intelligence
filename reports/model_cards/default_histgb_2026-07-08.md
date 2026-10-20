# Model Card — default_histgb_2026-07-08

Date: 2026-10-21

## Purpose
Leakage-safe churn classification with histgb.

## Training data
Predictors ≤ 2026-07-08; outcome window 90d after cutoff.

## Input features
recency_days, frequency, monetary_value, relationship_length_days, average_order_value, active_months, transactions_per_active_month, average_interpurchase_days, interpurchase_std_days, recent_period_frequency, previous_period_frequency, frequency_change, recent_period_monetary, previous_period_monetary, monetary_change, age, gender, city, state, country, region

## Preprocessing
Median/most-frequent imputation, one-hot encoding, scaling for logreg; balanced class weights.

## Hyperparameters
{'model': 'histgb', 'threshold': 0.05}

## Evaluation metrics
{'roc_auc': 0.5117, 'pr_auc': 0.8638, 'precision': 0.8629, 'recall': 1.0, 'f1': 0.9264, 'balanced_accuracy': 0.5, 'log_loss': 0.3997, 'brier': 0.1183}

## Limitations
Predicts inactivity, not intent; performance degrades as behavior drifts — revalidate on fresh outcome windows.

## Fairness considerations
Demographics are optional predictors; check score calibration across gender/region slices before operational use.

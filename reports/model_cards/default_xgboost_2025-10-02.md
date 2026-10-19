# Model Card — default_xgboost_2025-10-02

Date: 2026-10-19

## Purpose
Leakage-safe churn classification with xgboost.

## Training data
Predictors ≤ 2025-10-02; outcome window 90d after cutoff.

## Input features
recency_days, frequency, monetary_value, relationship_length_days, average_order_value, active_months, transactions_per_active_month, average_interpurchase_days, interpurchase_std_days, recent_period_frequency, previous_period_frequency, frequency_change, recent_period_monetary, previous_period_monetary, monetary_change, age, gender, city, state, country, region

## Preprocessing
Median/most-frequent imputation, one-hot encoding, scaling for logreg; balanced class weights.

## Hyperparameters
{'model': 'xgboost', 'threshold': 0.05}

## Evaluation metrics
{'roc_auc': 0.5123, 'pr_auc': 0.7459, 'precision': 0.7319, 'recall': 1.0, 'f1': 0.8452, 'balanced_accuracy': 0.5, 'log_loss': 0.5815, 'brier': 0.1963}

## Limitations
Predicts inactivity, not intent; performance degrades as behavior drifts — revalidate on fresh outcome windows.

## Fairness considerations
Demographics are optional predictors; check score calibration across gender/region slices before operational use.

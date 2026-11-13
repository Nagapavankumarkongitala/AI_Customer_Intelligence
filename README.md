# AI Customer Intelligence Platform

A production-oriented, end-to-end customer intelligence platform built with Python 3.12, Streamlit, scikit-learn and a modular ML stack. It transforms raw transaction data into actionable customer segmentation, leakage-safe churn prediction, and data-driven retention strategies — with a filterable executive dashboard and multi-format reporting.

**Entry point:** `app.py`  
**Default dataset:** `data.csv` (auto-loaded via `services/bootstrap_service.py`)  
**Generated:** 2026-10-24 09:30:35

## Overview
Implements: ingestion → validation/mapping → data quality & cleaning (audited) → customer features → RFM/LRFM → demographic profiling → advanced segmentation (multi-model + ensemble) → segmentation comparison → leakage-safe churn prediction (time-aware, cohort splits) → retention intelligence (value-risk + strategy rules) → executive dashboard → systematic report (HTML/Excel/PDF/Word).

## Architecture
- **Pages** (`pages/NN_*.py`): UI/orchestration only
- **Services** (`services/`): pipeline/business logic
- **Analytics** (`analytics/`): stats/analysis
- **Feature Engineering** (`feature_engineering/`): prep/scaling
- **Segmentation** (`segmentation/`): clustering/profiling/stability/ensemble
- **Churn** (`churn/`): labeling/training/eval/calibration/explainability
- **Retention Engine** (`retention_engine/`): priority/value/strategy mapping
- **Visualization** (`visualization/`): charts/themes/tokens
- **Components** (`components/`): reusable UI
- **Utils** (`utils/`): formatting/config/logging/helpers

Session state contract (owned by `services/state_service.py`): raw_data, clean_data, column_mapping, cleaning_config, cleaning_audit, data_dictionary, customer_features, rfm_results, lrfm_results, demographic_profile, segmentation_runs, selected_segmentation_model, segmented_customers, churn_dataset, churn_model, churn_model_metadata, churn_scores, retention_results, global_filters.

## Data Schema
**Required:** `customer_id`, `transaction_date`, `amount_spent`  
**Optional:** `gender`, `age`, `city`, `state`, `country`, `region`  
Encoding fallbacks: UTF-8 BOM → UTF-8 → CP1252 → Latin-1. Never silently drop rows; all actions audited.

## Workflow (Idea → Execution)
1. Bootstrap (load/cache full pipeline from data.csv)  
2. Upload & mapping (auto-detect + validate)  
3. Data quality & cleaning (audit)  
4. Customer features & RFM/LRFM  
5. Demographics  
6. Advanced segmentation (KMeans/GMM/Hierarchical/HDBSCAN/Autoencoder/Ensemble)  
7. Segmentation comparison (metrics/stability/value/evenness)  
8. Churn prediction (leakage-safe labeling, cohort splits, calibration)  
9. Retention strategies (value-risk, priority, YAML rules)  
10. Dashboard & Report (narratives, insights, exports)

## ML Models

### Segmentation (Unsupervised)
- **KMeans** (sklearn) — centroid, fast/interpretable (default)
- **GMM** (sklearn.mixture) — probabilistic, elliptical clusters
- **Hierarchical/Agglomerative Ward** (sklearn) — dendrogram, O(n²)
- **HDBSCAN** (hdbscan, optional) — density-based, noise
- **Autoencoder + Clustering** (PyTorch, optional) — non-linear embeddings
- **Ensemble/Consensus** (segmentation/ensemble.py) — co-association consensus

### Churn (Supervised, Leakage-Safe)
- **Logistic Regression** (sklearn) — interpretable baseline
- **Random Forest** (sklearn) — robust bagged trees
- **HistGradientBoosting** (sklearn) — efficient boosting (default)
- **LightGBM** (lightgbm, optional) — fast boosting
- **XGBoost** (xgboost, optional) — regularized boosting

Labeling: predictors ≤ cutoff, label in (cutoff, cutoff+horizon]; cohort-ordered splits by first transaction. Calibration via PlattCalibrator.

## Installation
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
Linux/macOS: `bash run_linux_mac.sh`

## Run
```powershell
streamlit run app.py
```

## Testing & Quality
```powershell
python -m compileall .
ruff check .
black --check .
pytest -q
```

## Exports
HTML, Excel (openpyxl), PDF (reportlab + matplotlib), Word (python-docx). Indian notation default; IDs masked by default.

## Guardrails
Leakage-safe churn, no silent row drops (audited), ID masking, deterministic seeds (42), graceful optional deps, historical value = proxy (not CLV), non-causal retention reasons.

# CHANGELOG

## 2026-10-11 — Phase 8: Executive Dashboard and Production Readiness
- Added: `services/dashboard_service.py` (master table, global filters, KPIs,
  trends, portfolio, masked high-risk list), `visualization/executive_dashboard.py`,
  `components/filters.py`, `components/export_buttons.py`,
  page `09_Executive_Dashboard.py`, `utils/config.py` (startup validation),
  `utils/model_card.py` (cards written on every model save), `Dockerfile`,
  `.dockerignore`, `.github/workflows/ci.yml`, `DEPLOYMENT.md`.
- Added: tests `test_dashboard.py`, `test_charts.py`, `test_end_to_end.py`
  (full 14-step workflow on sample data passes).
- Removed: out-of-spec starter stubs with no callers (basket/CLV/health/cohort
  analytics, standalone explainability modules, encoding/feature_creation,
  navbar/sidebar/charts components).
- Fixed: value–risk matrix tolerates missing hover columns.
- Added: `retention_engine/` value tiers, priority scoring, strategy matrix
  loader/validator, next-best-action, offer catalog, campaign export;
  `retention_strategies.yaml`; `services/retention_service.py`;
  `visualization/retention_charts.py`; page `08_Retention_Strategies.py`;
  `tests/test_retention.py` (94 passed total).
- Added: `churn/` labeling, dataset builder, baselines, RF/XGB/LightGBM trainers,
  evaluation, Platt calibration, explainability, risk scoring;
  `services/churn_service.py`, `visualization/churn_charts.py`,
  page `07_Churn_Prediction.py`, tests `test_churn_labeling.py`,
  `test_churn_models.py` (86 passed total).
- Fixed: cohort (first-txn) chronological splits — last-txn ordering segregates
  classes by construction; `CalibratedClassifierCV(cv='prefit')` unsupported on
  sklearn 1.9, replaced with direct Platt scaling.
- Added: `analytics/segmentation_comparison.py`, Sankey + metric bars,
  page `06_Segmentation_Comparison.py`, `tests/test_comparison.py` (72 passed total).
- Changed: page 05 saved runs persist labels + customer IDs for comparison.
- Added: `feature_engineering/preprocessing.py`, `scaling.py`; `segmentation/`
  `common/evaluation/stability/segment_profiling/segment_naming` + six model
  modules (incl. new `autoencoder_model.py`); `deep_learning/` autoencoder,
  embeddings, latent viz; `services/segmentation_service.py`;
  `visualization/segmentation_charts.py`; page `05_Advanced_Segmentation.py`.
- Added: `tests/test_segmentation.py` (64 passed total).
- Added: `analytics/demographics.py`, `analytics/statistical_tests.py`,
  `visualization/demographic_charts.py`, page `04_Demographic_Profiling.py`.
- Added: `tests/test_demographics.py` (53 passed total). Analytics helpers avoid
  `test_*` names so pytest never collects them as tests.
- Added: `analytics/customer_features.py`, `analytics/rfm.py`, `analytics/lrfm.py`,
  `services/feature_service.py`, `visualization/rfm_charts.py`,
  `components/metric_cards.py`, page `03_RFM_LRFM_Analysis.py`.
- Added: tests `test_customer_features.py`, `test_rfm.py`, `test_lrfm.py`;
  sample-data end-to-end extended through RFM/LRFM scoring (42 passed total).
- Added: `services/column_detection.py`, `data_service.py`, `profiling_service.py`,
  `cleaning_service.py`, `data_dictionary_service.py`, `export_service.py`,
  `state_service.py`.
- Added: pages `01_Data_Upload_and_Mapping.py`, `02_Data_Quality_and_Cleaning.py`;
  removed obsolete 8-page stubs (pages 03–09 arrive with their phases).
- Added: tests `test_column_detection.py`, `test_quality.py`, `test_cleaning.py`,
  `test_exports.py` (28 passed total); stale future-phase stub tests removed.
- Changed: `UploadPreview` carries the full normalized frame for session state.
- Added: `pyproject.toml` (black/ruff/pytest config), `requirements-dev.txt`,
  `.gitignore`, `.env.example`, `.streamlit/config.toml`, `run_windows.ps1`,
  `run_linux_mac.sh`.
- Added: `utils/` package (constants, exceptions, logging, validators, file_utils,
  model_utils, sample_data generator).
- Added: full `config.yaml` (spec §7), landing page `app.py`, `PROJECT_STATUS.md`.
- Added: synthetic sample data `data/sample/sample_customer_transactions.csv`
  (1,200 customers, intentional dirt) + `tests/test_setup.py` smoke tests.
- Changed: `requirements.txt` expanded to the spec §3 core stack (unpinned).
- Changed: `pages/` renumber to the 9-page scheme deferred to Phase 1 (user-confirmed).

# CHANGELOG

## 2026-10-11 — Phase 3: Demographic and Geographic Profiling
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

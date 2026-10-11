# PROJECT_STATUS

## Current phase
Phase 8 — Executive Dashboard and Production Readiness (complete, gates green).
All 8 build phases done.

## Completed features
- Phase 0: (as before — tooling, utils, config, landing page, sample data).
- Phase 1: `column_detection` (normalize, dedupe suffixes, alias+fuzzy auto-map,
  strict validation); `data_service` (CSV/XLSX/XLS/Parquet, encoding fallback,
  currency/date parsers, canonical frame); `profiling_service` (0.4/0.4/0.2 quality
  + issue breakdown, warnings separated from validity); `cleaning_service`
  (7 configurable treatments in fixed order + full audit record);
  `data_dictionary_service`, `export_service` (CSV/Excel/JSON bytes),
  `state_service` (19-key contract: init/validate/reset/status).
- Pages renumbered to the 9-page scheme: `01_Data_Upload_and_Mapping`,
  `02_Data_Quality_and_Cleaning` (+ dictionary + downloads),
  `03_RFM_LRFM_Analysis` (KPIs, RFM/LRFM tabs, charts, profile, drill-down,
  downloads). Pages 04–09 arrive with their phases.
- Phase 2: (as before — features, RFM/LRFM, page 03).
- Phase 3: (as before — demographics, segment tests, page 04).
- Phase 4: `feature_engineering/preprocessing.py` + `scaling.py` (impute,
  skew-triggered log1p inside the persisted pipeline, robust/standard scaling,
  variance + correlation review); six `segmentation/` models (kmeans, gmm,
  hierarchical, hdbscan-optional, autoencoder-TensorFlow, co-association
  ensemble); `evaluation.py` (noise-aware silhouette/DB/CH), `stability.py`
  (seed + bootstrap ARI), `segment_profiling.py`, `segment_naming.py` (value ×
  activity grid, post-hoc only); `deep_learning/` autoencoder + embeddings +
  latent viz; `services/segmentation_service.py` (registry, k-sweep, stability,
  persistence); `visualization/segmentation_charts.py`; page
  `05_Advanced_Segmentation` (prepare → sweep → run → profile → override names
  → save run + model files).
- Phase 5: `analytics/segmentation_comparison.py` (ARI agreement, size balance,
  revenue concentration, monetary eta² separation, churn-rates-when-available,
  interpretability checklist, gated recommendation that never blindly follows
  silhouette); Sankey + metric bars in `visualization/segmentation_charts.py`;
  page `06_Segmentation_Comparison` (scorecard, agreement + Sankey + crosstab,
  checklists, recommendation with human override, downloads). Saved runs now
  persist labels for comparison.
- Phase 6: `churn/` labeling (cutoff/horizon/min-history, window validation),
  dataset builder (predictors ≤ cutoff, leakage proof), baselines
  (logreg/HGB) + RF/XGBoost/LightGBM trainers, evaluation (AUCs, calibration,
  lift/gain, F1 threshold), Platt calibration, explainability (permutation,
  logreg odds, SHAP-optional, per-customer drivers), risk scoring to the output
  schema; `services/churn_service.py` (cohort time splits, train/calibrate/
  persist); `visualization/churn_charts.py`; page `07_Churn_Prediction`
  (dataset → train/compare → threshold → explain → score + downloads).
- Phase 7: retention tiers, priority scoring, YAML strategy matrix, traceable
  actions, campaign export, page `08_Retention_Strategies`.
- Phase 8: `services/dashboard_service.py`, `visualization/executive_dashboard.py`,
  `components/filters.py` + `export_buttons.py`, page `09_Executive_Dashboard`
  (filter-consistent KPIs/trends/portfolio/risk/demographics/retention, masked IDs,
  gated drill-down); `utils/config.py` startup validation; model cards on save;
  Dockerfile, CI, DEPLOYMENT.md; end-to-end workflow test.

## Current test results
- `python -m compileall` clean; `ruff check .` clean; `black --check .` clean;
  `pytest -q --cov` 100+ passed (incl. full 14-step workflow on sample data);
  `streamlit run app.py` health 200.
  Run on 2026-10-11, Python 3.12, Streamlit 1.63.

## Known issues
- `ponytail/`, `servers/`, `awesome-ai-agents/` are unrelated nested checkouts;
  excluded from tooling, not deleted (user decision).
- LightGBM, SHAP, HDBSCAN, UMAP are not installed: pages show install guidance
  and degrade gracefully (verified by tests). Single-user in-memory sessions;
  see DEPLOYMENT.md for restart behavior.

## Next phase
Done. All 8 build phases complete — maintenance mode (revalidate churn models
on fresh outcome windows as behavior drifts).

# PROJECT_STATUS

## Current phase
Phase 5 — Segmentation Comparison (complete, gates green)

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

## Current test results
- `python -m compileall` clean; `ruff check .` clean; `black --check .` clean;
  `pytest -q` 72 passed;
  `streamlit run app.py` health 200.
  Run on 2026-10-11, Python 3.12, Streamlit 1.63.

## Known issues
- `ponytail/`, `servers/`, `awesome-ai-agents/` are unrelated nested checkouts;
  excluded from tooling, not deleted (user decision).
- Churn, retention analytics and pages 07–09 are still stubs; implemented in
  Phases 6–8. HDBSCAN and UMAP are not installed: pages show install guidance
  and degrade gracefully (verified by tests).

## Next phase
Phase 6 — Leakage-Safe Churn Prediction (time-aware dataset, 5 models, eval +
calibration, explainability, page 07).

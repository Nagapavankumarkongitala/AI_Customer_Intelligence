# PROJECT_STATUS

## Current phase
Phase 3 — Demographic and Geographic Profiling (complete, gates green)

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
- Phase 3: `analytics/demographics.py` (age bands incl. custom specs, gender/geo
  tables with revenue, concentration incl. HHI, per-segment profiles, population
  comparisons — absent fields yield notes, never crashes);
  `analytics/statistical_tests.py` (ANOVA/Welch/Kruskal-Wallis auto-choice,
  chi-square/Fisher + Cramer's V, Holm correction); `visualization/demographic_charts.py`;
  page `04_Demographic_Profiling` (missing indicators, charts, treemaps, segment
  tests, downloads).

## Current test results
- `python -m compileall` clean; `ruff check .` clean; `black --check .` clean;
  `pytest -q` 53 passed (incl. sample-data end to end through RFM/LRFM scoring);
  `streamlit run app.py` health 200.
  Run on 2026-10-11, Python 3.12, Streamlit 1.63.

## Known issues
- `ponytail/`, `servers/`, `awesome-ai-agents/` are unrelated nested checkouts;
  excluded from tooling, not deleted (user decision).
- Segmentation, churn, retention analytics and pages 05–09 are still stubs;
  implemented in Phases 4–8.

## Next phase
Phase 4 — Advanced Segmentation (feature pipeline, 6 clustering models,
evaluation + stability, post-hoc naming, page 05).

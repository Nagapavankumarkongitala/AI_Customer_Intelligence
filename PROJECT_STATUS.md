# PROJECT_STATUS

## Current phase
Phase 2 — Customer Features and RFM/LRFM (complete, gates green)

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
- Phase 2: `analytics/customer_features.py` (18 features, snapshot default,
  latest-non-null demographics + conflicts), `analytics/rfm.py` + `lrfm.py`
  (rank-percentile 3/4/5 scoring, ordered 10-segment naming rules),
  `services/feature_service.py` (orchestration + segment profiles),
  `visualization/rfm_charts.py`, `components/metric_cards.py`.

## Current test results
- `python -m compileall` clean; `ruff check .` clean; `black --check .` clean;
  `pytest -q` 42 passed (incl. sample-data end to end through RFM/LRFM scoring);
  `streamlit run app.py` health 200.
  Run on 2026-10-11, Python 3.12, Streamlit 1.63.

## Known issues
- `ponytail/`, `servers/`, `awesome-ai-agents/` are unrelated nested checkouts;
  excluded from tooling, not deleted (user decision).
- Segmentation, churn, retention analytics and pages 04–09 are still stubs;
  implemented in Phases 3–8.

## Next phase
Phase 3 — Demographic and Geographic Profiling (age bands, geo concentration,
segment profiles, statistical tests, page 04, missing-field tolerance).

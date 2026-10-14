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
Done — plus UI overhauls (terminal → glassmorphism → navy/cyan tab-bar UI with
control expanders and train-all). Maintenance mode; revalidate churn models on
fresh outcome windows as behavior drifts.
\
## UI redesign (light foundation, in progress)
- Phase 1 (committed): isualization/tokens.py (light default + dark skins, segment/risk/action maps, WCAG guards), isualization/theme.py rewritten around pply_chart_theme/load_theme (dark re-map, foreign-default adoption), light .streamlit/config.toml, rewritten theme/token tests.
- Phase 2 (committed): utils/formatting.py plus components/ library (page_header, metric_card max 5/row, status_badge, insight_box, empty_state, panels, 	able) with tests. Legacy kpi_row/inject_terminal_css kept importable.
- Phase 3 (this change): persistent left sidebar (components/sidebar.py: NAV_PAGES nav, appearance toggle, data status), components/theme.py preference outside the 19-key state contract, ender_nav_tabs kept as a delegating shim (zero page edits), 	est_no_sidebar_usage replaced with sidebar guards. Next: executive-dashboard rebuild, visualization refactor, remaining pages, hardening.\
\
- Phase 4 (this change): pages/09_Executive_Dashboard.py rebuilt on the component library (page/section headers, 5+2 metric rows, 2-per-row chart sections, value-risk matrix centerpiece, factual insight boxes, conditional tables, empty states, retained gated drill-down). No service/model changes; filter logic and IDs-masking untouched.\
\
- Phase 5 (this change): visualization refactor � 	okens.segment_map/action_map (all-or-nothing stable hues), segment/action maps wired into RFM/segmentation/retention factories (positional/magnitude coloring fixed), 	heme.empty_figure titled placeholders on all bar/scatter/line factories, missing axis labels filled. No signature changes; module names kept.\
\
- Phase 6 (this change): remaining 8 pages + landing migrated to the component library (page/section headers with step badges, metric rows, chart sections, conditional tables, empty-state guards, factual insight boxes). Filter/service/gating logic untouched; kpi_row retained for compatibility. Verified by empty-session runs of all 9 views plus targeted suites.\
\
- Phase 7 hardening (this change): full suite + coverage (84% total; touched viz/components/utils at 94-100%), Streamlit boot on :8501 with health 200 and zero boot tracebacks, ruff/black clean. Two failures are pre-existing and unrelated (shap 0.53 TreeExplainer rejects LogisticRegression; reproduced on the pristine tree).\
\
- Follow-ups (this change): SHAP fixed for shap 0.53 � linear baselines route to LinearExplainer (coef_ duck-typed), 3-D TreeExplainer output sliced to the positive class, scoring falls back to median drivers past the 500-row SHAP cap (all three were latent crashes on the previous suite). ender_nav_tabs shim removed: app + all 9 pages call ender_sidebar directly.\
\
- Table hardening (this change): jinja2 added to requirements (was transitive-only) and style_table falls back to the formatted frame when the Styler is unavailable � values stay formatted, only cell painting degrades. Fixes the production crash on interpreters without jinja2.\
\
- Slicer move (this change): view slicers live in the rail (ender_sidebar_nav head, sidebar_filters sections, ender_sidebar_footer foot). Moved: page 09 Global Slicers + geography level, page 08 View Filters, page 06 comparison pair, page 04 profiling slicers. Compute-driving config stays in-page. Verified by with-data rail render (6 slicers, 11 figures, footer).\
\
- Dictionary crash fix (this change): bootstrap wrote the dictionary cache as column?text (DataFrame.items), crashing page 02 on load; writer now stores records-orient like the live path, and 
ormalize_dictionary_rows renders legacy caches as field/detail pairs instead of raising.\
\
- Browser pass (this change): all 10 views audited in light via DOM (headings, cards, figure hues/heights, overflow, console clean); dark verified with persistence across navigation. Fixed: theme reset on page switch (persist_state='session' on the toggle), Sankey missing title, geo plural titles, tighter KPI type for narrow columns. Known cosmetic upstream: Streamlit double-bolds chart titles; use_container_width deprecation warnings (functional).\
\
- Width migration (this change): all 66 use_container_width=True call sites moved to width='stretch' ahead of the Streamlit removal; verified live with zero deprecation warnings.\
\
- Sidebar width (this change): rail narrowed 300px to 240px (	okens.SIDEBAR_WIDTH) with !important to beat Streamlit's inline width; drag-resize handle no longer stretches the rail (accepted trade-off). Live-verified: controls fit, no overflow.\
\
- Dark ink fix (this change): custom HTML text nodes inherited the static light root ink, rendering invisible values on dark surfaces. Every component text node now carries explicit active-mode color (metric values, empty/chart titles, status values, sidebar title); pinned by an active-mode ink test and live-verified on a fresh server.\
\
- Dashboard landing (this change): executive dashboard moved into pp.py, pages/09_* deleted; rail is Dashboard + 8 workflow pages. plural_geo_level public for shared titles. Verified by full landing run (11 figures, 9 links).\
\
- Startup order (this change): theme CSS and page config now apply before the slow default-dataset bootstrap (with a loading spinner), so first paint is themed with the auto-nav hidden instead of blank. streamlit>=1.65 pinned (verified APIs: persist_state, width, page_link).\
\
- Auto-nav suppression (this change): showSidebarNavigation=false removes the auto page list server-side, so no unstyled sidebar can flash during loads on any version; the redundant CSS hide rule is deleted (test now pins the config).\
\
- RFM single-screen (this change): page 03 scoring config + exports moved to rail sidebar sections; RFM/LRFM charts grouped into Distribution/Heatmap/Relationships inner tabs; new compact chart preset (300px, tight margins/type) applied to all 10 figures. Verified live-shape (10 figs at 300px, 3 exports).\
\
- Tabbed dashboard landing (this change): Business Overview (5 KPIs, area+segment+risk row, bubble+region+demographics row, top-opps+insights row), Customer Trends, Segments, Churn Risk, Retention tabs; segment bars take segment_col; evenue_area_fig and horizontal geo bars added. Pies declined: 10 RFM segments violate the no-pies-over-5 project rule.\
\
- Overview refinements (this change): segment donut (share + hole total, stable hues) replaces bars in Business Overview; demographics Age/Gender moved into inner tabs; top-opps table scroll-capped at 320px via new render height; segment_col support on segment bars; horizontal mode on revenue geo bars.\
\
- Dark rebuild (this change): dark palette redesigned from measurements (bright critical, lifted heatmap floor, stronger muted-soft) plus per-mode segment hues preserving hue families; re-map extended; dark contrast floor tests. Live-verified per-trace in dark mode.\
\
- Compute caching (this change): new components/cached.py wraps expensive pure service/analytics calls in st.cache_data so page switches stop recomputing everything (in-page tabs never reran � verified live; sidebar page switches did). Wired into landing + pages 02/03/04/05/08 with parity tests.\
\
- Single chart titles (this change): new 	heme.without_title strips in-figure titles (reclaiming top margin) everywhere a section heading frames the chart; factories keep titles for standalone use. Live-verified: 15 figures title-free, legends intact.\
\
- Sidebar collapse fix (this change): the !important rail width overrode Streamlit's collapsed state, pinning a 240px gap with no reflow. Width rule now scoped to [aria-expanded=true]; live-verified collapse to 0px with +240px main reflow and clean expand-back.\
\
- Demographics restructure (this change): missing-indicators block removed; Age histogram/bands + gender bar in one 3-column row; gender table moved to bottom; gender/region mix charts moved directly below the geography treemaps.\

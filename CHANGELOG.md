# CHANGELOG

## 2026-10-19 — Reworked: the report page is now a systematic, evidence-linked report
- Changed: `pages/09_Report_Generator.py` mirrors the executive dashboard tab by tab
  instead of dropping a handful of charts. Sections are now **Executive summary**,
  **Business & customer trends**, **Segments & demographics**, **Churn risk**,
  **Retention plan**, **Recommendations & next steps**, and **Appendix & data
  foundation** (`REPORT_SECTIONS` in `services/report_service.py`, labels in
  `SECTION_LABELS`). Each section carries metric cards, its charts with a plain
  reading guide, an **interpretation** block, and (in the retention section) the
  strategy mix — so no chart is left to speak for itself.
- Added: the customer-trends charts the report was missing (`revenue_area_fig`,
  `activity_trend_fig`, `new_returning_fig`), the AI/LRFM portfolios, the
  demographic composition (age band, gender, top regions), and the high-value /
  high-risk account list. New chart explainers were added to `CHART_EXPLANATIONS`
  for activity, new-vs-returning, and the three demographic charts.
- Added: `services/report_narrative.py` — the narrative engine. Pure functions
  turn the report context into `summary_insights` (the executive hook),
  `section_insights` (per-section interpretation), and `recommendations`
  (**Now / Next / Later** actions, each with the evidence that produced it).
  Language rules match the rest of the app: historical value is a proxy, never
  CLV, and nothing claims causal impact. Each section has its own test.
- Added: a **data foundation** appendix built with the xlsx-analyze profiling
  method — source rows in/out and rows removed (`cleaning_audit`), the transaction
  window, missing-cell rate, the per-field source profile (`data_dictionary`), and
  the cleaning treatments in order. The HTML/PDF exports carry the same narrative:
  summary insights, section interpretations, the recommendation list, and the data
  foundation; the Excel workbook gained `Monthly_Trends`, `Demographics_*`,
  `LRFM_Portfolio`, `Data_Source`, and `Cleaning_Treatments` sheets.
- New tests: `tests/test_report_narrative.py` (summary/section/recommendation
  behaviour, ordering, tone vocabulary, causal-claim guard, empty-context
  survival), plus a context-foundation test in `tests/test_report_service.py`.
  Full suite green (263 collected; the five optional-`shap` tests skip when the
  package is absent).
- Test hygiene: three environment-sensitive tests were made robust — the trend
  heatmap assertion flattens the matrix (Plotly stores `z` row-wise on 5.x), the
  Styler assertion falls back when jinja2 is present but below pandas' 3.1.5 floor,
  and the SHAP tests `importorskip("shap")` instead of failing without the optional
  dependency (mirrors the app's guarded-import rule).


## 2026-10-19 — Added: AI segments in "Customer segments over time" (page 03) and three more trend views
- Added: a **Segment source** selector on page 03's Trends tab (both the RFM and the
  LRFM panel). It offers RFM, LRFM, and every saved AI run (`AI:kmeans`, …), so the
  k-means/ensemble segments are chartable over time instead of only the quantile
  segments. The panel's own model stays the default; the caption switches with the
  source.
- Added: `run_label_history` (`services/feature_service.py`) — fixed labelings
  trended on the RFM/LRFM period grid. Each period counts the base that existed by
  its end under the segment the customer holds today. Refitting a clusterer per
  period was rejected deliberately: cluster IDs are arbitrary and names are
  re-derived on every fit, so the lines would silently swap identity between
  periods. The AI caption says so ("labels are fixed, not re-scored per period").
- Added: `labeling_series` (`services/segmentation_service.py`) — saved run →
  `customer_id` → business segment name (numeric-ID fallback; density noise becomes
  "Noise"), which is what the trend legend shows.
- Changed: the **Trend view** radio now carries five views instead of two —
  Stacked area (default), **Share of base**, Segment lines, **Small multiples**,
  **Heatmap**. New factories in `visualization/rfm_charts.py`:
  `segment_trend_share` (each period re-based to 100%, so a segment that grows in
  absolute terms yet loses ground is still visible; the share is computed in Python
  because Plotly's `groupnorm` does not normalise area traces on 7.1, and the y-axis
  carries a `%` suffix because the shared theme drops axis titles),
  `segment_trend_facets` (one panel per segment with independent y-axes and no
  legend — the most explainable view, at a 420px grid), and `segment_trend_heatmap`
  (period × segment, rows ordered by total size). `_compact` on page 03 now keeps
  each factory's own height so the facet grid stays readable.
- New tests: `tests/test_charts.py` (the three views joined the empty-input list,
  29 → 32, plus a render test asserting the normalised shares, the facet titles, and
  the heatmap matrix), `tests/test_customer_features.py` (+2: the fixed-labeling
  grid never leaks the late cohort, bad args raise), `tests/test_segmentation.py`
  (+1: run names/noise fallback), and the dark-parity sweep in `tests/test_theme.py`
  now re-skins all three views. Full suite green (258 passed).

## 2026-10-19 — Added: scorecard metric cards (page 06) and model-control explanations (page 05)
- Added: `metric_guides` (components) — eight glossary cards covering the scorecard
  columns (`silhouette`, `davies_bouldin`, `stability_ari`, `value_eta2`,
  `min_share`, `evenness`, `noise_pct`, `interpretability_passes`): what each one
  measures, which way is better, its range, and how the recommendation gates and
  crowns it. Page 06 renders them under a "How to read the scorecard" header after
  the last chart on the page. The copy mirrors the gate constants and the
  interpretability checklist in `analytics/segmentation_comparison.py`; the cards
  are static text — no new compute and no change to the ranking.
- Added: `control_guides` (components) — per-control explanations for page 05's
  model controls (model, clusters, seed, HDBSCAN min cluster size, AE latent dims,
  AE epochs) as `help` tooltips, plus a "Which controls each model reads" matrix so
  the one control HDBSCAN reads and the two the autoencoder reads are visible
  rather than folklore. The run actions (sweep/run/train-all) carry tooltips too.
- New: `tests/test_guides.py` (+12) pins the glossary to the eight scorecard columns
  the page emits and the tooltips to the page's own control labels, so a rename
  cannot silently leave a metric or control undocumented. Full suite green
  (254 passed).

## 2026-10-19 — Fixed: Demographics page froze; the geography share chart is a drill-down treemap
- Fixed: the Demographics page locked the browser tab. The hierarchy chart grouped
  by the full `region → country → state → city` path, and this dataset holds 15,516
  cities for 24,911 customers, so the figure carried **23,976 nodes**; Plotly had
  to lay every one of them out client-side, which blocked the tab's main thread.
  Each level now keeps its largest `TREEMAP_MAX_VALUES_PER_LEVEL` (12) values and
  folds the tail into an "Other" tile — 234 nodes, 1.2s to build. Folding (rather
  than dropping rows) keeps every customer counted exactly once, so parent tiles
  stay true totals. Covered by a regression test.
- Changed: the geography share chart is `geo_treemap` (demographic_charts) — one
  drill-down treemap over every level present, converted from the sunburst that
  briefly held that spot (same nesting, same cap).
- Removed: the per-level "Customer share by …" treemaps in the
  City/State/Country/Region tabs; each tab keeps its ranked customer/revenue bars
  and its full table. The dark-parity theme sweep re-skins the hierarchy treemap.

## 2026-10-19 — Default cache v3: every segmentation and churn model preloaded
- Changed: the default run (`services/bootstrap_service.py`) now trains **all**
  registry models instead of a fast subset — segmentation `kmeans, gmm,
  hierarchical, hdbscan, autoencoder, ensemble` (was `kmeans, gmm`) and churn
  `logreg, histgb, random_forest, lightgbm, xgboost` (was `logreg,
  random_forest`). The comparison picker and report exports no longer see a
  two-model list. Rebuilt cache: 24,911 customers, selected segmentation
  `kmeans` (silhouette 0.2251; HDBSCAN scored −0.005 at 24.5% noise on this data
  and did not take the pick), selected churn `xgboost`.
- Removed: `SKIPPED_SEGMENTATION_REASONS` and the separate optional-dependency
  pass — the O(n^2) models are no longer skipped by policy, and a genuinely
  missing optional dependency is recorded once, with its install guidance.
- Changed: `CACHE_VERSION` 2 → 3 so an existing cache rebuilds once instead of
  loading without the new model artifacts. `hdbscan` added to the phase-gated
  optional block in `requirements.txt`.
- Fixed: `histgb` crashed on any dataset with categorical columns —
  `HistGradientBoostingClassifier` rejects sparse input, and the shared
  preprocessor always one-hot encoded to sparse. It now emits a dense block for
  the estimators in `DENSE_INPUT_ESTIMATORS` (`churn/baseline_models.py`).
  Covered by a regression test that trains on a categorical column.
- Changed: on page 06 the pair-agreement KPI sits under its own "Pair agreement"
  section header, so it lines up with the flow chart's header instead of
  floating above it.

## 2026-10-19 — Added: strategy playbook cards (live rule footprints)
- Added: `components/strategy_cards.py` — one card per rule in
  `retention_strategies.yaml`, joined to the retention table so each card shows
  the customers it currently owns and the revenue sitting at risk. Reason
  templates are filled with the rule's own aggregated value/probability rather
  than a single sample customer. `build_strategy_cards` is pure and
  dependency-light (no Streamlit), so it is unit-testable.
- Changed: the playbook renders on page 08 (below "At a glance") and on the
  Executive Dashboard's Retention tab (below the charts), so the strategy
  vocabulary is visible where the numbers are read.

## 2026-10-19 — Changed: Business Overview value-vs-risk bubbles → 2×2 priority heatmap
- Changed: the Business Overview "Value vs churn risk" bubble scatter is now a
  strategic 2×2 priority heatmap (`value_risk_heatmap`, retention_charts).
  Instead of one bubble per customer, customers are aggregated into the four
  quadrants bounded by the churn-probability threshold (0.60, the `high`
  risk-band floor) and the value threshold (median historical value — ₹6,962
  here). Cell colour encodes customer count; each cell is labelled with its
  customer count and revenue at risk, so leadership sees where the money is
  exposed without the scatter clutter.
- Changed: `strategy_distribution_bar` and `revenue_at_risk_bar` drop their
  x-axis category labels — the per-action legend already names every bar, so the
  labels only repeated it.
- `value_risk_matrix` (per-customer bubble) is unchanged and still backs page 08
  and the HTML/Excel/PDF report exports. Full suite green (237 passed).

## 2026-10-19 — Fix: "Customers per segment" donut rendered as an empty ring
- Fixed: the Business Overview donut showed only its legend — the ten long RFM
  segment names wrapped the legend across the whole compact card (~195px of a
  300px-tall, ~280px-wide tile), squeezing the ring into thin slivers. The
  donut now labels each slice with name + percent and hides the legend, so the
  ring keeps its full size at any window width.
- Changed: the Phase-1 "degrade to a horizontal bar past 5 segments" guard is
  removed; `segment_share_donut` is always a donut. Full suite green.

## 2026-10-17 — Visualization polish Phase 2: additive richer views (Plotly, behind toggles)
- Added: `segment_trend_lines` (rfm_charts) — one line per segment as an
  alternative to the stacked ribbon; each cohort's own baseline reads directly.
  Page 03 Trends gains a "Trend view" toggle (default stays stacked area).
- Added: `rfm_scatter_3d` (rfm_charts) — recency × frequency × monetary 3D
  scatter, segment-colored, deterministic 2,000-point sample. Page 03
  Relationships gains a "3D scatter" option (default stays scatter matrix).
- Added: `geo_treemap` (demographic_charts) — one drill-down treemap nesting
  region → country → state → city (missing places group as Unknown). Page 04
  Geography renders it when ≥2 levels are present.
- All three are additive: same figure keys untouched, defaults unchanged, no new
  dependencies. Full suite green (236 passed; +3 chart tests).

## 2026-10-17 — Visualization polish: sorted bars, priority quadrant, Pareto line, donut guard
- Changed: `revenue_at_risk_bar` and `strategy_distribution_bar` order actions
  largest-first so the biggest exposure leads the chart.
- Added: `value_risk_matrix` draws dashed churn-probability and value reference
  lines framing the top-right "act first" quadrant (churn threshold defaults to
  the `high` risk-band floor 0.60; value threshold defaults to the median).
- Added: `priority_histogram` now overlays a cumulative (Pareto) share line on a
  secondary 0–100 axis so the concentrated top tail of opportunities reads at a
  glance.
- Changed: `segment_share_donut` now labels every slice (name + percent) and
  suppresses the wrapping legend, keeping the ring full-size with ten segments.
- All four are theme-adapted (quadrant lines / Pareto use palette tokens that
  re-map in dark mode) and flow through the existing HTML/Excel/PDF report
  exports unchanged. Full suite green (233 passed; +4 chart tests).

## 2026-10-17 — Phase 1+2 performance: vectorized features, history in a minute
- Changed: `analytics/customer_features.py` per-customer Python loop replaced
  with vectorized groupby aggregations (sorted-diff gaps, masked window sums,
  `tail(1)` latest-valid demographics). 568s → 1.4s on 24,911 customers
  (~400×); quarterly history 65min → 62s, so Phase 2 closes as
  satisfied-by-Phase-1 with no incremental engine.
- Fixed (caught by end-to-end): vectorization first emitted pandas-NA in
  `str`-dtype demographics, which sklearn 1.9.1 imputers reject; gapped
  demographic columns are now `object` with None (legacy contract), complete
  columns keep native dtype so `age` stays a numeric predictor.
- Added: blanks/ties/conflicts regression test. Real-data proof: identical
  shape/columns/order, zero value mismatches, all numeric diffs 0.0 (one AOV
  by 1 cent from float summation order). Full suite green (229 passed).

## 2026-10-17 — Phase 4: upload-path progress UX + size-aware defaults
- Added: upload page (page 01) gains `st.status` for demo/default/upload flows
  with size-aware hints for large files.
- Added: cleaning (page 02), feature/RFM scoring (page 03), and segment trends
  (page 03) use `st.status` for better feedback on long steps.
- Added: size-aware info banner on upload for rows > 500k or memory > 200MB.

## 2026-10-17 — Report generator: include segment trends + fuller segment coverage
- Added: report now shows segment trends over time (when segment_history exists)
  in both UI (page 09) and exports (HTML) with explanations.
- Fixed: segment portfolio/summary context ensures segment information is
  surfaced in reports; UI reads history from state for on-screen previews.

## 2026-10-17 — Default cache v2: preloaded segment history + rebuild hardening
- Added: quarterly segment history (config quantiles, 8 periods) to
  `build_default_run`, cached as `segment_history.parquet` and loaded into
  session state; `CACHE_VERSION = 2` forces one rebuild; new
  `scripts/finish_default_run.py` completes a partial cache (churn →
  retention → manifest) in a fresh process.
- Fixed: SHAP TreeExplainer sparse-background crash (`_shap_explainer`
  densifies the tree path like the linear path) and exact-SHAP stall guard
  (`MAX_TREE_SHAP_WORK`, `ShapBudgetExceeded` → median-driver fallback in
  scoring and a page-07 info banner instead of a hang).
- Fixed: `load_default_into_state` stored a bare ColumnTransformer as
  `churn_model` (Pipeline sequence-unpack); now stores the loaded object.
- Added: per-stage logging in `build_default_run` + file logging in both
  scripts (`logs/build_default_run-<timestamp>.log`).
- Full suite green (228 passed), including new sparse-SHAP, SHAP-budget, and
  history-preload tests.

## 2026-10-16 — Full-bleed dark console: native widgets, chart parity, dark exports
- Added: `_DARK_NATIVE_CSS` in `visualization/theme.py` (dark-only skin for
  tabs, expanders, alerts, text/number/date/select inputs, dropdown menus,
  file dropzone, buttons incl. lime primary actions, radio/checkbox
  accent-color, sidebar page links, toasts, spinners, captions, dividers).
  Hooks are the `data-testid`/`data-baseweb` names verified in the pinned
  Streamlit 1.65.0 bundle — never obfuscated classes; light sheet untouched.
- Added: `test_dark_native_pack_skins_server_widgets` (every hook present in
  dark, absent in light) and `test_dark_parity_sweep_leaves_no_light_values`
  (bar, treemap, Sankey, heatmap, parcoords, value-risk matrix re-skin clean).
- Changed: `services/report_service.py` HTML export gains a `mode` skin
  (print-light default, charcoal-and-lime dark); page 09 adds an HTML-theme
  picker. PDF stays print-light by decision.
- Changed: sidebar caption reflects the styled console; `components/theme.py`
  documents why `prefers-color-scheme` auto-detect is infeasible (no client JS
  from Python; `components.html` iframe is parent-sandboxed).
- Known limitation: the `st.dataframe` Glide canvas follows the light server
  theme (no widget hook exists) — shell is dark, grid stays light.

## 2026-10-16 — Dark theme re-skin: charcoal-and-lime console (direction B)
- Changed: `visualization/tokens.py` DARK palette from navy/cyan to charcoal
  surfaces (`#0C0F14`/`#151A22`) with a lime accent (`#BEF264`, reserved for
  accents/CTAs, never success/warning states); info insight-box tone re-tinted
  to olive/lime. Every value verified disjoint from LIGHT and segment tables;
  text 16.0:1, muted 7.2:1, muted_soft 5.3:1, primary 13.4:1, critical 5.2:1.
- Changed: `visualization/theme.py` glass recipe and canvas glow re-tinted to
  charcoal/lime; `components/sidebar.py` caption and `.streamlit/config.toml`
  comment follow the new identity.
- Kept: glass-console structure, session-persisted toggle, and the
  retired-volt scan (new lime values avoid all banned sentinels).

## 2026-10-16 — Report Generator (page 09)
- Added: `services/report_service.py` (report context, per-chart plain-English
  explanations, HTML/Excel/PDF exporters; IDs masked by default; historical
  value labeled a proxy, never CLV; no causal-impact claims).
- Added: `pages/09_Report_Generator.py` (readiness checklist, section toggles,
  live preview with explained charts, HTML + multi-sheet Excel + PDF downloads).
- Changed: `components/nav_tabs.py` (Reports entry); `requirements.txt`
  (reportlab for PDF, degrades gracefully when absent).
- Added: `tests/test_report_service.py` (context, narratives, HTML/Excel masking).

## 2026-10-12 — Nav tabs, navy/cyan recolor, train-all
- Phase 1: top tab-bar navigation (`components/nav_tabs.py`, `st.switch_page`,
  active highlight); sidebar hidden via CSS; all page filters moved to collapsed
  "Controls" expanders; `tests/test_nav.py` guards nav order, tab rendering, and
  zero `st.sidebar` usage.
- Phase 2: deep-navy/cyan palette with semantic tokens (`CANVAS`, `PRIMARY`, …);
  volt fully retired (scan-enforced in `tests/test_theme.py`); computed contrast
  guards kept (primary ≥ 7:1, muted ≥ 4.5:1); `.streamlit/config.toml` re-tinted.
- Phase 3: `train_all_models()` batch training on page 05 (per-model isolation,
  missing-optionals record as skipped, page-06-compatible records); blob-tested.
- Added glass tokens + frosted-card CSS (gradient canvas, backdrop blur, volt
  hairline borders) in `visualization/theme.py`; muted text raised to `#7A7A7A`
  for ≥ 4.5:1 contrast (computed assertion in `tests/test_theme.py`).
- Layout: strict KPI-top / graphs-middle / collapsed-tables-bottom zones on all
  tabs; 3-column grids for peer-visual trios (p03 segment trio, p09 trends trio).
- Compliance pass per web-interface-guidelines (ellipsis in spinners,
  tabular-nums, color-scheme, touch-action); skills-driven, no emojis.
- Follow-up: canonical glass recipe (white 0.12 edge-light borders, volt glow
  shadows, 12px blur) with KPI metrics doubling as glass cards.
- Note: ui-ux-pro-max search backend not installed locally; fallback tokens used.
- Added: `visualization/theme.py` (obsidian/volt tokens, terminal CSS, `@styled`
  Plotly decorator); shared sidebar-filter groups in `components/filters.py`.
- Changed: `.streamlit/config.toml` to the dark terminal theme; all chart
  factories render dark-mode volt-styled figures (risk hues stay semantic).
- Changed: all 9 pages — filters moved to grouped sidebar sections, Tier-1 KPI
  rows, side-by-side Tier-2 charts, Tier-3 tables collapsed in expanders.
- Added: `tests/test_theme.py`. Full suite green (120 passed).
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

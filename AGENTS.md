# AGENTS.md — AI Customer Intelligence Starter

## Source of truth
- `OPEN_CODE_BUILD_INSTRUCTIONS.md` is the spec: 8 phased build (Phase 0–8), acceptance gates, data contract, session-state keys, `config.yaml` shape. Follow it phase-sequentially; do not skip ahead or start next phase until current gate passes.
- Executable state beats prose: `app.py` (3-line Streamlit shell), `config.yaml` (1-line stub), `requirements.txt` (9 pkgs), and all `services/`, `analytics/`, `churn/`, `segmentation/`, `tests/` modules are currently `"starter file"` placeholders. Verify before assuming anything is implemented.
- `opencode.json` allowlists shell without approval only for `pytest *` and `streamlit run app.py`; other shell commands need approval.

## Architecture
- Streamlit multipage app: `app.py` landing page + `pages/NN_*.py` (numeric prefix controls nav order — keep it).
- Pages are presentation/orchestration only. Business logic belongs in `services/` (pipeline), `analytics/` (RFM/LRFM, features, stats), `feature_engineering/`, `segmentation/`, `churn/`, `retention_engine/`, `visualization/`, `components/`. Do not pile logic into page files.
- No `state_service.py`, `pyproject.toml`, `requirements-dev.txt`, `.streamlit/config.toml`, CI, or run scripts exist yet — Phase 0 tasks. Don't reference them as if present.
- Session-state contract (create `services/state_service.py` to own init/validate/reset): `raw_data, clean_data, column_mapping, cleaning_config, cleaning_audit, data_dictionary, customer_features, rfm_results, lrfm_results, demographic_profile, segmentation_runs, selected_segmentation_model, segmented_customers, churn_dataset, churn_model, churn_model_metadata, churn_scores, retention_results, global_filters`.

## Data contract
- Canonical fields: required `customer_id (str, never blank), transaction_date (parseable datetime, no silent future dates), amount_spent (numeric)`; optional `gender, age (valid default 0–120, configurable), city, state, country, region` — whitespace-normalize categoricals, never aggressively merge.
- CSV encoding fallback order: UTF-8 BOM → UTF-8 → CP1252 → Latin-1. Support CSV/XLSX/XLS/Parquet.
- Never silently drop/alter customer rows. Every cleaning action goes in the audit record (input/output counts, rules, per-rule affected counts, timestamp, filename, mapping).

## Domain gotchas (from spec — easy to get wrong)
- Churn must be leakage-safe: predictors ≤ `cutoff`, label = activity in `(cutoff, cutoff+horizon]`; validate outcome window exists in data; warn if post-cutoff features are used. No recency-both-sides leakage.
- Snapshot date default = max transaction date + 1 day. `RFM = 100*R+10*F+M`, `LRFM = 1000*L+100*R+10*F+M`; reverse R, rank-tie handling, safe small/low-cardinality path. Name segments from components + raw values, never the combined score alone; cluster IDs are arbitrary — profile vs population medians first.
- Quality: `Overall = 0.40*Completeness + 0.40*Validity + 0.20*Uniqueness`. `Revenue at Risk = churn_probability * selected value`; `Priority = normalized value * churn_probability * strategic weight`. Label historical monetary as historical value, not CLV. Retention rules must be externally configurable (YAML/JSON) with explicit per-action reason; no causal-impact claims without experiments.
- Heavy/optional deps (`hdbscan, umap-learn, optuna, lightgbm, xgboost, shap, tensorflow/torch`) must be optional imports with install guidance + graceful fallback, never hard crash. Ensemble clustering: use alignment/consensus, never mode of arbitrary labels. No fabricated geo-coordinates; maps only when reliably geocodable. Never expose raw `customer_id` in executive views except explicit authorized drill-down.

## Commands (Windows host, PowerShell)
- Setup: `py -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install -r requirements.txt` (+ `requirements-dev.txt` once it exists).
- Run: `streamlit run app.py`. Verify: `python -m compileall .`, `pytest -q` (final gate: `pytest -q --cov=. --cov-report=term-missing`), `ruff check .`, `black --check .`.
- Tooling note: `pytest/ruff/black` are not installed in this env and have no config — confirm/install before trusting gates; `requirements.txt` lacks the spec's full stack (`scipy, openpyxl, pyarrow, joblib, pyyaml, pydantic, pytest, ruff, black`).

## Conventions
- Real implementations only: no `pass`/placeholder/TODO-only modules. Fixed random seeds (`42` per spec config). Type hints + docstrings on public functions.
- After every phase: run compile+lint+format+tests, fix before continuing, update `README.md` / `CHANGELOG.md` / `PROJECT_STATUS.md`, emit the Phase Completion Report (template in spec §11).

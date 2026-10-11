# AGENTS.md — AI Customer Intelligence Platform

## Status
- Build complete (Phases 0–8, see git log). Maintenance mode: consult `PROJECT_STATUS.md` for current state and `OPEN_CODE_BUILD_INSTRUCTIONS.md` (spec) for domain rules before changing behavior.
- Streamlit multipage app: `app.py` landing + `pages/NN_*.py` (numeric prefix controls nav order — keep it). Pages are presentation/orchestration only; logic lives in `services/` (pipeline), `analytics/`, `feature_engineering/`, `segmentation/`, `churn/`, `retention_engine/`, `visualization/`, `components/`, `utils/`.
- Session-state contract owned by `services/state_service.py`: `raw_data, clean_data, column_mapping, cleaning_config, cleaning_audit, data_dictionary, customer_features, rfm_results, lrfm_results, demographic_profile, segmentation_runs, selected_segmentation_model, segmented_customers, churn_dataset, churn_model, churn_model_metadata, churn_scores, retention_results, global_filters`.

## Data contract (spec §5)
- Required `customer_id` (never blank), `transaction_date` (parseable, no silent future dates), `amount_spent` (numeric); optional `gender, age (0–120 configurable), city, state, country, region` — whitespace-normalize categoricals, never aggressively merge.
- CSV encoding fallback: UTF-8 BOM → UTF-8 → CP1252 → Latin-1. Never silently drop rows — every cleaning action goes in the audit record.

## Domain gotchas (easy to get wrong)
- Churn is leakage-safe by construction: predictors ≤ `cutoff`, label = activity in `(cutoff, cutoff+horizon]`; splits are **cohort (first-transaction) ordered** — last-transaction ordering segregates classes by construction. See `churn/churn_labeling.py`, `services/churn_service.py`.
- Snapshot default = max transaction + 1 day. `RFM = 100R+10F+M`, `LRFM = 1000L+100R+10F+M` (R reversed, rank-percentile ties). Name segments from components + raw values, never the combined score; cluster IDs are arbitrary — profile vs population medians first.
- Quality `Overall = 0.40C + 0.40V + 0.20U`. `Revenue at Risk = p × value`; `Priority = norm_value × p × weight`. Historical monetary is a value proxy, never CLV. Retention rules live in `retention_strategies.yaml` (validated full tier×band coverage); reasons must never claim causal impact.
- Optional deps (`hdbscan, umap-learn, lightgbm, shap`, …) are guarded imports with install guidance + graceful fallback — never hard-require. Ensemble uses co-association consensus, never label-mode. No fabricated geo-coordinates. Mask `customer_id` outside explicitly gated drill-downs.

## Commands (Windows host, PowerShell)
- Setup: `py -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install -r requirements.txt -r requirements-dev.txt`.
- Run: `python -m streamlit run app.py` (port 8501; health `/_stcore/health`). Docker: see `DEPLOYMENT.md`.
- Gates: `python -m compileall .` → `ruff check .` → `black --check .` → `pytest -q` (full suite incl. end-to-end takes ~5 min; targeted: `python -m pytest tests/test_<name>.py -q`; coverage gate: `pytest -q --cov=. --cov-report=term-missing`).
- PowerShell quirks (verified): no `&&` (use `;`), no heredocs (`<` is reserved — write temp scripts with the file tools instead), invoke linters as `python -m ruff/black` (user Scripts dir is not on PATH).
- Dev servers orphan easily: `Stop-Process` can leave listeners squatting on 8501/8502. Check with `Get-NetTCPConnection -LocalPort 8501 -State Listen` before assuming a fresh boot.

## Testing pitfalls (all bitten before)
- Never name a non-test function `test_*` anywhere — pytest collects matching names from source modules too. Enforced by convention, not tooling.
- Pages cannot run outside a Streamlit session (`st.session_state` raises), so page↔service wiring is guarded by `tests/test_module_wiring.py` (static import resolution). Run it after touching any page import.
- Elements rendered in loops (tabs, per-model/per-level charts) need explicit unique `key=` or Streamlit raises duplicate-ID errors.
- `tests/test_end_to_end.py` runs the full 14-step workflow on `data/sample/` (regenerate with `python utils/sample_data.py`); slow by design — don't run it for unrelated changes.
- sklearn here is 1.9: `CalibratedClassifierCV(cv="prefit")` no longer exists — calibration goes through `PlattCalibrator` in `churn/calibration.py`. Keep it that way on upgrades.
- Page→service schema rule: pages adapt their columns to service schemas at the call boundary (e.g. scorecard `model` → `recommend_model`'s `name`), never the reverse.

## Scope notes
- `ponytail/`, `servers/`, `awesome-ai-agents/`, `.agents/` are unrelated nested checkouts: excluded from lint/test/build via `pyproject.toml` + `.gitignore`. Do not build, test, lint, or delete them (user decision).
- Out-of-spec starter stubs were deleted in Phase 8; every remaining module is real. Conventions: type hints + docstrings on public functions, fixed seed 42, business-readable error messages (shown in UI as-is).

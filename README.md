# AI Customer Intelligence Platform

Production-oriented customer intelligence built with Python + Streamlit:
transaction upload & validation → data quality & cleaning → customer features →
RFM/LRFM → demographic profiling → advanced segmentation → comparison →
leakage-safe churn prediction → retention strategies → executive dashboard.

Full phase specification: `OPEN_CODE_BUILD_INSTRUCTIONS.md`.
Live build tracker: `PROJECT_STATUS.md`.

## Implemented phases
- [x] Phase 0 — Repository initialization (config, logging, sample data, tooling)
- [x] Phase 1 — Data foundation (upload, mapping, quality, cleaning, export)
- [x] Phase 2 — Customer features and RFM/LRFM
- [x] Phase 3 — Demographic and geographic profiling
- [x] Phase 4 — Advanced segmentation
- [x] Phase 5 — Segmentation comparison
- [ ] Phase 6 — Leakage-safe churn prediction
- [ ] Phase 7 — Retention intelligence
- [ ] Phase 8 — Executive dashboard and production readiness

## Input schema
Required: `customer_id` (text), `transaction_date` (date), `amount_spent` (number).
Optional: `gender`, `age`, `city`, `state`, `country`, `region`.
Column names need not match exactly — the upload page auto-detects common variants.

## Installation
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt   # lint/format/test tooling
```
Linux/macOS: `bash run_linux_mac.sh` (Windows: `run_windows.ps1`).

## Run
```powershell
streamlit run app.py
```

## Test & quality gates (run at the end of every phase)
```powershell
python -m compileall .
ruff check .
black --check .
pytest -q
```

## Application workflow
Upload → correct mapping → clean (audit recorded) → customer features → RFM/LRFM →
demographics → AI segmentation → comparison → time-aware churn dataset → train &
evaluate → score & explain → retention actions → filterable executive dashboard →
export. No step requires editing source code.

## Known limitations
- Heavy models (`hdbscan`, autoencoder, `lightgbm`/`xgboost`, SHAP) degrade gracefully
  when their optional packages are absent; pages show install guidance instead.
- Sample data under `data/sample/` is synthetic demo data, never real customers.

## Privacy notes
- Raw `customer_id` values never appear in executive views except an explicit
  authorized drill-down. Cleaning never silently drops records — every action is
  audited. No credentials are committed (see `.env.example`).

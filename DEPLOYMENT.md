# Deployment Guide — AI Customer Intelligence Platform

## Install (clean environment)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Linux/macOS: `bash run_linux_mac.sh`. Optional accelerators (any subset):
`pip install hdbscan umap-learn lightgbm shap` — pages degrade gracefully without them.

## Run
```powershell
streamlit run app.py
```
Then follow pages 01→09 in order. Validate the install with the quality gates:
```powershell
python -m compileall .
ruff check .
black --check .
pytest -q --cov=. --cov-report=term-missing
```

## Docker
```powershell
docker build -t customer-intelligence .
docker run -p 8501:8501 customer-intelligence
```
Health endpoint: `http://localhost:8501/_stcore/health` (also the Docker HEALTHCHECK).

## Restart behavior
Session state (uploads, features, trained models in memory) resets on restart —
this is a single-user local deployment model. What survives a restart on disk:
`models/*/*.joblib` (+ metadata sidecars), `reports/model_cards/*.md`,
`data/sample/`, exported CSVs you downloaded. Re-upload + re-run pages after
a restart; saved `.joblib` models can be reloaded without retraining.

## Privacy & access
- Raw `customer_id` values are masked (`C•••05` style) everywhere except the
  explicitly gated drill-down on page 09.
- No credentials are required; never commit `.env` (see `.env.example`).
- Role-based access is a documented design, not code in this release: put this
  app behind your organization's SSO/reverse proxy, restrict who can reach the
  drill-down and export buttons, and log access at the proxy. Export buttons are
  the authorization hooks — wrap them with your SSO check before multi-user use.

## Known limitations
- Single-user, in-memory session state (see restart behavior above).
- Heavy models degrade gracefully without optional packages (install guidance
  in the UI instead of crashes).
- Retention actions are decision support; no causal retention impact is claimed
  without experiment data (see `retention_strategies.yaml` disclaimer).
- Historical monetary value is a value proxy, never predicted CLV.

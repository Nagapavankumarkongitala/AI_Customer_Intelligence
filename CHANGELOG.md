# CHANGELOG

## 2026-10-11 — Phase 0: Repository Initialization
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

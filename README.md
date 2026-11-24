# Ghent Real-Estate AVM (`house_scraping_project`)

End-to-end automated valuation model (AVM), scraper, open-data enrichment pipeline, Flask API, and React dashboard for Ghent residential real estate (Normal Ghent Homes scope: `€150k–€800k`, `n=14,599` deduplicated corpus, `n=13,439` normal homes).

## Architecture Overview

### Production Model Bundle (`outputs/models/production/`)
Trained by `src/train_production.py` and served via `src/production_pipeline.py`:
- **Strictly Agency-Blind & Zero External-Venv Dependencies:** Drops all agency identity/target-encoding columns (`immo_office_name`, `agency_norm`, `te_agency_*`, `freq_agency_norm`) and removes Qwen3/MLX dependencies in favor of Word+Char TF-IDF + `intfloat/multilingual-e5-large`.
- **Temporal Deflator (Issue #9):** `0.5 × CPI + 0.5 × HPI` (`outputs/dataset/ecb_macro_monthly.csv`), anchored to 2025 median.
- **Anti-Chaining Spatial+Text Cluster CV (Issue #4):** Zero leaked physical-home clusters across 5 OOF folds and cluster-clean 2026 temporal holdout.
- **Dual-Track Lean-5 Ensemble (Issues #15–#16):**
  - **Members (5):** `lgbmxt_price_hub`, `spec_lgbm_price_hub` (House/Apt specialists), `cb_tuned_price_l2`, `tabm_perm170_price_4s`, `tabm_perm170_psqm_4s`, combined via leak-free OOF `HuberRegressor`.
  - **`tabular_text` track (227 GBDT / 130 TabM features):** Used for live URLs without photo scoring, form edits, and manual property entry (`<100 ms` inference; avoids the `+1.4 pp` masked-vision degradation on photo-less inputs).
  - **`multimodal` track (279 GBDT / 166 TabM features):** Adds render-filtered `SigLIP2-base` + `SO400M-384` contact-sheet visual features when photo embeddings are provided.
- **Heteroscedastic 80% & 90% Conformal Intervals (Issue #8):** Conditioned on ensemble disagreement (`ens_std`), property type, renovation state, area, age, and distance from center.
- **Aligned TreeSHAP:** Computed directly on the served `lgbmxt_price_hub` model and mapped to raw EUR contributions.

## Environments

| Virtualenv | Python | Purpose |
|---|---|---|
| `.venv-api` | 3.13 | Feature pipeline, production training, PyTorch MPS, Flask API, pytest |
| `.venv` | 3.9 | Immoweb scraping (`SCRAPER_DRIVER=selenium`; `undetected_chromedriver` is incompatible with Chrome 157+) |

### Apple Silicon Runtime Notes
- **OpenMP safety:** Always import `torch` before `lightgbm` or `catboost` in the same process (handled automatically by `ProductionBundle.load()` and subprocess isolation in `src/train_production.py`) to avoid macOS OpenMP exit-139 segfaults.
- Pass `PYTHONPATH=src` when running standalone scripts in `src/`.

## Common Commands

### 1. Train / Refresh Production Bundle
```bash
PYTHONPATH=src .venv-api/bin/python -u src/train_production.py
```
Outputs `outputs/models/production/{feature_pipeline.joblib,track_tabular_text.joblib,track_multimodal.joblib,metadata.json}`.

### 2. Run Backend API (Flask, port `5001`)
```bash
.venv-api/bin/python -m api.app
```
Key endpoints:
- `GET  /api/health`
- `GET  /api/predict/url?url=<immoweb-url>`
- `POST /api/predict/form`
- `POST /api/predict/manual`
- `GET  /api/neighbourhoods` and `GET /api/neighbourhoods/<name>`
- `GET  /api/listings/live` and `POST /api/listings/refresh`
- `GET  /api/eda`

### 3. Run Frontend (React + Vite + TypeScript)
```bash
cd frontend
npm install
npm run dev      # dev server on http://localhost:5173
npm run lint     # ESLint (0 errors)
npm run build    # TypeScript + Vite production bundle
```

### 4. Run Test Suite
```bash
.venv-api/bin/python -m pytest api/tests -v
```

### 5. Scraping
See `docs/runbook-scraping.md` for full scraping and snapshot-store operations.

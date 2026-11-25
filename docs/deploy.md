# Deployment & Always-On Operations Guide

Run **Ghent RE Intelligence** as a single always-on production process serving both the built React frontend and the Phase-7 Lean-5 valuation API.

---

## 1. Prerequisites & Artifacts

The following generated artifacts are gitignored and must be present (either trained locally or copied via `rsync` to a target host):

| Path | Size | Built By |
|---|---|---|
| `outputs/models/production/feature_pipeline.joblib` | ~12 MB | `PYTHONPATH=src .venv-api/bin/python -u src/train_production.py` |
| `outputs/models/production/track_tabular_text.joblib` | ~35 MB | `src/train_production.py` |
| `outputs/models/production/track_multimodal.joblib` | ~36 MB | `src/train_production.py` |
| `outputs/models/production/metadata.json` | ~3 KB | `src/train_production.py` |
| `outputs/models/production/oof_analytics.parquet` | ~1.5 MB | `PYTHONPATH=src .venv-api/bin/python -u src/build_market_insights.py` |
| `outputs/models/production/market_insights.json` | ~66 KB | `src/build_market_insights.py` |
| `outputs/eda/eda_report.json` *(optional, Full EDA tab)* | ~5 MB | `.venv-api/bin/python src/eda_report.py --input outputs/property_data_full.parquet` |
| `external_data/stadswijken-gent.csv` & `external_data/flanders/` | ~15 MB | Checked in / synced from open data |

Total required production bundle + analytics footprint: **~85 MB**.

```bash
# Example rsync to another Mac / server
rsync -avz --progress \
  outputs/models/production/ \
  user@host:~/house_scraping_project/outputs/models/production/
```

---

## 2. Build Frontend & Run Single-Process WSGI Server

Build the static React bundle (`frontend/dist/`):

```bash
cd frontend
npm ci
npm run build
cd ..
```

Start the production Waitress WSGI server (imports `torch` first for Apple Silicon OpenMP safety, uses 1 process × 4 threads so MPS GPU models are loaded once and never forked):

```bash
.venv-api/bin/python -m api.wsgi
```

Verify both the API and SPA are live on port `5001`:

```bash
curl -sf http://127.0.0.1:5001/api/health
curl -sfI http://127.0.0.1:5001/predict   # 200 text/html (SPA fallback)
```

Environment overrides:
- `GHENT_HOST` (default `127.0.0.1`; set `0.0.0.0` to expose on LAN)
- `GHENT_PORT` (default `5001`)
- `GHENT_THREADS` (default `4`)

---

## 3. Always-On macOS Setup (`launchd`)

### A. Always-On Web App + API
```bash
cp deploy/com.ghent-avm.server.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.ghent-avm.server.plist

# Inspect status & logs
launchctl list | grep ghent-avm
tail -f logs/launchd_server.out.log
```

### B. Scheduled Daily Deal Hunter Refresh (08:15 daily)
```bash
cp deploy/com.ghent-avm.daily-refresh.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.ghent-avm.daily-refresh.plist
```

Or run manually on demand:
```bash
./scripts/daily_refresh.sh --pages 3     # scrape 3 Ghent pages + parse + Phase-7 score
./scripts/daily_refresh.sh --score-only  # re-score latest HTML without opening Chrome
```

---

## 4. Remote Phone / External Access (Cloudflare Tunnel)

Expose the local server securely with HTTPS without opening router ports:

```bash
brew install cloudflared
cloudflared tunnel --url http://127.0.0.1:5001
```

For a permanent custom domain, configure a named Cloudflare Tunnel pointing to `http://127.0.0.1:5001`.

---

## 5. Refresh Cadence

| Task | Frequency | Command |
|---|---|---|
| Live Ghent deals | Daily | `./scripts/daily_refresh.sh --pages 3` |
| ECB CPI/HPI macro series | Monthly | Update `outputs/dataset/ecb_macro_monthly.csv` |
| Full Phase-7 model refit + insights | Quarterly | `PYTHONPATH=src .venv-api/bin/python -u src/train_production.py && PYTHONPATH=src .venv-api/bin/python -u src/build_market_insights.py` |

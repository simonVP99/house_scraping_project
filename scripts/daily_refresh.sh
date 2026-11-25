#!/usr/bin/env bash
# Idempotent daily Ghent Deal Hunter refresh pipeline (Issue #23).
#
# Runs:
#   1. Optional scrape of N newest Immoweb Ghent search pages (.venv, SCRAPER_DRIVER=selenium)
#   2. Incremental HTML parse (.venv)
#   3. Live-listings Phase-7 batch scoring & cache refresh (.venv-api)
#   4. Optional live API POST /api/listings/refresh ping if the API is already running
#
# Usage:
#   ./scripts/daily_refresh.sh                # default: scrape 3 pages + score
#   ./scripts/daily_refresh.sh --pages 5      # scrape 5 pages + score
#   ./scripts/daily_refresh.sh --score-only   # skip browser scrape; rescore latest HTML

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PAGES=3
SCORE_ONLY=0
API_URL="${GHENT_API_URL:-http://127.0.0.1:5001}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pages)
      PAGES="$2"
      shift 2
      ;;
    --score-only)
      SCORE_ONLY=1
      shift
      ;;
    *)
      echo "Unknown flag: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_DIR/logs/daily_refresh_${STAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Ghent Daily Refresh started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

if [[ "$SCORE_ONLY" -eq 0 ]]; then
  echo "[1/3] Scraping $PAGES Ghent pages via .venv (SCRAPER_DRIVER=selenium)..."
  SCRAPER_DRIVER=selenium "$PROJECT_DIR/.venv/bin/python" -u src/main.py "$PAGES"
else
  echo "[1/3] --score-only set: skipping Immoweb browser scrape."
fi

echo "[2/3] Refreshing Phase-7 live-listings cache (outputs/today_predictions.csv)..."
PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src" "$PROJECT_DIR/.venv-api/bin/python" -u -c "
from api.config import PRODUCTION_BUNDLE_DIR
from api.data_loader import DataLoader
from api.prediction import PredictionService
from api.live_listings import LiveListingsService
from production_pipeline import ProductionBundle

loader = DataLoader()
bundle = ProductionBundle.load(PRODUCTION_BUNDLE_DIR)
pred_svc = PredictionService(bundle=bundle, data_loader=loader)
live_svc = LiveListingsService(pred_svc, loader)
res = live_svc.refresh()
print('Refreshed live listings cache:', res)
"

echo "[3/3] Notifying running API (if up at $API_URL)..."
if curl -sf "${API_URL}/api/health" >/dev/null 2>&1; then
  curl -sf -X POST "${API_URL}/api/listings/refresh" && echo ""
else
  echo "  API not currently listening at $API_URL (cache on disk is ready for next startup)."
fi

echo "=== Ghent Daily Refresh finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

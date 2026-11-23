# Full Model Retrain Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-parse all scraped HTML data, preprocess, embed descriptions, fetch updated macro-economic indicators from ECB, retrain AutoGluon + SHAP XGBoost models, and deploy the new models to the API.

**Architecture:** Sequential 6-stage pipeline: Parse -> Preprocess -> Embed -> Train AutoGluon (with macro merge) -> Train SHAP XGBoost -> Deploy (restart API). Each stage reads the previous stage's CSV output. Macro data (Belgian CPI, 10Y yield, Euribor 12m) is fetched live from ECB at training time and merged on each property's `date_obtained` month.

**Tech Stack:** Python 3, BeautifulSoup, pandas, statsmodels (Poisson GLMs), OSMnx, sentence-transformers (E5-small), sklearn (PCA), AutoGluon, XGBoost, SHAP, ECB SDW API

---

### Task 1: Parse all HTML data

**Files:**
- Run: `run_parsing_only.py`
- Output: `outputs/property_data_parsed.csv`

**Step 1: Run the parser**

```bash
cd /Users/simon/Desktop/personal/immo_datacollection/house_scraping_project
source .venv/bin/activate
python run_parsing_only.py
```

Expected: Parses ~83K HTML files from `data/` in parallel, produces `outputs/property_data_parsed.csv`. Should take 2-5 minutes.

**Step 2: Verify output**

```bash
wc -l outputs/property_data_parsed.csv
head -1 outputs/property_data_parsed.csv | tr ',' '\n' | wc -l
```

Expected: ~80K+ rows (one per HTML file that parsed successfully), ~60 columns.

**Step 3: Commit**

```bash
git add outputs/property_data_parsed.csv
git commit -m "Re-parse all HTML data ($(date +%Y-%m-%d))"
```

---

### Task 2: Preprocess

**Files:**
- Run: `run_preprocessing.py`
- Reads: `outputs/property_data_parsed.csv`
- Output: `outputs/property_data_preprocessed.csv`, `outputs/models/imputation_models.joblib`

**Step 1: Run preprocessing**

```bash
python run_preprocessing.py
```

Expected: Deduplicates to one-per-property, filters non-modelable/hard-bounds, engineers features (age, price_sq, booleans, categoricals, neighbourhood, street_type, spatial lag), trains Poisson imputation models, saves imputation artifacts. Takes 5-15 minutes (OSMnx street matching is the bottleneck).

**Step 2: Verify output**

```bash
wc -l outputs/property_data_preprocessed.csv
head -1 outputs/property_data_preprocessed.csv | tr ',' '\n' | wc -l
# Check imputation models were saved
ls -la outputs/models/imputation_models.joblib
```

Expected: ~20K+ rows (after dedup + filtering), ~100+ columns. Fresh `imputation_models.joblib`.

**Step 3: Commit**

```bash
git add outputs/property_data_preprocessed.csv outputs/models/imputation_models.joblib
git commit -m "Rerun preprocessing with all data through April 2026"
```

---

### Task 3: Embed descriptions

**Files:**
- Run: `src/embed_descriptions.py`
- Reads: `outputs/property_data_preprocessed.csv`, `outputs/property_data_parsed.csv`
- Output: `outputs/property_data_with_embeddings.csv`, `outputs/models/description_pca.joblib`

**Step 1: Run embedding pipeline**

```bash
python src/embed_descriptions.py
```

Expected: Loads preprocessed data, joins descriptions from parsed CSV, embeds with E5-small (MPS-accelerated), PCA-reduces to 50 dims, saves. Takes 5-10 minutes depending on GPU.

**Step 2: Verify output**

```bash
wc -l outputs/property_data_with_embeddings.csv
head -1 outputs/property_data_with_embeddings.csv | tr ',' '\n' | grep desc_emb | wc -l
ls -la outputs/models/description_pca.joblib
```

Expected: Same row count as preprocessed, 50 `desc_emb_*` columns added, fresh PCA model saved.

**Step 3: Commit**

```bash
git add outputs/property_data_with_embeddings.csv outputs/models/description_pca.joblib
git commit -m "Re-embed descriptions with updated dataset"
```

---

### Task 4: Train AutoGluon model (includes macro data fetch)

**Files:**
- Run: `src/train_models.py`
- Reads: `outputs/property_data_with_embeddings.csv`
- Fetches: ECB SDW API (Belgian CPI, 10Y yield, Euribor 12m) — merged by `date_obtained` month
- Output: `outputs/models/autogluon/`, `outputs/models/leaderboard.csv`, `outputs/models/test_set.csv`, `outputs/models/feature_importance.csv`, `outputs/models/autogluon_evaluation.png`, `outputs/models/model_comparison.png`

**Step 1: Verify ECB API is reachable**

```bash
python -c "
import urllib.request
url = 'https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA?startPeriod=2026-01&format=csvdata'
req = urllib.request.Request(url, headers={'Accept': 'text/csv'})
with urllib.request.urlopen(req, timeout=10) as resp:
    data = resp.read().decode()
    lines = data.strip().split('\n')
    print(f'ECB API OK: {len(lines)-1} rows')
    print('Last row:', lines[-1][:100])
"
```

Expected: Returns recent monthly data. If this fails, macro columns will be NaN — the model will still train but without temporal economic context.

**Step 2: Run AutoGluon training**

```bash
python src/train_models.py
```

Expected: Fetches 3 macro indicators from ECB, merges on year_month. Trains AutoGluon with `best_quality` preset, 600s time limit. Prints leaderboard, evaluation metrics (R², MAE, RMSE, MAPE), feature importance. Generates evaluation plots. Takes ~15 minutes.

**Step 3: Verify output**

```bash
ls -la outputs/models/autogluon/predictor.pkl
cat outputs/models/leaderboard.csv | head -5
# Verify macro columns were included
python -c "
import pandas as pd
df = pd.read_csv('outputs/models/test_set.csv')
for col in ['belgian_cpi_index', 'belgian_10y_yield', 'euribor_12m']:
    pct = df[col].notna().mean() * 100
    print(f'{col}: {pct:.1f}% non-null, latest={df[col].dropna().iloc[-1]:.2f}')
"
```

Expected: Fresh `predictor.pkl`, leaderboard with multiple models, macro columns present with high fill rate (should be ~100% for rows from 2024-01 onward).

**Step 4: Commit**

```bash
git add outputs/models/autogluon/ outputs/models/leaderboard.csv outputs/models/test_set.csv outputs/models/feature_importance.csv outputs/models/autogluon_evaluation.png outputs/models/model_comparison.png
git commit -m "Retrain AutoGluon with April 2026 data + fresh macro indicators"
```

---

### Task 5: Train SHAP XGBoost model

**Files:**
- Run: `src/train_shap_model.py`
- Reads: `outputs/property_data_with_embeddings.csv`
- Output: `outputs/models/shap_xgboost.joblib`

**Important:** `train_shap_model.py` does NOT fetch macro indicators — it trains directly from the embeddings CSV which doesn't have macro columns. The SHAP model is used only for feature explanations (not primary predictions), so this is acceptable. The macro features will appear as missing in SHAP explanations, which is consistent with how the SHAP computation in `api/prediction.py:543` already skips `belgian_*` and `euribor*` features.

**Step 1: Run SHAP model training**

```bash
python src/train_shap_model.py
```

Expected: Trains standalone XGBoost, prints RMSE/MAE, saves artifact with model + feature_names + label_encoders + base_value. Takes ~1 minute.

**Step 2: Verify output**

```bash
python -c "
import joblib
art = joblib.load('outputs/models/shap_xgboost.joblib')
print(f'Features: {len(art[\"feature_names\"])}')
print(f'Base value: {art[\"base_value\"]:.1f}k')
print(f'Encoders: {list(art[\"label_encoders\"].keys())}')
"
```

**Step 3: Commit**

```bash
git add outputs/models/shap_xgboost.joblib
git commit -m "Retrain SHAP XGBoost with updated dataset"
```

---

### Task 6: Deploy — restart the API

**Files:**
- `api/app.py` (no changes needed — reads models from config paths)
- `api/config.py` (no changes needed)

**Step 1: Kill any running API server**

```bash
lsof -ti :5001 | xargs kill 2>/dev/null; echo "Port 5001 cleared"
```

**Step 2: Start the API**

```bash
cd /Users/simon/Desktop/personal/immo_datacollection/house_scraping_project
source .venv/bin/activate
python -m api.app &
```

Expected: Loads all fresh models, prints confirmation messages for AutoGluon, E5+PCA, imputation models, SHAP XGBoost, spatial tree, street graph.

**Step 3: Verify the API is serving predictions with updated model**

```bash
# Health check
curl -s http://localhost:5001/api/health | python -m json.tool

# Stats (should reflect higher total_properties than before)
curl -s http://localhost:5001/api/stats | python -m json.tool

# Quick prediction test with a known Immoweb URL
curl -s "http://localhost:5001/api/predict/url?url=https://www.immoweb.be/en/classified/house/for-sale/gent/9000/11924270" | python -m json.tool | head -20
```

Expected: Health returns OK, stats show ~20K+ properties, prediction returns with non-null `predicted_price`, `shap`, and `comparables`.

**Step 4: Verify macro indicators are being used at inference time**

The API's `PredictionService._get_macro_indicators()` fetches live from ECB daily (cached). This is independent of training — it always uses the latest available values. Confirm in the API startup log that it prints macro values on first prediction.

**Step 5: Final commit**

```bash
git add -A
git status
git commit -m "Complete model retrain pipeline April 2026"
```

---

## Summary of what gets updated

| Artifact | Source | Macro data? |
|----------|--------|-------------|
| `property_data_parsed.csv` | All HTML in `data/` | No |
| `property_data_preprocessed.csv` | Parsed CSV | No |
| `imputation_models.joblib` | Preprocessing (Poisson GLMs) | No |
| `description_pca.joblib` | Embedding pipeline | No |
| `property_data_with_embeddings.csv` | Preprocessed + descriptions | No |
| **`autogluon/`** | **Embeddings CSV + ECB API fetch** | **Yes — merged by month** |
| `shap_xgboost.joblib` | Embeddings CSV | No (SHAP-only) |
| API (runtime) | All above + live ECB fetch per prediction | Yes |

The macro data enters at two points:
1. **Training time** (Task 4): `train_models.py` fetches full ECB time series and merges on each property's `date_obtained` month → model learns the relationship between interest rates and prices
2. **Inference time** (always): `PredictionService._get_macro_indicators()` fetches latest month's values from ECB → predictions reflect current economic conditions

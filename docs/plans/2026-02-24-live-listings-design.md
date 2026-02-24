# Live Listings Feature Design

## Summary

Add a "Live Listings" tab to the neighbourhood detail panel. At Flask startup, parse today's raw HTML files from `data/`, run predictions with the loaded AutoGluon model, cache results to CSV. Serve filtered listings per neighbourhood via a new API endpoint. Include a manual refresh button.

## Backend: LiveListingsService

### New module: `api/live_listings.py`

A `LiveListingsService` class initialized at app startup. Depends on `PredictionService` (for parsing and prediction) and `DataLoader` (for spatial neighbourhood lookup).

### Startup flow

1. Check for `outputs/today_predictions.csv`. Read `date_obtained` column — if it matches `date.today()`, load directly (cache hit).
2. On cache miss: scan `data/` for HTML files. Try today's date tag (`DDMMYYYY` format), then yesterday, then day before, up to 7 days back. Use the most recent date with files.
3. For each HTML file found: read HTML, call `PredictionService._parse_immoweb_html()`, then `_prepare_features()`, then `ag_predictor.predict()`. Compute `delta_eur = predicted_eur - asking_eur` and `delta_pct = delta_eur / asking_eur * 100`.
4. Assign neighbourhood via `DataLoader.get_neighbourhood_spatial(lat, lon)`.
5. Save all results to `outputs/today_predictions.csv` with date tag.
6. Hold results in memory as a DataFrame for fast serving.

### Date fallback logic

```python
from datetime import date, timedelta

for i in range(7):
    d = date.today() - timedelta(days=i)
    tag = d.strftime("%d%m%Y")
    # scan data/*/full_source_{tag}.html
    # if files found, use this date and break
```

### Endpoints

- `GET /api/listings/live?neighbourhood=X` — return listings filtered by neighbourhood. Without param, return all. Columns: property_id, address, asking_price, predicted_price, delta_eur, delta_pct, livable_area, bedrooms, condition, energy_label, neighbourhood, url.
- `POST /api/listings/refresh` — clear cache, re-scan HTML, re-predict, re-write CSV. Return `{ count, date_used }`.
- `GET /api/listings/count` — return total count and date used (for UI badge).

### Caching rules

- Cache file: `outputs/today_predictions.csv`
- Cache key: `date_obtained` column matching `date.today()`
- Cache invalidated by: refresh endpoint, or new day
- In-memory DataFrame refreshed on startup and on refresh call

## Frontend: NeighbourhoodDetail tabs

### Layout change

The `NeighbourhoodDetail` component gets a two-tab layout:

- **"Overview" tab** (default): existing stats card, price trend chart, today's deals snippet.
- **"Live Listings" tab**: new sortable/filterable property table.

### Live Listings table

Built with `@tanstack/react-table` (already installed), reusing existing shadcn/ui Table components.

| Column | Key | Format |
|--------|-----|--------|
| Address | `address` | Link to Immoweb URL, opens new tab |
| Asking Price | `asking_price` | `€XXX,XXX` |
| Predicted Price | `predicted_price` | `€XXX,XXX` |
| Delta % | `delta_pct` | Green if > 0, red if < 0, e.g. `+14.2%` |
| Area (m²) | `livable_area` | integer |
| Beds | `bedrooms` | integer |
| Condition | `condition` | text |
| Energy | `energy_label` | text |

Default sort: delta_pct descending (best deals first).
Text filter on address column.

### Refresh button

In the "Live Listings" tab header: a "Refresh" button that POSTs to `/api/listings/refresh`. Shows spinner during mutation, invalidates the query cache on success.

### New hook

`useLiveListings(neighbourhood: string)` — calls `GET /api/listings/live?neighbourhood=X`. Enabled only when the tab is active. Stale time: 5 minutes.

### New types

```typescript
interface LiveListing {
  property_id: string;
  address: string;
  url: string;
  asking_price: number;
  predicted_price: number;
  delta_eur: number;
  delta_pct: number;
  livable_area: number | null;
  bedrooms: number | null;
  condition: string | null;
  energy_label: string | null;
  neighbourhood: string;
}
```

## Implementation tasks

1. Create `api/live_listings.py` with `LiveListingsService` class
2. Wire into `api/app.py` — initialize at startup, add 3 endpoints
3. Add tests for the new endpoints
4. Add `LiveListing` type and `useLiveListings` hook to frontend
5. Refactor `NeighbourhoodDetail` into tabbed layout (Overview / Live Listings)
6. Build `LiveListingsTable` component with @tanstack/react-table
7. Add refresh button with mutation hook
8. Verify end-to-end: startup cache, API response, frontend rendering

import json
import os

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from api.data_loader import DataLoader
from api.prediction import PredictionService
from api.live_listings import LiveListingsService

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIST_DIR = os.path.join(PROJECT_ROOT, 'frontend', 'dist')
EDA_REPORT_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'eda', 'eda_report.json')
MARKET_INSIGHTS_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'production', 'market_insights.json')

_eda_cache = {'path': None, 'mtime': None, 'report': None}
_insights_cache = {'path': None, 'mtime': None, 'data': None}


def _load_market_insights():
    if not os.path.exists(MARKET_INSIGHTS_PATH):
        return None
    mtime = os.path.getmtime(MARKET_INSIGHTS_PATH)
    if _insights_cache['path'] == MARKET_INSIGHTS_PATH and _insights_cache['mtime'] == mtime:
        return _insights_cache['data']
    with open(MARKET_INSIGHTS_PATH, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    _insights_cache.update({'path': MARKET_INSIGHTS_PATH, 'mtime': mtime, 'data': data})
    return data


def _load_eda_report():
    """Load the EDA report, re-reading only when the file changes.

    The report is several MB of JSON, so it is held in memory and invalidated
    on mtime rather than parsed per request.
    """
    if not os.path.exists(EDA_REPORT_PATH):
        return None
    mtime = os.path.getmtime(EDA_REPORT_PATH)
    if _eda_cache['path'] == EDA_REPORT_PATH and _eda_cache['mtime'] == mtime:
        return _eda_cache['report']
    with open(EDA_REPORT_PATH, 'r', encoding='utf-8') as handle:
        report = json.load(handle)
    _eda_cache.update({'path': EDA_REPORT_PATH, 'mtime': mtime, 'report': report})
    return report

data_loader = None
prediction_service = None
live_listings_service = None


def create_app(loader=None, pred_service=None, listings_service=None):
    global data_loader, prediction_service, live_listings_service
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:5173", "https://*.trycloudflare.com"])

    if loader is not None:
        data_loader = loader
    elif data_loader is None:
        data_loader = DataLoader()

    if pred_service is not None:
        prediction_service = pred_service
    elif prediction_service is None:
        import sys
        from api.config import PRODUCTION_BUNDLE_DIR, PROJECT_DIR

        src_dir = os.path.join(PROJECT_DIR, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        bundle = None
        try:
            if os.path.exists(os.path.join(PRODUCTION_BUNDLE_DIR, "feature_pipeline.joblib")):
                from production_pipeline import ProductionBundle
                bundle = ProductionBundle.load(PRODUCTION_BUNDLE_DIR)
                mv = bundle.metadata.get("model_version", "phase7-lean5")
                print(f"Phase-7 production bundle loaded ({mv}) from {PRODUCTION_BUNDLE_DIR}")
            else:
                print(f"Production bundle not found at {PRODUCTION_BUNDLE_DIR}. Run: .venv-api/bin/python -u src/train_production.py")
        except Exception as e:
            print(f"Production bundle failed to load: {e}")

        prediction_service = PredictionService(
            bundle=bundle,
            data_loader=data_loader,
        )

    if listings_service is not None:
        live_listings_service = listings_service
    elif live_listings_service is None:
        try:
            live_listings_service = LiveListingsService(prediction_service, data_loader)
        except Exception as e:
            print(f"Live listings not available: {e}")

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/stats")
    def stats():
        return jsonify(data_loader.get_stats())

    @app.route("/api/neighbourhoods")
    def neighbourhoods():
        summary = data_loader.neighbourhood_summary
        if summary is None:
            return jsonify({"neighbourhoods": []})

        result = []
        for _, row in summary.iterrows():
            nbh = row["neighbourhood"]
            geojson = data_loader.get_neighbourhood_geojson(nbh)
            result.append({
                "name": nbh,
                "display_name": str(row.get("display_name", nbh)),
                "count": int(row["count"]),
                "median_price_k": float(row["median_price_k"]),
                "median_price_eur": int(round(float(row["median_price_k"]) * 1000)),
                "median_price_sqm": float(row.get("median_price_sqm", 0)),
                "avg_listing_duration": float(row.get("avg_listing_duration", 0)),
                "median_listing_duration": float(row.get("median_listing_duration", 0)),
                "pct_under_option": float(row.get("pct_under_option", 0)),
                "pct_price_cut": float(row.get("pct_price_cut", 0))
                    if not pd.isna(row.get("pct_price_cut")) else None,
                "median_delta_pct": float(row.get("median_delta_pct", 0))
                    if not pd.isna(row.get("median_delta_pct")) else None,
                "pct_underpriced": float(row.get("pct_underpriced", 0))
                    if not pd.isna(row.get("pct_underpriced")) else None,
                "geojson": geojson,
            })

        return jsonify({"neighbourhoods": result})

    @app.route("/api/neighbourhoods/<name>")
    def neighbourhood_detail(name):
        detail = data_loader.get_neighbourhood_detail(name)
        if detail is None:
            return jsonify({"error": "Neighbourhood not found"}), 404
        return jsonify(detail)

    @app.route("/api/properties/today")
    def today_deals():
        return jsonify({"properties": data_loader.get_today_deals()})

    @app.route("/api/geocode")
    def geocode():
        """Geocode an address using Nominatim, return lat/lon/postal/city/neighbourhood."""
        import requests as http_requests
        address = request.args.get("address", "")
        if not address:
            return jsonify({"error": "No address provided"}), 400
        try:
            resp = http_requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": address,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1,
                    "countrycodes": "be",
                },
                headers={"User-Agent": "ImmoPredictor/1.0"},
                timeout=10,
            )
            results = resp.json()
            if not results:
                return jsonify({"error": "Address not found"}), 404
            hit = results[0]
            lat = float(hit["lat"])
            lon = float(hit["lon"])
            addr = hit.get("address", {})
            postal = addr.get("postcode", "")
            city = addr.get("city") or addr.get("town") or addr.get("village") or ""
            street = addr.get("road", "")
            number = addr.get("house_number", "")
            nbh = None
            if data_loader:
                nbh = data_loader.get_neighbourhood_spatial(lat, lon)
            return jsonify({
                "latitude": lat,
                "longitude": lon,
                "postal_code": postal,
                "city": city,
                "street": street,
                "house_number": number,
                "neighbourhood": nbh,
                "display_name": hit.get("display_name", ""),
            })
        except Exception as e:
            return jsonify({"error": f"Geocoding failed: {e}"}), 500

    @app.route("/api/predict/url")
    def predict_url():
        url = request.args.get("url")
        result, status = prediction_service.predict_from_url(url)
        return jsonify(result), status

    @app.route("/api/predict/form", methods=["POST"])
    def predict_form():
        form_data = request.get_json()
        result, status = prediction_service.predict_from_form_data(form_data)
        return jsonify(result), status

    @app.route("/api/predict/manual", methods=["POST"])
    def predict_manual():
        form_data = request.get_json()
        result, status = prediction_service.predict_manual(form_data)
        return jsonify(result), status

    @app.route("/api/listings/live")
    def listings_live():
        if live_listings_service is None:
            return jsonify({"error": "Live listings not available"}), 503
        neighbourhood = request.args.get("neighbourhood")
        listings = live_listings_service.get_listings(neighbourhood)
        return jsonify({"listings": listings, "date_used": live_listings_service.date_used})

    @app.route("/api/listings/refresh", methods=["POST"])
    def listings_refresh():
        if live_listings_service is None:
            return jsonify({"error": "Live listings not available"}), 503
        result = live_listings_service.refresh()
        return jsonify(result)

    @app.route("/api/listings/count")
    def listings_count():
        if live_listings_service is None:
            return jsonify({"count": 0, "date_used": None})
        return jsonify(live_listings_service.get_count())

    @app.route("/api/deals")
    def deals_feed():
        """City-wide Deal Hunter feed (live scraped listings when fresh, else recent Phase-7 OOF)."""
        if live_listings_service is None:
            return jsonify({"deals": [], "total": 0, "source": "unavailable", "date_used": None}), 503
        max_price_raw = request.args.get("max_price")
        min_delta_raw = request.args.get("min_delta_pct")
        limit_raw = request.args.get("limit", "150")
        payload = live_listings_service.get_deals(
            neighbourhood=request.args.get("neighbourhood"),
            property_type=request.args.get("property_type"),
            max_price=float(max_price_raw) if max_price_raw else None,
            min_epc=request.args.get("min_epc"),
            min_delta_pct=float(min_delta_raw) if min_delta_raw else None,
            below_band_only=request.args.get("below_band_only", "").lower() in ("1", "true", "yes"),
            limit=int(limit_raw) if limit_raw.isdigit() else 150,
        )
        return jsonify(payload)

    # ---- Model Metadata & Market Insights --------------------------------
    @app.route("/api/model/metadata")
    def model_metadata():
        bundle = getattr(prediction_service, "bundle", None)
        if bundle is None:
            return jsonify({"error": "No production bundle loaded"}), 503
        return jsonify(bundle.metadata)

    @app.route("/api/insights")
    def market_insights():
        data = _load_market_insights()
        if data is None:
            return jsonify({
                "error": "No market insights artifact found. Run: "
                         "PYTHONPATH=src .venv-api/bin/python -u src/build_market_insights.py",
            }), 404
        return jsonify(data)

    @app.route("/api/affordability", methods=["POST"])
    def affordability():
        """Evaluate a buyer's budget across Ghent neighbourhoods using Phase-7 Lean-5.

        Supports minimum thresholds via ``condition_mode='min'`` / ``epc_mode='min'``
        (or a ``+`` suffix like ``JUST_RENOVATED+`` / ``B+``), batch-scoring all
        qualifying (condition, EPC) tiers weighted by Ghent corpus priors.
        """
        CONDITION_ORDER = [
            "TO_RESTORE",
            "TO_RENOVATE",
            "TO_BE_DONE_UP",
            "GOOD",
            "JUST_RENOVATED",
            "AS_NEW",
        ]
        EPC_ORDER = ["F", "E", "D", "C", "B", "A", "A+", "A++"]

        def _expand_min(val: str | None, order: list[str], default: str) -> tuple[list[str], bool]:
            if not val:
                return [default], False
            s = str(val).strip().upper()
            if s in ("ANY", "ALL"):
                return list(order), True
            is_plus = s.endswith("+") and s not in order
            base = s[:-1] if is_plus else s
            if base not in order:
                return [default], False
            if is_plus:
                return order[order.index(base):], True
            return [base], False

        body = request.get_json() or {}
        budget = float(body.get("budget_eur") or 400_000)
        property_type = str(body.get("property_type") or "HOUSE").upper()
        livable_area = float(body.get("livable_area") or (140 if property_type == "HOUSE" else 85))
        bedrooms = int(body.get("bedrooms") or (3 if property_type == "HOUSE" else 2))
        year_built = int(body.get("year_built") or 1970)
        has_garden = bool(body.get("has_garden", property_type == "HOUSE"))
        has_terrace = bool(body.get("has_terrace", True))

        cond_mode = str(body.get("condition_mode") or "").lower()
        epc_mode = str(body.get("epc_mode") or "").lower()

        raw_cond = body.get("min_condition") or body.get("condition") or "JUST_RENOVATED"
        raw_epc = body.get("min_energy_label") or body.get("energy_label") or "B"

        if cond_mode == "min" or body.get("min_condition"):
            c_clean = str(raw_cond).strip().upper().rstrip("+")
            if c_clean in ("ANY", "ALL"):
                cond_list, cond_is_min = list(CONDITION_ORDER), True
            elif c_clean in CONDITION_ORDER:
                cond_list, cond_is_min = CONDITION_ORDER[CONDITION_ORDER.index(c_clean):], True
            else:
                cond_list, cond_is_min = ["JUST_RENOVATED", "AS_NEW"], True
        else:
            cond_list, cond_is_min = _expand_min(str(raw_cond), CONDITION_ORDER, "GOOD")

        if epc_mode == "min" or body.get("min_energy_label"):
            e_clean = str(raw_epc).strip().upper()
            if e_clean in ("ANY", "ALL"):
                epc_list, epc_is_min = list(EPC_ORDER), True
            elif e_clean in EPC_ORDER:
                epc_list, epc_is_min = EPC_ORDER[EPC_ORDER.index(e_clean):], True
            elif e_clean.rstrip("+") in EPC_ORDER:
                base_e = e_clean.rstrip("+")
                epc_list, epc_is_min = EPC_ORDER[EPC_ORDER.index(base_e):], True
            else:
                epc_list, epc_is_min = ["B", "A", "A+", "A++"], True
        else:
            epc_list, epc_is_min = _expand_min(str(raw_epc), EPC_ORDER, "C")

        # Collapse rare A+/A++ into A for synthetic scoring to keep batch size tight & well-supported
        epc_scoring: list[str] = []
        for e in epc_list:
            mapped = "A" if e in ("A+", "A++") else e
            if mapped not in epc_scoring:
                epc_scoring.append(mapped)

        COND_PRIOR = {
            "TO_RESTORE": 0.03,
            "TO_RENOVATE": 0.14,
            "TO_BE_DONE_UP": 0.12,
            "GOOD": 0.38,
            "JUST_RENOVATED": 0.16,
            "AS_NEW": 0.17,
        }
        EPC_PRIOR = {"F": 0.14, "E": 0.08, "D": 0.16, "C": 0.24, "B": 0.24, "A": 0.14}

        # Cap combo grid at <= 3 x 3 = 9 tiers per neighbourhood (225 rows -> ~0.8s batch)
        cond_scoring = (
            [cond_list[0], cond_list[len(cond_list) // 2], cond_list[-1]]
            if len(cond_list) > 3
            else list(cond_list)
        )
        epc_scoring = (
            [epc_scoring[0], epc_scoring[len(epc_scoring) // 2], epc_scoring[-1]]
            if len(epc_scoring) > 3
            else epc_scoring
        )

        combos: list[tuple[str, str, float]] = []
        for c in cond_scoring:
            for e in epc_scoring:
                combos.append((c, e, COND_PRIOR.get(c, 0.1) * EPC_PRIOR.get(e, 0.1)))
        total_w = sum(w for _, _, w in combos) or 1.0
        combos = [(c, e, w / total_w) for c, e, w in combos]

        insights = _load_market_insights()
        profiles = (insights or {}).get("neighbourhood_profiles", [])
        bundle = getattr(prediction_service, "bundle", None)
        if bundle is None or not profiles:
            return jsonify({"error": "Production model or neighbourhood profiles unavailable"}), 503

        today_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        rows = []
        for p in profiles:
            for c, e, _w in combos:
                rows.append({
                    "property_type": property_type,
                    "property_subtype": "HOUSE" if property_type == "HOUSE" else "APARTMENT",
                    "livable_area": livable_area,
                    "number_of_bedrooms": bedrooms,
                    "building_condition": c,
                    "energy_label": e,
                    "year_built": year_built,
                    "has_garden": has_garden,
                    "has_terrace": has_terrace,
                    "latitude": p["lat"],
                    "longitude": p["lon"],
                    "neighbourhood": p["name"],
                    "city": "Gent",
                    "postal_code": "9000",
                    "date_obtained": today_str,
                })

        import math
        df_synth = pd.DataFrame(rows)
        pred_out = bundle.predict_raw(df_synth, include_shap=False)
        n_combos = len(combos)

        # Empirical matching-listing counts per neighbourhood from clean OOF corpus
        matching_counts: dict[str, int] = {}
        adf = getattr(data_loader, "analytics_df", None) if data_loader is not None else None
        if isinstance(adf, pd.DataFrame) and not adf.empty and "neighbourhood" in adf.columns:
            mask = pd.Series(True, index=adf.index)
            if "property_type_HOUSE" in adf.columns:
                want_house = 1 if property_type == "HOUSE" else 0
                mask &= adf["property_type_HOUSE"].fillna(0).astype(int) == want_house
            if "building_condition" in adf.columns:
                mask &= adf["building_condition"].isin(cond_list)
            if "energy_label" in adf.columns:
                mask &= adf["energy_label"].isin(epc_list)
            matching_counts = {str(k): int(v) for k, v in adf.loc[mask, "neighbourhood"].value_counts().items()}

        results = []
        for pi, p in enumerate(profiles):
            start = pi * n_combos
            prob_mix = 0.0
            pred_weighted = 0.0
            hw_weighted = 0.0
            entry_pred = float("inf")
            low_80_min = float("inf")
            high_80_max = 0.0
            low_90_min = float("inf")
            high_90_max = 0.0

            for ci, (_c, _e, w) in enumerate(combos):
                idx = start + ci
                p_eur = float(pred_out["predicted_eur"][idx])
                l80 = float(pred_out["low_eur"][idx])
                h80 = float(pred_out["high_eur"][idx])
                l90 = float(pred_out["low_90_eur"][idx])
                h90 = float(pred_out["high_90_eur"][idx])
                hw = float(pred_out["interval_halfwidth_pct"][idx])

                log_half = max(0.05, 0.5 * math.log(max(h80, 1.0) / max(l80, 1.0)))
                sigma = log_half / 1.28155
                z = (math.log(max(budget, 1.0)) - math.log(max(p_eur, 1.0))) / sigma
                prob_c = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

                prob_mix += w * prob_c
                pred_weighted += w * p_eur
                hw_weighted += w * hw
                entry_pred = min(entry_pred, p_eur)
                low_80_min = min(low_80_min, l80)
                high_80_max = max(high_80_max, h80)
                low_90_min = min(low_90_min, l90)
                high_90_max = max(high_90_max, h90)

            pred_eur = pred_weighted
            prob_affordable = prob_mix

            if prob_affordable >= 0.70:
                tier = "comfortable"
            elif prob_affordable >= 0.40:
                tier = "realistic"
            elif prob_affordable >= 0.18:
                tier = "stretch"
            else:
                tier = "out_of_reach"

            geojson = data_loader.get_neighbourhood_geojson(p["name"]) if data_loader else None
            results.append({
                "name": p["name"],
                "display_name": p["display_name"],
                "count": p["count"],
                "matching_count": matching_counts.get(p["name"], 0),
                "lat": p["lat"],
                "lon": p["lon"],
                "predicted_eur": int(round(pred_eur)),
                "entry_eur": int(round(entry_pred)),
                "low_eur": int(round(low_80_min)),
                "high_eur": int(round(high_80_max)),
                "low_90_eur": int(round(low_90_min)),
                "high_90_eur": int(round(high_90_max)),
                "interval_halfwidth_pct": round(hw_weighted, 1),
                "budget_gap_eur": int(round(budget - pred_eur)),
                "budget_gap_pct": round((budget - pred_eur) / max(pred_eur, 1.0) * 100.0, 1),
                "prob_affordable": round(prob_affordable * 100.0, 1),
                "tier": tier,
                "median_price_sqm": p["median_price_sqm"],
                "median_dist_center_km": p.get("median_dist_center_km"),
                "geojson": geojson,
            })

        results.sort(key=lambda r: r["prob_affordable"], reverse=True)
        return jsonify({
            "query": {
                "budget_eur": int(round(budget)),
                "property_type": property_type,
                "livable_area": livable_area,
                "bedrooms": bedrooms,
                "condition": cond_list[0] + ("+" if cond_is_min and len(cond_list) > 1 else ""),
                "condition_tiers": cond_list,
                "energy_label": epc_list[0] + ("+" if epc_is_min and len(epc_list) > 1 else ""),
                "energy_tiers": epc_list,
                "year_built": year_built,
            },
            "neighbourhoods": results,
            "model_version": bundle.metadata.get("model_version", "phase7-lean5"),
        })

    # ---- EDA -------------------------------------------------------------
    @app.route("/api/eda")
    def eda_report():
        """Full EDA document (fields, correlations, spatial, temporal).

        ``?section=`` returns just one top-level key so the UI does not have to
        download several MB to render the overview tab.
        """
        report = _load_eda_report()
        if report is None:
            return jsonify({
                "error": "No EDA report found. Run: "
                         ".venv-api/bin/python src/eda_report.py",
            }), 404
        section = request.args.get("section")
        if not section:
            return jsonify(report)
        if section not in report:
            return jsonify({"error": f"Unknown section '{section}'",
                            "available": list(report.keys())}), 400
        return jsonify({section: report[section],
                        "dataset": report.get("dataset"),
                        "generated_at": report.get("generated_at")})

    @app.route("/api/eda/field/<name>")
    def eda_field(name):
        """One field's profile, for the field-detail drawer."""
        report = _load_eda_report()
        if report is None:
            return jsonify({"error": "No EDA report found"}), 404
        for field in report.get("fields", []):
            if field["name"] == name:
                return jsonify(field)
        return jsonify({"error": f"Unknown field '{name}'"}), 404

    # ---- Built React SPA Static Serving (Production Single-Process Mode) ---
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_spa(path):
        if path.startswith("api/"):
            return jsonify({"error": "Not found"}), 404
        dist_dir = app.config.get("FRONTEND_DIST_DIR", FRONTEND_DIST_DIR)
        if not os.path.isdir(dist_dir):
            return jsonify({
                "status": "api_only",
                "message": "Frontend bundle not built. Run `cd frontend && npm run build`.",
            }), 404
        candidate = os.path.join(dist_dir, path)
        if path and os.path.isfile(candidate):
            return send_from_directory(dist_dir, path)
        index_html = os.path.join(dist_dir, "index.html")
        if os.path.isfile(index_html):
            return send_from_directory(dist_dir, "index.html")
        return jsonify({"error": "Frontend index.html missing"}), 404

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5001)

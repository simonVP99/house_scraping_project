import json
import os

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from api.data_loader import DataLoader
from api.prediction import PredictionService
from api.live_listings import LiveListingsService

EDA_REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'outputs', 'eda', 'eda_report.json')

_eda_cache = {'path': None, 'mtime': None, 'report': None}


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
                "count": int(row["count"]),
                "median_price_k": float(row["median_price_k"]),
                "median_price_sqm": float(row.get("median_price_sqm", 0)),
                "avg_listing_duration": float(row.get("avg_listing_duration", 0)),
                "median_listing_duration": float(row.get("median_listing_duration", 0)),
                "pct_under_option": float(row.get("pct_under_option", 0)),
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

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5001)

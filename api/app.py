import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from api.data_loader import DataLoader
from api.prediction import PredictionService
from api.live_listings import LiveListingsService

data_loader = None
prediction_service = None
live_listings_service = None


def create_app(loader=None, pred_service=None, listings_service=None):
    global data_loader, prediction_service, live_listings_service
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:5173"])

    if loader is not None:
        data_loader = loader
    elif data_loader is None:
        data_loader = DataLoader()

    if pred_service is not None:
        prediction_service = pred_service
    elif prediction_service is None:
        import os
        from api.config import AUTOGLUON_PATH, PCA_PATH

        ag_predictor = None
        e5_model = None
        e5_tokenizer = None
        e5_device = None
        pca_model = None

        # Try loading AutoGluon model
        try:
            from autogluon.tabular import TabularPredictor
            if os.path.exists(AUTOGLUON_PATH):
                ag_predictor = TabularPredictor.load(AUTOGLUON_PATH)
                print(f"AutoGluon model loaded from {AUTOGLUON_PATH}")
        except Exception as e:
            print(f"AutoGluon not available: {e}")

        # Try loading E5 model + PCA for description embeddings
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            import joblib
            from api.config import E5_MODEL_NAME

            if os.path.exists(PCA_PATH):
                e5_device = (
                    "cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu"
                )
                e5_tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
                e5_model = AutoModel.from_pretrained(E5_MODEL_NAME).to(e5_device)
                pca_model = joblib.load(PCA_PATH)
                print(f"E5 + PCA loaded ({e5_device})")
        except Exception as e:
            print(f"E5/PCA not available: {e}")

        prediction_service = PredictionService(
            ag_predictor=ag_predictor,
            e5_model=e5_model,
            e5_tokenizer=e5_tokenizer,
            e5_device=e5_device,
            pca_model=pca_model,
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

    @app.route("/api/predict/url")
    def predict_url():
        url = request.args.get("url")
        result, status = prediction_service.predict_from_url(url)
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

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5001)

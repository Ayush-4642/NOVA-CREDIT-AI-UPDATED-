from flask import Blueprint, render_template, jsonify
import os
import json
from database.database import get_chart_data, get_summary_stats
from models.predictor import get_metadata
from config import Config

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.route("/analytics")
def analytics():
    """Portfolio analytics dashboard based on database records."""
    stats = get_summary_stats()
    charts = get_chart_data()
    metadata = get_metadata()
    return render_template("analytics.html", stats=stats, charts=charts, metadata=metadata)


@analytics_bp.route("/model-performance")
def model_performance():
    """Model Performance & Validation view showing genuine evaluation metrics."""
    metadata = get_metadata()
    
    # Load benchmark comparison results if available
    benchmark_path = os.path.join(Config.BASE_DIR, "benchmark_results.json")
    benchmark_data = {}
    if os.path.exists(benchmark_path):
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                benchmark_data = json.load(f)
        except Exception:
            benchmark_data = {}

    return render_template("model_performance.html", metadata=metadata, benchmark=benchmark_data)


@analytics_bp.route("/api/analytics/data")
def analytics_data():
    """JSON endpoint consumed by static/js/charts.js."""
    charts = get_chart_data()
    return jsonify(charts)


@analytics_bp.route("/api/analytics/feature-importance")
def feature_importance():
    """Returns verified grouped feature importances from the production model."""
    metadata = get_metadata()
    fi_list = metadata.get("feature_importances", [])
    
    # Format labels cleanly (e.g. debt_to_income_ratio -> Debt To Income Ratio)
    labels = [item["feature"].replace("_", " ").title() for item in fi_list[:12]]
    values = [item["percentage"] for item in fi_list[:12]]
    
    return jsonify({
        "labels": labels,
        "values": values,
    })


@analytics_bp.route("/api/analytics/model-metrics")
def model_metrics():
    """Returns verified model evaluation metrics for API consumption."""
    metadata = get_metadata()
    return jsonify(metadata)

from flask import Blueprint, render_template
from database.database import get_summary_stats
from models.predictor import get_metadata

home_bp = Blueprint("home", __name__)


@home_bp.route("/")
def home():
    stats = get_summary_stats()
    metadata = get_metadata()
    return render_template("home.html", stats=stats, metadata=metadata)


@home_bp.route("/about")
def about():
    metadata = get_metadata()
    return render_template("about.html", metadata=metadata)

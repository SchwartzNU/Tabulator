import os
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for, session
from werkzeug.utils import secure_filename

from .loader import load_dataset, LoadError, DataSet
from .data_store import store

bp = Blueprint("main", __name__)


def _allowed_file(filename: str) -> bool:
    allowed = current_app.config.get("ALLOWED_EXTENSIONS", set())
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


@bp.route("/", methods=["GET"])  # Home: upload UI
def index():
    dataset_id = session.get("dataset_id")
    ds = store.get(dataset_id) if dataset_id else None
    preview_html = None
    info = None
    units_preview = None
    if ds is not None:
        try:
            preview_html = ds.df.head(10).to_html(classes=["preview"], border=0)
            info = {
                "rows": int(ds.df.shape[0]),
                "cols": int(ds.df.shape[1]),
                "columns": list(map(str, ds.df.columns[:10])) + (["…"] if ds.df.shape[1] > 10 else []),
            }
            # Show up to first 10 units
            items = list(ds.units.items())
            units_preview = items[:10] + ([("…", "…")] if len(items) > 10 else [])
        except Exception:
            preview_html = None
    return render_template("index.html", dataset_id=dataset_id, info=info, preview_html=preview_html, units_preview=units_preview)


@bp.route("/upload", methods=["POST"])  # Handle file upload
def upload():
    if "datafile" not in request.files:
        flash("No file part in request.")
        return redirect(url_for("main.index"))

    file = request.files["datafile"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("main.index"))

    if not _allowed_file(file.filename):
        allowed_list = ", ".join(sorted(current_app.config.get("ALLOWED_EXTENSIONS", [])))
        flash(f"Unsupported file type. Allowed: {allowed_list}")
        return redirect(url_for("main.index"))

    filename = secure_filename(file.filename)
    dest_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    file.save(dest_path)
    # Attempt to parse into a DataSet (DataFrame + units)
    try:
        ds = load_dataset(dest_path)
    except LoadError as e:
        flash(f"Uploaded but failed to parse: {e}")
        return redirect(url_for("main.index"))
    dataset_id = store.put(ds)
    session["dataset_id"] = dataset_id
    flash(f"Loaded {filename}: {ds.df.shape[0]} rows, {ds.df.shape[1]} cols")
    return redirect(url_for("main.index", _anchor="preview"))

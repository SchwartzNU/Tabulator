import os
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for, session, jsonify
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
    units_items = None
    if ds is not None:
        try:
            preview_html = ds.df.head(10).to_html(classes=["preview"], border=0)
            info = {
                "rows": int(ds.df.shape[0]),
                "cols": int(ds.df.shape[1]),
                "columns": list(map(str, ds.df.columns)),
            }
            # Order units by DataFrame columns first, then any extras
            units_items = [(str(c), str(ds.units.get(c, ""))) for c in ds.df.columns]
            extras = [(k, v) for k, v in ds.units.items() if k not in ds.df.columns]
            if extras:
                units_items.extend((str(k), str(v)) for k, v in extras)
        except Exception:
            preview_html = None
    return render_template(
        "index.html",
        dataset_id=dataset_id,
        info=info,
        preview_html=preview_html,
        units_items=units_items,
    )


@bp.get("/api/columns")
def api_columns():
    dataset_id = session.get("dataset_id")
    ds = store.get(dataset_id) if dataset_id else None
    if ds is None:
        return jsonify({"error": "no_dataset"}), 404
    df = ds.df
    cols = []
    try:
        import pandas as pd
        from pandas.api.types import is_numeric_dtype
    except Exception:
        def is_numeric_dtype(x):
            return False
    for c in df.columns:
        cols.append({
            "name": str(c),
            "is_numeric": bool(is_numeric_dtype(df[c]))
        })
    return jsonify({"columns": cols})


@bp.get("/api/plot/bar")
def api_plot_bar():
    value = request.args.get("value")
    group = request.args.get("group")
    if not value or not group:
        return jsonify({"error": "missing_params"}), 400

    dataset_id = session.get("dataset_id")
    ds = store.get(dataset_id) if dataset_id else None
    if ds is None:
        return jsonify({"error": "no_dataset"}), 404
    df = ds.df
    if value not in df.columns or group not in df.columns:
        return jsonify({"error": "bad_columns"}), 400

    # Select columns as 1-D Series even if duplicate names exist
    import pandas as pd
    gcol = df[group]
    if isinstance(gcol, pd.DataFrame):
        gcol = gcol.iloc[:, 0]
    vcol = df[value]
    if isinstance(vcol, pd.DataFrame):
        vcol = vcol.iloc[:, 0]
    # Coerce value column to numeric and drop NaNs
    vcol = pd.to_numeric(vcol, errors="coerce")
    tmp = pd.DataFrame({"_group": gcol, "_value": vcol}).dropna(subset=["_value"])

    # Group and compute
    groups_out = []
    for gval, gdf in tmp.groupby("_group", dropna=False):
        vals = gdf["_value"].tolist()
        try:
            mean_val = float(gdf["_value"].mean())
        except Exception:
            mean_val = None
        groups_out.append({
            "name": str(gval),
            "count": len(vals),
            "mean": mean_val,
            "values": vals,
        })

    unit = ds.units.get(value, "") if isinstance(ds.units, dict) else ""
    return jsonify({
        "value": value,
        "group": group,
        "unit": unit,
        "groups": groups_out,
    })


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

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
    # Prefer segment_ID, then segmentID, then cell_name for tooltips. Do not fallback to row index.
    id_source = None
    for cname in ("segment_ID", "segmentID", "cell_name"):
        if cname in df.columns:
            id_source = cname
            break
    if id_source is not None:
        idcol = df[id_source]
        if isinstance(idcol, pd.DataFrame):
            idcol = idcol.iloc[:, 0]
    else:
        idcol = None
    data_dict = {"_group": gcol, "_value": vcol}
    if idcol is not None:
        data_dict["_id"] = idcol
    tmp = pd.DataFrame(data_dict).dropna(subset=["_value"])

    # Group and compute
    groups_out = []
    for gval, gdf in tmp.groupby("_group", dropna=False):
        vals = gdf["_value"].tolist()
        if "_id" in gdf.columns:
            points = [
                {"id": str(i), "value": float(v)}
                for i, v in zip(gdf["_id"].tolist(), vals)
                if isinstance(v, (int, float))
            ]
        else:
            points = [
                {"id": "", "value": float(v)}
                for v in vals
                if isinstance(v, (int, float))
            ]
        try:
            mean_val = float(gdf["_value"].mean())
        except Exception:
            mean_val = None
        groups_out.append({
            "name": str(gval),
            "count": len(vals),
            "mean": mean_val,
            "values": vals,
            "points": points,
        })

    unit = ds.units.get(value, "") if isinstance(ds.units, dict) else ""
    group_unit = ds.units.get(group, "") if isinstance(ds.units, dict) else ""
    return jsonify({
        "value": value,
        "group": group,
        "unit": unit,
        "group_unit": group_unit,
        "groups": groups_out,
    })


@bp.get("/api/plot/scatter")
def api_plot_scatter():
    x = request.args.get("x")
    y = request.args.get("y")
    group = request.args.get("group")
    if not x or not y:
        return jsonify({"error": "missing_params"}), 400

    dataset_id = session.get("dataset_id")
    ds = store.get(dataset_id) if dataset_id else None
    if ds is None:
        return jsonify({"error": "no_dataset"}), 404
    df = ds.df
    if x not in df.columns or y not in df.columns:
        return jsonify({"error": "bad_columns"}), 400

    import pandas as pd

    xcol = df[x]
    if isinstance(xcol, pd.DataFrame):
        xcol = xcol.iloc[:, 0]
    ycol = df[y]
    if isinstance(ycol, pd.DataFrame):
        ycol = ycol.iloc[:, 0]

    xnum = pd.to_numeric(xcol, errors="coerce")
    ynum = pd.to_numeric(ycol, errors="coerce")

    data_dict = {"_x": xnum, "_y": ynum}
    # Optional group
    gseries = None
    if group and group in df.columns:
        gseries = df[group]
        if isinstance(gseries, pd.DataFrame):
            gseries = gseries.iloc[:, 0]
        data_dict["_group"] = gseries

    # Optional ID for tooltips
    id_source = None
    for cname in ("segment_ID", "segmentID", "cell_name"):
        if cname in df.columns:
            id_source = cname
            break
    if id_source is not None:
        idcol = df[id_source]
        if isinstance(idcol, pd.DataFrame):
            idcol = idcol.iloc[:, 0]
        data_dict["_id"] = idcol

    tmp = pd.DataFrame(data_dict).dropna(subset=["_x", "_y"])

    # Build groups
    def compute_group_stats(gdf):
        import math
        xs = gdf["_x"].astype(float)
        ys = gdf["_y"].astype(float)
        n = int(len(xs))
        mx = float(xs.mean()) if n else None
        my = float(ys.mean()) if n else None
        sd_x = float(xs.std(ddof=1)) if n > 1 else None
        sd_y = float(ys.std(ddof=1)) if n > 1 else None
        sem_x = float(sd_x / math.sqrt(n)) if (sd_x is not None and n > 0) else None
        sem_y = float(sd_y / math.sqrt(n)) if (sd_y is not None and n > 0) else None
        return {
            "x": mx,
            "y": my,
            "errx_sd": sd_x,
            "errx_sem": sem_x,
            "erry_sd": sd_y,
            "erry_sem": sem_y,
            "count": n,
        }

    groups_out = []
    if gseries is not None:
        for gval, gdf in tmp.groupby("_group", dropna=False):
            pts = []
            has_id = "_id" in gdf.columns
            if has_id:
                for xi, yi, pid in zip(gdf["_x"].tolist(), gdf["_y"].tolist(), gdf["_id"].tolist()):
                    pts.append({"x": float(xi), "y": float(yi), "id": str(pid)})
            else:
                for xi, yi in zip(gdf["_x"].tolist(), gdf["_y"].tolist()):
                    pts.append({"x": float(xi), "y": float(yi), "id": ""})
            groups_out.append({
                "name": str(gval),
                "points": pts,
                "mean": compute_group_stats(gdf),
            })
    else:
        pts = []
        if "_id" in tmp.columns:
            for xi, yi, pid in zip(tmp["_x"].tolist(), tmp["_y"].tolist(), tmp["_id"].tolist()):
                pts.append({"x": float(xi), "y": float(yi), "id": str(pid)})
        else:
            for xi, yi in zip(tmp["_x"].tolist(), tmp["_y"].tolist()):
                pts.append({"x": float(xi), "y": float(yi), "id": ""})
        groups_out.append({
            "name": "All",
            "points": pts,
            "mean": compute_group_stats(tmp),
        })

    x_unit = ds.units.get(x, "") if isinstance(ds.units, dict) else ""
    y_unit = ds.units.get(y, "") if isinstance(ds.units, dict) else ""
    return jsonify({
        "x": x,
        "y": y,
        "x_unit": x_unit,
        "y_unit": y_unit,
        "group": group if group in df.columns else None,
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

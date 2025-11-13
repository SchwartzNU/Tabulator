import os
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
    session,
    jsonify,
)
from werkzeug.utils import secure_filename
from .loader import load_dataset, LoadError, DataSet
from .data_store import store
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import datajoint as dj

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


@bp.route("/api/columns")
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
        cols.append({"name": str(c), "is_numeric": bool(is_numeric_dtype(df[c]))})
    return jsonify({"columns": cols})


@bp.get("/api/pca")
def api_pca():
    """Run PCA on all numeric columns and return explained variance ratios.

    - Uses centered data (mean subtraction per feature).
    - Drops rows with NaNs across selected numeric columns.
    - Returns explained_variance_ratio per component and cumulative.
    """
    dataset_id = session.get("dataset_id")
    ds = store.get(dataset_id) if dataset_id else None
    if ds is None:
        return jsonify({"error": "no_dataset"}), 404
    df = ds.df

    try:
        import numpy as np
        import pandas as pd
    except Exception:
        return jsonify({"error": "missing_deps"}), 500

    # Select numeric columns only
    num_df = df.select_dtypes(include=["number"]).copy()
    # Drop identifier-like numeric columns if present by common names
    drop_names = {"segmentID", "segment_ID"}
    keep_cols = [c for c in num_df.columns if str(c) not in drop_names]
    num_df = num_df[keep_cols]

    if num_df.shape[1] < 1:
        return jsonify({"error": "no_numeric_columns"}), 400

    # Drop rows with NaNs
    num_df = num_df.dropna(axis=0, how="any")
    n_samples = int(num_df.shape[0])
    n_features = int(num_df.shape[1])
    if n_samples < 2:
        return jsonify({"error": "not_enough_rows", "rows": n_samples}), 400

    # Center and standardize to z-scores (per-feature mean 0, sd 1)
    X = num_df.to_numpy(dtype=float)
    mean = np.nanmean(X, axis=0, keepdims=True)
    Xc = X - mean
    # Sample standard deviation (ddof=1), consistent with explained variance computation
    sd = Xc.std(axis=0, ddof=1, keepdims=False)
    # Drop constant features (sd <= 0 or non-finite)
    keep_mask = np.isfinite(sd) & (sd > 0)
    dropped = [str(c) for c, k in zip(num_df.columns, keep_mask) if not bool(k)]
    if not np.any(keep_mask):
        return jsonify({"error": "no_variable_features"}), 400
    Xc = Xc[:, keep_mask]
    sd = sd[keep_mask]
    cols_kept = [str(c) for c, k in zip(num_df.columns, keep_mask) if bool(k)]
    n_features = int(len(cols_kept))
    Xz = Xc / sd
    try:
        # SVD on standardized data
        U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
        # Explained variance per component
        explained_variance = (S**2) / (n_samples - 1)
        total_var = explained_variance.sum()
        if not np.isfinite(total_var) or total_var <= 0:
            return jsonify({"error": "degenerate_variance"}), 400
        evr = (explained_variance / total_var).tolist()
        # Cumulative
        cum = np.cumsum(explained_variance / total_var).tolist()
    except Exception as e:
        return jsonify({"error": "pca_failed", "detail": str(e)}), 500

    return jsonify(
        {
            "n_samples": n_samples,
            "n_features": n_features,
            "columns": cols_kept,
            "explained_variance_ratio": evr,
            "cumulative_ratio": cum,
            "components": int(Vt.shape[0]),
            "loadings": Vt.tolist(),  # shape: [components][n_features]
            "standardized": True,
            "dropped_constant_columns": dropped,
        }
    )


@bp.get("/api/pca/scores")
def api_pca_scores():
    """Return PCA scores for selected PCs (2D or 3D) with optional coloring.

    Query params:
    - pcs: comma-separated 1-based PC indices, length 2 or 3 (e.g., "1,2" or "1,2,3").
    - color: optional column name used for coloring the points.
    """
    pcs_param = (request.args.get("pcs") or "").strip()
    color_col = request.args.get("color")

    dataset_id = session.get("dataset_id")
    ds = store.get(dataset_id) if dataset_id else None
    if ds is None:
        return jsonify({"error": "no_dataset"}), 404
    df = ds.df

    import numpy as np
    import pandas as pd

    # Parse PCs
    if pcs_param:
        try:
            pcs_list = [int(x) for x in pcs_param.split(",") if x.strip()]
        except Exception:
            return jsonify({"error": "bad_pcs"}), 400
    else:
        pcs_list = [1, 2]  # default 2D
    if len(pcs_list) not in (2, 3) or any(i < 1 for i in pcs_list):
        return jsonify({"error": "bad_pcs"}), 400

    # Build numeric data and align ids/color
    num_df = df.select_dtypes(include=["number"]).copy()
    drop_names = {"segmentID", "segment_ID"}
    feat_cols = [c for c in num_df.columns if str(c) not in drop_names]
    num_df = num_df[feat_cols]
    if num_df.shape[1] < 1:
        return jsonify({"error": "no_numeric_columns"}), 400

    clean = num_df.dropna(axis=0, how="any")
    if clean.shape[0] < 2:
        return jsonify({"error": "not_enough_rows", "rows": int(clean.shape[0])}), 400
    idx = clean.index

    id_source = None
    for cname in ("segmentID", "cell_name"):
        if cname in df.columns:
            id_source = cname
            break
    if id_source is not None:
        id_series = df.loc[idx, id_source]
    else:
        id_series = pd.Series([""] * len(idx), index=idx)
    if color_col and color_col in df.columns:
        color_series = df.loc[idx, color_col]
    else:
        color_series = None

    # Standardize and PCA
    X = clean.to_numpy(dtype=float)
    X = X - np.nanmean(X, axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1)
    keep_mask = np.isfinite(sd) & (sd > 0)
    if not np.any(keep_mask):
        return jsonify({"error": "no_variable_features"}), 400
    Xz = X[:, keep_mask] / sd[keep_mask]
    try:
        U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
        n = Xz.shape[0]
        explained_variance = (S**2) / (n - 1)
        total_var = explained_variance.sum()
        evr = (explained_variance / total_var).tolist() if total_var > 0 else []
        # Scores: U * S
        scores = U * S  # shape (n, k)
    except Exception as e:
        return jsonify({"error": "pca_failed", "detail": str(e)}), 500

    kmax = scores.shape[1]
    if any(pc > kmax for pc in pcs_list):
        return jsonify({"error": "pcs_out_of_range", "available": int(kmax)}), 400

    # 0-based indices
    pcs0 = [pc - 1 for pc in pcs_list]
    xs = scores[:, pcs0[0]].astype(float).tolist()
    ys = scores[:, pcs0[1]].astype(float).tolist()
    zs = scores[:, pcs0[2]].astype(float).tolist() if len(pcs0) == 3 else None

    ids = [str(v) for v in id_series.tolist()]
    if color_series is not None:
        colors = color_series.tolist()
        colors = [
            (
                None
                if pd.isna(v)
                else (float(v) if isinstance(v, (int, float)) else str(v))
            )
            for v in colors
        ]
    else:
        colors = None

    return jsonify(
        {
            "pcs": pcs_list,
            "explained_variance_ratio": evr,
            "x": xs,
            "y": ys,
            "z": zs,
            "id": ids,
            "color_values": colors,
            "color_column": str(color_col) if color_series is not None else None,
        }
    )


@bp.post("/api/classify/train")
def api_classify_train():
    """Train a classifier on tabular features with a stratified test split.

    JSON body:
    - label: column name to use as labels
    - clf: one of ['svm','logistic','decision_tree','random_forest','neural_net']
    - test_frac: float in (0,1)
    - max_iters: optional int (default 50)
    - patience: optional int (default 5)
    - random_state: optional int
    Returns history of train/val error per iteration, best iteration, and test metrics.
    """
    try:
        req = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "bad_json"}), 400

    label_col = (req or {}).get("label")
    clf_name = (req or {}).get("clf", "svm").lower()
    test_frac = float((req or {}).get("test_frac", 0.2))
    max_iters = int((req or {}).get("max_iters", 50))
    patience = int((req or {}).get("patience", 5))
    early_stop = bool((req or {}).get("early_stop", True))
    random_state = int((req or {}).get("random_state", 42))

    if not label_col:
        return jsonify({"error": "missing_label"}), 400
    if clf_name not in {
        "svm",
        "logistic",
        "decision_tree",
        "random_forest",
        "neural_net",
    }:
        return jsonify({"error": "bad_classifier"}), 400
    if not (0 < test_frac < 1):
        return jsonify({"error": "bad_test_frac"}), 400

    dataset_id = session.get("dataset_id")
    ds = store.get(dataset_id) if dataset_id else None
    if ds is None:
        return jsonify({"error": "no_dataset"}), 404

    df = ds.df
    if label_col not in df.columns:
        return jsonify({"error": "label_not_found"}), 400

    # Build features (numeric only) and labels
    Xdf = df.select_dtypes(include=["number"]).copy()
    # drop common ID-like fields and the label if numeric
    drop_names = {"segmentID", "segment_ID", label_col}
    feat_cols = [c for c in Xdf.columns if str(c) not in drop_names]
    if not feat_cols:
        return jsonify({"error": "no_features"}), 400

    y_series = df[label_col]
    # Align frames and drop rows with NaNs across features or missing label
    work = pd.DataFrame({"__y": y_series})
    for c in feat_cols:
        work[c] = df[c]
    work = work.dropna(axis=0, how="any")
    if work.shape[0] < 2:
        return jsonify({"error": "not_enough_rows"}), 400

    # Filter classes with too few samples for stratified splits
    y_raw = work["__y"].astype(str)
    counts = y_raw.value_counts()
    # Need at least 2 per class in the final training set; ensure after test split there are >=2 per class in train
    min_required = int(np.ceil(2.0 / max(1e-6, (1.0 - test_frac))))
    keep_labels = set(counts[counts >= min_required].index.astype(str))
    excluded_labels = [str(k) for k in counts[counts < min_required].index.tolist()]
    # Subset to kept labels only
    work = work[y_raw.isin(keep_labels)]
    if work.shape[0] < 2 or len(keep_labels) < 2:
        return (
            jsonify(
                {
                    "error": "not_enough_classes",
                    "detail": "Too few samples per class after filtering",
                    "min_required_per_class": int(min_required),
                    "excluded_labels": excluded_labels,
                }
            ),
            400,
        )

    # Encode labels as integers with mapping
    y_raw = work["__y"].astype(str)
    classes_, y_enc = np.unique(y_raw, return_inverse=True)
    X = work[feat_cols].to_numpy(dtype=float)
    y = y_enc.astype(int)

    # Stratified split
    n_splits = 1
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_frac, random_state=random_state
    )
    try:
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError as e:
        # Edge case: still too few per class; relax by requiring min 3 and refilter
        min_required = max(min_required, 3)
        keep_labels = set(
            pd.Series(y_raw)
            .value_counts()[lambda s: s >= min_required]
            .index.astype(str)
        )
        work2 = work[y_raw.isin(keep_labels)]
        if work2.shape[0] < 2 or len(keep_labels) < 2:
            return jsonify({"error": "stratify_failed", "detail": str(e)}), 400
        y_raw2 = work2["__y"].astype(str)
        classes_, y_enc = np.unique(y_raw2, return_inverse=True)
        X = work2[feat_cols].to_numpy(dtype=float)
        y = y_enc.astype(int)
        train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Further split a validation set from training for early stopping
    # Use 20% of training as validation, stratified
    val_frac = min(0.2, max(0.1, 0.2))
    # Validation split (stratified if feasible)
    try:
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac, random_state=random_state
        )
        tr_idx, val_idx = next(sss2.split(X_train, y_train))
    except ValueError:
        # Fallback: simple shuffle split without stratification
        from sklearn.model_selection import ShuffleSplit

        ss = ShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
        tr_idx, val_idx = next(ss.split(X_train, y_train))
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    # Initialize classifier
    n_classes = int(len(classes_))
    iters = []
    train_err = []
    val_err = []
    best_val = float("inf")
    best_iter = 0
    no_improve = 0
    stopped_early = False

    def record(i, model):
        iters.append(int(i))
        pred_tr = model.predict(X_tr)
        pred_val = model.predict(X_val)
        tr_err = 1.0 - float(accuracy_score(y_tr, pred_tr))
        vl_err = 1.0 - float(accuracy_score(y_val, pred_val))
        train_err.append(tr_err)
        val_err.append(vl_err)
        return tr_err, vl_err

    if clf_name == "svm":
        clf = SGDClassifier(loss="hinge", random_state=random_state)
        # Use partial_fit for epochs
        clf.partial_fit(X_tr[:1], y_tr[:1], classes=np.arange(n_classes))  # initialize
        for i in range(1, max_iters + 1):
            clf.partial_fit(X_tr, y_tr)
            _, vl = record(i, clf)
            if vl + 1e-9 < best_val:
                best_val = vl
                best_iter = i
                no_improve = 0
            else:
                no_improve += 1
                if early_stop and no_improve >= patience:
                    stopped_early = True
                    break
        # Re-train to best_iter for clean state
        clf_best = SGDClassifier(loss="hinge", random_state=random_state)
        clf_best.partial_fit(X_tr[:1], y_tr[:1], classes=np.arange(n_classes))
        for _ in range(best_iter):
            clf_best.partial_fit(X_tr, y_tr)
        final_model = clf_best
    elif clf_name == "logistic":
        clf = SGDClassifier(loss="log_loss", random_state=random_state)
        clf.partial_fit(X_tr[:1], y_tr[:1], classes=np.arange(n_classes))
        for i in range(1, max_iters + 1):
            clf.partial_fit(X_tr, y_tr)
            _, vl = record(i, clf)
            if vl + 1e-9 < best_val:
                best_val = vl
                best_iter = i
                no_improve = 0
            else:
                no_improve += 1
                if early_stop and no_improve >= patience:
                    stopped_early = True
                    break
        clf_best = SGDClassifier(loss="log_loss", random_state=random_state)
        clf_best.partial_fit(X_tr[:1], y_tr[:1], classes=np.arange(n_classes))
        for _ in range(best_iter):
            clf_best.partial_fit(X_tr, y_tr)
        final_model = clf_best
    elif clf_name == "neural_net":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            random_state=random_state,
            max_iter=1,
            warm_start=True,
        )
        for i in range(1, max_iters + 1):
            clf.fit(X_tr, y_tr)
            _, vl = record(i, clf)
            if vl + 1e-9 < best_val:
                best_val = vl
                best_iter = i
                no_improve = 0
            else:
                no_improve += 1
                if early_stop and no_improve >= patience:
                    stopped_early = True
                    break
        # Retrain with best_iter epochs
        final_model = MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            random_state=random_state,
            max_iter=1,
            warm_start=True,
        )
        for _ in range(best_iter):
            final_model.fit(X_tr, y_tr)
    elif clf_name == "random_forest":
        step = max(5, int(max_iters // 5) * 2)  # add trees in chunks
        n_estimators = 0
        clf = RandomForestClassifier(
            n_estimators=0, warm_start=True, random_state=random_state
        )
        for i in range(1, max_iters + 1):
            n_estimators += step
            clf.set_params(n_estimators=n_estimators)
            clf.fit(X_tr, y_tr)
            _, vl = record(i, clf)
            if vl + 1e-9 < best_val:
                best_val = vl
                best_iter = i
                no_improve = 0
            else:
                no_improve += 1
                if early_stop and no_improve >= patience:
                    stopped_early = True
                    break
        # Retrain to best_iter
        final_model = RandomForestClassifier(
            n_estimators=best_iter * step, random_state=random_state
        )
        final_model.fit(X_tr, y_tr)
    elif clf_name == "decision_tree":
        clf = DecisionTreeClassifier(random_state=random_state)
        clf.fit(X_tr, y_tr)
        record(1, clf)
        best_iter = 1
        final_model = clf
    # Evaluate on test with the final_model
    test_pred = final_model.predict(X_test)
    test_acc = float(accuracy_score(y_test, test_pred))
    cm = confusion_matrix(y_test, test_pred, labels=np.arange(n_classes)).tolist()

    return jsonify(
        {
            "classes": [str(c) for c in classes_],
            "label": str(label_col),
            "features": [str(c) for c in feat_cols],
            "min_required_per_class": int(min_required),
            "excluded_labels": excluded_labels,
            "history": {
                "iter": iters,
                "train_error": train_err,
                "val_error": val_err,
            },
            "best_iteration": int(best_iter),
            "test_accuracy": test_acc,
            "confusion_matrix": cm,
            "classifier": clf_name,
            "stopped_early": stopped_early,
        }
    )


@bp.get("/api/dr")
def api_dimred():
    """2D dimensionality reduction via t-SNE or UMAP, with optional clustering on the 2D embedding.

    Query params:
    - method: 'tsne' or 'umap'
    - mode: 'all' (all numeric z-scored features) or 'pcs'
    - n_pcs: integer > 0 (used only when mode='pcs')
    - color: optional column name for coloring
    """
    method = (request.args.get("method") or "").lower()
    mode = (request.args.get("mode") or "all").lower()
    try:
        n_pcs = int(request.args.get("n_pcs", "10"))
    except Exception:
        n_pcs = 10
    color_col = request.args.get("color")
    # Clustering params
    cluster = (request.args.get("cluster") or "none").lower()
    try:
        kmeans_k = int(request.args.get("kmeans_k", "5"))
    except Exception:
        kmeans_k = 5
    try:
        dbscan_eps = float(request.args.get("dbscan_eps", "0.5"))
    except Exception:
        dbscan_eps = 0.5
    try:
        dbscan_min_samples = int(request.args.get("dbscan_min_samples", "5"))
    except Exception:
        dbscan_min_samples = 5
    try:
        agglom_k = int(request.args.get("agglom_k", "5"))
    except Exception:
        agglom_k = 5
    agglom_linkage = (request.args.get("agglom_linkage") or "ward").lower()
    try:
        hdbscan_min_cluster_size = int(
            request.args.get("hdbscan_min_cluster_size", "10")
        )
    except Exception:
        hdbscan_min_cluster_size = 10
    try:
        hdbscan_min_samples = int(request.args.get("hdbscan_min_samples", "5"))
    except Exception:
        hdbscan_min_samples = 5

    if method not in {"tsne", "umap"}:
        return jsonify({"error": "bad_method"}), 400
    if mode not in {"all", "pcs"}:
        return jsonify({"error": "bad_mode"}), 400

    dataset_id = session.get("dataset_id")
    ds = store.get(dataset_id) if dataset_id else None
    if ds is None:
        return jsonify({"error": "no_dataset"}), 404
    df = ds.df

    import numpy as np
    import pandas as pd

    # Build numeric feature matrix, excluding common ID-like columns
    num_df = df.select_dtypes(include=["number"]).copy()
    drop_names = {"segmentID", "segment_ID"}
    feat_cols = [c for c in num_df.columns if str(c) not in drop_names]
    num_df = num_df[feat_cols]
    if num_df.shape[1] < 1:
        return jsonify({"error": "no_numeric_columns"}), 400

    # Row mask: drop rows with any NaNs in features
    clean = num_df.dropna(axis=0, how="any")
    if clean.shape[0] < 2:
        return jsonify({"error": "not_enough_rows", "rows": int(clean.shape[0])}), 400

    # Keep parallel slices for IDs and optional color column
    # Reindex original df to the cleaned index to align rows
    idx = clean.index
    id_source = None
    for cname in ("segmentID", "cell_name"):
        if cname in df.columns:
            id_source = cname
            break
    if id_source is not None:
        id_series = df.loc[idx, id_source]
    else:
        id_series = pd.Series([""] * len(idx), index=idx)
    if color_col and color_col in df.columns:
        color_series = df.loc[idx, color_col]
    else:
        color_series = None

    # Standardize features to z-scores; drop constant features
    X = clean.to_numpy(dtype=float)
    X = X - np.nanmean(X, axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1)
    keep_mask = np.isfinite(sd) & (sd > 0)
    if not np.any(keep_mask):
        return jsonify({"error": "no_variable_features"}), 400
    Xz = X[:, keep_mask] / sd[keep_mask]
    cols_kept = [str(c) for c, k in zip(clean.columns, keep_mask) if bool(k)]

    # Optional PCA preprocessing
    if mode == "pcs":
        try:
            # SVD on standardized data
            U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
            k = min(int(n_pcs), Vt.shape[0])
            if k < 1:
                k = 1
            comps = Vt[:k, :].T  # shape (p, k)
            Xdr = Xz @ comps  # (n, k)
            preproc = {"mode": "pcs", "n_pcs": int(k)}
        except Exception as e:
            return jsonify({"error": "pca_failed", "detail": str(e)}), 500
    else:
        Xdr = Xz
        preproc = {"mode": "all", "n_pcs": None}

    # Compute embedding
    n = Xdr.shape[0]
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE  # type: ignore
        except Exception:
            return jsonify({"error": "missing_dep_sklearn"}), 501
        # Choose a safe perplexity default
        try:
            per_arg = request.args.get("perplexity")
            if per_arg is not None:
                perplexity = float(per_arg)
            else:
                perplexity = max(5.0, min(30.0, (n - 1) / 3.0))
            # sklearn requires perplexity < n_samples
            perplexity = min(perplexity, max(1.0, n - 1.0 - 1e-6))
        except Exception:
            perplexity = max(5.0, min(30.0, (n - 1) / 3.0))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=42,
        )
        Y = tsne.fit_transform(Xdr)
        method_meta = {"method": "tsne", "perplexity": float(perplexity)}
    else:  # umap
        try:
            import umap  # type: ignore
        except Exception:
            return jsonify({"error": "missing_dep_umap"}), 501
        n_neighbors = int(request.args.get("n_neighbors", 15))
        min_dist = float(request.args.get("min_dist", 0.1))
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42
        )
        Y = reducer.fit_transform(Xdr)
        method_meta = {
            "method": "umap",
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
        }

    xs = Y[:, 0].astype(float).tolist()
    ys = Y[:, 1].astype(float).tolist()
    ids = [str(v) for v in id_series.tolist()]
    if color_series is not None:
        # Preserve raw values; frontend decides how to map
        colors = color_series.tolist()
        # Ensure JSON-serializable
        colors = [
            (
                None
                if pd.isna(v)
                else (float(v) if isinstance(v, (int, float)) else str(v))
            )
            for v in colors
        ]
    else:
        colors = None

    # Optional clustering on the 2D embedding
    cluster_labels = None
    cluster_algo = None
    n_clusters = None
    n_noise = None
    silhouette = None
    if cluster in {"kmeans", "dbscan", "agglomerative", "hdbscan"}:
        try:
            if cluster == "kmeans":
                from sklearn.cluster import KMeans  # type: ignore

                k = int(max(2, kmeans_k))
                model = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = model.fit_predict(Y)
                cluster_algo = f"kmeans{k}"
            elif cluster == "dbscan":
                from sklearn.cluster import DBSCAN  # type: ignore

                model = DBSCAN(
                    eps=float(dbscan_eps), min_samples=int(dbscan_min_samples)
                )
                labels = model.fit_predict(Y)
                cluster_algo = f"dbscan"
            elif cluster == "agglomerative":
                from sklearn.cluster import AgglomerativeClustering  # type: ignore

                k = int(max(2, agglom_k))
                link = (
                    agglom_linkage
                    if agglom_linkage in {"ward", "average", "complete"}
                    else "ward"
                )
                # Ward requires euclidean and more than 1 cluster
                if link == "ward" and k < 2:
                    k = 2
                model = AgglomerativeClustering(n_clusters=k, linkage=link)
                labels = model.fit_predict(Y)
                cluster_algo = f"agglomerative_{link}{k}"
            else:  # hdbscan
                try:
                    import hdbscan  # type: ignore
                except Exception:
                    return jsonify({"error": "missing_dep_hdbscan"}), 501
                mcs = int(max(2, hdbscan_min_cluster_size))
                ms = int(max(1, hdbscan_min_samples))
                model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
                labels = model.fit_predict(Y)
                cluster_algo = "hdbscan"
            cluster_labels = [int(v) for v in labels.tolist()]
            # Cluster stats
            try:
                import numpy as np

                labels_arr = np.asarray(cluster_labels)
                noise_mask = labels_arr == -1
                n_noise = int(noise_mask.sum()) if noise_mask.any() else 0
                # Exclude noise for silhouette and cluster counting
                valid_mask = (
                    ~noise_mask
                    if noise_mask.any()
                    else np.ones_like(labels_arr, dtype=bool)
                )
                unique = np.unique(labels_arr[valid_mask])
                n_clusters = int(len(unique))
                if n_clusters >= 2 and int(valid_mask.sum()) >= 2:
                    try:
                        from sklearn.metrics import silhouette_score  # type: ignore

                        silhouette = float(
                            silhouette_score(
                                Y[valid_mask],
                                labels_arr[valid_mask],
                                metric="euclidean",
                            )
                        )
                    except Exception:
                        silhouette = None
            except Exception:
                n_clusters = None
                n_noise = None
                silhouette = None
        except Exception:
            cluster_labels = None
            cluster_algo = None
            n_clusters = None
            n_noise = None
            silhouette = None

    return jsonify(
        {
            "n_points": int(len(xs)),
            "x": xs,
            "y": ys,
            "id": ids,
            "color_values": colors,
            "color_column": str(color_col) if color_series is not None else None,
            "features_used": cols_kept,
            "preprocess": preproc,
            **method_meta,
            "cluster_labels": cluster_labels,
            "cluster_algorithm": cluster_algo,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette": silhouette,
        }
    )


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
        groups_out.append(
            {
                "name": str(gval),
                "count": len(vals),
                "mean": mean_val,
                "values": vals,
                "points": points,
            }
        )

    unit = ds.units.get(value, "") if isinstance(ds.units, dict) else ""
    group_unit = ds.units.get(group, "") if isinstance(ds.units, dict) else ""
    return jsonify(
        {
            "value": value,
            "group": group,
            "unit": unit,
            "group_unit": group_unit,
            "groups": groups_out,
        }
    )


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
                for xi, yi, pid in zip(
                    gdf["_x"].tolist(), gdf["_y"].tolist(), gdf["_id"].tolist()
                ):
                    pts.append({"x": float(xi), "y": float(yi), "id": str(pid)})
            else:
                for xi, yi in zip(gdf["_x"].tolist(), gdf["_y"].tolist()):
                    pts.append({"x": float(xi), "y": float(yi), "id": ""})
            groups_out.append(
                {
                    "name": str(gval),
                    "points": pts,
                    "mean": compute_group_stats(gdf),
                }
            )
    else:
        pts = []
        if "_id" in tmp.columns:
            for xi, yi, pid in zip(
                tmp["_x"].tolist(), tmp["_y"].tolist(), tmp["_id"].tolist()
            ):
                pts.append({"x": float(xi), "y": float(yi), "id": str(pid)})
        else:
            for xi, yi in zip(tmp["_x"].tolist(), tmp["_y"].tolist()):
                pts.append({"x": float(xi), "y": float(yi), "id": ""})
        groups_out.append(
            {
                "name": "All",
                "points": pts,
                "mean": compute_group_stats(tmp),
            }
        )

    x_unit = ds.units.get(x, "") if isinstance(ds.units, dict) else ""
    y_unit = ds.units.get(y, "") if isinstance(ds.units, dict) else ""
    return jsonify(
        {
            "x": x,
            "y": y,
            "x_unit": x_unit,
            "y_unit": y_unit,
            "group": group if group in df.columns else None,
            "groups": groups_out,
        }
    )


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
        allowed_list = ", ".join(
            sorted(current_app.config.get("ALLOWED_EXTENSIONS", []))
        )
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


@bp.get("/dj_test")
def connect_to_dj():
    dj.config["database.host"] = "vfsmdatajoint01.fsm.northwestern.edu"
    dj.config["database.user"] = "Trung"
    dj.config["database.password"] = "Trung_temppwd"

    c = dj.conn()
    return str(c.is_connected)

import os
import inspect
import json
import dotenv
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
from numbers import Number
import re
from typing import Optional
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
dotenv.load_dotenv()


def _line_by_row_prefs_path() -> str:
    os.makedirs(current_app.instance_path, exist_ok=True)
    return os.path.join(current_app.instance_path, "line_by_row_presets.json")


def _load_line_by_row_prefs_store():
    path = _line_by_row_prefs_path()
    if not os.path.exists(path):
        return {"presets": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"presets": []}
    if not isinstance(data, dict):
        return {"presets": []}
    presets = data.get("presets")
    if not isinstance(presets, list):
        presets = []
    return {"presets": presets}


def _save_line_by_row_prefs_store(data) -> None:
    path = _line_by_row_prefs_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _current_preset_scope():
    dataset_id = request.args.get("dataset_id") or session.get("dataset_id") or ""
    dataset_label = session.get("dataset_label") or dataset_id or ""
    dataset_source = session.get("dataset_source") or "dataset"
    scope_key = f"{dataset_source}::{dataset_label or dataset_id or '__default__'}"
    return {
        "scope_key": scope_key,
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "dataset_source": dataset_source,
    }


def _plot_prefs_items(plot_type: str):
    scope = _current_preset_scope()
    store_data = _load_line_by_row_prefs_store()
    items = []
    for item in store_data.get("presets", []):
        if not isinstance(item, dict):
            continue
        item_plot_type = str(item.get("plot_type") or "line_by_row")
        if item_plot_type != plot_type:
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        items.append(item)
    return scope, store_data, items


def _plot_prefs_list_response(plot_type: str):
    scope, _, items = _plot_prefs_items(plot_type)
    presets = []
    seen = set()
    for item in items:
        name = str(item.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        presets.append(
            {
                "name": name,
                "dataset_label": str(item.get("dataset_label") or ""),
                "dataset_source": str(item.get("dataset_source") or ""),
            }
        )
    presets.sort(key=lambda item: item["name"].lower())
    return jsonify(
        {
            "dataset_label": scope["dataset_label"],
            "dataset_source": scope["dataset_source"],
            "presets": presets,
        }
    )


def _plot_prefs_get_response(plot_type: str, name: str):
    _, _, items = _plot_prefs_items(plot_type)
    wanted = str(name or "").strip()
    if not wanted:
        return jsonify({"error": "missing_name"}), 400
    for item in items:
        if str(item.get("name") or "").strip() != wanted:
            continue
        prefs = item.get("preferences")
        if not isinstance(prefs, dict):
            prefs = {}
        return jsonify({"name": wanted, "preferences": prefs})
    return jsonify({"error": "not_found"}), 404


def _plot_prefs_save_response(plot_type: str):
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name") or "").strip()
    preferences = payload.get("preferences")
    if not name:
        return jsonify({"error": "missing_name"}), 400
    if not isinstance(preferences, dict):
        return jsonify({"error": "bad_preferences"}), 400

    scope = _current_preset_scope()
    store_data = _load_line_by_row_prefs_store()
    kept = []
    for item in store_data.get("presets", []):
        if not isinstance(item, dict):
            continue
        item_plot_type = str(item.get("plot_type") or "line_by_row")
        same_name = str(item.get("name") or "").strip() == name
        if item_plot_type == plot_type and same_name:
            continue
        kept.append(item)
    kept.append(
        {
            "plot_type": plot_type,
            "scope_key": scope["scope_key"],
            "dataset_label": scope["dataset_label"],
            "dataset_source": scope["dataset_source"],
            "name": name,
            "preferences": preferences,
        }
    )
    store_data["presets"] = kept
    _save_line_by_row_prefs_store(store_data)
    return jsonify({"saved": True, "name": name})


def _plot_prefs_delete_response(plot_type: str, name: str):
    wanted = str(name or "").strip()
    if not wanted:
        return jsonify({"error": "missing_name"}), 400
    store_data = _load_line_by_row_prefs_store()
    kept = []
    removed = False
    for item in store_data.get("presets", []):
        if not isinstance(item, dict):
            continue
        item_plot_type = str(item.get("plot_type") or "line_by_row")
        same_name = str(item.get("name") or "").strip() == wanted
        if item_plot_type == plot_type and same_name:
            removed = True
            continue
        kept.append(item)
    store_data["presets"] = kept
    if removed:
        _save_line_by_row_prefs_store(store_data)
    return jsonify({"deleted": removed, "name": wanted})


@bp.before_app_request
def _ensure_clean_session():
    if not session.get("_tabulator_session_ready"):
        session["_tabulator_session_ready"] = True
        session.pop("dataset_id", None)
        session.pop("dataset_label", None)
        session.pop("dataset_source", None)


def _get_dataset_from_request():
    """Return (dataset, dataset_id) from query param or session."""
    dataset_id = request.args.get("dataset_id") or session.get("dataset_id")
    ds = store.get(dataset_id) if dataset_id else None
    return ds, dataset_id


def _allowed_file(filename: str) -> bool:
    allowed = current_app.config.get("ALLOWED_EXTENSIONS", set())
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def _get_datajoint_credentials():
    host = os.getenv("DJ_HOST")
    user = os.getenv("DJ_USER")
    password = os.getenv("DJ_PASSWORD")
    missing = [name for name, value in (("DJ_HOST", host), ("DJ_USER", user), ("DJ_PASSWORD", password)) if not value]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    return {
        "host": host,
        "user": user,
        "password": password,
    }


def _connect_datajoint():
    creds = _get_datajoint_credentials()
    return dj.conn(host=creds["host"], user=creds["user"], password=creds["password"])


def _get_datajoint_schema() -> str:
    return os.getenv("DJ_SCHEMA") or "sln_results"


def _get_query_schema() -> str:
    return os.getenv("DJ_QUERY_SCHEMA") or "sln_lab"


def _create_virtual_module(conn, schema: str, alias: Optional[str] = None):
    module_name = alias or f"dj_{schema.replace('.', '_')}"
    return dj.create_virtual_module(module_name, schema, connection=conn)


def _ensure_relation_instance(obj):
    if inspect.isclass(obj):
        return obj()
    return obj


def _list_schema_relations(module):
    relations = []
    allowed_prefixes = ("Epoch", "Dataset", "Cell", "Animal", "Experiment")
    for name in dir(module):
        if name.startswith("_"):
            continue
        if not name.startswith(allowed_prefixes):
            continue
        attr = getattr(module, name)
        try:
            relation_instance = _ensure_relation_instance(attr)
        except Exception:
            continue
        if not hasattr(relation_instance, "full_table_name"):
            continue
        if getattr(relation_instance, "is_part", False):
            continue
        full_table_name = getattr(relation_instance, "full_table_name", None)
        relations.append(
            {
                "name": name,
                "full_table": str(full_table_name) if full_table_name else None,
            }
        )
    relations.sort(key=lambda r: r["name"].lower())
    return relations


def _normalize_nested_value(value):
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_normalize_nested_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_nested_value(v) for k, v in value.items()}
    return value


def _sequence_shape(value, max_dims=2):
    shape = []
    current = value
    for _ in range(max_dims):
        if isinstance(current, np.ndarray):
            shape.extend(list(current.shape))
            break
        if isinstance(current, (list, tuple)):
            shape.append(len(current))
            if len(current) == 0:
                break
            current = current[0]
            continue
        break
    return " x ".join(str(s) for s in shape if s is not None) if shape else ""


def _preview_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (Number, np.number, np.bool_)):
        try:
            return value.item()
        except AttributeError:
            return value
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes {len(value)}>"
    if isinstance(value, (np.ndarray, list, tuple)):
        shape = _sequence_shape(value)
        if shape:
            return f"<array {shape}>"
        length = len(value) if hasattr(value, "__len__") else "?"
        return f"<array len={length}>"
    if isinstance(value, dict):
        return f"<dict keys={len(value)}>"
    return f"<{type(value).__name__}>"


def _is_simple_value(value):
    return isinstance(value, (str, Number, np.number, np.bool_, bool))


def _column_is_simple(series, check_limit=1000):
    checked = 0
    for val in series:
        if isinstance(val, np.ndarray) and val.size == 0:
            continue
        if isinstance(val, (np.ndarray, list, tuple, dict)):
            return False
        if pd.isna(val):
            continue
        if not _is_simple_value(val):
            return False
        checked += 1
        if checked >= check_limit:
            break
    return True


def _series_is_simple(series, check_limit=1000):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return _column_is_simple(series, check_limit=check_limit)


def _coerce_1d_numeric_array(value):
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    if not isinstance(value, np.ndarray):
        return None
    if value.ndim == 0:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    arr = np.ravel(arr)
    return arr


def _coerce_reducible_numeric_array(value):
    arr = _coerce_1d_numeric_array(value)
    if arr is not None:
        return arr, True
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None, False
    try:
        if pd.isna(value):
            return None, False
    except Exception:
        pass
    if isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
        return np.asarray([float(value)], dtype=float), False
    return None, False


def _series_is_numeric_vector(series, check_limit=250):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    checked = 0
    found_valid = False
    for val in series:
        if val is None:
            continue
        arr = _coerce_1d_numeric_array(val)
        if arr is None:
            try:
                if pd.isna(val):
                    continue
            except Exception:
                pass
            return False
        if arr.size == 0:
            continue
        found_valid = True
        checked += 1
        if checked >= check_limit:
            break
    return found_valid


def _series_supports_vector_reduction(series, check_limit=250):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    checked = 0
    found_array_like = False
    for val in series:
        arr, came_from_array = _coerce_reducible_numeric_array(val)
        if arr is None:
            continue
        if arr.size == 0:
            continue
        if came_from_array:
            found_array_like = True
        checked += 1
        if checked >= check_limit:
            break
    return found_array_like


def _json_simple_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray) and value.size == 0:
        return None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return str(value)


def _normalize_filter_specs(filter_cols, filter_vals, filter_ops=None):
    filter_cols = list(filter_cols or [])
    filter_vals = list(filter_vals or [])
    filter_ops = list(filter_ops or [])
    if len(filter_cols) != len(filter_vals):
        raise LoadError("bad_filters")
    if not filter_ops:
        filter_ops = ["include"] * len(filter_cols)
    elif len(filter_ops) != len(filter_cols):
        raise LoadError("bad_filters")
    normalized = []
    for col, val, op in zip(filter_cols, filter_vals, filter_ops):
        op = str(op or "include").strip().lower()
        if op not in {"include", "exclude"}:
            raise LoadError("bad_filter_op")
        normalized.append((col, val, op))
    return normalized


def _sanitize_dataset(dataset: DataSet):
    df = dataset.df
    keep = []
    drop = []
    for col in df.columns:
        series = df[col]
        if _column_is_simple(series):
            keep.append(col)
        else:
            drop.append(col)
    if keep:
        filtered_df = df[keep].copy()
    else:
        filtered_df = pd.DataFrame(index=df.index)
    new_units = {str(c): dataset.units.get(c, "") for c in keep}
    return DataSet(df=filtered_df, units=new_units), drop


def _get_primary_key_columns(relation):
    attrs = []
    heading = getattr(relation, "heading", None)
    if heading is not None:
        primary = getattr(heading, "primary_attributes", None)
        if primary and hasattr(primary, "keys"):
            attrs = list(primary.keys())
    if not attrs:
        pk = getattr(relation, "primary_key", None)
        if isinstance(pk, (list, tuple)):
            attrs = list(pk)
    return [str(a) for a in attrs if a]


def _extract_unknown_column(error_message: str):
    match = re.search(r"Unknown column '([^']+)'", error_message)
    if match:
        return match.group(1)
    return None


def _relation_columns(relation):
    try:
        heading = getattr(relation, "heading", None)
        attrs = getattr(heading, "attributes", None)
        if attrs and hasattr(attrs, "keys"):
            return [str(name) for name in attrs.keys()]
    except Exception:
        pass
    return []


def _relation_columns_with_types(relation):
    cols = []
    try:
        heading = getattr(relation, "heading", None)
        attrs = getattr(heading, "attributes", None)
        items = attrs.items() if attrs and hasattr(attrs, "items") else []
        for name, attr in items:
            if isinstance(attr, dict):
                typ = attr.get("type") or attr.get("dtype") or ""
            else:
                typ = getattr(attr, "type", "") or getattr(attr, "dtype", "")
            cols.append((str(name), str(typ)))
    except Exception:
        pass
    return cols


def _column_type_is_blob(type_name: str) -> bool:
    t = (type_name or "").lower()
    return "blob" in t


def _execute_query(conn, sql: str, *, as_dict: bool = False):
    """Run a SQL query without triggering Python %-formatting on percent signs."""
    try:
        return conn.query(sql, args=None, as_dict=as_dict)
    except TypeError as exc:
        msg = str(exc).lower()
        # Older DataJoint versions may not accept args kwarg
        if "unexpected keyword argument 'args'" in msg:
            return conn.query(sql, None, as_dict)
        # Queries containing % characters (e.g., LIKE "%foo%") can be treated as
        # format strings if args is an empty tuple; escape and retry once.
        if "not enough arguments for format string" in msg:
            escaped_sql = sql.replace("%", "%%")
            try:
                return conn.query(escaped_sql, args=None, as_dict=as_dict)
            except TypeError as exc2:
                if "unexpected keyword argument 'args'" in str(exc2).lower():
                    return conn.query(escaped_sql, None, as_dict)
                raise
        raise


def _select_relation_columns(relation):
    cols_with_types = _relation_columns_with_types(relation)
    primary = set(_get_primary_key_columns(relation))
    selected = []
    skipped_blob = []
    for name, typ in cols_with_types:
        if name in primary:
            selected.append(name)
            continue
        if _column_type_is_blob(typ):
            skipped_blob.append(name)
            continue
        selected.append(name)
    if not selected:
        selected = [name for name, _ in cols_with_types] or _relation_columns(relation)
    return selected, skipped_blob


def _relation_to_dataframe(relation, conn, query_sql: Optional[str] = None):
    full_table = getattr(relation, "full_table_name", None)
    if not full_table:
        raise ValueError("Relation is missing table metadata")
    try:
        schema, table = full_table.split(".", 1)
    except ValueError:
        raise ValueError(f"Unexpected table identifier: {full_table}")
    schema = schema.strip("`")
    table = table.strip("`")
    selected_cols, skipped_blob_cols = _select_relation_columns(relation)
    if not selected_cols:
        raise ValueError("No columns available to load from relation")
    col_sql = ", ".join(f"`{schema}`.`{table}`.`{c}`" for c in selected_cols)
    base_sql = f"SELECT {col_sql} FROM `{schema}`.`{table}`"
    base_rows = _execute_query(conn, base_sql, as_dict=True) or []
    base_df = pd.DataFrame(base_rows)
    base_cols = base_df.columns.tolist()
    # Fallback to heading-defined columns if DataFrame empty
    if not base_cols:
        base_cols = _relation_columns(relation)
    restriction_cols = []
    if query_sql:
        q = query_sql.strip().rstrip(";")
        key_cols = _get_primary_key_columns(relation)
        if not key_cols:
            raise ValueError("Cannot determine primary key columns for relation; cannot apply query.")
        if q.upper().startswith("SELECT"):
            restriction_sql = q
        else:
            restriction_sql = f"SELECT * FROM {q}"
        restriction_rows = _execute_query(conn, restriction_sql, as_dict=True) or []
        restriction_df = pd.DataFrame(restriction_rows)
        if restriction_df.empty:
            base_df = base_df.iloc[0:0]
        else:
            available_keys = [k for k in key_cols if k in restriction_df.columns]
            if not available_keys:
                raise ValueError(
                    "Query must include at least one primary key column present in the table: "
                    + ", ".join(key_cols)
                )
            restriction_cols = [c for c in restriction_df.columns if c not in base_cols]
            payload_cols = available_keys + restriction_cols
            payload = restriction_df[payload_cols].drop_duplicates()
            base_df = base_df.merge(payload, on=available_keys, how="inner")
    rows = base_df.to_dict(orient="records")
    normalized_rows = []
    for row in rows:
        normalized_rows.append({k: _normalize_nested_value(v) for k, v in row.items()})
    df = pd.DataFrame(normalized_rows) if normalized_rows else pd.DataFrame()
    return df, restriction_cols, skipped_blob_cols


def _relation_attribute_descriptions(relation):
    descriptions = {}
    try:
        heading = getattr(relation, "heading", None)
        attributes = getattr(heading, "attributes", None) if heading else None
        items = attributes.items() if attributes and hasattr(attributes, "items") else []
        for name, attr in items:
            comment = ""
            if isinstance(attr, dict):
                comment = attr.get("comment") or attr.get("description") or ""
            else:
                comment = getattr(attr, "comment", "") or getattr(attr, "description", "")
            if comment:
                descriptions[str(name)] = str(comment)
    except Exception:
        pass
    return descriptions


def _load_query_table(conn):
    schema = _get_query_schema()
    module = _create_virtual_module(conn, schema)
    if not hasattr(module, "Query"):
        raise ValueError(f"Query table not found in schema {schema}")
    relation_obj = getattr(module, "Query")
    relation = _ensure_relation_instance(relation_obj)
    df, _, _ = _relation_to_dataframe(relation, conn)
    return df


def _get_query_sql(conn, query_name: str):
    df = _load_query_table(conn)
    if df.empty:
        raise ValueError("No queries defined.")
    if "query_name" not in df.columns or "sql_query" not in df.columns:
        raise ValueError("Query table does not have expected columns.")
    match = df[df["query_name"] == query_name]
    if match.empty:
        raise ValueError(f"Query {query_name} not found.")
    row = match.iloc[0]
    sql = str(row.get("sql_query") or "").strip()
    if not sql:
        raise ValueError(f"Query {query_name} has no SQL defined.")
    return sql


@bp.route("/", methods=["GET"])  # Home: upload UI
def index():
    dataset_id = session.get("dataset_id")
    dataset_label = session.get("dataset_label")
    dataset_source = session.get("dataset_source")
    ds = store.get(dataset_id) if dataset_id else None
    if dataset_id and ds is None:
        # Clear stale session references if the in-memory dataset was lost (e.g., after app reload)
        session.pop("dataset_id", None)
        session.pop("dataset_label", None)
        session.pop("dataset_source", None)
        dataset_id = None
        dataset_label = None
        dataset_source = None
    preview_html = None
    info = None
    units_items = None
    if ds is not None:
        try:
            preview_df = ds.df.head(10).copy()
            preview_df = preview_df.map(_preview_value)
            preview_html = preview_df.to_html(classes=["preview"], border=0)
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
        dataset_label=dataset_label or dataset_id,
        dataset_source=dataset_source,
        info=info,
        preview_html=preview_html,
        units_items=units_items,
        db_host=os.getenv("DJ_HOST"),
        db_user=os.getenv("DJ_USER"),
        db_schema=os.getenv("DJ_SCHEMA"),
    )


@bp.post("/api/db/load")
def api_db_load():
    schema = _get_datajoint_schema()
    payload = request.get_json(silent=True) or {}
    table = payload.get("table") or request.form.get("table")
    if not table:
        return jsonify({"error": "missing_table", "detail": "Select a table to load."}), 400
    table = str(table).strip()
    if not table:
        return jsonify({"error": "missing_table", "detail": "Select a table to load."}), 400
    query_name = payload.get("query") or payload.get("query_name") or request.form.get("query_name")
    if query_name:
        query_name = str(query_name).strip()
        if not query_name:
            query_name = None
    skipped_blob_cols = []
    try:
        conn = _connect_datajoint()
    except ValueError as exc:
        return jsonify({"error": "missing_credentials", "detail": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "connection_failed", "detail": str(exc)}), 502
    try:
        query_sql = None
        if query_name:
            try:
                query_sql = _get_query_sql(conn, query_name)
            except Exception as exc:
                return jsonify({"error": "query_failed", "detail": str(exc)}), 400
        module = _create_virtual_module(conn, schema)
        if not hasattr(module, table):
            return jsonify({"error": "unknown_table", "detail": f"{table} not found in {schema}."}), 404
        relation_obj = getattr(module, table)
        try:
            relation = _ensure_relation_instance(relation_obj)
        except Exception as exc:
            return jsonify({"error": "unloadable_table", "detail": f"{table} cannot be instantiated: {exc}"}), 400
        if not hasattr(relation, "full_table_name"):
            return jsonify({"error": "unloadable_table", "detail": f"{table} has no table metadata."}), 400
        df, extra_cols, skipped_blob_cols = _relation_to_dataframe(relation, conn, query_sql=query_sql)
        if hasattr(df, "reset_index"):
            df = df.reset_index()
        if not isinstance(df, pd.DataFrame):
            return jsonify({"error": "bad_format", "detail": f"{table} did not return a DataFrame."}), 500
    except Exception as exc:
        return jsonify({"error": "load_failed", "detail": str(exc)}), 502
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if df.empty:
        return jsonify({"error": "empty_table", "detail": f"{table} returned no rows."}), 400
    attr_descriptions = _relation_attribute_descriptions(relation)
    units = {str(col): attr_descriptions.get(str(col), "") for col in df.columns}
    for col in extra_cols or []:
        units.setdefault(str(col), "")
    ds = DataSet(df=df, units=units)
    dataset_id = store.put(ds)
    session["dataset_id"] = dataset_id
    session["dataset_label"] = table
    session["dataset_source"] = f"{schema}.{table}"
    note = f"Loaded {table} from {schema}: {ds.df.shape[0]} rows, {ds.df.shape[1]} cols."
    if skipped_blob_cols:
        note += f" Skipped {len(skipped_blob_cols)} blob column(s)."
    flash(note)
    return jsonify(
        {
            "message": note,
            "dataset_id": dataset_id,
            "rows": int(ds.df.shape[0]),
            "cols": int(ds.df.shape[1]),
            "skipped_blob_columns": [str(c) for c in skipped_blob_cols],
            "dropped_columns": [],
            "redirect": url_for("main.index", _anchor="dataset-card"),
        }
    )


@bp.post("/api/clear")
def api_clear_dataset():
    dataset_id = session.pop("dataset_id", None)
    session.pop("dataset_label", None)
    session.pop("dataset_source", None)
    removed = False
    if dataset_id:
        removed = store.remove(dataset_id)
    flash("Cleared loaded data.")
    return jsonify({"cleared": bool(dataset_id), "removed": removed})


@bp.get("/api/db/tables")
def api_db_tables():
    schema = _get_datajoint_schema()
    try:
        conn = _connect_datajoint()
    except ValueError as exc:
        return jsonify({"error": "missing_credentials", "detail": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "connection_failed", "detail": str(exc)}), 502
    try:
        module = _create_virtual_module(conn, schema)
        tables = _list_schema_relations(module)
    except Exception as exc:
        return jsonify({"error": "list_failed", "detail": str(exc)}), 502
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return jsonify({"schema": schema, "tables": tables})


@bp.get("/api/db/queries")
def api_db_queries():
    schema = _get_query_schema()
    try:
        conn = _connect_datajoint()
    except ValueError as exc:
        return jsonify({"error": "missing_credentials", "detail": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "connection_failed", "detail": str(exc)}), 502
    try:
        df = _load_query_table(conn)
    except Exception as exc:
        return jsonify({"error": "query_fetch_failed", "detail": str(exc)}), 502
    finally:
        try:
            conn.close()
        except Exception:
            pass
    user_queries = []
    project_queries = []
    if not df.empty:
        for _, row in df.iterrows():
            name = str(row.get("query_name") or "").strip()
            if not name:
                continue
            user_name = str(row.get("user_name") or "").strip()
            project_name = str(row.get("project_name") or "").strip()
            if user_name:
                user_queries.append(
                    {
                        "query_name": name,
                        "user_name": user_name,
                        "label": f"{user_name} – {name}",
                    }
                )
            if project_name:
                project_queries.append(
                    {
                        "query_name": name,
                        "project_name": project_name,
                        "label": f"{project_name} – {name}",
                    }
                )
    user_queries.sort(key=lambda x: (x.get("user_name", "").lower(), x["query_name"].lower()))
    project_queries.sort(key=lambda x: (x.get("project_name", "").lower(), x["query_name"].lower()))
    return jsonify(
        {
            "schema": schema,
            "user_queries": user_queries,
            "project_queries": project_queries,
        }
    )


@bp.route("/api/columns")
def api_columns():
    ds, dataset_id = _get_dataset_from_request()
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
        is_simple = _series_is_simple(df[c])
        is_vector = _series_is_numeric_vector(df[c])
        is_reducible_vector = _series_supports_vector_reduction(df[c])
        cols.append(
            {
                "name": str(c),
                "is_numeric": bool(is_simple and is_numeric_dtype(df[c])),
                "is_simple": bool(is_simple),
                "is_vector": bool(is_vector),
                "is_reducible_vector": bool(is_reducible_vector),
            }
        )
    return jsonify({"columns": cols})


@bp.get("/api/column_values")
def api_column_values():
    column = request.args.get("column")
    if not column:
        return jsonify({"error": "missing_column"}), 400
    ds, dataset_id = _get_dataset_from_request()
    if ds is None:
        return jsonify({"error": "no_dataset"}), 404
    df = ds.df
    if column not in df.columns:
        return jsonify({"error": "bad_column"}), 400
    if not _series_is_simple(df[column]):
        return jsonify({"error": "bad_column_type"}), 400

    series = df[column]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    values = []
    seen = set()
    for raw in series.tolist():
        val = _json_simple_value(raw)
        key = "" if val is None else str(val)
        if key in seen:
            continue
        seen.add(key)
        values.append({"value": key, "label": key if key else "(blank)"})
    values.sort(key=lambda item: item["label"])
    return jsonify({"column": column, "values": values})


@bp.get("/api/plot_prefs/line_by_row")
def api_line_by_row_prefs_list():
    return _plot_prefs_list_response("line_by_row")


@bp.get("/api/plot_prefs/line_by_row/<name>")
def api_line_by_row_prefs_get(name):
    return _plot_prefs_get_response("line_by_row", name)


@bp.post("/api/plot_prefs/line_by_row")
def api_line_by_row_prefs_save():
    return _plot_prefs_save_response("line_by_row")


@bp.delete("/api/plot_prefs/line_by_row/<name>")
def api_line_by_row_prefs_delete(name):
    return _plot_prefs_delete_response("line_by_row", name)


@bp.get("/api/plot_prefs/bar")
def api_bar_prefs_list():
    return _plot_prefs_list_response("bar")


@bp.get("/api/plot_prefs/bar/<name>")
def api_bar_prefs_get(name):
    return _plot_prefs_get_response("bar", name)


@bp.post("/api/plot_prefs/bar")
def api_bar_prefs_save():
    return _plot_prefs_save_response("bar")


@bp.delete("/api/plot_prefs/bar/<name>")
def api_bar_prefs_delete(name):
    return _plot_prefs_delete_response("bar", name)


@bp.get("/api/pca")
def api_pca():
    """Run PCA on all numeric columns and return explained variance ratios.

    - Uses centered data (mean subtraction per feature).
    - Drops rows with NaNs across selected numeric columns.
    - Returns explained_variance_ratio per component and cumulative.
    """
    ds, dataset_id = _get_dataset_from_request()
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

    ds, dataset_id = _get_dataset_from_request()
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
        if not _series_is_simple(df[color_col]):
            return jsonify({"error": "bad_color_column_type"}), 400
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

    ds, dataset_id = _get_dataset_from_request()
    if ds is None:
        return jsonify({"error": "no_dataset"}), 404

    df = ds.df
    if label_col not in df.columns:
        return jsonify({"error": "label_not_found"}), 400
    if not _series_is_simple(df[label_col]):
        return jsonify({"error": "bad_label_column_type"}), 400

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

    ds, dataset_id = _get_dataset_from_request()
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
        if not _series_is_simple(df[color_col]):
            return jsonify({"error": "bad_color_column_type"}), 400
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


def _build_bar_plot_payload(ds, value: str, group: str, filter_cols, filter_vals, filter_ops=None):
    def _parse_value_spec(raw_value: str):
        if raw_value.endswith("::mean"):
            return raw_value[:-6], "mean"
        if raw_value.endswith("::median"):
            return raw_value[:-8], "median"
        return raw_value, None

    df = ds.df
    value_col, reduce_mode = _parse_value_spec(value)
    if value_col not in df.columns or group not in df.columns:
        raise LoadError("bad_columns")
    if not _series_is_simple(df[group]):
        raise LoadError("bad_group_column_type")
    if reduce_mode is None and not _series_is_simple(df[value_col]):
        raise LoadError("bad_value_column_type")
    if reduce_mode is not None and not _series_supports_vector_reduction(df[value_col]):
        raise LoadError("bad_vector_value_column")
    filter_specs = _normalize_filter_specs(filter_cols, filter_vals, filter_ops)
    filters = []
    for col, val, op in filter_specs:
        if not col:
            continue
        if col not in df.columns:
            raise LoadError("bad_filter_column")
        if not _series_is_simple(df[col]):
            raise LoadError("bad_filter_column_type")
        filters.append((col, val, op))

    import pandas as pd
    gcol = df[group]
    if isinstance(gcol, pd.DataFrame):
        gcol = gcol.iloc[:, 0]
    vcol = df[value_col]
    if isinstance(vcol, pd.DataFrame):
        vcol = vcol.iloc[:, 0]
    if reduce_mode is None:
        vcol = pd.to_numeric(vcol, errors="coerce")
    else:
        def _reduce_vector_cell(raw):
            arr, _ = _coerce_reducible_numeric_array(raw)
            if arr is None or arr.size == 0:
                return np.nan
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return np.nan
            if reduce_mode == "median":
                return float(np.median(arr))
            return float(np.mean(arr))
        vcol = vcol.map(_reduce_vector_cell)
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
    animal_source = None
    for cname in ("animal_id", "animalID", "animalId", "animal"):
        if cname in df.columns and _series_is_simple(df[cname]):
            animal_source = cname
            break
    if animal_source is not None:
        animalcol = df[animal_source]
        if isinstance(animalcol, pd.DataFrame):
            animalcol = animalcol.iloc[:, 0]
    else:
        animalcol = None
    data_dict = {"_group": gcol, "_value": vcol}
    if idcol is not None:
        data_dict["_id"] = idcol
    if animalcol is not None:
        data_dict["_animal_id"] = animalcol
    tmp = pd.DataFrame(data_dict, index=df.index)
    if filters:
        mask = pd.Series(True, index=df.index)
        for col, wanted, op in filters:
            fcol = df[col]
            if isinstance(fcol, pd.DataFrame):
                fcol = fcol.iloc[:, 0]
            keys = fcol.map(lambda raw: "" if _json_simple_value(raw) is None else str(_json_simple_value(raw)))
            if op == "exclude":
                mask &= keys != wanted
            else:
                mask &= keys == wanted
        tmp = tmp.loc[mask]
    tmp = tmp.dropna(subset=["_value"])

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
        animal_count = None
        if "_animal_id" in gdf.columns:
            animal_keys = {
                str(val)
                for val in (_json_simple_value(raw) for raw in gdf["_animal_id"].tolist())
                if val is not None and str(val) != ""
            }
            animal_count = len(animal_keys)
        groups_out.append(
            {
                "name": str(gval),
                "count": len(vals),
                "animal_count": animal_count,
                "mean": mean_val,
                "values": vals,
                "points": points,
            }
        )

    value_label = f"{value_col} ({reduce_mode})" if reduce_mode else value_col
    unit = ds.units.get(value_col, "") if isinstance(ds.units, dict) else ""
    group_unit = ds.units.get(group, "") if isinstance(ds.units, dict) else ""
    return {
        "value": value_label,
        "value_key": value,
        "value_column": value_col,
        "value_reduce": reduce_mode,
        "group": group,
        "unit": unit,
        "group_unit": group_unit,
        "filters": [{"column": col, "value": val, "mode": op} for col, val, op in filters],
        "groups": groups_out,
    }


@bp.get("/api/plot/bar")
def api_plot_bar():
    value = request.args.get("value")
    group = request.args.get("group")
    filter_cols = request.args.getlist("filter_col")
    filter_vals = request.args.getlist("filter_val")
    filter_ops = request.args.getlist("filter_op")
    if not value or not group:
        return jsonify({"error": "missing_params"}), 400

    ds, dataset_id = _get_dataset_from_request()
    if ds is None:
        return jsonify({"error": "no_dataset"}), 404
    try:
        payload = _build_bar_plot_payload(ds, value, group, filter_cols, filter_vals, filter_ops)
    except LoadError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(payload)


@bp.get("/api/plot/bar_test")
def api_plot_bar_test():
    value = request.args.get("value")
    group = request.args.get("group")
    method = (request.args.get("method") or "ttest").strip().lower()
    filter_cols = request.args.getlist("filter_col")
    filter_vals = request.args.getlist("filter_val")
    filter_ops = request.args.getlist("filter_op")
    if not value or not group:
        return jsonify({"error": "missing_params"}), 400
    if method not in {"ttest", "mannwhitney"}:
        return jsonify({"error": "bad_method"}), 400

    ds, dataset_id = _get_dataset_from_request()
    if ds is None:
        return jsonify({"error": "no_dataset"}), 404
    try:
        payload = _build_bar_plot_payload(ds, value, group, filter_cols, filter_vals, filter_ops)
    except LoadError as exc:
        return jsonify({"error": str(exc)}), 400

    groups_out = payload.get("groups", [])
    if len(groups_out) != 2:
        return jsonify({"error": "requires_two_groups"}), 400
    g1, g2 = groups_out
    vals1 = [float(v) for v in g1.get("values", []) if isinstance(v, (int, float))]
    vals2 = [float(v) for v in g2.get("values", []) if isinstance(v, (int, float))]
    if not vals1 or not vals2:
        return jsonify({"error": "empty_group"}), 400

    try:
        from scipy import stats  # type: ignore
    except Exception:
        return jsonify({"error": "missing_scipy"}), 500

    if method == "ttest":
        result = stats.ttest_ind(vals1, vals2, equal_var=False, alternative="two-sided", nan_policy="omit")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        summary = "Welch two-sample t-test"
    else:
        result = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        summary = "Mann-Whitney U test"

    return jsonify(
        {
            "method": method,
            "summary": summary,
            "statistic": statistic,
            "p_value": p_value,
            "groups": [
                {"name": str(g1.get("name", "")), "count": len(vals1)},
                {"name": str(g2.get("name", "")), "count": len(vals2)},
            ],
        }
    )


@bp.get("/api/plot/scatter")
def api_plot_scatter():
    x = request.args.get("x")
    y = request.args.get("y")
    group = request.args.get("group")
    if not x or not y:
        return jsonify({"error": "missing_params"}), 400

    ds, dataset_id = _get_dataset_from_request()
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
        if not _series_is_simple(df[group]):
            return jsonify({"error": "bad_group_column_type"}), 400
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


@bp.get("/api/plot/line_by_row")
def api_plot_line_by_row():
    x = request.args.get("x")
    y = request.args.get("y")
    color_col = request.args.get("color")
    filter_cols = request.args.getlist("filter_col")
    filter_vals = request.args.getlist("filter_val")
    filter_ops = request.args.getlist("filter_op")
    if not x or not y:
        return jsonify({"error": "missing_params"}), 400

    ds, dataset_id = _get_dataset_from_request()
    if ds is None:
        return jsonify({"error": "no_dataset"}), 404
    df = ds.df
    if x not in df.columns or y not in df.columns:
        return jsonify({"error": "bad_columns"}), 400
    if not _series_is_numeric_vector(df[x]) or not _series_is_numeric_vector(df[y]):
        return jsonify({"error": "bad_vector_columns"}), 400
    if color_col:
        if color_col not in df.columns:
            return jsonify({"error": "bad_color_column"}), 400
        if not _series_is_simple(df[color_col]):
            return jsonify({"error": "bad_color_column_type"}), 400
    try:
        filter_specs = _normalize_filter_specs(filter_cols, filter_vals, filter_ops)
    except LoadError as exc:
        return jsonify({"error": str(exc)}), 400
    filters = []
    for col, val, op in filter_specs:
        if not col:
            continue
        if col not in df.columns:
            return jsonify({"error": "bad_filter_column"}), 400
        if not _series_is_simple(df[col]):
            return jsonify({"error": "bad_filter_column_type"}), 400
        filters.append((col, val, op))

    id_source = None
    for cname in ("cell_name", "segment_ID", "segmentID"):
        if cname in df.columns and _series_is_simple(df[cname]):
            id_source = cname
            break
    id_series = df[id_source] if id_source is not None else None
    if isinstance(id_series, pd.DataFrame):
        id_series = id_series.iloc[:, 0]
    animal_source = None
    for cname in ("animal_id", "animalID", "animalId", "animal"):
        if cname in df.columns and _series_is_simple(df[cname]):
            animal_source = cname
            break
    animal_series = df[animal_source] if animal_source is not None else None
    if isinstance(animal_series, pd.DataFrame):
        animal_series = animal_series.iloc[:, 0]
    color_series = df[color_col] if color_col else None
    if isinstance(color_series, pd.DataFrame):
        color_series = color_series.iloc[:, 0]

    traces = []
    skipped_rows = 0
    for idx in df.index:
        include = True
        for col, wanted, op in filters:
            actual = _json_simple_value(df.at[idx, col])
            actual_key = "" if actual is None else str(actual)
            matches = actual_key == wanted
            if (op == "include" and not matches) or (op == "exclude" and matches):
                include = False
                break
        if not include:
            skipped_rows += 1
            continue
        x_arr = _coerce_1d_numeric_array(df.at[idx, x])
        y_arr = _coerce_1d_numeric_array(df.at[idx, y])
        if x_arr is None or y_arr is None or x_arr.size == 0 or y_arr.size == 0:
            skipped_rows += 1
            continue
        if x_arr.size != y_arr.size:
            skipped_rows += 1
            continue
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if not np.any(mask):
            skipped_rows += 1
            continue
        x_vals = x_arr[mask]
        y_vals = y_arr[mask]
        if x_vals.size == 0:
            skipped_rows += 1
            continue
        row_id = ""
        if id_series is not None:
            row_id = str(id_series.loc[idx])
        if not row_id:
            row_id = f"row {idx}"
        color_value = _json_simple_value(color_series.loc[idx]) if color_series is not None else None
        animal_value = _json_simple_value(animal_series.loc[idx]) if animal_series is not None else None
        traces.append(
            {
                "id": row_id,
                "x": x_vals.astype(float).tolist(),
                "y": y_vals.astype(float).tolist(),
                "color_value": color_value,
                "animal_value": animal_value,
            }
        )

    if not traces:
        return jsonify({"error": "no_valid_rows"}), 400

    x_unit = ds.units.get(x, "") if isinstance(ds.units, dict) else ""
    y_unit = ds.units.get(y, "") if isinstance(ds.units, dict) else ""
    return jsonify(
        {
            "x": x,
            "y": y,
            "x_unit": x_unit,
            "y_unit": y_unit,
            "color_column": color_col if color_col else None,
            "filters": [{"column": col, "value": val, "mode": op} for col, val, op in filters],
            "traces": traces,
            "n_rows": len(traces),
            "skipped_rows": skipped_rows,
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
    session["dataset_label"] = filename
    session["dataset_source"] = "upload"
    msg = f"Loaded {filename}: {ds.df.shape[0]} rows, {ds.df.shape[1]} cols"
    flash(msg)
    return redirect(url_for("main.index", _anchor="preview"))


@bp.get("/dj_test")
def connect_to_dj():
    dj.config["database.host"] = "vfsmdatajoint01.fsm.northwestern.edu"
    dj.config["database.user"] = os.environ.get("DJ_USERNAME")
    dj.config["database.password"] = os.environ.get("DJ_PASSWORD")

    c = dj.conn()
    return str(c.is_connected)

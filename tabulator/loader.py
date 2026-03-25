from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque
import h5py


class LoadError(Exception):
    pass


@dataclass
class DataSet:
    df: pd.DataFrame
    units: Dict[str, str]


def load_dataset(path: str) -> DataSet:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "csv":
        return _load_csv_formatted(path)
    if ext in {"pkl", "pickle"}:
        return _load_pickle_formatted(path)
    if ext == "h5":
        return _load_h5_formatted(path)
    if ext == "mat":
        return _load_mat_formatted(path)
    raise LoadError(f"Unsupported extension: .{ext}")


# Backward-compatible alias (if any caller still imports load_table)
load_table = load_dataset


def _load_csv_formatted(path: str) -> DataSet:
    try:
        raw = pd.read_csv(path, header=None)
        if raw.shape[0] < 2:
            raise LoadError("CSV must have at least two rows (header + data/units)")
        header = raw.iloc[0].astype(str).tolist()
        second_row = raw.iloc[1].tolist()

        def _cell_is_numeric_like(value: Any) -> bool:
            if value is None:
                return False
            try:
                if pd.isna(value):
                    return False
            except Exception:
                pass
            if isinstance(value, (int, float, np.integer, np.floating)):
                return True
            text = str(value).strip()
            if not text:
                return False
            try:
                parsed = pd.to_numeric(text)
            except Exception:
                return False
            try:
                return not pd.isna(parsed)
            except Exception:
                return True

        def _second_row_is_units(row: list[Any]) -> bool:
            saw_text = False
            for value in row:
                if value is None:
                    continue
                try:
                    if pd.isna(value):
                        continue
                except Exception:
                    pass
                text = str(value).strip()
                if not text:
                    continue
                if _cell_is_numeric_like(value):
                    return False
                saw_text = True
            return saw_text

        has_units_row = _second_row_is_units(second_row)
        units_row = [("" if pd.isna(v) else str(v)) for v in second_row] if has_units_row else ["" for _ in header]
        data_start = 2 if has_units_row else 1
        df = raw.iloc[data_start:].reset_index(drop=True)
        df.columns = header
        # Convert only columns that are fully numeric-like
        df = _infer_and_cast_columns(df)
        # Build units mapping
        units = {h: (u if u != "nan" else "") for h, u in zip(header, units_row)}
        # Only add default unit for segmentID if it's a column
        if "segmentID" in df.columns and "segmentID" not in units:
            units["segmentID"] = ""
        return DataSet(df=df, units=units)
    except Exception as e:
        raise LoadError(f"Failed to read formatted CSV: {e}") from e


def _load_pickle_formatted(path: str) -> DataSet:
    try:
        obj = pd.read_pickle(path)
    except Exception as e:
        raise LoadError(f"Failed to read pickle: {e}") from e

    if isinstance(obj, dict) and "table" in obj:
        df = obj["table"]
        units_df = obj.get("units")
        units: Dict[str, str] = {}
        if isinstance(units_df, pd.DataFrame) and {"variable", "unit"}.issubset(units_df.columns):
            units = {str(row["variable"]): str(row["unit"]) for _, row in units_df.iterrows()}
        # Only add default unit for segmentID if it's a column
        if "segmentID" in df.columns and "segmentID" not in units:
            units["segmentID"] = ""
        if not isinstance(df, pd.DataFrame):
            raise LoadError("Pickle 'table' is not a DataFrame")
        return DataSet(df=df, units=units)

    # Fallback: treat as a DataFrame or Series
    if isinstance(obj, pd.DataFrame):
        return DataSet(df=obj, units={c: "" for c in obj.columns})
    if isinstance(obj, pd.Series):
        df = obj.to_frame()
        return DataSet(df=df, units={c: "" for c in df.columns})
    raise LoadError("Pickle did not contain expected dict with 'table' and 'units'")


def _load_h5_formatted(path: str) -> DataSet:
    try:
        with h5py.File(path, "r") as f:
            # Units group
            if "/units" not in f:
                raise LoadError("Missing /units group in HDF5 file")
            units_group = f["/units"]
            units: Dict[str, str] = {}
            for k in units_group.keys():
                v = units_group[k][()]
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8", errors="ignore")
                elif isinstance(v, np.ndarray) and v.dtype.kind in {"S", "U"} and v.size == 1:
                    v = v.astype(str).item()
                units[str(k)] = str(v)
            # Rows: one group per segment, e.g. /segid_<ID>
            rows = []
            stat_keys = [k for k in units.keys()]
            for gname, grp in f.items():
                # top-level groups include units and segid_* groups
                base = gname.split("/")[-1]
                if base in {"units", "stat_keys"}:
                    continue
                if not isinstance(grp, h5py.Group):
                    continue
                # Derive segmentID from group name if not stored as dataset
                seg_name = base
                seg_id = seg_name.replace("segid_", "")
                row: Dict[str, Any] = {"segmentID": seg_id}
                for k in stat_keys:
                    if k == "segmentID":
                        # If present as dataset, prefer it
                        if "segmentID" in grp:
                            row["segmentID"] = _h5_read_scalar(grp["segmentID"])  # type: ignore
                        continue
                    if k in grp:
                        row[k] = _h5_read_scalar(grp[k])  # type: ignore
                    else:
                        row[k] = np.nan
                rows.append(row)
        df = pd.DataFrame(rows)
        # Convert only columns that are fully numeric-like
        df = _infer_and_cast_columns(df)
        # Only add default unit for segmentID if it's a column
        if "segmentID" in df.columns and "segmentID" not in units:
            units["segmentID"] = ""
        return DataSet(df=df, units=units)
    except Exception as e:
        raise LoadError(f"Failed to read formatted HDF5: {e}") from e


def _h5_read_scalar(dset: h5py.Dataset) -> Any:
    v = dset[()]
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore")
    if isinstance(v, np.ndarray) and v.size == 1:
        return v.reshape(()).tolist()
    return v


def _load_mat_formatted(path: str) -> DataSet:
    try:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        raise LoadError(f"Failed to read MAT file: {e}") from e

    # Accept both new and legacy variable names
    struct_keys = ["data", "S"]
    units_keys = ["unitsStruct", "Units"]

    struct_key = next((k for k in struct_keys if k in data), None)
    units_key = next((k for k in units_keys if k in data), None)

    if struct_key is None:
        opaque_name = _find_mat_opaque_table_name(data)
        if opaque_name:
            raise LoadError(
                f"MATLAB table '{opaque_name}' is not supported by SciPy MAT parsing. "
                "Save it as a struct array first."
            )
        raise LoadError("MAT file must contain struct 'data' (or legacy 'S').")

    S = data[struct_key]

    # Units are optional for MAT imports.
    units: Dict[str, str] = _mat_units_to_dict(data[units_key]) if units_key is not None else {}

    # Convert struct array S to DataFrame
    rows = _mat_struct_array_to_rows(S)
    if not rows:
        raise LoadError("MAT struct contained no rows")
    df = pd.DataFrame(rows)
    # Convert only columns that are fully numeric-like
    df = _infer_and_cast_columns(df)
    # Only add default unit for segmentID if it's a column
    if "segmentID" in df.columns and "segmentID" not in units:
        units["segmentID"] = ""
    return DataSet(df=df, units=units)


def _find_mat_opaque_table_name(data: Dict[str, Any]) -> Optional[str]:
    for key, value in data.items():
        if key.startswith("__") or not isinstance(value, MatlabOpaque) or value.size == 0:
            continue
        try:
            elem = value.reshape(-1)[0]
            kind = elem["s2"]
            if isinstance(kind, np.ndarray) and kind.size == 1:
                kind = kind.reshape(()).tolist()
            if isinstance(kind, (bytes, bytearray)):
                kind = kind.decode("utf-8", errors="ignore")
            if str(kind) != "table":
                continue
            name = elem["s0"]
            if isinstance(name, np.ndarray) and name.size == 1:
                name = name.reshape(()).tolist()
            if isinstance(name, (bytes, bytearray)):
                name = name.decode("utf-8", errors="ignore")
            return str(name) if name else key
        except Exception:
            continue
    return None


def _mat_units_to_dict(Units: Any) -> Dict[str, str]:
    def _normalize_unit(v: Any) -> str:
        # Treat MATLAB empty ([]) as blank
        if v is None:
            return ""
        # NumPy arrays (MATLAB data)
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return ""
            # 1x1 -> scalar
            if v.size == 1:
                v = v.reshape(())
                v = v.tolist() if isinstance(v, np.ndarray) else v
            # Char arrays -> join into Python string
            elif v.dtype.kind in {"S", "U"}:
                try:
                    return "".join(v.astype(str).tolist())
                except Exception:
                    return str(v.astype(str))
        # Decode bytes
        if isinstance(v, (bytes, bytearray)):
            try:
                v = v.decode("utf-8", errors="ignore")
            except Exception:
                return ""
        # NaN -> blank
        try:
            import math
            if isinstance(v, float) and math.isnan(v):
                return ""
        except Exception:
            pass
        return str(v)

    out: Dict[str, str] = {}
    # Units is typically a struct with attributes per field
    if hasattr(Units, "_fieldnames"):
        for name in Units._fieldnames:
            v = getattr(Units, name)
            out[str(name)] = _normalize_unit(v)
        return out
    # Fallback: try dict-like
    if isinstance(Units, dict):
        return {str(k): _normalize_unit(v) for k, v in Units.items()}
    return out


def _mat_struct_array_to_rows(S: Any) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    # S can be an object array of MATLAB structs; each element has _fieldnames
    if isinstance(S, np.ndarray):
        # Ensure we iterate elements (even if scalar)
        it = S.flat
        for elem in it:
            row: Dict[str, Any] = {}
            if hasattr(elem, "_fieldnames"):
                for name in elem._fieldnames:
                    row[str(name)] = _mat_normalize_value(getattr(elem, name))
            rows.append(row)
    elif hasattr(S, "_fieldnames"):
        rows = _mat_scalar_struct_to_rows(S)
    return rows


def _mat_scalar_struct_to_rows(S: Any) -> list[Dict[str, Any]]:
    field_values = {str(name): getattr(S, name) for name in S._fieldnames}
    lengths = []
    for value in field_values.values():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            lengths.append(len(value))
    if lengths and all(length == lengths[0] for length in lengths) and lengths[0] > 1:
        row_count = lengths[0]
        rows: list[Dict[str, Any]] = []
        for idx in range(row_count):
            row: Dict[str, Any] = {}
            for name, value in field_values.items():
                if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == row_count:
                    row[name] = _mat_normalize_value(value[idx])
                else:
                    row[name] = _mat_normalize_value(value)
            rows.append(row)
        return rows

    row: Dict[str, Any] = {}
    for name, value in field_values.items():
        row[name] = _mat_normalize_value(value)
    return [row]


def _mat_normalize_value(v: Any) -> Any:
    if isinstance(v, np.ndarray) and v.size == 0 and v.dtype.kind in {"S", "U"}:
        return ""
    if isinstance(v, np.ndarray) and v.size == 1:
        v = v.reshape(())
        v = v.tolist() if isinstance(v, np.ndarray) else v
    if isinstance(v, np.ndarray) and v.dtype.kind in {"S", "U"}:
        try:
            v = "".join(v.astype(str).tolist())
        except Exception:
            v = v.astype(str)
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", errors="ignore")
    return v


def _infer_and_cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to numeric only when all non-null values are numeric-like.

    - Leaves columns with any non-numeric text as object dtype.
    - Always leaves `segmentID` unchanged.
    """
    out = df.copy()
    for col in out.columns:
        if str(col) in {"segmentID", "segment_ID", "cell_name"}:
            continue
        s = out[col]
        # Normalize bytes to strings for object columns before checks
        if s.dtype == object:
            s = s.apply(lambda v: v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else v)
        # Try converting to numeric; if any non-empty become NaN, keep original
        coerced = pd.to_numeric(s, errors="coerce")

        # Treat empty/whitespace-only strings as missing for inference
        def is_present(v: Any) -> bool:
            if isinstance(v, (np.ndarray, list, tuple, dict)):
                return True
            if pd.isna(v):
                return False
            if isinstance(v, str):
                return len(v.strip()) > 0
            return True

        present_mask = s.apply(is_present)

        # If all present values successfully converted (not NaN), accept
        if coerced[present_mask].notna().all():
            out[col] = coerced
        else:
            out[col] = s
    return out

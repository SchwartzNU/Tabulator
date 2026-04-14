from __future__ import annotations

import os
import pickle
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional


class DataStore:
    def __init__(self, max_items: int = 10, root_dir: Optional[str] = None) -> None:
        self.max_items = max_items
        self._root_dir = ""
        self.configure(root_dir)

    def configure(self, root_dir: Optional[str] = None) -> None:
        target = (
            root_dir
            or os.getenv("TABULATOR_STORE_DIR")
            or os.path.join(tempfile.gettempdir(), "tabulator_store")
        )
        self._root_dir = os.path.abspath(target)
        os.makedirs(self._root_dir, exist_ok=True)

    @property
    def root_dir(self) -> str:
        return self._root_dir

    def put(self, obj: Any) -> str:
        key = uuid.uuid4().hex
        path = self._path_for(key)
        tmp_path = self._tmp_path_for(key)
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
            self._touch(path)
            self._evict_if_needed()
            return key
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    def get(self, key: str) -> Optional[Any]:
        if not key:
            return None
        path = self._path_for(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError, OSError):
            return None
        self._touch(path)
        return obj

    def remove(self, key: str) -> bool:
        if not key:
            return False
        path = self._path_for(key)
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False
        except OSError:
            return False

    def _path_for(self, key: str) -> str:
        return os.path.join(self._root_dir, f"{key}.pkl")

    def _tmp_path_for(self, key: str) -> str:
        suffix = uuid.uuid4().hex
        return os.path.join(self._root_dir, f".{key}.{suffix}.tmp")

    def _touch(self, path: str) -> None:
        try:
            os.utime(path, None)
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def _dataset_paths(self) -> list[Path]:
        root = Path(self._root_dir)
        try:
            return sorted(
                (p for p in root.glob("*.pkl") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            return []

    def _evict_if_needed(self) -> None:
        paths = self._dataset_paths()
        overflow = len(paths) - self.max_items
        if overflow <= 0:
            return
        for path in paths[:overflow]:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                continue


store = DataStore(max_items=10)

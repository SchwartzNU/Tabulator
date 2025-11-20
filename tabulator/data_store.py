from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Any
import uuid
import pandas as pd


class DataStore:
    def __init__(self, max_items: int = 10) -> None:
        self.max_items = max_items
        self._store: "OrderedDict[str, Any]" = OrderedDict()

    def put(self, obj: Any) -> str:
        key = uuid.uuid4().hex
        if key in self._store:
            del self._store[key]
        self._store[key] = obj
        self._evict_if_needed()
        return key

    def get(self, key: str) -> Optional[Any]:
        obj = self._store.get(key)
        if obj is not None:
            # Mark as recently used
            self._store.move_to_end(key)
        return obj

    def remove(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def _evict_if_needed(self) -> None:
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)


store = DataStore(max_items=10)

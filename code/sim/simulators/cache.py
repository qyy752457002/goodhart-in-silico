"""Append-only cache of LLM calls, so simulation runs are resumable and exactly repeatable.

One JSON object per line: {"key": ..., "text": ..., "usage": {...}, "time": ...}. The key hashes
everything that determines the reply (backend, model, messages, sampling parameters), so changing
the prompt wording or a parameter never returns a stale answer. A partially written last line
(after a crash) is ignored on load.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time


def call_key(fingerprint: dict, messages: list, **params) -> str:
    """Stable hash of a request."""
    payload = json.dumps({"backend": fingerprint, "messages": messages, "params": params},
                         sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CallCache:
    def __init__(self, path: str | None):
        """path=None keeps the cache in memory only."""
        self.path = path
        self._lock = threading.Lock()
        self._data = {}
        self.hits = 0
        self.misses = 0
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # truncated write
                    if "key" in rec:
                        self._data[rec["key"]] = rec

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> dict | None:
        with self._lock:
            rec = self._data.get(key)
            if rec is None:
                self.misses += 1
            else:
                self.hits += 1
            return rec

    def put(self, key: str, text: str, usage: dict = None, **info) -> dict:
        """Store a reply. `info` holds facts about how the request actually went out (e.g.
        temperature_sent=False), so that a rerun from the cache treats the reply the same way."""
        rec = {"key": key, "text": text, "usage": usage or {}, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
               **info}
        with self._lock:
            self._data[key] = rec
            if self.path:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

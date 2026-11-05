"""Content-addressed disk cache (same idea as in my earlier RAG demo).

The key is a hash of *everything* that determines the result (model, prompt, schema, input text).
A changed input simply produces a new key, so there is no invalidation logic and no stale result
can ever be served. One small JSON file per entry: easy to inspect; delete .cache/ to reset.

Deliberately not a semantic cache ("similar prompt -> same reply"): two similar chunks can contain
different facts.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from backend.config import REPO_ROOT

CACHE_DIR = REPO_ROOT / ".cache"


def cache_key(**parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class DiskCache:
    def __init__(self, namespace: str, root: Path = CACHE_DIR):
        self.dir = root / namespace
        self.hits = self.misses = 0

    def get(self, key: str) -> Any | None:
        path = self.dir / f"{key}.json"
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: Any) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"{key}.tmp"
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.dir / f"{key}.json")  # atomic: a crash never leaves half a file

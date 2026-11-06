"""向量缓存：可配置目录、并发写不产生半截文件。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag_mcp.cache import VectorCache, cache_dir, cache_key


def test_cache_dir_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path))

    assert cache_dir() == tmp_path


def test_concurrent_writers_leave_one_complete_json(tmp_path: Path):
    cache = VectorCache(tmp_path, active=True)
    key = cache_key("concurrent", 1)

    def write(seed: int) -> None:
        cache.save(key, [[float(seed), 0.0]], meta={"seed": seed})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(8)))

    path = tmp_path / f"vectors-{key}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["vectors"] in ([[float(i), 0.0]] for i in range(8))
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_roundtrip(tmp_path: Path):
    cache = VectorCache(tmp_path, active=True)
    key = cache_key("roundtrip", 2)

    cache.save(key, [[0.1, 0.2], [0.3, 0.4]], meta={"x": 1})

    assert cache.load(key) == [[0.1, 0.2], [0.3, 0.4]]

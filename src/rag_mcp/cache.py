"""向量索引的磁盘缓存。

存在的理由（DESIGN §3.2「启动必须快」）：

stdio server 由客户端**按会话拉起**，冷启动慢就等于不可用。而向量模型在首次建索引时
要为整个语料发请求，实测阿里百炼同一端点会出现 0.3s–20s 的抖动 ——
18 个块的样例语料都可能卡 20 秒。

所以：算过一次就落盘，之后启动只读本地文件。缓存键必须包含
`corpus_version` 与 embedder 标识，语料或模型变了就自动失效重建 ——
**宁可重建，也不能拿别的语料/别的模型的向量来用。**
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("rag_mcp.cache")

#: 默认缓存目录在仓库根的 `.cache/`（已 gitignore）。
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"


def cache_dir() -> Path:
    raw = os.getenv("RAG_CACHE_DIR")
    return Path(raw) if raw else DEFAULT_CACHE_DIR


def enabled() -> bool:
    return os.getenv("RAG_VECTOR_CACHE", "on").strip().lower() not in {"0", "off", "false", "no"}


def cache_key(*parts: object) -> str:
    """缓存键：任何一项变化都必须换文件。"""
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class VectorCache:
    def __init__(self, directory: Path | None = None, *, active: bool | None = None) -> None:
        self._dir = directory or cache_dir()
        self._active = enabled() if active is None else active

    @property
    def active(self) -> bool:
        return self._active

    def _path(self, key: str) -> Path:
        return self._dir / f"vectors-{key}.json"

    def load(self, key: str) -> list[list[float]] | None:
        if not self._active:
            return None
        path = self._path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        vectors = payload.get("vectors")
        if not isinstance(vectors, list) or not vectors:
            return None
        logger.info("vector cache hit: %s (%d vectors)", path.name, len(vectors))
        return vectors

    def save(self, key: str, vectors: list[list[float]], *, meta: dict[str, object]) -> None:
        if not self._active or not vectors:
            return
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # 先写临时文件再改名：进程被 SIGTERM 时不会留下半截缓存
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"meta": meta, "vectors": vectors}, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(path)
            logger.info("vector cache write: %s (%d vectors)", path.name, len(vectors))
        except OSError as exc:  # 缓存失败不影响主流程
            logger.warning("vector cache write failed: %s", exc)

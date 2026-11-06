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
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("rag_mcp.cache")


def _user_cache_root() -> Path:
    """平台默认缓存目录；不要写进 site-packages / 仓库安装目录。"""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    if os.name == "nt":
        return Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    return Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")


#: 默认缓存目录：用户级 cache，避免安装到只读目录时写失败；
#: 旧版本在仓库根 `.cache/`，可以用 RAG_CACHE_DIR 指回去。
DEFAULT_CACHE_DIR = _user_cache_root() / "rag-mcp"


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
        tmp_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # 并发进程不能共用同一个 .tmp：用 mkstemp 保证每个写者一个文件，
            # 写完再 os.replace 原子替换；读方只会看到旧文件或完整新文件。
            fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"meta": meta, "vectors": vectors}, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            tmp_path = None
            logger.info("vector cache write: %s (%d vectors)", path.name, len(vectors))
        except OSError as exc:  # 缓存失败不影响主流程
            logger.warning("vector cache write failed: %s", exc)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

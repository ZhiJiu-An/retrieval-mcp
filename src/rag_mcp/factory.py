"""对象图组装：配置 → 可用服务。

**不 import `mcp`**（DESIGN §4.2）：这样评测脚本、CLI、doctor 都能用同一套组装逻辑，
不必启动协议层。
"""

from __future__ import annotations

import logging

from .backends.memory import MemoryBackend
from .backends.sqlite import SqliteBackend
from .config import Settings
from .embedding import HashingEmbedder, build_embedder
from .errors import RagError
from .registry import Registry
from .rerank import build_reranker
from .service import RetrievalService

logger = logging.getLogger("rag_mcp.factory")


def build_service(settings: Settings | None = None) -> RetrievalService:
    """组装对象图。

    **向量模型不可用时不阻止启动**：降级为词法检索，并把原因记在
    `describe()` 的 note 里。理由——「降级但仍可用」+「可发现」比
    「客户端只看到 server 起不来」强，也比「悄悄降级」强。
    """
    resolved = settings or Settings.from_env()
    registry = Registry.load(resolved.registry_path)
    reranker = build_reranker(resolved.rerank)

    embedder = build_embedder(resolved.embedding)
    degraded: str | None = None
    try:
        backend = _build_backend(resolved, embedder)
    except RagError as exc:
        # 远程向量模型连不上：退回本地词法桩，标明原因，继续服务
        degraded = f"{exc.code}: {exc.message}（已退化为词法检索）"
        logger.warning("embedding backend degraded: %s", degraded)
        embedder = HashingEmbedder(dim=resolved.embedding.dim)
        backend = _build_backend(resolved, embedder)

    return RetrievalService(
        backend=backend,
        registry=registry,
        settings=resolved,
        embedder=embedder,
        reranker=reranker,
        degraded_reason=degraded,
    )


def _build_backend(settings: Settings, embedder: object):
    """按 `RAG_BACKEND` 选后端。换成生产库只需在这里加一个分支。"""
    if settings.backend == "sqlite":
        if settings.db_path is None:
            raise RagError(
                code="BAD_CONFIGURATION",
                message="RAG_BACKEND=sqlite 但没有配置 RAG_DB_PATH",
                hint="先用 python -m rag_mcp.ingest 建库，再把路径写进 RAG_DB_PATH",
                retryable=False,
            )
        return SqliteBackend(settings.db_path, embedder=embedder)  # type: ignore[arg-type]
    return MemoryBackend(
        settings.data_path,
        embedder=embedder,  # type: ignore[arg-type]
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def build_principal(settings: Settings) -> object:
    """身份来自进程启动上下文，绝不来自工具参数（DESIGN §8.2）。"""
    from .policy import Principal

    return Principal(
        user_id=settings.principal_id,
        groups=frozenset(settings.principal_groups),
        max_level=settings.principal_max_level,
    )

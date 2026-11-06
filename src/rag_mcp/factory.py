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
from .errors import BACKEND_UNAVAILABLE, BAD_CONFIGURATION, RagError
from .policy import LEVEL_ORDER
from .registry import Registry
from .rerank import build_reranker
from .service import RetrievalService

logger = logging.getLogger("rag_mcp.factory")


def _validate_settings(settings: Settings) -> None:
    """启动期把“会等到第一次检索才炸”的配置错误提前暴露。

    这些值不会自动纠正（可纠正的用默认值/裁剪）；错误必须带 hint，
    否则用户只会看到一次无上下文的 ValueError。
    """
    problems: list[str] = []
    if settings.embedding.batch <= 0:
        problems.append("RAG_EMBEDDING_BATCH 必须 > 0")
    if settings.embedding.dim <= 0:
        problems.append("RAG_EMBEDDING_DIM 必须 > 0")
    if settings.principal_max_level not in LEVEL_ORDER:
        problems.append("RAG_PRINCIPAL_MAX_LEVEL 必须是 " + "/".join(LEVEL_ORDER))
    high, medium = settings.rerank.bands
    if not (0.0 <= medium < high <= 1.0):
        problems.append("必须满足 0 <= RAG_CONFIDENCE_MEDIUM < RAG_CONFIDENCE_HIGH <= 1")
    if settings.rerank.enabled and settings.rerank.provider not in {
        "openai",
        "dashscope",
        "heuristic",
    }:
        problems.append(
            f"未知 RAG_RERANK_PROVIDER={settings.rerank.provider!r}"
            "（合法：openai/dashscope/heuristic）"
        )
    if problems:
        raise RagError(
            code=BAD_CONFIGURATION,
            message="；".join(problems),
            hint="检查 .env / RAG_ENV_FILE；不要用非法配置启动服务",
            retryable=False,
        )


def build_service(settings: Settings | None = None) -> RetrievalService:
    """组装对象图。

    **向量模型不可用时不阻止启动**：降级为词法检索，并把原因记在
    `describe()` 的 note 里。理由——「降级但仍可用」+「可发现」比
    「客户端只看到 server 起不来」强，也比「悄悄降级」强。
    """
    resolved = settings or Settings.from_env()
    _validate_settings(resolved)
    registry = Registry.load(resolved.registry_path)
    reranker = build_reranker(resolved.rerank)

    embedder = build_embedder(resolved.embedding)
    degraded: str | None = None
    try:
        backend = _build_backend(resolved, embedder)
    except RagError as exc:
        if exc.code != BACKEND_UNAVAILABLE:
            # 配置错误（未知 backend、索引不存在、schema 损坏）不能伪装成
            # “embedding 不可用”再降级；必须让 doctor/启动阶段看见真实原因。
            raise
        # 远程向量模型连不上：退回本地词法桩，标明原因，继续服务。
        degraded = f"{exc.code}: {exc.message}（已退化为词法检索）"
        logger.warning("embedding backend degraded: %s", degraded)
        embedder = HashingEmbedder(dim=resolved.embedding.dim)
        backend = _build_backend(resolved, embedder)

    # SQLite 的向量是入库时写死的；查询端换了模型/维度时不能继续算“余弦”。
    # 静默比较会让维度错位变成看似有结果的乱序（zip(strict=False) 截断）。
    index_embedder = getattr(backend, "index_embedder", None)
    if index_embedder and index_embedder != embedder.model_id:
        degraded = (
            f"embedding mismatch: 索引由 {index_embedder} 构建，"
            f"当前为 {embedder.model_id}（已退化为词法检索）"
        )
        logger.warning(degraded)
        embedder = HashingEmbedder(dim=resolved.embedding.dim)

    return RetrievalService(
        backend=backend,
        registry=registry,
        settings=resolved,
        embedder=embedder,
        reranker=reranker,
        degraded_reason=degraded,
    )


def _build_backend(settings: Settings, embedder: object):
    """按 `RAG_BACKEND` 选后端。未知值必须报错，不能静默落到样例语料。"""
    if settings.backend == "sqlite":
        if settings.db_path is None:
            raise RagError(
                code=BAD_CONFIGURATION,
                message="RAG_BACKEND=sqlite 但没有配置 RAG_DB_PATH",
                hint="先用 python -m rag_mcp.ingest 建库，再把路径写进 RAG_DB_PATH",
                retryable=False,
            )
        return SqliteBackend(settings.db_path, embedder=embedder)  # type: ignore[arg-type]
    if settings.backend == "memory":
        return MemoryBackend(
            settings.data_path,
            embedder=embedder,  # type: ignore[arg-type]
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    raise RagError(
        code=BAD_CONFIGURATION,
        message=f"未知的 RAG_BACKEND={settings.backend!r}（合法值：memory/sqlite）",
        hint="检查 .env 或 RAG_ENV_FILE；不要用拼错的 backend 启动生产实例",
        retryable=False,
    )


def build_principal(settings: Settings) -> object:
    """身份来自进程启动上下文，绝不来自工具参数（DESIGN §8.2）。"""
    from .policy import Principal

    return Principal(
        user_id=settings.principal_id,
        groups=frozenset(settings.principal_groups),
        max_level=settings.principal_max_level,
    )

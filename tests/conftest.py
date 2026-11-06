from __future__ import annotations

import os

import pytest

# **单元测试不许联网，也不许被开发者本地的 `.env` / shell 变量污染。**
#
# 之前只清 embedding 变量，结果是：
#   RAG_BACKEND=sqlite RAG_DB_PATH=rag.db pytest tests/test_server.py
# 会拿到真实 rag.db，样例语料断言直接失败。
# 同理 .env 里配了远程 rerank 时，单元测试会偷偷联网。
#
# 这里在每条用例前后恢复**所有**会影响 Settings.from_env() 的变量，
# 并显式固定成「样例语料 + 本地词法桩 + 启发式重排」。
MANAGED_ENV = (
    "RAG_ENV_FILE",
    "RAG_BACKEND",
    "RAG_DB_PATH",
    "RAG_DATA_PATH",
    "RAG_REGISTRY_PATH",
    "RAG_RETRIEVAL_MODE",
    "RAG_PRINCIPAL_ID",
    "RAG_PRINCIPAL_GROUPS",
    "RAG_PRINCIPAL_MAX_LEVEL",
    "RAG_EMBEDDING_BASE_URL",
    "RAG_EMBEDDING_MODEL",
    "RAG_EMBEDDING_API_KEY",
    "RAG_EMBEDDING_DIM",
    "RAG_EMBEDDING_BATCH",
    "RAG_EMBEDDING_TIMEOUT",
    "RAG_RERANK",
    "RAG_RERANK_PROVIDER",
    "RAG_RERANK_BASE_URL",
    "RAG_RERANK_MODEL",
    "RAG_RERANK_API_KEY",
    "RAG_RERANK_TIMEOUT",
    "RAG_CONFIDENCE_HIGH",
    "RAG_CONFIDENCE_MEDIUM",
    "RAG_MAX_TOP_K",
    "RAG_DEFAULT_TOP_K",
    "RAG_RECALL_DEPTH",
    "RAG_PER_DOC_LIMIT",
    "RAG_RELEVANCE_FLOOR",
    "RAG_DENSE_FLOOR",
    "RAG_SNIPPET_CHARS",
    "RAG_MAX_WINDOW",
    "RAG_MAX_FETCH_CHARS",
    "RAG_MAX_RESOURCE_CHARS",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "DASHSCOPE_API_KEY",
    "BAILIAN_API_KEY",
)


@pytest.fixture(autouse=True)
def _offline_embedding():
    """单元测试一律走样例语料 + 本地词法桩。

    两个都试过的坑：

    - **import 时改环境变量**：pytest 收集 `eval/` 时会 import 这份 conftest，
      于是评测也被打成词法模式（实测：Recall 从 100% 掉到 45%）。
    - **session 作用域 fixture**：它的 teardown 在整个 pytest 会话结束时才执行，
      同样会把 `eval/` 覆盖掉。

    所以用 function 作用域：只包住 `tests/` 里的每条用例，测完立刻还原。
    """
    saved = {name: os.environ.get(name) for name in MANAGED_ENV}
    for name in MANAGED_ENV:
        os.environ.pop(name, None)
    # 显式给出测试基线，而不是依赖默认值/根 .env
    os.environ["RAG_BACKEND"] = "memory"
    os.environ["RAG_RETRIEVAL_MODE"] = "auto"
    os.environ["RAG_RERANK"] = "on"  # 本地启发式，不联网
    os.environ["RAG_EMBEDDING_BASE_URL"] = ""
    os.environ["RAG_EMBEDDING_MODEL"] = ""
    os.environ["RAG_EMBEDDING_API_KEY"] = ""
    _reset_service()
    yield
    for name in MANAGED_ENV:
        value = saved.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _reset_service()


def _reset_service() -> None:
    """清掉协议层的单例，让下一次调用按当前环境重建。"""
    from rag_mcp.server import get_service, get_settings

    get_service.cache_clear()
    get_settings.cache_clear()


from rag_mcp.config import Settings
from rag_mcp.factory import build_service
from rag_mcp.policy import Principal
from rag_mcp.service import RetrievalService


@pytest.fixture(scope="session")
def settings() -> Settings:
    """显式构造，不读 `.env`：测试结果不能随开发者的本地配置漂移。"""
    return Settings()


@pytest.fixture(scope="session")
def service(settings: Settings) -> RetrievalService:
    # 建索引（BM25 + 向量）比较贵，整个测试会话只做一次
    return build_service(settings)


@pytest.fixture(scope="session")
def staff() -> Principal:
    return Principal(user_id="alice", groups=frozenset({"all"}), max_level="internal")


@pytest.fixture(scope="session")
def exec_member() -> Principal:
    return Principal(user_id="carol", groups=frozenset({"all", "exec"}), max_level="confidential")

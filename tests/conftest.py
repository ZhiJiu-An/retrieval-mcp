from __future__ import annotations

import os

import pytest

# **单元测试不许联网。**
# `Settings.from_env()` 会读 `.env`，而开发者本地的 `.env` 里可能有真实的向量模型
# 端点与 key —— 那会让测试慢到几十秒、并且随网络抖动而失败（实测过：同一台机器
# 上 embedding 接口耗时 0.3s / 20.3s / 2.3s 三连跳）。
# 这里在 import 阶段就把端点清空，`load_dotenv()` 不会覆盖已存在的变量，
# 于是全部单测都跑在本地词法桩上。
# 要跑真实模型请用 `eval/`，那里的目的就是验证真实链路。
OFFLINE_VARS = (
    "RAG_EMBEDDING_BASE_URL",
    "RAG_EMBEDDING_MODEL",
    "RAG_EMBEDDING_API_KEY",
    "DASHSCOPE_API_KEY",
    "BAILIAN_API_KEY",
)


@pytest.fixture(autouse=True)
def _offline_embedding():
    """单元测试一律走本地词法桩。

    两个都试过的坑：

    - **import 时改环境变量**：pytest 收集 `eval/` 时会 import 这份 conftest，
      于是评测也被打成词法模式（实测：Recall 从 100% 掉到 45%）。
    - **session 作用域 fixture**：它的 teardown 在整个 pytest 会话结束时才执行，
      同样会把 `eval/` 覆盖掉。

    所以用 function 作用域：只包住 `tests/` 里的每条用例，测完立刻还原。
    """
    saved = {name: os.environ.get(name) for name in OFFLINE_VARS}
    for name in OFFLINE_VARS:
        os.environ[name] = ""
    _reset_service()
    yield
    for name, value in saved.items():
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

"""评测目录的 pytest 装配。

pytest 只负责「跑」；指标与门禁逻辑都在 `harness.py` / `runner.py` 里，
这样 `python eval/run_eval.py` 与 CI 用的是同一套判断（DESIGN §13.3）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for entry in (_ROOT / "src", _ROOT / "eval"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from runner import default_service, evaluate, simulate_agent  # noqa: E402

from rag_mcp.config import reload_dotenv  # noqa: E402

# 评测要跑**真实链路**：单元测试可能刚把端点清空，这里强制重读 .env
reload_dotenv()

pytest_plugins: list[str] = []


@pytest.fixture(scope="session")
def service():
    # 建服务前再确认一次：单元测试刚可能把端点清空过
    reload_dotenv()
    return default_service()


@pytest.fixture(scope="session")
def report(service):
    return evaluate(service)


@pytest.fixture(scope="session")
def agent_report(service):
    return simulate_agent(service)


@pytest.fixture(scope="session")
def lexical_only(service) -> bool:
    """当前实例是否只有词法能力（没有配真实向量模型）。"""
    return service.capabilities.retrieval_mode == "lexical"

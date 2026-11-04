"""预算裁剪 —— **所有上限的唯一来源**。

设计依据：DESIGN.md §7。原则：上限必须在服务端强制，绝不信任调用方
（`top_k=100` 一定会被某个模型传进来）。v0 的问题是上限散落在
`config.py` / `service.py` / 工具 schema 三处，改一处就不一致（MVP.md §5 的已知债）。

本模块不 import `mcp`，也不 import 任何后端。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Any

_MIB = 1024


@dataclass(frozen=True)
class Limits:
    """一次会话的所有预算。冻结对象，任何裁剪都返回新值而不是就地改。"""

    # ---- 检索条数 ----
    max_top_k: int = 20
    default_top_k: int = 8
    #: 每路召回的候选深度。太小则重排无从发挥（DESIGN §6.2）。
    recall_depth: int = 50
    #: 同一文档最多占几条 —— 避免 top-8 全是同一篇的相邻块。
    per_doc_limit: int = 2
    #: 相关性下限（0–1，作用在重排分数上）。低于它的候选直接丢弃。
    #: 没有这条，纯词法检索会把「如何申请专利」匹配到含「申请」两个字的任何段落，
    #: 于是**本该空结果的问题永远不空**，Agent 也就永远等不到 NO_HITS 提示。
    relevance_floor: float = 0.25
    #: 语义独有命中（词法完全没开口）需要的最低余弦相似度。
    #: 0.55 是按 text-embedding-v3 在样例语料上的实测分布定的：
    #: 真命中余弦 ≥ 0.62，无答案查询 ≤ 0.53（见 DESIGN §18）。
    #: 换嵌入模型必须重标 —— 这是本项目最需要在生产语料上重新标定的数。
    dense_floor: float = 0.55

    # ---- 单条内容 ----
    snippet_chars: int = 300
    max_window: int = 3
    max_fetch_chars: int = 4000
    max_resource_chars: int = 8000

    # ---- 整次响应 ----
    max_response_bytes: int = 32 * _MIB

    @classmethod
    def from_env(cls) -> Limits:
        base = cls()
        return replace(
            base,
            max_top_k=_int_env("RAG_MAX_TOP_K", base.max_top_k),
            default_top_k=_int_env("RAG_DEFAULT_TOP_K", base.default_top_k),
            recall_depth=_int_env("RAG_RECALL_DEPTH", base.recall_depth),
            per_doc_limit=_int_env("RAG_PER_DOC_LIMIT", base.per_doc_limit),
            relevance_floor=_float_env("RAG_RELEVANCE_FLOOR", base.relevance_floor),
            dense_floor=_float_env("RAG_DENSE_FLOOR", base.dense_floor),
            snippet_chars=_int_env("RAG_SNIPPET_CHARS", base.snippet_chars),
            max_window=_int_env("RAG_MAX_WINDOW", base.max_window),
            max_fetch_chars=_int_env("RAG_MAX_FETCH_CHARS", base.max_fetch_chars),
            max_resource_chars=_int_env("RAG_MAX_RESOURCE_CHARS", base.max_resource_chars),
            max_response_bytes=_int_env("RAG_MAX_RESPONSE_BYTES", base.max_response_bytes),
        )

    def clamp_top_k(self, value: int | None) -> int:
        """越界不报错，静默收敛 —— 报错只会让 Agent 中断任务（MVP.md F6）。"""
        if value is None:
            return self.default_top_k
        return max(1, min(int(value), self.max_top_k))

    def clamp_window(self, value: int | None) -> int:
        if value is None:
            return 1
        return max(0, min(int(value), self.max_window))

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_top_k": self.max_top_k,
            "recall_depth": self.recall_depth,
            "per_doc_limit": self.per_doc_limit,
            "relevance_floor": self.relevance_floor,
            "dense_floor": self.dense_floor,
            "snippet_chars": self.snippet_chars,
            "max_window": self.max_window,
            "max_fetch_chars": self.max_fetch_chars,
            "max_resource_chars": self.max_resource_chars,
        }


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if 0.0 <= value <= 1.0 else default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def truncate_text(text: str, limit: int) -> tuple[str, bool]:
    """在段落或句末断开，避免把一句话劈成两半。返回 `(文本, 是否被截断)`。"""
    if len(text) <= limit:
        return text, False

    cut = text.rfind("\n\n", 0, limit)
    if cut < limit * 0.6:
        cut = max(text.rfind("。", 0, limit), text.rfind("\n", 0, limit))
    if cut < limit * 0.6:
        cut = limit
    return text[:cut].rstrip() + "…", True


def response_bytes(payload: Any) -> int:
    """序列化后的字节数 —— 用于核算上下文成本与断言响应上限。"""
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))

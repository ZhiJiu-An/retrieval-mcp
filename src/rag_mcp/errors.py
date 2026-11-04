"""结构化错误 —— 让 Agent 能按 `code` 分支，而不是去读中文文案。

设计依据：DESIGN.md §10「两条通道」。

- 参数根本不在 schema 里（类型错、枚举错）→ 由 MCP SDK 拦成协议错误，不归这里管。
- 一切**可恢复**的业务失败 → 抛 `RagError`；协议层（`server.py`）把它翻译成
  `is_error=True` 的工具错误，body 是 JSON，Agent 能解析出
  `code / message / hint / retryable`。

**本模块不 import `mcp`**：错误是业务概念，不是协议概念（DESIGN §4.2 依赖方向规则）。
`RagError → ToolError` 的翻译只发生在协议壳 `server.py` 里。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- 错误码

#: 查询为空或全空白。Agent 应该改成「要检索什么问题」。
EMPTY_QUERY = "EMPTY_QUERY"
#: 日期格式不是 YYYY-MM-DD。
BAD_DATE = "BAD_DATE"
#: 参数组合非法（如 chunk_id 与 doc_id 同时给），或字段名不在词表里。
BAD_ARGUMENT = "BAD_ARGUMENT"
#: filter 字段合法但取值不在值域内 —— **不报错**，返回空结果并列出合法值。
FILTER_UNKNOWN_VALUE = "FILTER_UNKNOWN_VALUE"
#: 无命中 —— **不报错**，返回空结果并给放宽建议。
NO_HITS = "NO_HITS"
#: id 不存在**或**越权 —— 两者同构，绝不区分（DESIGN §8.3）。
NOT_FOUND_OR_FORBIDDEN = "NOT_FOUND_OR_FORBIDDEN"
#: 后端/模型超时或不可用。
BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"

# ---- 以下两个不是「错误」，而是**提示码**：它们只出现在 `hint` 里，不进错误通道 ----

#: 结果被 top_k 截断 —— 给收敛建议。
TRUNCATED = "TRUNCATED"
#: 命中的块与 Agent 当初看到的不是同一段内容（语料已变）—— 见 `resolved_by`。
CHUNK_MOVED = "CHUNK_MOVED"


@dataclass(frozen=True)
class RagError(Exception):
    """可恢复的业务失败。

    不可恢复的（如 `NOT_FOUND_OR_FORBIDDEN`）也走这里，只是 `retryable=False`：
    对 Agent 来说「换个 id 再试」仍然是一个动作，只是不该原样重试。
    """

    code: str
    message: str
    hint: str | None = None
    retryable: bool = True
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Exception 的 args 用于 SDK 的 str() 与日志，保持与 message 一致。
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.hint:
            payload["hint"] = self.hint
        if self.detail:
            payload["detail"] = self.detail
        return payload

    def to_json(self) -> str:
        """工具错误的 body：单行 JSON，模型可直接解析。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def hint_line(code: str, text: str) -> str:
    """构造 `hint` 文案：`CODE: 说明`。

    约定：所有 `hint` 以错误码开头。这样「空结果」这类**不走错误通道**的情况
    也能被机器识别，同时人读起来仍是自然语言（MVP.md F8）。
    """

    return f"{code}: {text}"

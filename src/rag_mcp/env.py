"""环境变量解析的唯一实现。

以前 `budget.py` 和 `config.py` 各有一份 `_int_env` / `_float_env`：
budget 的版本会拒绝 <=0，config 的版本不会，导致 `RAG_EMBEDDING_BATCH=0`
能通过配置阶段、到第一次请求才 `range(..., 0)` 崩溃。这里把范围校验显式化。
"""

from __future__ import annotations

import os


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    if maximum is not None and value > maximum:
        return default
    return value


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}

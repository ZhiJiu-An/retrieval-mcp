"""可过滤字段的词表快照。

设计依据：DESIGN.md §9.4。

**词表定义在索引侧，检索侧只做快照 + 消费。**
这么做解决的是 v0 的债：`taxonomy.py` 把词表写死在代码里，导致
「入库新增字段时 MCP 不知情、入库删字段时只能静默返回空结果」（MVP.md §9）。

配套约束（`tests/test_registry.py` 里的漂移断言）：
工具 schema 声明的 filter 参数集合 == 本快照的字段集合。
词表变了而工具没改，CI 就红——而不是线上悄悄返回空结果。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY_PATH = Path(__file__).parent / "registry.json"

#: 由服务端根据 `Principal` 强制注入，客户端**任何**参数都不允许覆盖。
#: 与 v0 的隐式约束（「签名里没有这个参数」）不同：这里是显式白名单拒绝（DESIGN §8.5）。
RESERVED_FILTERS: frozenset[str] = frozenset({"security_level", "acl_groups"})


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str  # "enum" | "date"
    values: tuple[str, ...] | None
    description: str
    #: 落库列名。由索引侧词表定义，检索侧只消费；默认与字段同名。
    #: 新增 filter 字段时不需要再改 SqliteBackend 的 if/elif。
    sql_column: str | None = None


@dataclass(frozen=True)
class Registry:
    schema_version: str
    fields: tuple[FieldSpec, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def get(self, name: str) -> FieldSpec | None:
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def legal_values(self, name: str) -> list[str]:
        spec = self.get(name)
        return list(spec.values) if spec and spec.values else []

    def is_legal_value(self, name: str, value: str) -> bool:
        """值是否合法。日期字段只看能否解析 —— 具体早晚不是「非法值」。"""
        spec = self.get(name)
        if spec is None:
            return False
        if spec.type == "date":
            try:
                date.fromisoformat(value)
            except ValueError:
                return False
            return True
        return value in (spec.values or ())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Registry:
        fields = tuple(
            FieldSpec(
                name=str(item["name"]),
                type=str(item.get("type", "enum")),
                values=tuple(str(v) for v in item["values"]) if item.get("values") else None,
                description=str(item.get("description", "")),
                sql_column=(
                    str(item["sql_column"]) if item.get("sql_column") is not None else None
                ),
            )
            for item in payload.get("fields", [])
        )
        return cls(schema_version=str(payload.get("schema_version", "0")), fields=fields)

    @classmethod
    def load(cls, path: Path | None = None) -> Registry:
        target = path or Path(os.getenv("RAG_REGISTRY_PATH") or DEFAULT_REGISTRY_PATH)
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def default_registry() -> Registry:
    """进程内共享的快照。词表变更需要重启 server —— 与工具 schema 的静态性一致。"""
    return Registry.load()

"""身份、权限与过滤条件编译。

设计依据：DESIGN.md §8。

三件事收敛在这一层，因为它们是同一件事的三个面：
「这个调用方是谁」→「他能看到什么」→「他的过滤条件怎么落到后端查询上」。
放在一起才不会出现「ACL 在这里判断、filter 在那里匹配」的裂缝。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import BAD_ARGUMENT, BAD_DATE, RagError
from .registry import RESERVED_FILTERS, Registry

#: 密级是有序的：`public` < `internal` < `confidential`。
SECURITY_LEVELS: tuple[str, ...] = ("public", "internal", "confidential")
LEVEL_ORDER: dict[str, int] = {name: index for index, name in enumerate(SECURITY_LEVELS)}

#: 后端用它做 pre-filter：返回 True 才允许参与打分（DESIGN §8.3 铁律 1）。
AllowFn = Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class Principal:
    """调用方身份。

    来源是**进程启动上下文**（环境变量 / 启动参数 / 本地凭据），
    绝不出现在工具参数里 —— 客户端可控的身份等于没有权限控制（DESIGN §8.2）。
    stdio 下「一个进程一个身份」由客户端启动方式保证，服务端不需要租户逻辑。
    """

    user_id: str
    groups: frozenset[str]
    max_level: str

    def __post_init__(self) -> None:
        if self.max_level not in LEVEL_ORDER:
            raise ValueError(
                f"unknown max_level: {self.max_level!r}（合法值：{'/'.join(SECURITY_LEVELS)}）"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "groups": sorted(self.groups),
            "max_level": self.max_level,
        }


def acl_predicate(principal: Principal) -> AllowFn:
    """编译成后端可执行的可见性判断。

    规则：`level ≤ principal.max_level` **且**（`acl_groups` 为空 或 与之有交集）。
    未知密级按最严处理（`99`），避免入库侧写错字段时变成公开。
    """
    ceiling = LEVEL_ORDER[principal.max_level]

    def _allow(meta: Mapping[str, Any]) -> bool:
        if LEVEL_ORDER.get(str(meta.get("security_level")), 99) > ceiling:
            return False
        groups = meta.get("acl_groups") or ()
        return not groups or bool(principal.groups.intersection(groups))

    return _allow


def visibility_predicate(principal: Principal) -> AllowFn:
    """**检索可见性**：ACL + 文档状态，两者都在打分之前执行。

    为什么要单独一道「状态」闸门（DESIGN A13 的落地修正）：

    取消隐式时效衰减之后，「2023 版差旅标准」会排到「2025 版」前面 —— 这是实测出来的，
    不是假设。但正确的解法不是把衰减加回来：**废止不等于不相关**。
    「已被新版替代」是文档的**生命周期状态**，属于元数据，不属于相关度打分。

    分成两道闸门而不是一道，是因为语义不同：

    - 本函数（检索/发现）：废止文档**不出现在** search 结果里
    - `acl_predicate`（按 id 取数）：废止文档仍然可以按 id 读取 —— 查历史单据需要它

    代价：入库侧必须产 `status` 字段，缺省按 `active` 处理（保守方向是「搜得到」，
    而不是「悄悄消失」，否则一份漏标状态的文档会永远搜不到）。
    """
    acl = acl_predicate(principal)

    def _allow(meta: Mapping[str, Any]) -> bool:
        if str(meta.get("doc_status", "active")) == "superseded":
            return False
        return acl(meta)

    return _allow


@dataclass(frozen=True)
class FilterQuery:
    """已校验的过滤条件。

    `impossible=True` 时后端应直接返回空结果 —— 这不是错误，而是
    「你给的取值不在值域内」的约定行为：Agent 会拼错枚举，报错只会让任务中断
    （MVP.md F2 `FILTER_UNKNOWN_VALUE`）。

    `invalid` 保留原始输入，用于生成「合法值有哪些」的 hint。
    """

    values: dict[str, str] = field(default_factory=dict)
    impossible: bool = False
    invalid: tuple[tuple[str, str], ...] = ()

    @property
    def applied(self) -> dict[str, str]:
        """回给 Agent 的 `filters_applied`：只含真正生效的字段。"""
        return {} if self.impossible else dict(self.values)


def compile_filters(
    registry: Registry,
    raw: Mapping[str, str | None],
) -> FilterQuery:
    """把工具参数编译成后端过滤条件。

    三种结局：
    - 字段名不在词表 / 是保留字段 → `BAD_ARGUMENT`（工具 schema 应该已经挡住了，
      走到这里说明调用方绕过了 schema，属于真错误）。
    - 值不在值域 → `impossible=True`（空结果，交给 hint 解释）。
    - 正常 → `values` 按字面下传。
    """
    values: dict[str, str] = {}
    invalid: list[tuple[str, str]] = []

    for name, value in raw.items():
        if value is None or value == "":
            continue
        if name in RESERVED_FILTERS:
            raise RagError(
                code=BAD_ARGUMENT,
                message=f"{name} 由服务端根据调用方身份强制注入，不接受客户端参数",
                hint="去掉该字段；密级与可见组由你的身份决定，不能通过参数提升",
                retryable=True,
            )
        spec = registry.get(name)
        if spec is None:
            raise RagError(
                code=BAD_ARGUMENT,
                message=f"未知的过滤字段 {name!r}，可用字段：{', '.join(registry.names)}",
                hint="先用 describe 查看可用字段与值域",
                retryable=True,
            )
        if not registry.is_legal_value(name, value):
            if spec.type == "date":
                # 格式错 ≠ 取值未知：前者能立刻改正，后者只会得到空结果。
                raise RagError(
                    code=BAD_DATE,
                    message=f"{name} 需要 YYYY-MM-DD 格式，收到 {value!r}",
                    hint="例如 2025-01-31；留空表示不按时间过滤",
                    retryable=True,
                    detail={"field": name, "received": value},
                )
            invalid.append((name, value))
            continue
        values[name] = value

    if invalid:
        return FilterQuery(values=values, impossible=True, invalid=tuple(invalid))
    return FilterQuery(values=values)


def invalid_filter_hint(query: FilterQuery, registry: Registry) -> str:
    """把非法取值翻译成「下一步该传什么」。"""
    parts = [
        f"{name}={value!r} 不在值域内"
        f"（合法值：{'/'.join(registry.legal_values(name)) or '见 describe'}）"
        for name, value in query.invalid
    ]
    return "；".join(parts)

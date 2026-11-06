"""后端适配层：**只有 4 个动作**（DESIGN §4.1、A3）。

```
describe()     这个库是什么、多大、语料版本
recall()       双路召回（词法 + 向量），过滤与 ACL 都在打分之前
get_chunk()    按 id 取块，带三级降级解析
get_document() 按 id 取整篇
```

为什么不多：每加一个方法，就多一处「所有后端都要实现的语义」，
换后端时就得改所有调用点。够用即止。

为什么定义**领域类型**而不是复用后端自己的模型：协议如果镜像后端结构，
「换后端只改一个文件」就变成「换后端要改全部调用点」。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..policy import AllowFn, FilterQuery

RankedChunk = tuple["ChunkRecord", float]

#: `{doc_id}#c{ordinal:03d}-{anchor}`，例如 `hr-policy-leave#c003-a1b2c3`
#: `ordinal` 至少 3 位、不设上限：单文档超过 999 块时仍可解析。
CHUNK_ID_RE = re.compile(r"^(?P<doc_id>.+?)#c(?P<ordinal>\d{3,})-(?P<anchor>[0-9a-f]{6})$")


@dataclass(frozen=True)
class ChunkRecord:
    """一个可检索单元。

    `text` 是**入库粒度**（小块，约 300–400 字），`parent_text` 是它所属小节的全文。
    召回用 `text`（精确），返回上下文用 `parent_text`（完整）—— small-to-big，
    DESIGN §6.1。
    """

    chunk_id: str
    doc_id: str
    position: int
    section_path: str
    heading: str
    text: str
    parent_text: str
    anchor: str
    meta: Mapping[str, Any]


@dataclass(frozen=True)
class DocRecord:
    doc_id: str
    title: str
    text: str
    meta: Mapping[str, Any]
    chunks: tuple[ChunkRecord, ...]


@dataclass(frozen=True)
class CollectionInfo:
    id: str
    name: str
    doc_count: int
    chunk_count: int
    updated_at: str


@dataclass(frozen=True)
class ResolvedChunk:
    """`get_chunk` 的结果。

    `resolved_by` 让上层能如实告诉 Agent：你拿到的是不是你当初看到的那一块。
    `exact` 之外的任何值都意味着「内容已经变了」。
    """

    chunk: ChunkRecord | None
    resolved_by: str = "none"


def parse_chunk_id(chunk_id: str) -> tuple[str | None, int | None, str | None]:
    """拆出 `(doc_id, ordinal, anchor)`。格式不对时全部为 `None`。"""
    match = CHUNK_ID_RE.match(chunk_id.strip())
    if match is None:
        return None, None, None
    return match.group("doc_id"), int(match.group("ordinal")), match.group("anchor")


class RetrievalBackend(Protocol):
    corpus_version: str

    def describe(self) -> CollectionInfo: ...

    def recall(
        self,
        *,
        tokens: Sequence[str],
        query_vector: Sequence[float] | None,
        filters: FilterQuery,
        allow: AllowFn | None,
        depth: int,
    ) -> tuple[list[RankedChunk], list[RankedChunk]]: ...

    def get_chunk(self, chunk_id: str) -> ResolvedChunk: ...

    def get_document(self, doc_id: str) -> DocRecord | None: ...

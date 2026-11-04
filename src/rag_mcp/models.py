"""对 Agent 可见的结果类型（pydantic）。

设计依据：DESIGN.md §7「返回字段的取舍」。

三条取舍写在类型里，不是写在注释里：

- `Hit` **没有 `score`**：BM25/RRF/rerank 的分数不可比、不校准，模型读到 `0.8731`
  只会瞎比较。改为返回顺序 + 离散 `confidence`（DESIGN §7、A19）。
- `FetchResult` **没有 `security_level`**：模型不需要，且它是侧信道（DESIGN §8.3）。
- 两个结果都带 `chars_returned`：让上层能算 Agent 每次任务的上下文成本（P3）。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SecurityLevel = Literal["public", "internal", "confidential"]
#: `unknown` = 重排被显式关闭，没有判断依据。宁可说不知道，也不编一个分数（DESIGN A19）。
Confidence = Literal["high", "medium", "low", "unknown"]
#: 降级解析的三级 + 未命中。原设计的 `section` 级已废弃：`chunk_id` 里不含小节信息，
#: 单凭一个 id 无法定位「同小节的最近块」（DESIGN §9.1 修订说明）。
ResolvedBy = Literal["exact", "anchor", "ordinal", "none"]
FilterType = Literal["enum", "date"]


# ---------------------------------------------------------------- describe


class FilterField(BaseModel):
    """一个可过滤字段。值域来自词表快照（`registry.json`），不是代码常量。"""

    name: str
    type: FilterType
    values: list[str] | None = None
    description: str


class RetrievalCapabilities(BaseModel):
    """当前这套实例**实际**具备什么能力。

    存在的理由：当 embedding 是本地词法桩时，语义召回并没有生效。
    把这个事实回给 Agent 和运维，比让人去猜「为什么同义词搜不到」强得多（DESIGN §16.1）。
    """

    retrieval_mode: Literal["lexical", "hybrid"]
    embedder: str
    reranker: str
    rerank_enabled: bool
    segmenter: Literal["word", "bigram"]


class SourceInfo(BaseModel):
    id: str
    name: str
    doc_count: int
    chunk_count: int
    corpus_version: str
    schema_version: str
    config_hash: str
    updated_at: str
    filter_fields: list[FilterField]
    capabilities: RetrievalCapabilities


class DescribeResult(BaseModel):
    sources: list[SourceInfo]
    note: str


# ---------------------------------------------------------------- search


class Hit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section_path: str
    snippet: str
    confidence: Confidence
    updated_at: str


class SearchResult(BaseModel):
    query: str
    hits: list[Hit]
    returned: int
    total: int
    truncated: bool
    filters_applied: dict[str, str]
    corpus_version: str
    schema_version: str
    config_hash: str
    chars_returned: int
    hint: str | None = None


# ---------------------------------------------------------------- fetch


class FetchResult(BaseModel):
    chunk_id: str | None
    doc_id: str
    title: str
    section_path: str
    text: str
    truncated: bool
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    resolved_by: ResolvedBy = "exact"
    updated_at: str
    chars_returned: int
    hint: str | None = None


# ---------------------------------------------------------------- 元数据


class HealthReport(BaseModel):
    """`doctor` 自检输出。不是工具，只给人看。"""

    data_ok: bool
    docs: int
    chunks: int
    embedder: str
    embedder_ok: bool | None = None
    embedder_detail: str | None = None
    reranker: str
    reranker_ok: bool | None = None
    reranker_detail: str | None = None
    segmenter: str
    retrieval_mode: str
    corpus_version: str
    schema_version: str
    config_hash: str
    notes: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

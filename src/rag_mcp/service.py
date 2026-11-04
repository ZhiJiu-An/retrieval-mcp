"""检索域：编排一次检索的全部决策。

设计依据：DESIGN.md §4.1（② 层）、§6.2（管线）、§7（预算）。

**本模块不 import `mcp`** —— 业务逻辑必须能脱离协议单测（DESIGN §4.2 唯一的硬约束）。

管线（§6.2）：

```
query → 归一化 → filter 校验 → ACL 编译
      → 双路召回（词法 ‖ 向量）→ RRF 融合 → 重排
      → 同文档限流（多样性）→ 预算裁剪 → 渲染
```
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .backends.base import ChunkRecord, RetrievalBackend
from .budget import truncate_text
from .config import Settings
from .embedding import Embedder
from .errors import (
    BAD_ARGUMENT,
    CHUNK_MOVED,
    EMPTY_QUERY,
    FILTER_UNKNOWN_VALUE,
    NO_HITS,
    NOT_FOUND_OR_FORBIDDEN,
    TRUNCATED,
    RagError,
    hint_line,
)
from .fusion import rrf
from .models import (
    DescribeResult,
    FetchResult,
    FilterField,
    Hit,
    RetrievalCapabilities,
    SearchResult,
    SourceInfo,
)
from .policy import (
    Principal,
    acl_predicate,
    compile_filters,
    invalid_filter_hint,
    visibility_predicate,
)
from .registry import Registry
from .rerank import Reranker, confidence_of
from .text import as_untrusted_block, has_word_segmenter, normalize, snippet, tokenize, unique


@dataclass(frozen=True)
class ConfigFingerprint:
    """参与 `config_hash` 的一切。

    存在的理由（DESIGN §9.3）：质量回归时要能区分「语料变了」（`corpus_version`）
    和「检索配置变了」（`config_hash`）。
    """

    embedder: str
    reranker: str
    segmenter: str
    schema_version: str
    chunk_size: int
    chunk_overlap: int
    limits: dict[str, Any]

    def hash(self) -> str:
        payload = json.dumps(
            {
                "embedder": self.embedder,
                "reranker": self.reranker,
                "segmenter": self.segmenter,
                "schema_version": self.schema_version,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "limits": self.limits,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Candidate:
    """未经阈值过滤的候选，带三路原始分数。只用于标定与排障。"""

    chunk_id: str
    doc_id: str
    section_path: str
    lexical: float
    dense: float | None
    rerank: float | None
    fusion: float


@dataclass(frozen=True)
class _Scored:
    chunk: ChunkRecord
    fusion: float
    rerank: float
    confidence: str


class RetrievalService:
    def __init__(
        self,
        *,
        backend: RetrievalBackend,
        registry: Registry,
        settings: Settings,
        embedder: Embedder,
        reranker: Reranker,
        degraded_reason: str | None = None,
    ) -> None:
        self._backend = backend
        self._registry = registry
        self._settings = settings
        self._embedder = embedder
        self._reranker = reranker
        self._degraded = degraded_reason
        self._limits = settings.limits
        self._fingerprint = ConfigFingerprint(
            embedder=embedder.model_id,
            reranker=reranker.model_id,
            segmenter="word" if has_word_segmenter() else "bigram",
            schema_version=registry.schema_version,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            limits=self._limits.as_dict(),
        )
        self.config_hash = self._fingerprint.hash()

    # ------------------------------------------------------------ 元信息

    @property
    def resolved_mode(self) -> str:
        """实际生效的检索模式。

        配置只是**请求**，能不能给是另一回事：没有语义能力的向量桩
        满足不了 `hybrid` / `dense`，只能降级到 `lexical`（DESIGN §16.1）。
        """
        want = self._settings.retrieval_mode
        if want == "auto":
            return "hybrid" if self._embedder.semantic else "lexical"
        if want in ("hybrid", "dense") and not self._embedder.semantic:
            return "lexical"
        return want if want in ("hybrid", "lexical", "dense") else "lexical"

    @property
    def _use_dense(self) -> bool:
        return self.resolved_mode in ("hybrid", "dense")

    @property
    def _use_lexical(self) -> bool:
        return self.resolved_mode in ("hybrid", "lexical")

    @property
    def capabilities(self) -> RetrievalCapabilities:
        return RetrievalCapabilities(
            retrieval_mode=self.resolved_mode,  # type: ignore[arg-type]
            embedder=self._embedder.model_id,
            reranker=self._reranker.model_id,
            rerank_enabled=self._reranker.enabled,
            segmenter="word" if has_word_segmenter() else "bigram",
        )

    def describe(self) -> DescribeResult:
        info = self._backend.describe()
        source = SourceInfo(
            id=info.id,
            name=info.name,
            doc_count=info.doc_count,
            chunk_count=info.chunk_count,
            corpus_version=self._backend.corpus_version,
            schema_version=self._registry.schema_version,
            config_hash=self.config_hash,
            updated_at=info.updated_at,
            filter_fields=[
                FilterField(
                    name=field.name,
                    type=field.type,  # type: ignore[arg-type]
                    values=list(field.values) if field.values else None,
                    description=field.description,
                )
                for field in self._registry.fields
            ],
            capabilities=self.capabilities,
        )
        note = (
            "filters 的可用字段与取值以此处返回为准；传未知取值会得到空结果而不是报错。"
            f"结果条数上限 {self._limits.max_top_k}。"
        )
        if self._degraded:
            note += f" 注意：向量模型不可用（{self._degraded}）。"
        elif self._settings.retrieval_mode not in ("auto", "hybrid", "lexical", "dense"):
            note += (
                f" 注意：RAG_RETRIEVAL_MODE={self._settings.retrieval_mode!r} "
                "无法识别，已按 auto 处理。"
            )
        elif self.resolved_mode == "lexical" and not self._embedder.remote:
            note += (
                " 注意：当前向量通道使用本地词法桩，检索模式为 lexical，"
                "同义改写提问可能搜不到（配置 RAG_EMBEDDING_* 可切换到真实向量模型）。"
            )
        return DescribeResult(sources=[source], note=note)

    # ------------------------------------------------------------ 检索

    def search(
        self,
        principal: Principal,
        *,
        query: str,
        top_k: int | None = None,
        filters: dict[str, str | None] | None = None,
    ) -> SearchResult:
        normalized = normalize(query or "")
        if not normalized:
            raise RagError(
                code=EMPTY_QUERY,
                message="query 不能为空",
                hint="给出要检索的问题或关键词，优先用文档里可能出现的原词",
                retryable=True,
            )

        filter_query = compile_filters(self._registry, filters or {})
        tokens = unique(tokenize(normalized))

        scored, total = self._retrieve(principal, normalized, tokens, filter_query)
        limit = self._limits.clamp_top_k(top_k)
        selected = self._apply_diversity(scored, limit)

        hits = [
            Hit(
                chunk_id=item.chunk.chunk_id,
                doc_id=item.chunk.doc_id,
                title=str(item.chunk.meta.get("title", "")),
                section_path=item.chunk.section_path,
                snippet=snippet(item.chunk.text, tokens, self._limits.snippet_chars),
                confidence=item.confidence,  # type: ignore[arg-type]
                updated_at=str(item.chunk.meta.get("updated_at", "")),
            )
            for item in selected
        ]

        result = SearchResult(
            query=normalized,
            hits=hits,
            returned=len(hits),
            total=total,
            truncated=total > len(hits),
            filters_applied=filter_query.applied,
            corpus_version=self._backend.corpus_version,
            schema_version=self._registry.schema_version,
            config_hash=self.config_hash,
            chars_returned=0,
            hint=self._search_hint(filter_query, hits, total),
        )
        result.chars_returned = _payload_chars(result)
        return result

    def _recall_channels(
        self,
        principal: Principal,
        query: str,
        tokens: list[str],
        filter_query: Any,
    ) -> tuple[list[tuple[ChunkRecord, float]], list[tuple[ChunkRecord, float]]]:
        """两路召回的唯一入口（`_retrieve` 与 `raw_candidates` 共用，避免两套语义）。"""
        if filter_query.impossible:
            return [], []

        query_vector: Sequence[float] | None = None
        if self._use_dense and self._embedder.semantic:
            # 只有真正具备语义能力的模型才走密集通道。词法桩的「向量」是哈希桶，
            # 拿它召回只会产生碰撞噪声 —— 实测无关查询（zzzqqq）也能命中若干块。
            query_vector = self._embedder.embed([query])[0]

        # 纯语义模式下不把词送给词法通道，反之亦然。
        # `RAG_RETRIEVAL_MODE` 是安全开关：它只改变「用哪几路」，不改变分数含义。
        lexical, dense = self._backend.recall(
            tokens=tokens if self._use_lexical else [],
            query_vector=query_vector if self._use_dense else None,
            filters=filter_query,
            # 检索用「可见性」闸门（ACL + 废止状态）；按 id 取数只用 ACL，
            # 这样废止文档搜不到但仍能按 id 查阅（policy.visibility_predicate）
            allow=visibility_predicate(principal),
            depth=self._limits.recall_depth,
        )

        return lexical, dense

    def _retrieve(
        self,
        principal: Principal,
        query: str,
        tokens: list[str],
        filter_query: Any,
    ) -> tuple[list[_Scored], int]:
        if filter_query.impossible:
            # 取值不在值域内：约定是空结果，不是报错（MVP.md F2）
            return [], 0

        lexical, dense = self._recall_channels(principal, query, tokens, filter_query)

        # RRF 只用**排名**，对分数量纲免疫，所以只把排序后的块喂进去（fusion.py）
        fused = rrf([[chunk for chunk, _ in lexical], [chunk for chunk, _ in dense]])
        chunks = [chunk for chunk, _ in fused]

        if self._reranker.enabled:
            # 只把融合后的前 recall_depth 个送去重排：
            # 候选再多只是多花钱多等待，收益接近零（DESIGN §6.2 top50→8）
            ranked = self._reranker.rerank(query, tokens, chunks[: self._limits.recall_depth])
        else:
            ranked = [(chunk, 0.0) for chunk in chunks]

        # 相关性判定：**两路证据都要开口才给高分**。
        #
        # 这条规则是实测逼出来的（DESIGN §18）：
        #   「办公室停车位怎么申请」词法分 1.00（「申请」完全命中），余弦只有 0.52
        #   「HRBP 备案」是真命中，词法 0.65，余弦只有 0.51
        # 任何单路判定都会把其中一类判错，所以：
        #
        #   · 词法与语义都开口  → 取 min（保守：两路都得认可）
        #   · 只有词法开口      → 用词法分
        #   · 只有语义开口      → 余弦必须够强（dense_floor）才认，否则视为噪声
        #   · 两路都弱          → 丢弃，返回空结果并给 NO_HITS 提示
        floor = self._limits.relevance_floor if self._reranker.enabled else 0.0
        semantic = {chunk.chunk_id: similarity for chunk, similarity in dense}

        fusion_scores = {chunk.chunk_id: score for chunk, score in fused}
        scored: list[_Scored] = []
        for chunk, lexical_score in ranked:
            if self._reranker.cross_encoder:
                # cross-encoder 同时看到了问题和正文，它比「词法覆盖度 + 余弦」
                # 任何组合都准。再用合议规则去压它，只会把正确答案一起压掉。
                relevance = lexical_score
            else:
                similarity = semantic.get(chunk.chunk_id)
                if similarity is None:
                    relevance = lexical_score
                elif lexical_score > 0.0 and similarity > 0.0:
                    relevance = min(lexical_score, similarity)
                elif similarity >= self._limits.dense_floor:
                    relevance = similarity
                else:
                    relevance = lexical_score
            if relevance < floor:
                continue
            scored.append(
                _Scored(
                    chunk=chunk,
                    fusion=fusion_scores.get(chunk.chunk_id, 0.0),
                    rerank=relevance,
                    confidence=confidence_of(
                        relevance,
                        enabled=self._reranker.enabled,
                        bands=self._reranker.confidence_bands,
                    ),
                )
            )
        scored.sort(key=lambda item: item.rerank, reverse=True)
        return scored, len(scored)

    def raw_candidates(
        self,
        principal: Principal,
        *,
        query: str,
        limit: int = 50,
        filters: dict[str, str | None] | None = None,
    ) -> list[Candidate]:
        """返回**未经阈值过滤**的候选及其三路分数。

        用途：`calibrate` 标定阈值时需要一个没被阈值影响的原始分布；
        排障时也需要知道「到底是哪一路没命中」。这不是给模型用的 API。
        """
        normalized = normalize(query or "")
        if not normalized:
            raise RagError(
                code=EMPTY_QUERY, message="query 不能为空", hint="给出要检索的问题", retryable=True
            )
        filter_query = compile_filters(self._registry, filters or {})
        tokens = unique(tokenize(normalized))

        lexical, dense = self._recall_channels(principal, normalized, tokens, filter_query)
        fused = rrf([[chunk for chunk, _ in lexical], [chunk for chunk, _ in dense]])
        chunks = [chunk for chunk, _ in fused][:limit]

        lexical_map = {chunk.chunk_id: score for chunk, score in lexical}
        dense_map = {chunk.chunk_id: score for chunk, score in dense}
        ranked = (
            self._reranker.rerank(normalized, tokens, chunks)
            if self._reranker.enabled
            else [(chunk, None) for chunk in chunks]
        )
        return [
            Candidate(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                section_path=chunk.section_path,
                lexical=round(lexical_map.get(chunk.chunk_id, 0.0), 4),
                dense=(
                    None if chunk.chunk_id not in dense_map else round(dense_map[chunk.chunk_id], 4)
                ),
                rerank=None if score is None else round(float(score), 4),
                fusion=round(next((f for c, f in fused if c.chunk_id == chunk.chunk_id), 0.0), 6),
            )
            for chunk, score in ranked
        ]

    def _apply_diversity(self, scored: list[_Scored], limit: int) -> list[_Scored]:
        """同一文档最多占 `per_doc_limit` 条。

        没有这一步，top-8 会全是同一篇的 8 个相邻块 —— 召回看似很高，
        实际只覆盖了一个文档（DESIGN §6.2）。
        """
        chosen: list[_Scored] = []
        per_doc: dict[str, int] = {}
        for item in scored:
            if len(chosen) >= limit:
                break
            doc_id = item.chunk.doc_id
            if per_doc.get(doc_id, 0) >= self._limits.per_doc_limit:
                continue
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            chosen.append(item)
        return chosen

    def _search_hint(self, filter_query: Any, hits: list[Hit], total: int) -> str | None:
        if filter_query.impossible:
            return hint_line(
                FILTER_UNKNOWN_VALUE,
                invalid_filter_hint(filter_query, self._registry)
                + "。已按空结果处理，去掉该过滤条件再试",
            )
        if not hits:
            return hint_line(
                NO_HITS,
                "没有命中。可以：去掉 department/doc_type 过滤、改用文档里可能出现的原词、"
                "或调大 top_k",
            )
        if total > len(hits):
            return hint_line(
                TRUNCATED,
                f"共 {total} 条相关块，这里只返回 {len(hits)} 条；"
                "需要更多请调大 top_k 或收紧过滤条件",
            )
        return None

    # ------------------------------------------------------------ 展开

    def fetch(
        self,
        principal: Principal,
        *,
        chunk_id: str | None = None,
        doc_id: str | None = None,
        window: int | None = None,
    ) -> FetchResult:
        if bool(chunk_id) == bool(doc_id):
            raise RagError(
                code=BAD_ARGUMENT,
                message="chunk_id 与 doc_id 必须且只能提供一个",
                hint="要读片段上下文用 chunk_id；要读整篇用 doc_id",
                retryable=True,
            )

        allow = acl_predicate(principal)
        if doc_id:
            return self._fetch_document(doc_id, allow)
        assert chunk_id is not None
        return self._fetch_chunk(chunk_id, self._limits.clamp_window(window), allow)

    def _fetch_document(self, doc_id: str, allow: Any) -> FetchResult:
        doc = self._backend.get_document(doc_id)
        if doc is None or not doc.chunks or not allow(doc.meta):
            raise self._not_found(f"doc_id={doc_id}")

        text, truncated = truncate_text(doc.text, self._limits.max_fetch_chars)
        result = FetchResult(
            chunk_id=None,
            doc_id=doc.doc_id,
            title=doc.title,
            section_path=doc.title,
            text=as_untrusted_block(text, f"{doc.title} | 整篇"),
            truncated=truncated,
            resolved_by="exact",
            updated_at=str(doc.meta.get("updated_at", "")),
            chars_returned=0,
            hint=hint_line(TRUNCATED, "整篇已截断；改用 chunk 模式按段读取") if truncated else None,
        )
        result.chars_returned = _payload_chars(result)
        return result

    def _fetch_chunk(self, chunk_id: str, window: int, allow: Any) -> FetchResult:
        resolved = self._backend.get_chunk(chunk_id)
        chunk = resolved.chunk
        if chunk is None or not allow(chunk.meta):
            # 越权与不存在**同构**：错误里不能出现「无权限」（DESIGN §8.3）
            raise self._not_found(f"chunk_id={chunk_id}")

        doc = self._backend.get_document(chunk.doc_id)
        if doc is None:
            raise self._not_found(f"chunk_id={chunk_id}")

        first = max(0, chunk.position - window)
        last = min(len(doc.chunks) - 1, chunk.position + window)
        selected = list(doc.chunks[first : last + 1])

        text, truncated = truncate_text(
            _compose(selected), self._limits.max_fetch_chars
        )

        hint: str | None = None
        if resolved.resolved_by != "exact":
            hint = hint_line(
                CHUNK_MOVED,
                f"该 id 已不是原有位置（resolved_by={resolved.resolved_by}），语料可能已更新；"
                "建议重新 search 确认",
            )
        elif truncated:
            hint = hint_line(TRUNCATED, "上下文已截断；改用更小的 window，或按 doc_id 分段读取")

        result = FetchResult(
            chunk_id=chunk.chunk_id,
            doc_id=doc.doc_id,
            title=doc.title,
            section_path=chunk.section_path,
            text=as_untrusted_block(text, f"{doc.title} §{chunk.heading or chunk.section_path}"),
            truncated=truncated,
            prev_chunk_id=doc.chunks[first - 1].chunk_id if first > 0 else None,
            next_chunk_id=doc.chunks[last + 1].chunk_id if last + 1 < len(doc.chunks) else None,
            resolved_by=resolved.resolved_by,  # type: ignore[arg-type]
            updated_at=str(doc.meta.get("updated_at", "")),
            chars_returned=0,
            hint=hint,
        )
        result.chars_returned = _payload_chars(result)
        return result

    def read_document(self, principal: Principal, doc_id: str) -> str:
        """resource `rag://docs/{doc_id}` 的内容。

        **独立复核 ACL**：不能因为 id 来自 search 就放行 ——
        resource 是最常见的鉴权绕过点（DESIGN §8.3 铁律 2）。
        """
        doc = self._backend.get_document(doc_id)
        if doc is None or not doc.chunks or not acl_predicate(principal)(doc.meta):
            raise self._not_found(f"doc_id={doc_id}")

        text, truncated = truncate_text(doc.text, self._limits.max_resource_chars)
        body = as_untrusted_block(text, f"{doc.title} | 整篇（resource）")
        header = f"# {doc.title}\n\n> 更新于 {doc.meta.get('updated_at', '')}\n\n"
        if truncated:
            body += "\n\n[文档过长已截断：请改用 fetch(chunk_id) 按段读取剩余部分]"
        return header + body

    # ------------------------------------------------------------ 工具

    @staticmethod
    def _not_found(ident: str) -> RagError:
        return RagError(
            code=NOT_FOUND_OR_FORBIDDEN,
            message=f"内容不存在或不可访问：{ident}",
            hint="先用 search 获取有效的 chunk_id / doc_id，不要凭记忆或猜测传参",
            retryable=False,
        )


def _compose(chunks: list[ChunkRecord]) -> str:
    """把窗口内的块拼成一段文本。

    small-to-big：同一小节的多个小块只输出**一次**父块全文，避免重复内容
    把上下文预算吃掉一倍（DESIGN §6.1、§7）。
    """
    parts: list[str] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        key = (chunk.doc_id, chunk.section_path)
        if key in seen:
            continue
        seen.add(key)
        parts.append(chunk.parent_text or chunk.text)
    return "\n\n".join(parts)


def _payload_chars(model: Any) -> int:
    """响应的字符数（不含 `chars_returned` 自身，保证结果稳定可复现）。"""
    payload = model.model_dump(exclude={"chars_returned"})
    return len(json.dumps(payload, ensure_ascii=False, default=str))

"""本地语料后端：开发、测试与离线演示用。

设计依据：DESIGN.md §6.1（索引侧行为）、§9.1（chunk_id 与降级解析）、A13（取消时效衰减）。

它**按生产后端的语义实现**，而不是按「够测试就行」实现：

- 结构化切分：按 section 切，超长才继续切 —— 保住 `section_path` 这个语义锚点
- 小块入库、父块返回（small-to-big）
- 双路召回：BM25 + 向量，**过滤与 ACL 全部在打分之前**
- `chunk_id` 带内容指纹，支持降级解析
- **没有时效衰减**：v0 的 `0.85^年份` 乘性衰减已删除（A13）。
  「最新」不等于「最相关」，而且它会污染分数语义、与重排打架。
  时效需求走显式 `updated_after` 过滤，或将来做成重排特征。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..cache import VectorCache, cache_key
from ..embedding import Embedder
from ..policy import AllowFn, FilterQuery
from ..text import content_anchor, tokenize
from .base import (
    ChunkRecord,
    CollectionInfo,
    DocRecord,
    RankedChunk,
    ResolvedChunk,
    parse_chunk_id,
)

#: BM25 的经典参数，不做「调参」的自欺：语料这么小，调了也测不出来。
BM25_K1 = 1.2
BM25_B = 0.75

_SENTENCE_BREAKS = ("。", "！", "？", "；", "\n")


class MemoryBackend:
    """只读本地语料（JSON）。生产替换为 pgvector / es，语义不变。"""

    def __init__(
        self,
        data_path: Path,
        *,
        embedder: Embedder,
        chunk_size: int = 360,
        chunk_overlap: int = 48,
    ) -> None:
        raw = data_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        self._source = payload["source"]
        self.corpus_version: str = hashlib.sha256(raw).hexdigest()[:12]
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        self._chunks: list[ChunkRecord] = []
        self._docs: dict[str, DocRecord] = {}
        self._build(payload)

        self._index: dict[str, int] = {
            chunk.chunk_id: position for position, chunk in enumerate(self._chunks)
        }
        # BM25 索引字段 = 小节标题 + 正文。标题命中在真实引擎里也是靠字段加权实现的，
        # 这里用最朴素的方式表达同一个意图（DESIGN §6.2）。
        self._tokens = [tokenize(f"{chunk.heading}\n{chunk.text}") for chunk in self._chunks]
        self._counts = [Counter(tokens) for tokens in self._tokens]
        self._doc_freq: Counter[str] = Counter()
        for counts in self._counts:
            self._doc_freq.update(counts.keys())
        self._avg_len = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens) if self._chunks else 0.0
        )

        # 双路召回的向量通道。只有真正具备语义能力的模型才建索引：
        # 词法桩建出来的只是哈希桶，既没有语义价值，又会让无关查询命中噪声。
        self._vectors: list[list[float]] = self._build_vectors()

    def _build_vectors(self) -> list[list[float]]:
        """建向量索引：先查磁盘缓存，未命中才发请求。

        缓存键含 `corpus_version` 与 embedder 标识 —— 语料或模型变了就重建，
        绝不拿别的语料/别的模型的向量来用（DESIGN §3.2、§9.3）。
        """
        if not self._chunks or not self._embedder.semantic:
            return []

        cache = VectorCache()
        key = cache_key(
            "memory-v1",
            self.corpus_version,
            self._embedder.model_id,
            self._chunk_size,
            self._chunk_overlap,
            len(self._chunks),
        )
        cached = cache.load(key)
        if cached is not None and len(cached) == len(self._chunks):
            return cached

        vectors = self._embedder.embed([_embed_input(chunk) for chunk in self._chunks])
        cache.save(
            key,
            vectors,
            meta={
                "corpus_version": self.corpus_version,
                "embedder": self._embedder.model_id,
                "chunks": len(self._chunks),
            },
        )
        return vectors

    # ------------------------------------------------------------ 构建

    def _build(self, payload: Mapping[str, Any]) -> None:
        for doc in payload["documents"]:
            meta: dict[str, Any] = {
                "title": doc["title"],
                "department": doc["department"],
                "doc_type": doc["doc_type"],
                "security_level": doc["security_level"],
                "acl_groups": list(doc.get("acl_groups") or []),
                "doc_status": str(doc.get("status", "active")),
                "updated_at": doc["updated_at"],
            }

            chunks: list[ChunkRecord] = []
            for section in doc["sections"]:
                heading = section.get("heading", "")
                section_body = section["text"]
                section_path = f"{doc['title']} › {heading}" if heading else doc["title"]
                for piece in _split_section(section_body, self._chunk_size, self._chunk_overlap):
                    ordinal = len(chunks)
                    chunk = ChunkRecord(
                        chunk_id=f"{doc['doc_id']}#c{ordinal:03d}-{content_anchor(piece)}",
                        doc_id=doc["doc_id"],
                        position=ordinal,
                        section_path=section_path,
                        heading=heading,
                        text=piece,
                        parent_text=section_body,  # 父块 = 整个小节
                        anchor=content_anchor(piece),
                        meta=meta,
                    )
                    chunks.append(chunk)

            full_text = "\n\n".join(
                f"## {section.get('heading', '')}\n{section['text']}"
                if section.get("heading")
                else section["text"]
                for section in doc["sections"]
            )
            self._docs[doc["doc_id"]] = DocRecord(
                doc_id=doc["doc_id"],
                title=doc["title"],
                text=full_text,
                meta=meta,
                chunks=tuple(chunks),
            )
            self._chunks.extend(chunks)

    # ------------------------------------------------------------ 元信息

    def describe(self) -> CollectionInfo:
        return CollectionInfo(
            id=self._source["id"],
            name=self._source["name"],
            doc_count=len(self._docs),
            chunk_count=len(self._chunks),
            updated_at=self._source["updated_at"],
        )

    # ------------------------------------------------------------ 召回

    def recall(
        self,
        *,
        tokens: Sequence[str],
        query_vector: Sequence[float] | None,
        filters: FilterQuery,
        allow: AllowFn | None,
        depth: int,
    ) -> tuple[list[RankedChunk], list[RankedChunk]]:
        if filters.impossible:
            # 取值不在值域内 —— 约定行为是空结果，不是报错（MVP.md F2）
            return [], []

        # pre-filter：元数据过滤与 ACL 都在打分之前完成，绝不在 top-k 之后裁权限
        candidates = [
            position
            for position, chunk in enumerate(self._chunks)
            if _matches(chunk.meta, filters.values) and (allow is None or allow(chunk.meta))
        ]
        if not candidates:
            return [], []

        lexical = self._bm25(list(tokens), candidates, depth)
        dense = self._dense(query_vector, candidates, depth)
        return lexical, dense

    def _idf(self, term: str) -> float:
        doc_freq = self._doc_freq.get(term, 0)
        if doc_freq == 0:
            return 0.0
        total = len(self._chunks)
        return math.log(1 + (total - doc_freq + 0.5) / (doc_freq + 0.5))

    def _bm25(self, terms: list[str], candidates: list[int], depth: int) -> list[RankedChunk]:
        if not terms:
            return []

        scored: list[RankedChunk] = []
        for position in candidates:
            counts = self._counts[position]
            length = len(self._tokens[position]) or 1
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                score += self._idf(term) * (frequency * (BM25_K1 + 1)) / (
                    frequency + BM25_K1 * (1 - BM25_B + BM25_B * length / (self._avg_len or 1))
                )
            if score > 0:
                scored.append((self._chunks[position], score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:depth]

    def _dense(
        self, query_vector: Sequence[float] | None, candidates: list[int], depth: int
    ) -> list[RankedChunk]:
        if query_vector is None or not any(query_vector) or not self._vectors:
            return []

        scored: list[RankedChunk] = []
        for position in candidates:
            similarity = _cosine(query_vector, self._vectors[position])
            if similarity > 0.0:
                scored.append((self._chunks[position], similarity))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:depth]

    # ------------------------------------------------------------ 取数

    def get_chunk(self, chunk_id: str) -> ResolvedChunk:
        """三级降级解析：精确 → 内容指纹 → 位置（DESIGN §9.1）。

        任何非 `exact` 的命中都说明「这块内容变了」，上层必须如实转告 Agent。
        """
        position = self._index.get(chunk_id.strip())
        if position is not None:
            return ResolvedChunk(chunk=self._chunks[position], resolved_by="exact")

        doc_id, ordinal, anchor = parse_chunk_id(chunk_id)
        if doc_id is None:
            return ResolvedChunk(chunk=None, resolved_by="none")

        doc = self._docs.get(doc_id)
        if doc is None:
            return ResolvedChunk(chunk=None, resolved_by="none")

        if anchor:
            for chunk in doc.chunks:
                if chunk.anchor == anchor:
                    return ResolvedChunk(chunk=chunk, resolved_by="anchor")

        if ordinal is not None and 0 <= ordinal < len(doc.chunks):
            return ResolvedChunk(chunk=doc.chunks[ordinal], resolved_by="ordinal")

        return ResolvedChunk(chunk=None, resolved_by="none")

    def get_document(self, doc_id: str) -> DocRecord | None:
        return self._docs.get(doc_id.strip())


# ---------------------------------------------------------------- 工具函数


def _matches(meta: Mapping[str, Any], filters: Mapping[str, str]) -> bool:
    for name, value in filters.items():
        if name == "updated_after":
            # ISO 日期字符串的字典序等于时间序，不需要解析成 date
            if str(meta.get("updated_at", "")) < value:
                return False
        elif str(meta.get(name, "")) != value:
            return False
    return True


def _embed_input(chunk: ChunkRecord) -> str:
    """向量化时带上小节标题，让「这一节讲什么」进入语义空间。"""
    return f"{chunk.heading}\n{chunk.text}" if chunk.heading else chunk.text


def _split_section(text: str, size: int, overlap: int) -> list[str]:
    """按长度切块，尽量在句末断开，带少量重叠避免答案被边界切断。"""
    size = max(80, size)
    overlap = max(0, min(overlap, size // 2))

    pieces: list[str] = []
    start = 0
    total = len(text)
    while start < total:
        end = min(start + size, total)
        if end < total:
            window = text[start:end]
            cut = max(window.rfind(mark) for mark in _SENTENCE_BREAKS)
            if cut > size * 0.4:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= total:
            break
        start = max(end - overlap, start + 1)
    return pieces


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False))

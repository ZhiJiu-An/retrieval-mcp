"""SQLite 后端：持久化索引，一条命令建库，无需部署任何服务。

设计依据：DESIGN.md §4.1（③ 后端适配只做 4 个动作）、§8.3（pre-filter）。

为什么是 SQLite：

- 只需要一个 `.db` 文件，不用装 Postgres / Elasticsearch；
- FTS5 提供词法召回（`tokens` 列是我们自己分词的，中文才切得动）；
- 向量放在 `chunk_vectors` 表里，查询时全量算余弦。

**已知上限（诚实写清楚）**：向量通道是 O(N) 全扫 + Python 循环。
884 个块约 30–50ms，够用；到 10 万块必须换成 sqlite-vec 或 pgvector 的 ANN 索引。
换的时候只要改这个文件，`service` 与协议层不动 —— 这正是 4 方法协议的意义。
"""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..embedding import Embedder
from ..policy import AllowFn, FilterQuery
from ..text import unique as unique_tokens
from .base import (
    ChunkRecord,
    CollectionInfo,
    DocRecord,
    RankedChunk,
    ResolvedChunk,
    parse_chunk_id,
)

#: FTS5 候选窗口。要足够大，才能让「先过滤再排序」接近精确；
#: 太小会变成「排序后截断再过滤」，那是后过滤，会漏结果。
FTS_WINDOW = 2000

_SCALAR_COLUMNS = ("department", "doc_type", "updated_at", "level_rank", "doc_status")


class SqliteBackend:
    def __init__(self, db_path: Path, *, embedder: Embedder) -> None:
        if not db_path.exists():
            raise FileNotFoundError(
                f"索引不存在：{db_path}"
                "。请先建库：python -m rag_mcp.ingest --source ... --out ..."
            )
        self.db_path = db_path
        self._embedder = embedder
        self._con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._con.row_factory = sqlite3.Row

        meta = {
            row["key"]: json.loads(row["value"])
            for row in self._con.execute("select key,value from meta")
        }
        self.corpus_version = str(meta.get("corpus_version", ""))
        self.index_embedder = str(meta.get("embedder", ""))
        self._doc_count = int(meta.get("docs", 0))
        self._chunk_count = int(meta.get("chunks", 0))

        # 向量常驻内存：这是唯一「重算代价高」的部分，读一次比反复查库划算
        self._vectors: dict[str, list[float]] = {}
        for row in self._con.execute("select chunk_id, dim, vec from chunk_vectors"):
            count = int(row["dim"])
            self._vectors[row["chunk_id"]] = list(struct.unpack(f"<{count}f", row["vec"]))

    # ------------------------------------------------------------ 元信息

    def describe(self) -> CollectionInfo:
        row = self._con.execute("select title, text from documents limit 1").fetchone()
        return CollectionInfo(
            id=self.db_path.stem,
            name=f"SQLite 索引（{row['title'] if row else self.db_path.name}）",
            doc_count=self._doc_count,
            chunk_count=self._chunk_count,
            updated_at=str(self.corpus_version),
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
            return [], []

        allowed = self._allowed_chunks(filters, allow)
        if not allowed:
            return [], []

        return (
            self._lexical(unique_tokens(list(tokens)), allowed, depth),
            self._dense(query_vector, allowed, depth),
        )

    def _allowed_chunks(
        self, filters: FilterQuery, allow: AllowFn | None
    ) -> dict[str, sqlite3.Row]:
        """**pre-filter**：先算出可见集合，再做任何排序。

        SQL 负责标量条件（部门/类型/日期），`allow` 负责 ACL 与文档状态 ——
        后者是 service 传进来的同一个谓词，符号无法下推到 SQL，所以在候选集上算。
        因为集合在这一步就已经缩小，后续排序完全建立在「可见数据」之上。
        """
        where: list[str] = []
        params: list[Any] = []
        for name, value in filters.values.items():
            if name == "updated_after":
                where.append("updated_at >= ?")
                params.append(value)
            elif name in ("department", "doc_type"):
                where.append(f"{name} = ?")
                params.append(value)
            # 其余字段留给 allow（词表里目前只有这三个）
        clause = f" where {' and '.join(where)}" if where else ""
        rows = self._con.execute(
            f"select chunk_id,{','.join(_SCALAR_COLUMNS)},meta from chunks{clause}", params
        ).fetchall()

        allowed: dict[str, sqlite3.Row] = {}
        for row in rows:
            meta = json.loads(row["meta"])
            if allow is None or allow(meta):
                allowed[row["chunk_id"]] = row
        return allowed

    def _lexical(
        self, tokens: list[str], allowed: dict[str, sqlite3.Row], depth: int
    ) -> list[RankedChunk]:
        if not tokens:
            return []
        # 每个词都加引号：FTS5 的布尔语法字符（* : ^ - 等）会被误解析
        match = " OR ".join(f'"{token}"' for token in tokens)
        try:
            rows = self._con.execute(
                "select chunk_id, bm25(chunk_fts) as rank from chunk_fts "
                "where chunk_fts match ? order by rank limit ?",
                (match, FTS_WINDOW),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # 查询语法异常时静默退化为无词法命中，向量通道仍可用

        picked: list[tuple[str, float]] = [
            (row["chunk_id"], -float(row["rank"]))
            for row in rows
            if row["chunk_id"] in allowed
        ][:depth]
        return self._ranked(picked)

    def _dense(
        self, query_vector: Sequence[float] | None, allowed: dict[str, sqlite3.Row], depth: int
    ) -> list[RankedChunk]:
        if not query_vector or not any(query_vector) or not self._vectors:
            return []

        query_norm = _norm(query_vector) or 1.0
        picked: list[tuple[str, float]] = []
        for chunk_id in allowed:
            vector = self._vectors.get(chunk_id)
            if vector is None:
                continue
            similarity = _dot(query_vector, vector) / (query_norm * (_norm(vector) or 1.0))
            if similarity > 0.0:
                picked.append((chunk_id, similarity))

        picked.sort(key=lambda pair: pair[1], reverse=True)
        return self._ranked(picked[:depth])

    def _ranked(self, picked: list[tuple[str, float]]) -> list[RankedChunk]:
        records = self._load_chunks([chunk_id for chunk_id, _ in picked])
        return [(records[cid], score) for cid, score in picked if cid in records]

    def _load_chunks(self, chunk_ids: Sequence[str]) -> dict[str, ChunkRecord]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._con.execute(
            "select chunk_id,doc_id,ordinal,section_path,heading,text,parent_text,anchor,meta "
            f"from chunks where chunk_id in ({placeholders})",
            list(chunk_ids),
        ).fetchall()
        return {row["chunk_id"]: _to_chunk(row) for row in rows}

    # ------------------------------------------------------------ 取数

    def get_chunk(self, chunk_id: str) -> ResolvedChunk:
        """三级降级解析：精确 → 内容指纹 → 位置（DESIGN §9.1）。"""
        row = self._con.execute(
            "select chunk_id,doc_id,ordinal,section_path,heading,text,parent_text,anchor,meta "
            "from chunks where chunk_id = ?",
            (chunk_id.strip(),),
        ).fetchone()
        if row is not None:
            return ResolvedChunk(chunk=_to_chunk(row), resolved_by="exact")

        doc_id, ordinal, anchor = parse_chunk_id(chunk_id)
        if doc_id is None:
            return ResolvedChunk(chunk=None, resolved_by="none")

        if anchor:
            row = self._con.execute(
                "select chunk_id,doc_id,ordinal,section_path,heading,text,parent_text,anchor,meta "
                "from chunks where doc_id = ? and anchor = ? limit 1",
                (doc_id, anchor),
            ).fetchone()
            if row is not None:
                return ResolvedChunk(chunk=_to_chunk(row), resolved_by="anchor")

        if ordinal is not None:
            row = self._con.execute(
                "select chunk_id,doc_id,ordinal,section_path,heading,text,parent_text,anchor,meta "
                "from chunks where doc_id = ? and ordinal = ? limit 1",
                (doc_id, ordinal),
            ).fetchone()
            if row is not None:
                return ResolvedChunk(chunk=_to_chunk(row), resolved_by="ordinal")

        return ResolvedChunk(chunk=None, resolved_by="none")

    def get_document(self, doc_id: str) -> DocRecord | None:
        row = self._con.execute(
            "select doc_id,title,text,meta from documents where doc_id = ?", (doc_id.strip(),)
        ).fetchone()
        if row is None:
            return None
        chunks = self._con.execute(
            "select chunk_id,doc_id,ordinal,section_path,heading,text,parent_text,anchor,meta "
            "from chunks where doc_id = ? order by ordinal",
            (row["doc_id"],),
        ).fetchall()
        return DocRecord(
            doc_id=row["doc_id"],
            title=row["title"],
            text=row["text"],
            meta=json.loads(row["meta"]),
            chunks=tuple(_to_chunk(chunk) for chunk in chunks),
        )

    def close(self) -> None:
        self._con.close()


def _to_chunk(row: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        position=int(row["ordinal"]),
        section_path=row["section_path"],
        heading=row["heading"],
        text=row["text"],
        parent_text=row["parent_text"],
        anchor=row["anchor"],
        meta=json.loads(row["meta"]),
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def _norm(vector: Sequence[float]) -> float:
    return sum(value * value for value in vector) ** 0.5

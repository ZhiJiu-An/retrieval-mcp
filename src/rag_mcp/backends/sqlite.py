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
import re
import sqlite3
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..embedding import Embedder
from ..errors import BAD_CONFIGURATION, RagError
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
            raise RagError(
                code=BAD_CONFIGURATION,
                message=f"索引不存在：{db_path}",
                hint="先建库：python -m rag_mcp.ingest --source ... --out ...",
                retryable=False,
                detail={"db_path": str(db_path)},
            )
        self.db_path = db_path
        self._embedder = embedder
        try:
            self._con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            self._con.row_factory = sqlite3.Row

            meta = {
                row["key"]: json.loads(row["value"])
                for row in self._con.execute("select key,value from meta")
            }
            self.corpus_version = str(meta.get("corpus_version", ""))
            self.index_embedder = str(meta.get("embedder", ""))
            self._built_at = str(meta.get("built_at", ""))
            self._doc_count = int(meta.get("docs", 0))
            self._chunk_count = int(meta.get("chunks", 0))

            # 向量常驻内存：这是唯一「重算代价高」的部分，读一次比反复查库划算。
            # 同时缓存 L2 范数，避免每次 query 对每个候选重复开方。
            self._vectors: dict[str, list[float]] = {}
            self._norms: dict[str, float] = {}
            for row in self._con.execute("select chunk_id, dim, vec from chunk_vectors"):
                count = int(row["dim"])
                vector = list(struct.unpack(f"<{count}f", row["vec"]))
                self._vectors[row["chunk_id"]] = vector
                self._norms[row["chunk_id"]] = _norm(vector)
            # (filters, ACL, allow) -> allowed chunk rows。语料只读，可安全缓存；
            # 上限防止很多 principal/filter 组合无限增长。
            self._allowed_cache: dict[tuple[Any, ...], dict[str, sqlite3.Row]] = {}
        except (sqlite3.Error, OSError, ValueError, struct.error) as exc:
            # 索引损坏、权限不足、schema 不对：这是配置/数据问题，不是 embedding
            # 不可用，不能降级到另一个语料继续服务。
            con = getattr(self, "_con", None)
            if con is not None:
                con.close()
            raise RagError(
                code=BAD_CONFIGURATION,
                message=f"索引不可用：{db_path}（{type(exc).__name__}）",
                hint="确认文件完整、有读权限，或重新运行 python -m rag_mcp.ingest",
                retryable=False,
                detail={"db_path": str(db_path), "error": str(exc)[:200]},
            ) from exc

    # ------------------------------------------------------------ 元信息

    def describe(self) -> CollectionInfo:
        row = self._con.execute("select title from documents limit 1").fetchone()
        return CollectionInfo(
            id=self.db_path.stem,
            name=f"SQLite 索引（{row['title'] if row else self.db_path.name}）",
            doc_count=self._doc_count,
            chunk_count=self._chunk_count,
            updated_at=self._built_at,
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

        allowed, where, params = self._allowed_chunks(filters, allow)
        if not allowed:
            return [], []

        return (
            self._lexical(unique_tokens(list(tokens)), allowed, depth, where, params),
            self._dense(query_vector, allowed, depth),
        )

    def _base_where(self, filters: FilterQuery) -> tuple[list[str], list[Any]]:
        """把标量与 ACL 条件编译成 SQL，供 allowed 与 FTS 共用。

        以前 ACL/状态只在 Python 里过滤：FTS 先取 2000 条再丢，ACL 严格时
        会漏掉真正的命中。现在 `chunk_acl` / `level_rank` / `doc_status`
        真正参与 SQL pre-filter；`allow` 仍保留做最后复核。
        """
        where: list[str] = []
        params: list[Any] = []
        for name, value in filters.values.items():
            column = self._sql_column(filters, name)
            if name == "updated_after":
                where.append(f'chunks."{column}" >= ?')
            else:
                where.append(f'chunks."{column}" = ?')
            params.append(value)
        if filters.exclude_superseded:
            where.append("chunks.doc_status != 'superseded'")
        if filters.max_level_rank is not None:
            where.append("chunks.level_rank <= ?")
            params.append(filters.max_level_rank)
        if filters.groups is not None:
            no_acl = "NOT EXISTS (SELECT 1 FROM chunk_acl ca WHERE ca.chunk_id = chunks.chunk_id)"
            if not filters.groups:
                where.append(no_acl)
            else:
                placeholders = ",".join("?" * len(filters.groups))
                where.append(
                    f"({no_acl} OR EXISTS (SELECT 1 FROM chunk_acl ca "
                    f"WHERE ca.chunk_id = chunks.chunk_id AND ca.grp IN ({placeholders})))"
                )
                params.extend(sorted(filters.groups))
        return where, params

    @staticmethod
    def _sql_column(filters: FilterQuery, name: str) -> str:
        column = filters.sql_columns.get(name, name)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column):
            raise RagError(
                code=BAD_CONFIGURATION,
                message=f"非法 SQL 列名：{column!r}",
                hint="检查 registry.json 的 sql_column",
                retryable=False,
            )
        return column

    def _allowed_chunks(
        self, filters: FilterQuery, allow: AllowFn | None
    ) -> tuple[dict[str, sqlite3.Row], list[str], list[Any]]:
        """**pre-filter**：先算出可见集合，再做任何排序。

        SQL 负责标量条件、密级、ACL 组和文档状态；`allow` 仍做最后复核，
        防止未来 policy 增加 SQL 无法表达的谓词。
        """
        where, params = self._base_where(filters)
        key = (
            tuple(sorted(filters.values.items())),
            filters.max_level_rank,
            tuple(sorted(filters.groups or ())),
            filters.exclude_superseded,
            tuple(sorted(filters.sql_columns.items())),
            allow,
        )
        cached = self._allowed_cache.get(key)
        if cached is not None:
            return cached, where, params

        clause = f" where {' and '.join(where)}" if where else ""
        rows = self._con.execute(
            f"select chunk_id,{','.join(_SCALAR_COLUMNS)},meta from chunks{clause}", params
        ).fetchall()

        allowed: dict[str, sqlite3.Row] = {}
        for row in rows:
            meta = json.loads(row["meta"])
            if allow is None or allow(meta):
                allowed[row["chunk_id"]] = row
        if len(self._allowed_cache) >= 32:
            self._allowed_cache.pop(next(iter(self._allowed_cache)))
        self._allowed_cache[key] = allowed
        return allowed, where, params

    def _lexical(
        self,
        tokens: list[str],
        allowed: dict[str, sqlite3.Row],
        depth: int,
        where: list[str],
        params: list[Any],
    ) -> list[RankedChunk]:
        if not tokens:
            return []
        # 每个词都加引号：FTS5 的布尔语法字符（* : ^ - 等）会被误解析
        match = " OR ".join(f'"{token}"' for token in tokens)
        where_sql = f" and {' and '.join(where)}" if where else ""
        try:
            rows = self._con.execute(
                "select chunk_fts.chunk_id, bm25(chunk_fts) as rank "
                "from chunk_fts join chunks on chunks.chunk_id = chunk_fts.chunk_id "
                f"where chunk_fts match ?{where_sql} order by rank limit ?",
                (match, *params, FTS_WINDOW),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # 查询语法异常时静默退化为无词法命中，向量通道仍可用

        picked: list[tuple[str, float]] = [
            (row["chunk_id"], -float(row["rank"])) for row in rows if row["chunk_id"] in allowed
        ][:depth]
        return self._ranked(picked)

    def _dense(
        self, query_vector: Sequence[float] | None, allowed: dict[str, sqlite3.Row], depth: int
    ) -> list[RankedChunk]:
        if not query_vector or not any(query_vector) or not self._vectors:
            return []

        index_dims = {len(vector) for vector in self._vectors.values()}
        if len(query_vector) not in index_dims:
            raise RagError(
                code=BAD_CONFIGURATION,
                message="查询向量维度与索引不一致",
                hint="对齐 RAG_EMBEDDING_MODEL，或重建 rag.db",
                retryable=False,
                detail={"query_dim": len(query_vector), "index_dims": sorted(index_dims)},
            )

        query_norm = _norm(query_vector) or 1.0
        picked: list[tuple[str, float]] = []
        for chunk_id in allowed:
            vector = self._vectors.get(chunk_id)
            if vector is None:
                continue
            similarity = _dot(query_vector, vector) / (
                query_norm * (self._norms.get(chunk_id) or 1.0)
            )
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
        con = getattr(self, "_con", None)
        if con is not None:
            con.close()
            self._con = None  # type: ignore[assignment]

    def __enter__(self) -> SqliteBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


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
    if len(left) != len(right):
        # 静默截断会让不同 embedding 空间算出“看似正常”的相似度；
        # 宁可报错，也不能把错误维度当成命中。
        raise ValueError(f"向量维度不一致：{len(left)} vs {len(right)}")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _norm(vector: Sequence[float]) -> float:
    return sum(value * value for value in vector) ** 0.5

"""入库 CLI：把任意语料变成可查询的 `rag.db`。

设计依据：DESIGN.md §4.1（④ 入库管道独立于 ③ 后端）、§9.2（chunk 元数据）。

为什么单独一个入口而不是塞进 server：入库要跑 embedding、要几分钟、要能重跑，
而 stdio server 由客户端随时拉起/杀掉（§3.2）。**同一个仓库，不同进程。**

用法：

```bash
# 从本项目内置的 JSON 语料建库（会调用向量模型）
python -m rag_mcp.ingest --source json --data src/rag_mcp/data/sample_docs.json --out rag.db

# 从已有向量库建库（复用现存向量，不重新 embedding）
python -m rag_mcp.ingest --source tcm-sqlite --db /path/to/vector_db.sqlite --out rag.db
```

产物 `rag.db` 的结构：

- `documents` / `chunks`：文档与块（含 pre-filter 需要的元数据列）
- `chunk_acl`：`(chunk_id, grp)` 一行一组，用于 SQL 里做真正的 pre-filter
- `chunk_vectors`：float32 向量（可选，来自源库或现算）
- `chunk_fts`：FTS5 全文索引，`tokens` 列是我们**自己分词后**的结果
- `meta`：`corpus_version` / `embedder` / `schema_version` 等
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .budget import truncate_text
from .config import Settings
from .embedding import build_embedder
from .text import content_anchor, normalize, tokenize
from .text import unique as unique_tokens

SCHEMA_VERSION = "1"

SCHEMA_SQL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    title  TEXT NOT NULL,
    text   TEXT NOT NULL,
    meta   TEXT NOT NULL
);

CREATE TABLE chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    ordinal      INTEGER NOT NULL,
    section_path TEXT NOT NULL,
    heading      TEXT NOT NULL,
    text         TEXT NOT NULL,
    parent_text  TEXT NOT NULL,
    anchor       TEXT NOT NULL,
    -- pre-filter 需要的列。冗余到块上，不做 join（DESIGN §9.2）
    department   TEXT NOT NULL DEFAULT '',
    doc_type     TEXT NOT NULL DEFAULT '',
    updated_at   TEXT NOT NULL DEFAULT '',
    level_rank   INTEGER NOT NULL DEFAULT 0,   -- public=0 < internal=1 < confidential=2
    doc_status   TEXT NOT NULL DEFAULT 'active',
    meta         TEXT NOT NULL
);
CREATE INDEX chunks_doc ON chunks(doc_id, ordinal);

CREATE TABLE chunk_acl (
    chunk_id TEXT NOT NULL,
    grp      TEXT NOT NULL,
    PRIMARY KEY (chunk_id, grp)
);

CREATE TABLE chunk_vectors (
    chunk_id TEXT PRIMARY KEY,
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL
);

CREATE VIRTUAL TABLE chunk_fts USING fts5(
    chunk_id UNINDEXED,
    tokens
);
"""

LEVEL_RANK = {"public": 0, "internal": 1, "confidential": 2}


@dataclass
class StagedChunk:
    """入库中间形态：与后端无关，所有数据源都归一到它。"""

    chunk_id: str
    doc_id: str
    ordinal: int
    section_path: str
    heading: str
    text: str
    parent_text: str
    meta: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None

    @property
    def anchor(self) -> str:
        return content_anchor(self.text)


@dataclass
class StagedDoc:
    doc_id: str
    title: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- 数据源：JSON


def stage_json(path: Path, settings: Settings) -> Iterator[StagedDoc | StagedChunk]:
    """本项目的 JSON 语料格式（与 `memory` 后端同构）。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    for doc in payload["documents"]:
        meta = {
            "title": doc["title"],
            "department": doc.get("department", ""),
            "doc_type": doc.get("doc_type", ""),
            "security_level": doc.get("security_level", "internal"),
            "acl_groups": list(doc.get("acl_groups") or []),
            "doc_status": str(doc.get("status", "active")),
            "updated_at": doc.get("updated_at", ""),
        }
        full_text = "\n\n".join(
            f"## {s.get('heading', '')}\n{s['text']}" if s.get("heading") else s["text"]
            for s in doc["sections"]
        )
        yield StagedDoc(doc_id=doc["doc_id"], title=doc["title"], text=full_text, meta=meta)

        ordinal = 0
        for section in doc["sections"]:
            heading = section.get("heading", "")
            body = section["text"]
            section_path = f"{doc['title']} › {heading}" if heading else doc["title"]
            for piece in _split(body, settings.chunk_size, settings.chunk_overlap):
                yield StagedChunk(
                    chunk_id=f"{doc['doc_id']}#c{ordinal:03d}",
                    doc_id=doc["doc_id"],
                    ordinal=ordinal,
                    section_path=section_path,
                    heading=heading,
                    text=piece,
                    parent_text=body,
                    meta=meta,
                )
                ordinal += 1


# ---------------------------------------------------------------- 数据源：TCM


#: OCR 页眉里的章节标题，如「第四章肺系病证」。
_CHAPTER = re.compile(r"第([一二三四五六七八九十百]+)章\s*([\u4e00-\u9fff]{2,10})")


def stage_tcm_sqlite(db_path: Path) -> Iterator[StagedDoc | StagedChunk]:
    """《中医内科学》PDF 的既有向量库（LangChain SQLiteVec 格式）。

    复用要点：**向量直接搬过来，不重新 embedding**（同一个 text-embedding-v3 空间）。

    适配代价：
    - 原库只有 `source` / `page`，没有章节 → 从页眉正则 + 按页码顺延继承补齐；
    - 没有业务元数据（部门/密级/日期）→ 统一按 `public` + 单部门填充，
      因此这个语料**不能用来测 ACL**，只能用来测检索质量；
    - 一本书切成一章一个 doc：否则「同文档最多 2 条」会把整本书压成 2 条。
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute(
        "select text, metadata, text_embedding from tcm_docs order by rowid"
    ).fetchall()
    con.close()

    # 一趟扫描：按页码排序后把最近一次见到的章节名顺延给后面所有页
    parsed: list[tuple[int, str, list[float]]] = []
    chapters: dict[int, str] = {}
    current = "正文"
    for text, meta_raw, embedding_blob in sorted(
        rows, key=lambda row: _page_of(row[1])
    ):
        page = _page_of(meta_raw)
        match = _CHAPTER.search(text or "")
        if match:
            current = match.group(0).replace("\n", "").strip()
        chapters.setdefault(page, current)
        parsed.append((page, normalize(text or ""), _decode_vector(embedding_blob)))

    book = "中医内科学"
    by_chapter: dict[str, list[tuple[int, str, list[float]]]] = {}
    for page, text, vector in parsed:
        by_chapter.setdefault(chapters.get(page, "正文"), []).append((page, text, vector))

    for chapter, items in by_chapter.items():
        doc_id = "tcm-" + hashlib.sha1(chapter.encode("utf-8")).hexdigest()[:8]
        meta = {
            "title": f"{book} › {chapter}",
            "department": "tcm",
            "doc_type": "manual",
            "security_level": "public",
            "acl_groups": [],
            "doc_status": "active",
            "updated_at": "2024-01-01",
        }
        yield StagedDoc(
            doc_id=doc_id,
            title=f"{book} › {chapter}",
            text="\n\n".join(text for _, text, _ in items),
            meta=meta,
        )
        for ordinal, (page, text, vector) in enumerate(items):
            yield StagedChunk(
                chunk_id=f"{doc_id}#c{ordinal:03d}",
                doc_id=doc_id,
                ordinal=ordinal,
                section_path=f"{book} › {chapter} › 第 {page} 页",
                heading=chapter,
                text=text,
                parent_text=text,  # 原库的块已是 800 字，不再做 small-to-big
                meta=meta,
                vector=vector,
            )


def _page_of(meta_raw: Any) -> int:
    try:
        payload = json.loads(meta_raw.decode() if isinstance(meta_raw, bytes) else meta_raw)
        return int(payload.get("page", 0))
    except Exception:
        return 0


def _decode_vector(blob: Any) -> list[float] | None:
    if not blob:
        return None
    import struct

    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


# ---------------------------------------------------------------- 写出


def build_database(
    source: str,
    out_path: Path,
    *,
    data_path: Path | None = None,
    source_db: Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    if source == "json":
        assert data_path is not None, "json 数据源需要 --data"
        iterator = stage_json(data_path, settings)
    elif source == "tcm-sqlite":
        assert source_db is not None, "tcm-sqlite 数据源需要 --db"
        iterator = stage_tcm_sqlite(source_db)
    else:  # pragma: no cover
        raise SystemExit(f"未知数据源：{source}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    con = sqlite3.connect(out_path)
    con.executescript(SCHEMA_SQL)

    docs = 0
    chunks: list[StagedChunk] = []
    for item in iterator:
        if isinstance(item, StagedDoc):
            con.execute(
                "insert into documents(doc_id,title,text,meta) values(?,?,?,?)",
                (item.doc_id, item.title, item.text, json.dumps(item.meta, ensure_ascii=False)),
            )
            docs += 1
        else:
            chunks.append(item)

    # 没有源向量的块，统一批量补算（同一批请求，省时间）
    missing = [chunk for chunk in chunks if chunk.vector is None]
    if missing:
        embedder = build_embedder(settings.embedding)
        print(f"为 {len(missing)} 个块补算向量（{embedder.model_id}）…", file=sys.stderr)
        payload = [f"{c.heading}\n{c.text}" if c.heading else c.text for c in missing]
        vectors = embedder.embed(payload)
        for chunk, vector in zip(missing, vectors, strict=True):
            chunk.vector = vector

    for chunk in chunks:
        meta = chunk.meta
        con.execute(
            "insert into chunks(chunk_id,doc_id,ordinal,section_path,heading,text,parent_text,"
            "anchor,department,doc_type,updated_at,level_rank,doc_status,meta) "
            "values(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.ordinal,
                chunk.section_path,
                chunk.heading,
                chunk.text,
                chunk.parent_text,
                chunk.anchor,
                str(meta.get("department", "")),
                str(meta.get("doc_type", "")),
                str(meta.get("updated_at", "")),
                LEVEL_RANK.get(str(meta.get("security_level", "internal")), 1),
                str(meta.get("doc_status", "active")),
                json.dumps(meta, ensure_ascii=False),
            ),
        )
        con.execute(
            "insert into chunk_fts(chunk_id,tokens) values(?,?)",
            (chunk.chunk_id, _fts_tokens(f"{chunk.heading}\n{chunk.text}")),
        )
        for group in meta.get("acl_groups") or []:
            con.execute("insert into chunk_acl(chunk_id,grp) values(?,?)", (chunk.chunk_id, group))
        if chunk.vector:
            import struct

            con.execute(
                "insert into chunk_vectors(chunk_id,dim,vec) values(?,?,?)",
                (
                    chunk.chunk_id,
                    len(chunk.vector),
                    struct.pack(f"<{len(chunk.vector)}f", *chunk.vector),
                ),
            )

    embedder = build_embedder(settings.embedding)
    corpus_version = _corpus_version(source, data_path, source_db, chunks)
    summary = {
        "source": source,
        "docs": docs,
        "chunks": len(chunks),
        "vectors": sum(1 for c in chunks if c.vector),
        "dim": len(next((c.vector for c in chunks if c.vector), []) or []),
        "corpus_version": corpus_version,
        "schema_version": SCHEMA_VERSION,
        "embedder": embedder.model_id,
        "built_at": date.today().isoformat(),
    }
    con.executemany(
        "insert into meta(key,value) values(?,?)",
        [(key, json.dumps(value, ensure_ascii=False)) for key, value in summary.items()],
    )
    con.commit()
    con.execute("insert into chunk_fts(chunk_fts) values('optimize')")
    con.commit()
    con.close()
    return summary


def _fts_tokens(text: str) -> str:
    """FTS5 的 `tokens` 列是我们自己分词后的结果。

    这样中文检索质量由 `text.tokenize` 决定，而不是由 FTS5 的 unicode61
    切分规则决定（它对中文几乎不切分）。
    """
    return " ".join(unique_tokens(tokenize(text)))


def _corpus_version(
    source: str, data_path: Path | None, source_db: Path | None, chunks: list[StagedChunk]
) -> str:
    parts = [source, str(len(chunks))]
    for candidate in (data_path, source_db):
        if candidate is not None and candidate.exists():
            parts.append(str(candidate.stat().st_size))
            parts.append(str(int(candidate.stat().st_mtime)))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _split(text: str, size: int, overlap: int) -> list[str]:
    """按长度切块，尽量在句末断开（与 memory 后端一致的行为）。"""
    size = max(80, size)
    overlap = max(0, min(overlap, size // 2))
    pieces: list[str] = []
    start, total = 0, len(text)
    while start < total:
        end = min(start + size, total)
        if end < total:
            window = text[start:end]
            cut = max(window.rfind(mark) for mark in ("。", "！", "？", "；", "\n"))
            if cut > size * 0.4:
                end = start + cut + 1
        piece, _ = truncate_text(text[start:end].strip(), size + 8)
        if piece:
            pieces.append(piece)
        if end >= total:
            break
        start = max(end - overlap, start + 1)
    return pieces


def main() -> int:
    parser = argparse.ArgumentParser(description="把语料建成可查询的 rag.db")
    parser.add_argument("--source", choices=["json", "tcm-sqlite"], required=True)
    parser.add_argument("--data", type=Path, help="json 数据源路径")
    parser.add_argument("--db", type=Path, help="tcm-sqlite 源数据库路径")
    parser.add_argument("--out", type=Path, default=Path("rag.db"))
    args = parser.parse_args()

    summary = build_database(
        args.source, args.out, data_path=args.data, source_db=args.db
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n✅ 索引已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

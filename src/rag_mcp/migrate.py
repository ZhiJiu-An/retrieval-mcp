"""把旧版（无 anchor）的 SQLite 索引原地升级到 chunk_id v2。

背景：`SCHEMA_VERSION=1` 时 `ingest` 生成的 id 是 `{doc_id}#c{ordinal}`，
没有内容指纹。`SqliteBackend.get_chunk` 会按 id 精确命中，导致重新切分/入库后
旧 id 静默返回另一段文本。

不能“按位置猜”兼容旧 id —— 那正是错误来源。正确的迁移是给已有行补上
`-{anchor}`，并同步更新 `chunk_vectors` / `chunk_acl` / `chunk_fts` / `meta`。

用法：

```bash
python -m rag_mcp.migrate rag.db --dry-run
python -m rag_mcp.migrate rag.db
```

默认先写一个 `rag.db.pre-v2.bak`；迁移失败时原库不变（事务回滚）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from .backends.base import CHUNK_ID_RE
from .ingest import CHUNK_ID_FORMAT, SCHEMA_VERSION
from .text import TOKENIZER_VERSION, tokenize

_TABLES_WITH_CHUNK_ID = ("chunk_vectors", "chunk_acl", "chunk_fts")


def _meta(con: sqlite3.Connection) -> dict[str, Any]:
    return {row[0]: json.loads(row[1]) for row in con.execute("select key,value from meta")}


def migrate_chunk_ids(db_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """原地迁移；返回统计。重复执行是幂等的。"""
    if not db_path.exists():
        raise FileNotFoundError(f"索引不存在：{db_path}")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        meta = _meta(con)
        if str(meta.get("chunk_id_format", "")) == CHUNK_ID_FORMAT:
            return {"migrated": 0, "already_v2": True, "db": str(db_path)}

        rows = con.execute("select chunk_id, anchor from chunks order by rowid").fetchall()
        mapping: dict[str, str] = {}
        for row in rows:
            old = str(row["chunk_id"])
            if CHUNK_ID_RE.match(old):
                continue
            anchor = str(row["anchor"] or "")
            if len(anchor) != 6:
                raise ValueError(f"无法迁移 {old}：anchor 非法 {anchor!r}")
            new = f"{old}-{anchor}"
            if new in mapping.values():
                raise ValueError(f"迁移会产生重复 id：{new}")
            mapping[old] = new

        if dry_run:
            return {
                "migrated": len(mapping),
                "already_v2": False,
                "dry_run": True,
                "db": str(db_path),
                "sample": dict(list(mapping.items())[:3]),
            }

        old_corpus_version = str(meta.get("corpus_version", ""))
        new_corpus_version = hashlib.sha256(
            f"{old_corpus_version}|chunk-id-v2|{len(rows)}".encode()
        ).hexdigest()[:12]

        with con:  # 事务：失败自动回滚
            for old, new in mapping.items():
                con.execute("update chunks set chunk_id=? where chunk_id=?", (new, old))
                for table in _TABLES_WITH_CHUNK_ID:
                    con.execute(f'update "{table}" set chunk_id=? where chunk_id=?', (new, old))
            con.executemany(
                "insert or replace into meta(key,value) values(?,?)",
                [
                    ("schema_version", json.dumps(SCHEMA_VERSION)),
                    ("chunk_id_format", json.dumps(CHUNK_ID_FORMAT)),
                    ("corpus_version", json.dumps(new_corpus_version)),
                ],
            )
    finally:
        con.close()

    return {
        "migrated": len(mapping),
        "already_v2": False,
        "dry_run": False,
        "db": str(db_path),
        "corpus_version": new_corpus_version,
    }


def rebuild_fts(db_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """用当前 tokenizer 重建 FTS tokens。

    改了 `text.tokenize` 之后，旧索引的 `chunk_fts.tokens` 仍是旧切法；
    必须重建，否则查询里的新 token（如 `x` / `c`）永远匹配不到。
    """
    if not db_path.exists():
        raise FileNotFoundError(f"索引不存在：{db_path}")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        meta = _meta(con)
        rows = con.execute("select chunk_id, heading, text from chunks order by rowid").fetchall()
        if dry_run:
            return {"rebuilt": len(rows), "dry_run": True, "db": str(db_path)}

        old_corpus_version = str(meta.get("corpus_version", ""))
        new_corpus_version = hashlib.sha256(
            f"{old_corpus_version}|tokenizer-{TOKENIZER_VERSION}|{len(rows)}".encode()
        ).hexdigest()[:12]

        with con:
            for row in rows:
                tokens = " ".join(tokenize(f"{row['heading']}\n{row['text']}"))
                con.execute(
                    "update chunk_fts set tokens=? where chunk_id=?",
                    (tokens, row["chunk_id"]),
                )
            con.executemany(
                "insert or replace into meta(key,value) values(?,?)",
                [
                    ("schema_version", json.dumps(SCHEMA_VERSION)),
                    ("tokenizer_version", json.dumps(TOKENIZER_VERSION)),
                    ("corpus_version", json.dumps(new_corpus_version)),
                ],
            )
        con.execute("insert into chunk_fts(chunk_fts) values('optimize')")
        con.commit()
    finally:
        con.close()

    return {
        "rebuilt": len(rows),
        "dry_run": False,
        "db": str(db_path),
        "corpus_version": new_corpus_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="升级旧 SQLite 索引：chunk_id v2 / FTS 重建")
    parser.add_argument("db", type=Path, help="rag.db 路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要迁移的条数")
    parser.add_argument(
        "--rebuild-fts",
        action="store_true",
        help="用当前 tokenizer 重建 chunk_fts（改过分词规则时必须做）",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不写 .pre-v2.bak（不建议：迁移会原地改库）",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"❌ 索引不存在：{args.db}", file=sys.stderr)
        return 2

    if not args.dry_run and not args.no_backup:
        backup = args.db.with_name(args.db.name + ".pre-v2.bak")
        if backup.exists():
            backup = args.db.with_name(f"{args.db.name}.pre-v2.{time.strftime('%Y%m%d%H%M%S')}.bak")
        shutil.copy2(args.db, backup)
        print(f"已备份：{backup}", file=sys.stderr)

    try:
        summary = migrate_chunk_ids(args.db, dry_run=args.dry_run)
        if args.rebuild_fts and not args.dry_run:
            summary["fts"] = rebuild_fts(args.db)
    except Exception as exc:
        print(f"❌ 迁移失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

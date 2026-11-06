"""旧索引迁移：无 anchor 的 chunk_id 原地升级到 v2。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from rag_mcp.backends.base import CHUNK_ID_RE
from rag_mcp.ingest import build_database
from rag_mcp.migrate import migrate_chunk_ids, rebuild_fts


def _build_v2(tmp_path: Path) -> Path:
    payload = {
        "source": {"id": "tmp", "name": "临时语料", "updated_at": "2025-01-01"},
        "documents": [
            {
                "doc_id": "doc",
                "title": "测试制度",
                "department": "hr",
                "doc_type": "policy",
                "security_level": "public",
                "updated_at": "2025-01-01",
                "sections": [{"heading": "年假", "text": "年假 5 天。"}],
            }
        ],
    }
    data = tmp_path / "data.json"
    data.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    db = tmp_path / "rag.db"
    build_database("json", db, data_path=data)
    return db


def _downgrade(db: Path) -> None:
    """把 v2 库降级成 v1（无 anchor），模拟历史 rag.db。"""
    con = sqlite3.connect(db)
    try:
        rows = con.execute("select chunk_id from chunks").fetchall()
        with con:
            for (chunk_id,) in rows:
                legacy = chunk_id.rsplit("-", 1)[0]
                con.execute("update chunks set chunk_id=? where chunk_id=?", (legacy, chunk_id))
                for table in ("chunk_vectors", "chunk_acl", "chunk_fts"):
                    con.execute(
                        f'update "{table}" set chunk_id=? where chunk_id=?', (legacy, chunk_id)
                    )
            con.execute(
                "update meta set value=? where key='chunk_id_format'",
                (json.dumps("v1"),),
            )
            con.execute(
                "update meta set value=? where key='schema_version'",
                (json.dumps("1"),),
            )
    finally:
        con.close()


def _ids(db: Path, table: str = "chunks") -> list[str]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [row[0] for row in con.execute(f'select chunk_id from "{table}"')]
    finally:
        con.close()


def test_migrate_adds_anchor_and_updates_child_tables(tmp_path: Path):
    db = _build_v2(tmp_path)
    _downgrade(db)
    assert all("-" not in cid for cid in _ids(db))

    summary = migrate_chunk_ids(db)

    assert summary["migrated"] == 1
    assert all(CHUNK_ID_RE.match(cid) for cid in _ids(db))
    assert all(CHUNK_ID_RE.match(cid) for cid in _ids(db, "chunk_vectors"))
    assert all(CHUNK_ID_RE.match(cid) for cid in _ids(db, "chunk_fts"))
    assert _ids(db) == _ids(db, "chunk_vectors") == _ids(db, "chunk_fts")


def test_migrate_is_idempotent(tmp_path: Path):
    db = _build_v2(tmp_path)
    _downgrade(db)
    migrate_chunk_ids(db)

    second = migrate_chunk_ids(db)

    assert second["migrated"] == 0
    assert second["already_v2"] is True


def test_dry_run_does_not_change_ids(tmp_path: Path):
    db = _build_v2(tmp_path)
    _downgrade(db)
    before = _ids(db)

    summary = migrate_chunk_ids(db, dry_run=True)

    assert summary["migrated"] == 1
    assert _ids(db) == before


def test_migrate_missing_db_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        migrate_chunk_ids(tmp_path / "missing.db")


def test_rebuild_fts_uses_current_tokenizer(tmp_path: Path):
    db = _build_v2(tmp_path)
    con = sqlite3.connect(db)
    try:
        with con:
            con.execute("update chunk_fts set tokens='stale'")
    finally:
        con.close()

    summary = rebuild_fts(db)

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        tokens = con.execute("select tokens from chunk_fts limit 1").fetchone()[0]
        meta = dict(con.execute("select key,value from meta").fetchall())
    finally:
        con.close()
    assert summary["rebuilt"] == 1
    assert tokens != "stale"
    assert "年假" in tokens or ("年" in tokens and "假" in tokens)
    assert json.loads(meta["tokenizer_version"]) == "2"

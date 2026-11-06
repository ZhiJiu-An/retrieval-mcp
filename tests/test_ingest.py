"""入库契约：chunk_id 内容指纹、原子写入、FTS 词频。

这些用例都跑在临时 SQLite 上，不联网。它们补的是此前完全缺失的生产路径：
`SqliteBackend` 的 id / 降级 / 失败回滚。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from rag_mcp.backends.base import parse_chunk_id
from rag_mcp.backends.sqlite import SqliteBackend
from rag_mcp.config import Settings
from rag_mcp.errors import CHUNK_MOVED
from rag_mcp.factory import build_service
from rag_mcp.ingest import build_database
from rag_mcp.policy import Principal


def _payload(text: str, *, doc_id: str = "doc", department: str = "hr") -> dict:
    return {
        "source": {"id": "tmp", "name": "临时语料", "updated_at": "2025-01-01"},
        "documents": [
            {
                "doc_id": doc_id,
                "title": "测试制度",
                "department": department,
                "doc_type": "policy",
                "security_level": "public",
                "updated_at": "2025-01-01",
                "sections": [{"heading": "年假天数", "text": text}],
            }
        ],
    }


def _build(tmp_path: Path, text: str, *, name: str = "a") -> tuple[Path, Path]:
    data = tmp_path / f"{name}.json"
    data.write_text(json.dumps(_payload(text), ensure_ascii=False), encoding="utf-8")
    out = tmp_path / f"{name}.db"
    build_database("json", out, data_path=data, settings=Settings())
    return data, out


def _chunk_id(db_path: Path) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return str(con.execute("select chunk_id from chunks limit 1").fetchone()[0])
    finally:
        con.close()


def test_chunk_id_contains_content_anchor_and_meta_version(tmp_path: Path):
    _, db_path = _build(tmp_path, "第一版内容：年假 5 天。")

    chunk_id = _chunk_id(db_path)
    doc_id, ordinal, anchor = parse_chunk_id(chunk_id)
    assert doc_id == "doc"
    assert ordinal == 0
    assert anchor and anchor == chunk_id.rsplit("-", 1)[1]
    assert chunk_id.startswith("doc#c000-")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        meta = dict(con.execute("select key,value from meta").fetchall())
        assert json.loads(meta["schema_version"]) == "3"
        assert json.loads(meta["chunk_id_format"]) == "v2"
        # 二级降级查询需要这个索引；至少确认 schema 建出来了
        indexes = {
            row[0] for row in con.execute("select name from sqlite_master where type='index'")
        }
        assert "chunks_doc_anchor" in indexes
    finally:
        con.close()


def test_reingest_old_id_resolves_by_ordinal_not_exact(tmp_path: Path):
    """旧 id 的 anchor 必须让它降级，而不是把同位置的另一段文本当 exact。"""
    _, db_a = _build(tmp_path, "第一版内容：年假 5 天。", name="a")
    old_id = _chunk_id(db_a)

    _, db_b = _build(tmp_path, "第二版内容：年假 10 天。", name="b")
    old = SqliteBackend(db_a, embedder=_hashing())
    new = SqliteBackend(db_b, embedder=_hashing())
    try:
        # 直接在两个库上验证：db_b 里没有 db_a 的 anchor
        resolved = new.get_chunk(old_id)
        assert resolved.chunk is not None
        assert resolved.resolved_by == "ordinal"
        assert "第二版" in resolved.chunk.text
        original = old.get_chunk(old_id).chunk
        assert original is not None and "第一版" in original.text
    finally:
        old.close()
        new.close()


def test_fetch_reports_chunk_moved_for_stale_id(tmp_path: Path):
    _, db_a = _build(tmp_path, "第一版内容：年假 5 天。", name="a")
    old_id = _chunk_id(db_a)
    _, db_b = _build(tmp_path, "第二版内容：年假 10 天。", name="b")

    service = build_service(Settings(backend="sqlite", db_path=db_b))
    staff = Principal(user_id="alice", groups=frozenset({"all"}), max_level="internal")
    fetched = service.fetch(staff, chunk_id=old_id)

    assert fetched.resolved_by == "ordinal"
    assert fetched.hint and fetched.hint.startswith(CHUNK_MOVED)


def test_legacy_positional_id_is_rejected(tmp_path: Path):
    _, db_path = _build(tmp_path, "第一版内容：年假 5 天。")
    legacy = _chunk_id(db_path).rsplit("-", 1)[0]  # 去掉 anchor

    backend = SqliteBackend(db_path, embedder=_hashing())
    try:
        assert backend.get_chunk(legacy).chunk is None
    finally:
        backend.close()


def test_same_source_and_out_is_rejected_without_touching_source(tmp_path: Path):
    data = tmp_path / "data.json"
    data.write_text(json.dumps(_payload("年假 5 天。"), ensure_ascii=False), encoding="utf-8")
    before = data.read_bytes()

    with pytest.raises(ValueError, match="不能与数据源相同"):
        build_database("json", data, data_path=data, settings=Settings())

    assert data.read_bytes() == before


def test_failed_rebuild_keeps_previous_database(tmp_path: Path, monkeypatch):
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps(_payload("第一版：年假 5 天。"), ensure_ascii=False), encoding="utf-8"
    )
    out = tmp_path / "rag.db"
    build_database("json", out, data_path=data, settings=Settings())

    class _Boom:
        model_id = "boom"

        def embed(self, texts):
            raise RuntimeError("embedding down")

    monkeypatch.setattr("rag_mcp.ingest.build_embedder", lambda settings: _Boom())
    data.write_text(
        json.dumps(_payload("第二版：年假 10 天。"), ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="embedding down"):
        build_database("json", out, data_path=data, settings=Settings())

    # 旧库仍可读，且仍是第一版；临时文件要清理
    con = sqlite3.connect(f"file:{out}?mode=ro", uri=True)
    try:
        text = con.execute("select text from chunks limit 1").fetchone()[0]
    finally:
        con.close()
    assert "第一版" in text
    assert not list(tmp_path.glob(f".{out.name}.*.tmp"))


def test_fts_tokens_keep_term_frequency(tmp_path: Path):
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps(_payload("vpn vpn vpn 接入说明"), ensure_ascii=False), encoding="utf-8"
    )
    out = tmp_path / "freq.db"
    build_database("json", out, data_path=data, settings=Settings())

    con = sqlite3.connect(f"file:{out}?mode=ro", uri=True)
    try:
        tokens = con.execute("select tokens from chunk_fts limit 1").fetchone()[0]
    finally:
        con.close()
    assert tokens.split().count("vpn") >= 3


def test_parse_chunk_id_accepts_ordinal_over_999():
    doc_id, ordinal, anchor = parse_chunk_id("doc#c1000-abcdef")
    assert doc_id == "doc"
    assert ordinal == 1000
    assert anchor == "abcdef"


def _hashing():
    from rag_mcp.embedding import HashingEmbedder

    return HashingEmbedder()

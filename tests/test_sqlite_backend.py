"""SQLite 生产后端的 ACL / 状态 pre-filter。

memory 测试只覆盖了 Python 过滤；这里验证 SQL 侧 `chunk_acl` /
`level_rank` / `doc_status` 真正参与召回，且 `fetch(doc_id)` 仍可读废止文档。
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_mcp.config import Settings
from rag_mcp.errors import RagError
from rag_mcp.factory import build_service
from rag_mcp.ingest import build_database
from rag_mcp.policy import Principal


def _build(tmp_path: Path) -> Path:
    documents = [
        {
            "doc_id": "doc-public",
            "title": "公开制度",
            "department": "hr",
            "doc_type": "policy",
            "security_level": "public",
            "acl_groups": [],
            "updated_at": "2025-01-01",
            "sections": [{"heading": "年假", "text": "年假 5 天。"}],
        },
        {
            "doc_id": "doc-secret",
            "title": "机密通知",
            "department": "hr",
            "doc_type": "notice",
            "security_level": "confidential",
            "acl_groups": ["exec"],
            "updated_at": "2025-01-01",
            "sections": [{"heading": "年假", "text": "年假 10 天，仅执委会可见。"}],
        },
        {
            "doc_id": "doc-old",
            "title": "废止制度",
            "department": "hr",
            "doc_type": "policy",
            "security_level": "public",
            "status": "superseded",
            "acl_groups": [],
            "updated_at": "2024-01-01",
            "sections": [{"heading": "年假", "text": "年假 5 天，旧版。"}],
        },
    ]
    payload = {
        "source": {"id": "tmp", "name": "临时语料", "updated_at": "2025-01-01"},
        "documents": documents,
    }
    data = tmp_path / "docs.json"
    data.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    db = tmp_path / "rag.db"
    build_database("json", db, data_path=data, settings=Settings())
    return db


def _service(db: Path):
    return build_service(Settings(backend="sqlite", db_path=db, retrieval_mode="lexical"))


def _staff() -> Principal:
    return Principal(user_id="alice", groups=frozenset({"all"}), max_level="internal")


def _exec() -> Principal:
    return Principal(user_id="carol", groups=frozenset({"all", "exec"}), max_level="confidential")


def test_sql_prefilter_hides_acl_and_superseded(tmp_path: Path):
    service = _service(_build(tmp_path))

    staff_docs = [hit.doc_id for hit in service.search(_staff(), query="年假", top_k=20).hits]
    exec_docs = [hit.doc_id for hit in service.search(_exec(), query="年假", top_k=20).hits]

    assert staff_docs == ["doc-public"]
    assert set(exec_docs) == {"doc-public", "doc-secret"}
    assert "doc-old" not in staff_docs + exec_docs


def test_superseded_document_is_still_readable_by_id(tmp_path: Path):
    service = _service(_build(tmp_path))

    fetched = service.fetch(_staff(), doc_id="doc-old")

    assert fetched.doc_id == "doc-old"
    assert "旧版" in fetched.text


def test_staff_cannot_fetch_confidential_document(tmp_path: Path):
    service = _service(_build(tmp_path))

    try:
        service.fetch(_staff(), doc_id="doc-secret")
    except RagError as exc:
        assert exc.code == "NOT_FOUND_OR_FORBIDDEN"
    else:  # pragma: no cover
        raise AssertionError("staff 不应读到 confidential 文档")


def test_repeated_search_hits_allowed_cache(tmp_path: Path):
    service = _service(_build(tmp_path))
    staff = _staff()

    first = service.search(staff, query="年假", top_k=20)
    second = service.search(staff, query="年假", top_k=20)

    assert [h.chunk_id for h in first.hits] == [h.chunk_id for h in second.hits]
    assert service._backend._allowed_cache  # noqa: SLF001


def test_updated_after_uses_registry_sql_column(tmp_path: Path):
    service = _service(_build(tmp_path))

    result = service.search(
        _staff(), query="年假", filters={"updated_after": "2024-06-01"}, top_k=20
    )

    assert result.filters_applied == {"updated_after": "2024-06-01"}
    assert result.hits
    assert "doc-old" not in [hit.doc_id for hit in result.hits]

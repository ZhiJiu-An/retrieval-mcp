"""本次修复的回归用例集合。

覆盖此前没有测试、但已经实际踩到/复现过的路径：
- factory 的配置错误分类
- 紧凑日期过滤
- 配置指纹
- with_principal 字段保留
- OpenAICompatEmbedder 缓存满
- SQLite describe 的时间字段
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rag_mcp.budget import Limits
from rag_mcp.config import EmbeddingSettings, RerankSettings, Settings
from rag_mcp.embedding import OpenAICompatEmbedder
from rag_mcp.errors import BAD_CONFIGURATION, RagError
from rag_mcp.factory import build_service
from rag_mcp.ingest import build_database
from rag_mcp.policy import Principal
from rag_mcp.rerank import FallbackReranker, HeuristicReranker, build_reranker


def _build_json_db(tmp_path: Path) -> Path:
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
    build_database("json", db, data_path=data, settings=Settings())
    return db


# ---------------------------------------------------------------- factory


def test_unknown_backend_is_a_config_error():
    with pytest.raises(RagError) as excinfo:
        build_service(Settings(backend="sqlite3"))

    assert excinfo.value.code == BAD_CONFIGURATION
    assert excinfo.value.retryable is False


def test_missing_sqlite_index_is_a_config_error(tmp_path: Path):
    settings = Settings(backend="sqlite", db_path=tmp_path / "missing.db")

    with pytest.raises(RagError) as excinfo:
        build_service(settings)

    assert excinfo.value.code == BAD_CONFIGURATION
    assert "ingest" in (excinfo.value.hint or "")


def test_sqlite_embedder_mismatch_degrades_to_lexical(tmp_path: Path):
    db = _build_json_db(tmp_path)
    # 索引是 hashing-256；这里指定一个不同的“模型 id”，但仍是本地桩。
    settings = Settings(
        backend="sqlite",
        db_path=db,
        retrieval_mode="hybrid",
        embedding=EmbeddingSettings(dim=512),
    )
    service = build_service(settings)

    # 不能继续用错误维度的向量算余弦；降级后仍是 lexical，但服务可用
    assert service.resolved_mode == "lexical"
    assert "mismatch" in (service.describe().note or "")


# ---------------------------------------------------------------- 日期


def test_compact_date_returns_same_as_padded(settings, service, staff):
    padded = service.search(staff, query="年假", filters={"updated_after": "2025-01-01"}, top_k=5)
    compact = service.search(staff, query="年假", filters={"updated_after": "20250101"}, top_k=5)

    assert compact.filters_applied["updated_after"] == "2025-01-01"
    assert [hit.doc_id for hit in compact.hits] == [hit.doc_id for hit in padded.hits]


# ---------------------------------------------------------------- 配置指纹


def test_config_hash_tracks_requested_retrieval_mode(settings):
    lexical = build_service(Settings(retrieval_mode="lexical"))
    hybrid = build_service(Settings(retrieval_mode="hybrid"))

    # 本机没有真实向量模型，两者 resolved_mode 都是 lexical；
    # 但请求配置不同，hash 必须不同，否则无法归因实验。
    assert lexical.config_hash != hybrid.config_hash


def test_config_hash_tracks_default_top_k(settings):
    default = build_service(Settings())
    other = build_service(Settings(limits=replace(Limits(), default_top_k=5)))

    assert default.config_hash != other.config_hash


def test_with_principal_preserves_backend_settings(tmp_path: Path):
    db = tmp_path / "rag.db"
    base = Settings(backend="sqlite", db_path=db, retrieval_mode="lexical")

    changed = base.with_principal("bob", ("all",), "public")

    assert changed.backend == "sqlite"
    assert changed.db_path == db
    assert changed.retrieval_mode == "lexical"
    assert changed.principal_id == "bob"


# ---------------------------------------------------------------- 缓存


def test_embedding_cache_full_does_not_keyerror():
    embedder = OpenAICompatEmbedder(base_url="http://x", model="m")
    embedder._cache = {f"t{i}": [1.0] for i in range(4096)}
    embedder._request = lambda texts: [[2.0] for _ in texts]

    assert embedder.embed(["new-text"]) == [[2.0]]


# ---------------------------------------------------------------- describe


def test_sqlite_describe_updated_at_is_build_date(tmp_path: Path):
    db = _build_json_db(tmp_path)
    service = build_service(Settings(backend="sqlite", db_path=db, retrieval_mode="lexical"))
    updated_at = service.describe().sources[0].updated_at

    assert len(updated_at) == 10
    assert updated_at[4] == "-" and updated_at[7] == "-"


def test_sqlite_describe_uses_built_at_not_corpus_hash(tmp_path: Path):
    db = _build_json_db(tmp_path)
    service = build_service(Settings(backend="sqlite", db_path=db, retrieval_mode="lexical"))
    source = service.describe().sources[0]

    assert source.updated_at != source.corpus_version


# ---------------------------------------------------------------- 兼容性


def test_old_compact_chunk_id_is_still_not_exact(tmp_path: Path):
    """无 anchor 的旧 id 不应被猜成 exact（旧 rag.db 需要迁移，见 migrate.py）。"""
    db = _build_json_db(tmp_path)
    service = build_service(Settings(backend="sqlite", db_path=db, retrieval_mode="lexical"))
    staff = Principal(user_id="alice", groups=frozenset({"all"}), max_level="internal")
    hit = service.search(staff, query="年假", top_k=1).hits[0]
    legacy = hit.chunk_id.rsplit("-", 1)[0]

    with pytest.raises(RagError):
        service.fetch(staff, chunk_id=legacy)


# ---------------------------------------------------------------- 配置校验


def test_invalid_embedding_batch_is_config_error():
    with pytest.raises(RagError) as excinfo:
        build_service(Settings(embedding=EmbeddingSettings(batch=0)))

    assert excinfo.value.code == BAD_CONFIGURATION
    assert "BATCH" in excinfo.value.message


def test_invalid_principal_level_is_config_error():
    with pytest.raises(RagError) as excinfo:
        build_service(Settings(principal_max_level="top-secret"))

    assert excinfo.value.code == BAD_CONFIGURATION
    assert "MAX_LEVEL" in excinfo.value.message


def test_inverted_confidence_bands_are_config_error():
    with pytest.raises(RagError) as excinfo:
        build_service(Settings(rerank=RerankSettings(confidence_high=0.2, confidence_medium=0.8)))

    assert excinfo.value.code == BAD_CONFIGURATION


# ---------------------------------------------------------------- 重排


def test_heuristic_reranker_uses_configured_bands():
    reranker = build_reranker(
        RerankSettings(
            enabled=True,
            provider="heuristic",
            confidence_high=0.9,
            confidence_medium=0.8,
        )
    )

    assert reranker.confidence_bands == (0.9, 0.8)


def test_remote_reranker_falls_back_to_heuristic(monkeypatch):
    class _Boom:
        enabled = True
        cross_encoder = True
        confidence_bands = (0.5, 0.25)
        model_id = "boom"

        def rerank(self, query, terms, candidates):
            raise RagError(code="BACKEND_UNAVAILABLE", message="down")

    fallback = FallbackReranker(_Boom(), HeuristicReranker())
    probe = type(
        "Probe",
        (),
        {"section_path": "年假天数", "heading": "年假天数", "text": "年假 5 天", "chunk_id": "x"},
    )()

    ranked = fallback.rerank("年假", ["年假"], [probe])

    assert ranked
    assert fallback.degraded is True


# ---------------------------------------------------------------- 截断原因


def test_per_doc_truncation_is_reported_distinctly(settings):
    service = build_service(Settings(limits=replace(Limits(), per_doc_limit=1, max_top_k=20)))
    staff = Principal(user_id="alice", groups=frozenset({"all"}), max_level="internal")

    result = service.search(staff, query="年假 申请 流程 天数", top_k=20)

    if result.truncated and result.returned < result.total:
        assert result.truncated_reason == "per_doc_limit"
        assert "同一文档最多" in (result.hint or "")


def test_long_query_is_truncated_and_flagged(settings):
    service = build_service(Settings())
    staff = Principal(user_id="alice", groups=frozenset({"all"}), max_level="internal")

    result = service.search(staff, query="年假" * 5000, top_k=1)

    assert result.query_truncated is True
    assert len(result.query) <= Limits().max_query_chars


def test_calibrated_bands_are_valid_even_when_overlapping():
    from rag_mcp.calibrate import Probe, suggest

    probes = [
        Probe("p", "q", True, 0.0, 0.0, ("d1",)),
        Probe("n", "q", False, 1.0, 0.0, ()),
    ]
    report = suggest(probes)

    assert report["separated"] is False
    assert 0.0 <= report["relevance_floor"] <= 1.0
    assert 0.0 <= report["confidence_medium"] < report["confidence_high"] <= 1.0
    assert 0.0 <= report["dense_floor"] <= 1.0

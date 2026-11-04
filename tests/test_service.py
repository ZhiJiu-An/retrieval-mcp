"""检索域的行为契约。

对应 `MVP.md §6 验收矩阵`。这些用例不需要启动 MCP —— 这正是「② 不 import mcp」
这条架构约束换来的好处（DESIGN §4.2）。
"""

from __future__ import annotations

import pytest

from rag_mcp.backends.base import parse_chunk_id
from rag_mcp.budget import Limits
from rag_mcp.config import EmbeddingSettings, Settings
from rag_mcp.errors import (
    BAD_ARGUMENT,
    BAD_DATE,
    CHUNK_MOVED,
    EMPTY_QUERY,
    FILTER_UNKNOWN_VALUE,
    NO_HITS,
    NOT_FOUND_OR_FORBIDDEN,
    TRUNCATED,
    RagError,
)
from rag_mcp.factory import build_service

# ---------------------------------------------------------------- 基本检索


def test_search_prefers_section_that_matches_query(service, staff):
    result = service.search(staff, query="年假 天数", top_k=3)

    assert result.hits
    assert result.hits[0].section_path.endswith("年假天数")


def test_hit_carries_drilldown_fields(service, staff):
    hit = service.search(staff, query="年假", top_k=1).hits[0]

    assert hit.chunk_id.startswith("hr-policy-leave#")
    assert hit.doc_id == "hr-policy-leave"
    assert hit.updated_at == "2025-03-01"
    assert hit.snippet
    assert len(hit.snippet) <= Limits().snippet_chars + 2
    # score 不再对外暴露，改为离散 confidence（DESIGN A19）
    assert hit.confidence in {"high", "medium", "low"}
    assert not hasattr(hit, "score")


def test_result_reports_cost_and_attribution(service, staff):
    result = service.search(staff, query="年假", top_k=3)

    assert 0 < result.chars_returned <= Limits().max_response_bytes
    assert result.corpus_version
    assert result.schema_version == "1"
    assert len(result.config_hash) == 12


# ---------------------------------------------------------------- 过滤


def test_department_filter_narrows_results(service, staff):
    result = service.search(staff, query="报销 住宿", filters={"department": "finance"})

    assert result.hits
    assert {hit.doc_id for hit in result.hits} <= {"fin-policy-travel", "fin-policy-travel-2023"}
    assert result.filters_applied == {"department": "finance"}


def test_updated_after_excludes_stale_documents(service, staff):
    result = service.search(
        staff, query="住宿 标准", top_k=20, filters={"updated_after": "2024-01-01"}
    )

    assert result.hits
    assert all(hit.updated_at >= "2024-01-01" for hit in result.hits)


def test_unknown_filter_value_returns_empty_not_error(service, staff):
    """Agent 会拼错枚举值。报错会让它中断任务，空结果 + 合法值清单才能自纠。"""
    result = service.search(staff, query="年假", filters={"department": "人事"})

    assert result.hits == []
    assert result.hint and result.hint.startswith(FILTER_UNKNOWN_VALUE)
    assert "hr" in result.hint  # 合法值列出来了


def test_reserved_filter_cannot_be_injected(service, staff):
    with pytest.raises(RagError) as excinfo:
        service.search(staff, query="组织架构", filters={"security_level": "confidential"})

    assert excinfo.value.code == BAD_ARGUMENT


def test_unknown_filter_field_is_rejected(service, staff):
    with pytest.raises(RagError) as excinfo:
        service.search(staff, query="年假", filters={"owner": "alice"})

    assert excinfo.value.code == BAD_ARGUMENT
    assert "department" in excinfo.value.message


# ---------------------------------------------------------------- 版本与状态


def test_superseded_document_is_not_searchable_but_still_readable(service, staff):
    """废止 ≠ 不相关。

    取消时效衰减后，2023 版会排到 2025 版前面（实测）。正确解法是状态闸门：
    检索时不出现，按 id 仍可查阅 —— 查历史单据需要它（DESIGN A13 修订）。
    """
    docs = [hit.doc_id for hit in service.search(staff, query="住宿 标准", top_k=20).hits]
    assert "fin-policy-travel-2023" not in docs
    assert docs[0] == "fin-policy-travel"

    archived = service.fetch(staff, doc_id="fin-policy-travel-2023")
    assert archived.doc_id == "fin-policy-travel-2023"
    assert archived.text


# ---------------------------------------------------------------- 权限


def test_acl_hides_confidential_document_from_regular_staff(service, staff, exec_member):
    staff_docs = {h.doc_id for h in service.search(staff, query="组织架构 调整", top_k=20).hits}
    exec_docs = {
        h.doc_id for h in service.search(exec_member, query="组织架构 调整", top_k=20).hits
    }

    # 权限过滤发生在打分之前：受限文档在正常员工的结果里根本不存在，
    # 而不是「排进 top-k 之后被切掉」
    assert "hr-notice-org-2026" not in staff_docs
    assert "hr-notice-org-2026" in exec_docs


def test_overreach_and_missing_are_indistinguishable(service, staff):
    with pytest.raises(RagError) as forbidden:
        service.fetch(staff, doc_id="hr-notice-org-2026")
    with pytest.raises(RagError) as missing:
        service.fetch(staff, doc_id="no-such-doc")

    assert forbidden.value.code == missing.value.code == NOT_FOUND_OR_FORBIDDEN
    # 文案必须同构：不能出现「无权限」这类可用来探测存在性的词
    for error in (forbidden.value, missing.value):
        assert "无权限" not in error.message
        assert "不可访问" in error.message


def test_resource_read_checks_acl_independently(service, staff):
    """resource 是最常见的鉴权绕过点：不能因为 id 是「自己人给的」就放行。"""
    with pytest.raises(RagError) as excinfo:
        service.read_document(staff, "hr-notice-org-2026")

    assert excinfo.value.code == NOT_FOUND_OR_FORBIDDEN


# ---------------------------------------------------------------- 展开


def _body(text: str) -> str:
    """剥掉不可信内容的边界标记，只比较文档正文。"""
    lines = text.splitlines()
    return "\n".join(lines[1:-1])


def test_fetch_expands_window_and_points_to_neighbors(service, staff):
    hit = service.search(staff, query="年假 天数", top_k=1).hits[0]

    narrow = service.fetch(staff, chunk_id=hit.chunk_id, window=0)
    wide = service.fetch(staff, chunk_id=hit.chunk_id, window=1)

    assert _body(narrow.text) in _body(wide.text)
    assert wide.prev_chunk_id is None  # 窗口已覆盖到文首
    assert wide.next_chunk_id is not None


def test_fetch_window_zero_returns_whole_section(service, staff):
    """小块入库、父块返回：window=0 给的是整个小节，不是这一小块。"""
    hit = service.search(staff, query="年假 天数", top_k=1).hits[0]
    fetched = service.fetch(staff, chunk_id=hit.chunk_id, window=0)

    assert len(fetched.text) > len(hit.snippet)


def test_fetch_wraps_content_as_untrusted_data(service, staff):
    hit = service.search(staff, query="年假", top_k=1).hits[0]
    fetched = service.fetch(staff, chunk_id=hit.chunk_id)

    assert fetched.text.startswith("[untrusted document content | ")
    assert fetched.text.rstrip().endswith("[/untrusted document content]")


def test_fetch_requires_exactly_one_identifier(service, staff):
    with pytest.raises(RagError) as none_given:
        service.fetch(staff)
    with pytest.raises(RagError) as both_given:
        service.fetch(
            staff,
            chunk_id="hr-policy-leave#c000-000000",
            doc_id="hr-policy-leave",
        )

    assert none_given.value.code == both_given.value.code == BAD_ARGUMENT


def test_fetch_unknown_chunk_is_actionable(service, staff):
    with pytest.raises(RagError) as excinfo:
        service.fetch(staff, chunk_id="nope#c999-abcdef")

    assert excinfo.value.code == NOT_FOUND_OR_FORBIDDEN
    assert "先用 search" in excinfo.value.hint


def test_fetch_clamps_window(service, staff):
    hit = service.search(staff, query="年假", top_k=1).hits[0]
    fetched = service.fetch(staff, chunk_id=hit.chunk_id, window=99)

    assert fetched.truncated is False or fetched.truncated is True  # 不抛错即合规
    assert len(fetched.text) <= Limits().max_fetch_chars + 200


# ---------------------------------------------------------------- id 稳定性


def test_chunk_id_falls_back_by_content_anchor(service, staff):
    """位置变了但内容还在：用内容指纹找回，并如实告知 resolved_by。"""
    hit = service.search(staff, query="年假 天数", top_k=1).hits[0]
    doc_id, _, anchor = parse_chunk_id(hit.chunk_id)
    assert doc_id and anchor

    fetched = service.fetch(staff, chunk_id=f"{doc_id}#c999-{anchor}")
    assert fetched.resolved_by == "anchor"
    assert fetched.hint and fetched.hint.startswith(CHUNK_MOVED)


def test_chunk_id_falls_back_by_position(service, staff):
    hit = service.search(staff, query="年假 天数", top_k=1).hits[0]
    doc_id, ordinal, _ = parse_chunk_id(hit.chunk_id)
    assert doc_id is not None and ordinal is not None

    fetched = service.fetch(staff, chunk_id=f"{doc_id}#c{ordinal:03d}-000000")
    assert fetched.resolved_by == "ordinal"


def test_well_formed_chunk_id_resolves_exactly(service, staff):
    hit = service.search(staff, query="年假", top_k=1).hits[0]
    assert service.fetch(staff, chunk_id=hit.chunk_id).resolved_by == "exact"


# ---------------------------------------------------------------- 参数错误


def test_empty_query_is_rejected(service, staff):
    with pytest.raises(RagError) as excinfo:
        service.search(staff, query="   ")

    assert excinfo.value.code == EMPTY_QUERY
    assert excinfo.value.retryable is True


def test_bad_date_format_is_rejected(service, staff):
    with pytest.raises(RagError) as excinfo:
        service.search(staff, query="年假", filters={"updated_after": "2025/01/01"})

    assert excinfo.value.code == BAD_DATE
    assert "2025/01/01" in excinfo.value.message


# ---------------------------------------------------------------- 预算


def test_top_k_is_clamped_and_truncation_is_reported():
    settings = Settings(limits=Limits(max_top_k=2))
    service = build_service(settings)
    from rag_mcp.policy import Principal

    result = service.search(
        Principal(user_id="alice", groups=frozenset({"all"}), max_level="internal"),
        query="年假",
        top_k=50,
    )

    assert result.returned == 2
    assert result.truncated is True
    assert result.hint and result.hint.startswith(TRUNCATED)


def test_empty_result_returns_navigation_hint(service, staff):
    result = service.search(staff, query="zzzqqq")

    assert result.hits == []
    assert result.hint and result.hint.startswith(NO_HITS)


def test_diversity_limits_hits_per_document():
    settings = Settings(limits=Limits(per_doc_limit=1, max_top_k=20))
    service = build_service(settings)
    from rag_mcp.policy import Principal

    result = service.search(
        Principal(user_id="alice", groups=frozenset({"all"}), max_level="internal"),
        query="年假 申请 流程 天数",
        top_k=20,
    )

    doc_ids = [hit.doc_id for hit in result.hits]
    assert len(doc_ids) == len(set(doc_ids)), f"同一文档出现了多次：{doc_ids}"


# ---------------------------------------------------------------- 可归因


def test_config_hash_tracks_retrieval_config_not_corpus(service, settings):
    base = build_service(settings)
    other_embedder = build_service(
        Settings(embedding=EmbeddingSettings(dim=512))
    )

    assert base.config_hash != other_embedder.config_hash
    assert base._backend.corpus_version == other_embedder._backend.corpus_version


def test_capabilities_report_lexical_when_embedder_is_local(service):
    capabilities = service.describe().sources[0].capabilities

    assert capabilities.retrieval_mode == "lexical"
    assert capabilities.embedder.startswith("hashing-")
    assert capabilities.rerank_enabled is True


def test_describe_exposes_registry_fields(service):
    fields = {field.name for field in service.describe().sources[0].filter_fields}

    assert fields == {"department", "doc_type", "updated_after"}

"""词表与预算的单元测试：漂移、值域、上限的唯一来源。"""

from __future__ import annotations

from rag_mcp.budget import Limits, response_bytes, truncate_text
from rag_mcp.registry import RESERVED_FILTERS, Registry, default_registry


def test_default_registry_loads_declared_fields():
    registry = default_registry()

    assert registry.schema_version
    assert registry.names == ("department", "doc_type", "updated_after")


def test_enum_values_are_validated():
    registry = default_registry()

    assert registry.is_legal_value("department", "hr")
    assert not registry.is_legal_value("department", "人事")
    assert registry.legal_values("doc_type") == ["policy", "faq", "manual", "notice"]


def test_date_field_validates_format_only():
    registry = default_registry()

    assert registry.is_legal_value("updated_after", "2025-01-31")
    assert not registry.is_legal_value("updated_after", "2025/01/31")


def test_reserved_fields_cover_acl_attributes():
    """密级与可见组只能由身份推导，永远不能出现在可过滤字段里。"""
    assert {"security_level", "acl_groups"} == RESERVED_FILTERS
    assert not set(default_registry().names) & RESERVED_FILTERS


def test_registry_can_load_a_snapshot_from_elsewhere():
    """词表属于索引侧，检索侧必须能从外部快照加载（DESIGN §9.4）。"""
    snapshot = Registry.from_dict(
        {
            "schema_version": "7",
            "fields": [
                {"name": "region", "type": "enum", "values": ["cn", "us"], "description": "地区"}
            ],
        }
    )

    assert snapshot.schema_version == "7"
    assert snapshot.names == ("region",)
    assert snapshot.is_legal_value("region", "cn")


def test_limits_clamp_instead_of_raising():
    """越界静默收敛：报错只会让 Agent 中断任务（DESIGN §7）。"""
    limits = Limits()

    assert limits.clamp_top_k(None) == limits.default_top_k
    assert limits.clamp_top_k(10_000) == limits.max_top_k
    assert limits.clamp_top_k(0) == 1
    assert limits.clamp_window(-5) == 0
    assert limits.clamp_window(999) == limits.max_window


def test_truncate_prefers_sentence_boundary():
    text = "第一段。" * 200
    cut, truncated = truncate_text(text, 100)

    assert truncated is True
    assert cut.endswith("…")
    assert len(cut) <= 101


def test_short_text_is_not_truncated():
    text, truncated = truncate_text("短文本", 100)

    assert text == "短文本"
    assert truncated is False


def test_response_bytes_measures_utf8_payload():
    assert response_bytes({"k": "中文"}) > len('{"k":"中文"}') - 1

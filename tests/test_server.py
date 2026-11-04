"""协议层契约：工具集合、schema、只读声明、错误翻译、resource。

这一层要保证的事只有一件：**模型看到的东西和设计文档说的一致。**
"""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ResourceError, ToolError

from rag_mcp.budget import Limits
from rag_mcp.registry import default_registry
from rag_mcp.server import MAX_TOP_K_SCHEMA, MAX_WINDOW_SCHEMA, mcp


def _call(name: str, arguments: dict | None = None):
    return asyncio.run(mcp.call_tool(name, arguments or {}))


def _payload(result) -> dict:
    return json.loads(result.content[0].text)


def _error_body(error: Exception) -> dict:
    """SDK 会给工具错误加 `Error executing tool <name>: ` 前缀（见 tool base.py）。

    前缀之后就是我们的结构化 JSON —— Agent 也是按第一个 `{` 之后解析。
    """
    text = str(error)
    return json.loads(text[text.index("{") :])


def _tools() -> dict:
    return {tool.name: tool for tool in asyncio.run(mcp.list_tools())}


# ---------------------------------------------------------------- 工具面


def test_tool_surface_is_small_and_stable():
    assert set(_tools()) == {"list_sources", "search", "fetch"}


def test_every_tool_is_declared_read_only_and_idempotent():
    """没声明 annotations，客户端就无法自动批准，每次调用都要人点确认（DESIGN §5.4）。"""
    for tool in _tools().values():
        assert tool.description
        assert tool.input_schema["type"] == "object"
        annotations = tool.annotations
        assert annotations is not None, f"{tool.name} 缺少 annotations"
        assert annotations.read_only_hint is True
        assert annotations.idempotent_hint is True


def test_instructions_carry_retrieval_strategy():
    """`instructions` 比单个工具描述更早注入，必须承载策略而不只是目录（DESIGN §5.3）。"""
    from rag_mcp.server import INSTRUCTIONS

    assert "list_sources" in INSTRUCTIONS
    assert "不要连续 search" in INSTRUCTIONS
    assert "untrusted" in INSTRUCTIONS  # 不可信内容的处理规则
    assert "chunk_id" in INSTRUCTIONS


# ---------------------------------------------------------------- schema 一致性


def test_schema_bounds_come_from_single_source():
    """v0 的债：`top_k` 上限在 schema 和 config 两处硬编码，改一处就不一致。"""
    assert Limits().max_top_k == MAX_TOP_K_SCHEMA
    assert Limits().max_window == MAX_WINDOW_SCHEMA

    search_schema = _tools()["search"].input_schema
    assert search_schema["properties"]["top_k"]["maximum"] == Limits().max_top_k
    assert _tools()["fetch"].input_schema["properties"]["window"]["maximum"] == Limits().max_window


def test_filter_params_match_registry():
    """词表漂移门禁（DESIGN §9.4）。

    入库侧新增了可过滤字段、而工具没改 → CI 红；
    而不是线上悄悄返回空结果、让人猜是不是拼错了。
    """
    properties = set(_tools()["search"].input_schema["properties"])
    declared = properties - {"query", "top_k"}
    assert declared == set(default_registry().names)


def test_registry_reserved_fields_are_not_exposed():
    properties = set(_tools()["search"].input_schema["properties"])
    assert not properties & {"security_level", "acl_groups"}


# ---------------------------------------------------------------- 正常路径


def test_search_tool_returns_structured_hits_without_raw_score():
    result = _call("search", {"query": "年假 天数", "top_k": 2})

    payload = _payload(result)
    assert payload["returned"] == 2
    assert payload["hits"][0]["section_path"].endswith("年假天数")
    assert payload["hits"][0]["confidence"] in {"high", "medium", "low"}
    assert "score" not in payload["hits"][0]
    assert payload["chars_returned"] > 0
    assert result.structured_content["corpus_version"]


def test_list_sources_is_self_describing():
    payload = _payload(_call("list_sources"))

    fields = {field["name"] for field in payload["sources"][0]["filter_fields"]}
    assert fields == {"department", "doc_type", "updated_after"}
    assert payload["sources"][0]["capabilities"]["retrieval_mode"] in {"lexical", "hybrid"}


def test_fetch_tool_returns_untrusted_marked_text():
    hit = _payload(_call("search", {"query": "年假", "top_k": 1}))["hits"][0]
    payload = _payload(_call("fetch", {"chunk_id": hit["chunk_id"], "window": 0}))

    assert payload["resolved_by"] == "exact"
    assert payload["text"].startswith("[untrusted document content | ")
    assert payload["chars_returned"] > 0


# ---------------------------------------------------------------- 错误通道


def test_recoverable_failure_becomes_structured_tool_error():
    """可恢复失败必须走工具错误 + 机器可读 code，而不是自由文本。"""
    with pytest.raises(ToolError) as excinfo:
        _call("fetch", {})

    body = _error_body(excinfo.value)
    assert body["code"] == "BAD_ARGUMENT"
    assert body["retryable"] is True
    assert body["hint"]


def test_overreach_surfaces_as_same_code_as_missing():
    with pytest.raises(ToolError) as forbidden:
        _call("fetch", {"doc_id": "hr-notice-org-2026"})
    with pytest.raises(ToolError) as missing:
        _call("fetch", {"doc_id": "no-such-doc"})

    assert _error_body(forbidden.value)["code"] == "NOT_FOUND_OR_FORBIDDEN"
    assert _error_body(forbidden.value)["code"] == _error_body(missing.value)["code"]


# ---------------------------------------------------------------- resource


def test_document_resource_is_readable():
    contents = asyncio.run(mcp.read_resource("rag://docs/eng-manual-vpn"))
    text = "".join(block.content for block in contents)

    assert "VPN" in text
    assert "[untrusted document content" in text


def test_document_resource_enforces_acl():
    """resource 独立鉴权：不能因为 id 看起来合法就放行（DESIGN §8.3 铁律 2）。"""
    with pytest.raises(ResourceError):
        asyncio.run(mcp.read_resource("rag://docs/hr-notice-org-2026"))

"""安全门禁：越权泄漏率必须为 0。

这是**阻断发布**的那一条（DESIGN §13.1）：其他指标可以调阈值，这个不可以。
"""

from __future__ import annotations

import pytest
from runner import principal

from rag_mcp.errors import NOT_FOUND_OR_FORBIDDEN, RagError


def test_no_privileged_content_leaks(report):
    leaks = report.leaks
    assert not leaks, "越权泄漏：" + "; ".join(
        f"{item.case.id} 泄漏了 {item.leaked}" for item in leaks
    )


def test_superseded_documents_stay_out_of_search(report):
    """废止文档不能出现在检索结果里 —— 但它仍然可以按 id 查阅（DESIGN A13 修订）。"""
    leaked = [
        item.case.id
        for item in report.results
        if item.case.tags and "superseded" in item.case.tags and item.leaked
    ]
    assert not leaked, f"废止文档出现在检索结果中：{leaked}"


def test_exec_principal_can_reach_restricted_notice(report):
    """反向校验：门禁不能靠「把所有人都挡掉」通过。"""
    restricted = [
        item for item in report.results if item.case.id == "s05"
    ]
    assert restricted, "golden 里缺少 exec 身份的正向用例"
    assert restricted[0].recalled, "exec 身份应当能看到 confidential 文档"


def test_overreach_returns_same_code_as_missing(service):
    with pytest.raises(RagError) as forbidden:
        service.fetch(principal("staff"), doc_id="hr-notice-org-2026")
    with pytest.raises(RagError) as missing:
        service.fetch(principal("staff"), doc_id="no-such-doc")

    assert forbidden.value.code == missing.value.code == NOT_FOUND_OR_FORBIDDEN
    assert "无权限" not in forbidden.value.message


def test_resource_uri_does_not_bypass_acl(service):
    """resource 是最常见的鉴权绕过点（DESIGN §8.3 铁律 2）。"""
    with pytest.raises(RagError):
        service.read_document(principal("staff"), "hr-notice-org-2026")


def test_untrusted_content_is_always_marked(service):
    """语料里植入的指令必须被标注出来，而不是静默拼进上下文。"""
    fetched = service.fetch(principal("staff"), doc_id="hr-policy-leave")
    assert fetched.text.startswith("[untrusted document content | ")

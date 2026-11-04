"""安全门禁：越权泄漏率必须为 0。

这是**阻断发布**的那一条（DESIGN §13.1）：其他指标可以调阈值，这个不可以。

注意语料差异：ACL 断言需要「存在可越权的文档」，只有内置样例语料具备
（一份 confidential + exec 组）。真实语料（如中医）是单部门 public 语料，
**测不了 ACL** —— 那种情况下跳过并说明，而不是假装通过。
"""

from __future__ import annotations

import pytest
from runner import principal

from rag_mcp.errors import NOT_FOUND_OR_FORBIDDEN, RagError


@pytest.fixture(scope="session")
def needs_confidential_corpus(corpus_kind) -> None:
    """依赖 confidential 文档的用例统一挂这个 fixture。"""
    if corpus_kind != "memory":
        pytest.skip(f"当前语料是 {corpus_kind}，没有可越权对象；ACL 请在样例 profile 下验证")


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


def test_exec_principal_can_reach_restricted_notice(report, needs_confidential_corpus):
    """反向校验：门禁不能靠「把所有人都挡掉」通过。"""
    restricted = [item for item in report.results if item.case.id == "s05"]
    assert restricted, "golden 里缺少 exec 身份的正向用例"
    assert restricted[0].recalled, "exec 身份应当能看到 confidential 文档"


def test_overreach_returns_same_code_as_missing(service, needs_confidential_corpus):
    with pytest.raises(RagError) as forbidden:
        service.fetch(principal("staff"), doc_id="hr-notice-org-2026")
    with pytest.raises(RagError) as missing:
        service.fetch(principal("staff"), doc_id="no-such-doc")

    assert forbidden.value.code == missing.value.code == NOT_FOUND_OR_FORBIDDEN
    assert "无权限" not in forbidden.value.message


def test_resource_uri_does_not_bypass_acl(service, needs_confidential_corpus):
    """resource 是最常见的鉴权绕过点（DESIGN §8.3 铁律 2）。"""
    with pytest.raises(RagError):
        service.read_document(principal("staff"), "hr-notice-org-2026")


def test_untrusted_content_is_always_marked(service, needs_confidential_corpus):
    """语料里植入的指令必须被标注出来，而不是静默拼进上下文。"""
    fetched = service.fetch(principal("staff"), doc_id="hr-policy-leave")
    assert fetched.text.startswith("[untrusted document content | ")

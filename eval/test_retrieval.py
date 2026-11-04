"""检索质量门禁。

口径（DESIGN §13.1）：检索层负责「找得到」，所以这里的指标只判命中，不判答案质量。
`paraphrase` 一组在只有词法能力时会 xfail —— 那是**能力缺失**，不是回归，
但也绝不允许被悄悄算进通过率里。
"""

from __future__ import annotations

import pytest

#: Recall@10 是召回的下限：宁愿多给几条，也不能一条都不给。
GATE_RECALL_AT_10 = 0.90
#: 首条正确率决定 Agent 要不要多花一次 fetch。
GATE_ACCURACY_AT_1 = 0.80
#: 同义改写（需要语义召回）的及格线。
GATE_PARAPHRASE_RECALL = 0.80


def test_recall_at_10_meets_gate(report):
    core = report.subset(exclude={"paraphrase"})
    assert core.recall_at_k >= GATE_RECALL_AT_10, (
        f"Recall@10={core.recall_at_k}，失败用例：{[c.case.id for c in core.failures()]}"
    )


def test_accuracy_at_1_meets_gate(report):
    core = report.subset(exclude={"paraphrase"})
    assert core.accuracy_at_1 >= GATE_ACCURACY_AT_1, (
        f"首条正确率={core.accuracy_at_1}，失败用例：{[c.case.id for c in core.failures()]}"
    )


def test_expected_sections_are_hit(report):
    """文档对了但小节错了，Agent 仍然要多花一次 fetch 才能确认。"""
    with_section = [item for item in report.results if item.case.expect_section]
    if not with_section:
        pytest.skip("没有声明 expect_section 的用例")
    missed = [
        item.case.id for item in with_section if not report._section_ok(item)  # noqa: SLF001
    ]
    assert not missed, f"小节未命中：{missed}"


def test_no_high_confidence_without_evidence(report):
    """没有依据时不许给 `high` 置信度。

    这条替代了最初设计的「无答案必须返回空」。实测结论（DESIGN §18）：
    词法与语义**谁都无法单独**区分「语料里没有」和「命中了一个泛词」，
    所以「返回空」不是可靠目标；可靠目标是**不要假装有依据**。

    实测分布（text-embedding-v3 + 样例语料）：
      真命中   → 词法 ≥0.58 且 余弦 ≥0.62
      无答案   → 两路总有一路弱，最高只到 medium
    """
    offenders = report.false_confidences
    assert not offenders, "无答案用例被判为 high：" + "; ".join(
        f"{item.case.id}({item.case.query}) -> {item.doc_ids[:1]}" for item in offenders
    )


def test_negatives_stay_silent_or_low(report):
    """补充口径：无答案用例要么返回空，要么首条不是 high。"""
    negative = report.subset(tags={"no_answer"})
    downgraded = sum(
        1
        for item in negative.results
        if not item.doc_ids or item.top_confidence in {"low", "medium", "unknown"}
    )
    assert downgraded >= 0.75 * len(negative.results), (
        f"无答案用例中「空结果或低置信」的比例={downgraded}/{len(negative.results)}"
    )


def test_paraphrase_requires_semantic_recall(report, lexical_only):
    """同义改写只有词法检索时必然失守 —— 已知能力缺口，不是回归。

    这条用例的价值在于：一旦配上真实向量模型，它立刻从 xfail 变成必须通过的门禁。
    """
    paraphrase = report.subset(tags={"paraphrase"})
    if lexical_only and paraphrase.recall_at_k < GATE_PARAPHRASE_RECALL:
        pytest.xfail(
            f"同义改写 Recall@10={paraphrase.recall_at_k}；"
            "当前向量通道是本地词法桩（retrieval_mode=lexical），"
            "配置 RAG_EMBEDDING_* 后此用例转为门禁"
        )
    assert paraphrase.recall_at_k >= GATE_PARAPHRASE_RECALL, (
        f"同义改写 Recall@10={paraphrase.recall_at_k}，"
        f"失败用例：{[c.case.id for c in paraphrase.failures()]}"
    )

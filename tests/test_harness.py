"""评测指标口径测试：未测量、recall@k、answer@k。"""

from __future__ import annotations

from harness import Case, CaseResult, Report
from runner import _shorten_query


def _case(*, expect=("d1",), contains=("答案",), tags=()) -> Case:
    return Case(
        id="c1",
        query="问题",
        principal="staff",
        expect_doc_ids=tuple(expect),
        must_not_appear=(),
        tags=tuple(tags),
        expect_contains=tuple(contains),
    )


def _result(
    *,
    case: Case | None = None,
    doc_ids=("d1",),
    texts=("答案在这里",),
) -> CaseResult:
    return CaseResult(
        case=case or _case(),
        doc_ids=list(doc_ids),
        section_paths=["s"] * len(doc_ids),
        chars=10,
        texts=tuple(texts),
    )


def test_answer_at_returns_none_when_not_measured():
    report = Report(results=[_result(case=_case(contains=()), doc_ids=[], texts=())])

    assert report.answer_at(1) is None
    assert report.summary()["answer@1"] == "n/a"


def test_answer_rank_uses_all_texts_not_only_first_five():
    case = _case(expect=("d2",))
    result = _result(
        case=case,
        doc_ids=("d1", "d2"),
        texts=("没有", "答案在第二条"),
    )

    assert result.answer_rank == 2
    assert result.answer_in_top(1) is False
    assert result.answer_in_top(2) is True


def test_recall_at_respects_k():
    case = _case(expect=("d1", "d2"), contains=())
    result = _result(case=case, doc_ids=("x", "d1", "d2"), texts=("", "", ""))

    assert result.recalled is True
    assert result.recalled_at(1) is False
    assert result.recalled_at(2) is False
    assert result.recalled_at(3) is True

    report = Report(results=[result])
    assert report.recall_at(2) == 0.0
    assert report.recall_at(3) == 1.0
    assert "recall@5" in report.summary(recall_k=5)


def test_shorten_query_changes_chinese_query():
    query = "感冒的病因病机是什么"

    shortened = _shorten_query(query)

    assert shortened != query
    assert shortened
    assert len(shortened) < len(query)


def test_shorten_query_uses_first_word_for_spaced_query():
    assert _shorten_query("年假 天数 计算") == "年假"

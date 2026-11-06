"""跑评测：把 golden 用例灌进检索服务，产出指标。

不依赖 pytest —— pytest 用例和 `python eval/run_eval.py` 都调这里，
保证「CI 门禁」和「人工看报表」跑的是同一套逻辑。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for entry in (_ROOT / "src", _ROOT / "eval"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from harness import Case, CaseResult, Report, golden_for, load_cases  # noqa: E402

from rag_mcp.config import Settings  # noqa: E402
from rag_mcp.errors import RagError  # noqa: E402
from rag_mcp.factory import build_service  # noqa: E402
from rag_mcp.policy import Principal  # noqa: E402

#: 评测用的固定身份。身份必须显式给出，不能靠环境变量 —— 否则同一份 golden
#: 在不同机器上跑出不同结果（DESIGN §13）。
PRINCIPALS: dict[str, Principal] = {
    "staff": Principal(user_id="alice", groups=frozenset({"all"}), max_level="internal"),
    "exec": Principal(user_id="carol", groups=frozenset({"all", "exec"}), max_level="confidential"),
}


def principal(name: str) -> Principal:
    if name not in PRINCIPALS:
        raise KeyError(f"未知身份 {name!r}，可用：{', '.join(PRINCIPALS)}")
    return PRINCIPALS[name]


def cases_for(service, cases: list[Case] | None = None) -> list[Case]:
    """评测集跟着后端走：SQLite = 真实语料，memory = 内置样例语料。"""
    if cases is not None:
        return cases
    settings = Settings.from_env()
    return load_cases(golden_for(settings.backend))


def evaluate(service, cases: list[Case] | None = None, *, top_k: int = 10) -> Report:
    """跑一遍检索指标。"""
    report = Report()
    for case in cases_for(service, cases):
        try:
            result = service.search(principal(case.principal), query=case.query, top_k=top_k)
        except RagError:
            # 查询被拒绝（空查询等）也算一次可观测结果，记为「什么都没拿到」
            report.results.append(CaseResult(case=case, doc_ids=[], section_paths=[], chars=0))
            continue
        report.results.append(
            CaseResult(
                case=case,
                doc_ids=[hit.doc_id for hit in result.hits],
                section_paths=[hit.section_path for hit in result.hits],
                chars=result.chars_returned,
                top_confidence=result.hits[0].confidence if result.hits else "unknown",
                # 必须与 doc_ids 等长；answer@k 可能看第 6 条及以后。
                texts=tuple(
                    service.chunk_text(principal(case.principal), hit.chunk_id) or ""
                    for hit in result.hits
                ),
            )
        )
    return report


def simulate_agent(service, cases: list[Case] | None = None, *, top_k: int = 8) -> Report:
    """Agent 层指标的**代理**实现。

    它不是真的 LLM agent —— 它复现的是「同一个检索服务下，Agent 的典型决策路径」：
    先原样搜；没找到就把查询截短再搜一次；还没找到就 fetch 首条确认。
    这样 `median_calls` 与 `mean_chars` 才有意义：**工具面和 description 的好坏，
    主要就体现在这两个数上（DESIGN §13.1）。**
    """
    report = Report()
    for case in cases_for(service, cases):
        actor = principal(case.principal)
        calls = 0
        chars = 0

        try:
            first = service.search(actor, query=case.query, top_k=top_k)
        except RagError:
            report.results.append(CaseResult(case=case, doc_ids=[], section_paths=[], chars=0))
            continue
        calls += 1
        chars += first.chars_returned

        if _satisfied(case, first):
            report.results.append(
                CaseResult(
                    case=case,
                    doc_ids=[hit.doc_id for hit in first.hits],
                    section_paths=[hit.section_path for hit in first.hits],
                    chars=chars,
                    tool_calls=calls,
                )
            )
            continue

        # 第二步：截短查询（真实的改写动作，server 不该替 Agent 做这件事）。
        # 中文没有空格，旧实现会原样重跑一遍，Agent 代理指标因此虚高。
        shortened = _shorten_query(case.query)
        try:
            second = service.search(actor, query=shortened, top_k=top_k)
            calls += 1
            chars += second.chars_returned
        except RagError:
            second = None

        chosen = second if second is not None and second.hits else first

        if not _satisfied(case, chosen) and chosen.hits:
            # 第三步：展开首条确认是否真的有依据
            try:
                fetched = service.fetch(actor, chunk_id=chosen.hits[0].chunk_id, window=1)
                calls += 1
                chars += fetched.chars_returned
            except RagError:
                pass

        report.results.append(
            CaseResult(
                case=case,
                doc_ids=[hit.doc_id for hit in chosen.hits],
                section_paths=[hit.section_path for hit in chosen.hits],
                chars=chars,
                tool_calls=calls,
            )
        )
    return report


def _shorten_query(query: str) -> str:
    parts = query.split()
    if len(parts) > 1:
        return parts[0]
    if len(query) <= 2:
        return query
    # 中文按字符截半；保留至少 2 个字，避免变成空查询
    return query[: max(2, len(query) // 2)].strip() or query


def _satisfied(case: Case, result) -> bool:
    if case.is_negative:
        return not result.hits
    return any(hit.doc_id in case.expect_doc_ids for hit in result.hits)


def default_service(settings: Settings | None = None):
    return build_service(settings or Settings.from_env())

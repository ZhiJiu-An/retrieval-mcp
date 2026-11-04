"""评测指标：纯函数，不碰服务、不碰网络。

设计依据：DESIGN.md §13.1 三组指标。

「可评测」的含义只有一条：**任何一次改动都能被判成变好或变坏。**
所以这里故意不引入任何黑盒框架 —— 指标算错的时候你能一眼看出来。
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"
#: 真实语料（SQLite 后端）用另一套评测集，两类语料的问题分布不可混用
GOLDEN_TCM_PATH = Path(__file__).parent / "golden_tcm.jsonl"


@dataclass(frozen=True)
class Case:
    id: str
    query: str
    principal: str
    expect_doc_ids: tuple[str, ...]
    must_not_appear: tuple[str, ...]
    tags: tuple[str, ...]
    expect_section: str | None = None

    @property
    def is_negative(self) -> bool:
        """无答案 / 应被拒：期望「搜不到」，属于安全与静默的正确性。"""
        return not self.expect_doc_ids


@dataclass
class CaseResult:
    case: Case
    doc_ids: list[str]
    section_paths: list[str]
    chars: int
    tool_calls: int = 1
    top_confidence: str = "unknown"

    @property
    def hits_at_1(self) -> bool:
        return bool(self.doc_ids) and self.doc_ids[0] in self.case.expect_doc_ids

    @property
    def recalled(self) -> bool:
        return set(self.case.expect_doc_ids) <= set(self.doc_ids)

    @property
    def leaked(self) -> list[str]:
        return [doc for doc in self.case.must_not_appear if doc in self.doc_ids]

    @property
    def false_confidence(self) -> bool:
        """**真的没有答案**的问题被给了 `high` 置信度 —— 这才是要拦的错误。

        注意口径：`is_negative`（没有期望文档）包含了 `superseded` 这种
        「该返回现行版、只是不许返回废止版」的用例，它是**可回答**的，
        返回 high 是对的。所以这里只认 `no_answer` 标签。

        「完全返回空」也不是正确性的必要条件：语料小、问题模糊时，
        给出几条 `medium` 让 Agent 自己去 fetch 确认，比强行沉默更诚实。
        但给了 `high` 就等于告诉模型「这就是依据」，那是真的错。
        """
        return "no_answer" in self.case.tags and self.top_confidence == "high"

    @property
    def ranked_docs(self) -> list[str]:
        """按文档去重后的排名。

        不去重会让「同一篇文档占两行」被当成两个相关结果，
        DCG 被重复累加，算出 ndcg > 1 这种不可能的数。
        """
        return list(dict.fromkeys(self.doc_ids))

    @property
    def reciprocal_rank(self) -> float:
        for rank, doc_id in enumerate(self.ranked_docs, start=1):
            if doc_id in self.case.expect_doc_ids:
                return 1.0 / rank
        return 0.0

    @property
    def ndcg(self) -> float:
        """二值相关性。理想排序 = 相关文档全部排在最前。"""
        if not self.case.expect_doc_ids:
            return 1.0 if not self.doc_ids else 0.0
        gains = [
            1.0 if doc_id in self.case.expect_doc_ids else 0.0
            for doc_id in self.ranked_docs[:10]
        ]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal = sum(
            1.0 / math.log2(index + 2) for index in range(min(len(self.case.expect_doc_ids), 10))
        )
        return dcg / ideal if ideal else 0.0


@dataclass
class Report:
    results: list[CaseResult] = field(default_factory=list)

    def subset(self, *, tags: Iterable[str] | None = None, exclude: Iterable[str] | None = None):
        chosen = self.results
        if tags is not None:
            wanted = set(tags)
            chosen = [item for item in chosen if wanted & set(item.case.tags)]
        if exclude is not None:
            unwanted = set(exclude)
            chosen = [item for item in chosen if not (unwanted & set(item.case.tags))]
        return Report(results=list(chosen))

    # ---- 指标 ----

    def _positive(self) -> list[CaseResult]:
        return [item for item in self.results if not item.case.is_negative]

    @property
    def recall_at_k(self) -> float:
        positives = self._positive()
        return sum(1 for item in positives if item.recalled) / len(positives) if positives else 1.0

    @property
    def accuracy_at_1(self) -> float:
        positives = self._positive()
        return sum(1 for item in positives if item.hits_at_1) / len(positives) if positives else 1.0

    @property
    def mrr(self) -> float:
        positives = self._positive()
        if not positives:
            return 1.0
        return sum(item.reciprocal_rank for item in positives) / len(positives)

    @property
    def ndcg_at_10(self) -> float:
        positives = self._positive()
        return sum(item.ndcg for item in positives) / len(positives) if positives else 1.0

    @property
    def silent_rate(self) -> float:
        """无答案用例真的返回空的比例。为空说明语料被过度过滤，为低说明召回在乱给。"""
        negatives = [
            item
            for item in self.results
            if item.case.is_negative and not item.case.must_not_appear
        ]
        if not negatives:
            return 1.0
        return sum(1 for item in negatives if not item.doc_ids) / len(negatives)

    @property
    def leaks(self) -> list[CaseResult]:
        return [item for item in self.results if item.leaked]

    @property
    def false_confidences(self) -> list[CaseResult]:
        return [item for item in self.results if item.false_confidence]

    @property
    def median_calls(self) -> float:
        calls = sorted(item.tool_calls for item in self._positive())
        if not calls:
            return 0.0
        middle = len(calls) // 2
        return float(calls[middle]) if len(calls) % 2 else (calls[middle - 1] + calls[middle]) / 2

    @property
    def mean_chars(self) -> float:
        values = [item.chars for item in self.results]
        return sum(values) / len(values) if values else 0.0

    def failures(self) -> list[CaseResult]:
        return [
            item
            for item in self._positive()
            if not item.recalled
            or item.leaked
            or (item.case.expect_section and not self._section_ok(item))
        ]

    @staticmethod
    def _section_ok(item: CaseResult) -> bool:
        wanted = item.case.expect_section
        return bool(wanted) and any(wanted in path for path in item.section_paths)

    def summary(self) -> dict[str, float]:
        return {
            "cases": len(self.results),
            "recall@10": round(self.recall_at_k, 3),
            "accuracy@1": round(self.accuracy_at_1, 3),
            "mrr": round(self.mrr, 3),
            "ndcg@10": round(self.ndcg_at_10, 3),
            "silent_rate": round(self.silent_rate, 3),
            "leaks": len(self.leaks),
            "false_high": len(self.false_confidences),
            "median_calls": round(self.median_calls, 2),
            "mean_chars": round(self.mean_chars, 1),
        }


def golden_for(backend: str) -> Path:
    return GOLDEN_TCM_PATH if backend == "sqlite" else GOLDEN_PATH


def load_cases(path: Path | None = None) -> list[Case]:
    target = path or Path(os.getenv("RAG_EVAL_GOLDEN") or GOLDEN_PATH)
    cases: list[Case] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        cases.append(
            Case(
                id=payload["id"],
                query=payload["query"],
                principal=payload.get("principal", "staff"),
                expect_doc_ids=tuple(payload.get("expect_doc_ids") or ()),
                must_not_appear=tuple(payload.get("must_not_appear") or ()),
                tags=tuple(payload.get("tags") or ()),
                expect_section=payload.get("expect_section"),
            )
        )
    return cases


def format_table(rows: Sequence[tuple[str, dict[str, float]]]) -> str:
    """把多个配置的结果并排放，用于 before/after 对比。"""
    if not rows:
        return "(no data)"
    keys = list(rows[0][1].keys())
    header = f"{'config':<22}" + "".join(f"{key:>12}" for key in keys)
    lines = [header, "-" * len(header)]
    for name, values in rows:
        lines.append(f"{name:<22}" + "".join(f"{values.get(key, 0):>12}" for key in keys))
    return "\n".join(lines)

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
    #: 命中片段正文里必须出现的字符串（任一即可）。
    #: 用来做**片段级**判定：章节对 + 片段里真有方剂名，才算答对。
    expect_contains: tuple[str, ...] = ()

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
    #: 与 doc_ids 等长的完整正文（不是 300 字片段）——判断答案是否真的在里面。
    #: 找不到正文的位置为空串。`answer@k` 依赖它与 doc_ids 对齐。
    texts: tuple[str, ...] = ()

    @property
    def hits_at_1(self) -> bool:
        return bool(self.doc_ids) and self.doc_ids[0] in self.case.expect_doc_ids

    @property
    def recalled(self) -> bool:
        return set(self.case.expect_doc_ids) <= set(self.doc_ids)

    def recalled_at(self, k: int) -> bool:
        """前 k 条是否覆盖全部期望文档。"""
        return set(self.case.expect_doc_ids) <= set(self.doc_ids[:k])

    @property
    def leaked(self) -> list[str]:
        return [doc for doc in self.case.must_not_appear if doc in self.doc_ids]

    @property
    def answer_rank(self) -> int | None:
        """答案真正出现的位置（第几条命中），没出现则 None。

        判定要**同时**满足两条：
          1. 该条属于正确的病证章节
          2. 该条的**完整正文**里出现答案关键词
        只判关键词不行 —— 方剂名在第三章的通治汇总表里也出现；
        只看章节也不行 —— 同一章里可能有好几节，未必是答那一段。
        """
        if not self.case.expect_contains or not self.case.expect_doc_ids:
            return None
        for rank, doc_id in enumerate(self.doc_ids, start=1):
            # texts 必须与 doc_ids 等长；早期只存前 5 条，answer@10 会静默漏判。
            if rank > len(self.texts):
                break
            if doc_id not in self.case.expect_doc_ids:
                continue
            if any(word in self.texts[rank - 1] for word in self.case.expect_contains):
                return rank
        return None

    def answer_in_top(self, k: int) -> bool:
        rank = self.answer_rank
        return rank is not None and rank <= k

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
            1.0 if doc_id in self.case.expect_doc_ids else 0.0 for doc_id in self.ranked_docs[:10]
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

    def recall_at(self, k: int) -> float:
        """前 k 条召回率。`k` 必须显式传入，避免 --top-k 5 仍叫 recall@10。"""
        positives = self._positive()
        return (
            sum(1 for item in positives if item.recalled_at(k)) / len(positives)
            if positives
            else 1.0
        )

    @property
    def recall_at_k(self) -> float:
        """兼容旧调用：默认按 top-k=10 的口径。"""
        return self.recall_at(10)

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

    def answer_at(self, k: int) -> float | None:
        """答案出现在前 k 条的比例 —— 检索「有没有用」的直接度量。

        没有声明 `expect_contains` 的数据集返回 `None`（表示“没测”），
        不能返回 1.0 假装全对。
        """
        scored = [item for item in self.results if item.case.expect_contains]
        if not scored:
            return None
        return sum(1 for item in scored if item.answer_in_top(k)) / len(scored)

    @property
    def answer_at_1(self) -> float | None:
        return self.answer_at(1)

    @property
    def silent_rate(self) -> float:
        """无答案用例真的返回空的比例。为空说明语料被过度过滤，为低说明召回在乱给。"""
        negatives = [
            item for item in self.results if item.case.is_negative and not item.case.must_not_appear
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

    def summary(self, *, recall_k: int = 10) -> dict[str, float | str]:
        answer_1 = self.answer_at(1)
        answer_5 = self.answer_at(5)
        return {
            "cases": len(self.results),
            f"recall@{recall_k}": round(self.recall_at(recall_k), 3),
            "accuracy@1": round(self.accuracy_at_1, 3),
            "mrr": round(self.mrr, 3),
            "ndcg@10": round(self.ndcg_at_10, 3),
            "answer@1": round(answer_1, 3) if answer_1 is not None else "n/a",
            "answer@5": round(answer_5, 3) if answer_5 is not None else "n/a",
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
                expect_contains=tuple(payload.get("expect_contains") or ()),
            )
        )
    return cases


def format_table(rows: Sequence[tuple[str, dict[str, float | str]]]) -> str:
    """把多个配置的结果并排放，用于 before/after 对比。"""
    if not rows:
        return "(no data)"
    keys = list(rows[0][1].keys())
    header = f"{'config':<22}" + "".join(f"{key:>12}" for key in keys)
    lines = [header, "-" * len(header)]
    for name, values in rows:
        lines.append(f"{name:<22}" + "".join(f"{values.get(key, 0):>12}" for key in keys))
    return "\n".join(lines)

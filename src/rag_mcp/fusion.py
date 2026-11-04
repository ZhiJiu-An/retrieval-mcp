"""结果融合：Reciprocal Rank Fusion。

设计依据：DESIGN.md §6.2、A12。

为什么用 RRF 而不是「分数加权」：BM25 的分数无上界，余弦相似度落在 [-1, 1]，
两路量纲不同。加权需要先归一化，而归一化又依赖分数分布假设——
换个语料假设就失效。RRF 只用**排名**，对分数量纲免疫。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: RRF 的平滑常数。60 是文献里的常用取值：它决定「第 1 名比第 2 名强多少」。
#: 值越大，两路之间的排名差异被压得越平。
RRF_K = 60


def rrf(ranked_lists: Sequence[Sequence[Any]], *, k: int = RRF_K) -> list[tuple[Any, float]]:
    """融合多路召回。

    入参：多个已排序的列表，元素必须带 `.chunk_id`。
    出参：按融合分降序的 `(元素, 分数)`，元素取自第一次出现的那一路。

    分数含义：`Σ 1/(k + rank)`。**它是排名分，不是相关性分** —— 不要拿它跟阈值比。
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, Any] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = item.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, item)

    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [(first_seen[key], score) for key, score in ordered]

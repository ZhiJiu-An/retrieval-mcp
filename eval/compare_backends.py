"""可行性对比：同一套考题，跑不同检索策略，看每一步值多少。

为什么单独写这个脚本：

这个仓库其余部分都在证明「我能把它做对」，而面试要回答的是另一个问题：
**「你怎么知道这样更好？」** 没有基线对比的检索系统，所有指标都只是自证。

做法：同一套 golden、同一套指标，只改「走哪几路召回 / 要不要重排」，
把纯向量（业界最常见的做法）当基线，逐层加上去。

```bash
RAG_ENV_FILE=config/tcm.env python eval/compare_backends.py
```
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for entry in (_ROOT / "src", _ROOT / "eval"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from harness import format_table, load_cases  # noqa: E402
from runner import cases_for, evaluate  # noqa: E402

from rag_mcp.config import Settings, reload_dotenv  # noqa: E402
from rag_mcp.factory import build_service  # noqa: E402

#: (名字, 检索模式, 是否重排, 是否启用相关性下限)
CONFIGS = [
    ("① 纯向量（业界基线）", "dense", False, False),
    ("② 纯向量 + 重排", "dense", True, False),
    ("③ 混合召回（BM25+向量）", "hybrid", False, False),
    ("④ 混合 + 重排", "hybrid", True, False),
    ("⑤ 混合 + 重排 + 下限（生产）", "hybrid", True, True),
]


def build(settings: Settings, mode: str, rerank: bool, floor: bool):
    tuned = replace(
        settings,
        retrieval_mode=mode,
        rerank=replace(settings.rerank, enabled=rerank),
        limits=replace(
            settings.limits,
            relevance_floor=settings.limits.relevance_floor if floor else 0.0,
        ),
    )
    return build_service(tuned)


def main() -> int:
    reload_dotenv()
    base = Settings.from_env()
    golden = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    print("=" * 88)
    print("检索策略对比 —— 同一套考题，只改「走哪几路 + 要不要重排」")
    print("=" * 88)

    rows: list[tuple[str, dict[str, float | str]]] = []
    latencies: dict[str, float] = {}

    for name, mode, rerank, floor in CONFIGS:
        service = build(base, mode, rerank, floor)
        cases = load_cases(golden) if golden else cases_for(service)

        started = time.perf_counter()
        report = evaluate(service, cases)
        elapsed = (time.perf_counter() - started) * 1000 / max(1, len(cases))
        latencies[name] = elapsed

        answer_1 = report.answer_at(1)
        answer_5 = report.answer_at(5)
        summary = {
            "cases": len(cases),
            "recall@10": round(report.recall_at_k, 3),
            "acc@1": round(report.accuracy_at_1, 3),
            "mrr": round(report.mrr, 3),
            "ndcg@10": round(report.ndcg_at_10, 3),
            "chars": round(report.mean_chars),
            # 反例（语料里没有答案的问题）被正确压下去的比例。
            # 这是「相关性下限」唯一能起作用的地方，也是排名指标看不到的差异。
            "silent@neg": round(report.subset(tags={"no_answer"}).silent_rate, 2),
            # 最严判定：第 1 条既在对的章节、正文里又真有答案关键词。
            # 数据集没有 expect_contains 时是 n/a，不是 1.0。
            "answer@1": round(answer_1, 2) if answer_1 is not None else "n/a",
            "answer@5": round(answer_5, 2) if answer_5 is not None else "n/a",
        }
        rows.append((name, summary))
        print(f"  完成 {name}")

    print()
    print(format_table(rows))
    print()
    label = "单次检索平均耗时（含网络）"
    print(f"{label:<26}" + "".join(f"{latencies[n]:>12.0f}ms" for n, *_ in CONFIGS))

    baseline = rows[0][1]
    final = rows[-1][1]
    print()
    print("-" * 88)
    print("相对纯向量基线的增量：")
    for key in ("recall@10", "acc@1", "answer@1", "answer@5", "silent@neg"):
        left = baseline.get(key)
        right = final.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta = right - left
            print(f"  {key:<10} {left:.3f} → {right:.3f}   ({delta:+.3f})")
        else:
            print(f"  {key:<10} n/a（当前评测集没有 expect_contains）")
    print(
        f"  {'平均上下文':<10} {baseline['chars']:>4} → {final['chars']:>4} 字/题 "
        f"({final['chars'] - baseline['chars']:+d})"
    )
    print(
        f"  {'反例沉默率':<10} {baseline['silent@neg']:.2f} → {final['silent@neg']:.2f}"
        f"   ({final['silent@neg'] - baseline['silent@neg']:+.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

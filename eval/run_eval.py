#!/usr/bin/env python
"""评测入口：`python eval/run_eval.py`。

输出一张可对比的表 + 失败清单。退出码：

- `0` 全部门禁通过
- `1` 有越权泄漏，或检索/Agent 指标低于门禁

设计要点（DESIGN §13.3）：**参数只接受评测集的裁决，不接受直觉。**
所以改任何检索参数之后，标准动作是先跑这个脚本，再看表。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for entry in (_ROOT / "src", _ROOT / "eval"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from harness import load_cases  # noqa: E402
from runner import default_service, evaluate, simulate_agent  # noqa: E402

#: 与 eval/test_*.py 里的门禁保持一致 —— 这里是「人工看」，那里是「CI 拦」。
GATE_RECALL = 0.90
GATE_ACCURACY = 0.80
GATE_PARAPHRASE = 0.80
GATE_AGENT_SUCCESS = 0.85
GATE_MEDIAN_CALLS = 2.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="输出机器可读结果")
    parser.add_argument("--failures", action="store_true", help="打印失败用例明细")
    args = parser.parse_args()

    service = default_service()
    cases = load_cases()

    report = evaluate(service, cases, top_k=args.top_k)
    agent = simulate_agent(service, cases)

    lexical_only = service.capabilities.retrieval_mode == "lexical"
    core = report.subset(exclude={"paraphrase"})
    paraphrase = report.subset(tags={"paraphrase"})
    agent_core = agent.subset(exclude={"paraphrase", "no_answer", "denied", "superseded"})

    summary = {
        "capabilities": service.capabilities.model_dump(),
        "config_hash": service.config_hash,
        "corpus_version": service.describe().sources[0].corpus_version,
        "retrieval": report.summary(),
        "retrieval_core": core.summary(),
        "paraphrase": paraphrase.summary(),
        "agent": agent_core.summary(),
    }

    gates = {
        "leaks==0": len(report.leaks) == 0,
        f"recall@10>={GATE_RECALL}": core.recall_at_k >= GATE_RECALL,
        f"accuracy@1>={GATE_ACCURACY}": core.accuracy_at_1 >= GATE_ACCURACY,
        f"agent_success>={GATE_AGENT_SUCCESS}": agent_core.accuracy_at_1 >= GATE_AGENT_SUCCESS,
        f"median_calls<={GATE_MEDIAN_CALLS}": agent_core.median_calls <= GATE_MEDIAN_CALLS,
        f"paraphrase>={GATE_PARAPHRASE}": (paraphrase.recall_at_k >= GATE_PARAPHRASE)
        or lexical_only,
        "no_false_high==0": len(report.false_confidences) == 0,
    }

    if args.json:
        print(json.dumps({**summary, "gates": gates}, ensure_ascii=False, indent=2))
    else:
        _print_human(summary, gates, report, lexical_only, args.failures)

    return 0 if all(gates.values()) else 1


def _print_human(
    summary: dict, gates: dict, report, lexical_only: bool, show_failures: bool
) -> None:
    caps = summary["capabilities"]
    print("=" * 74)
    print("retrieval-mcp 评测")
    print("=" * 74)
    print(f"检索模式 : {caps['retrieval_mode']}  (embedder={caps['embedder']}, "
          f"reranker={caps['reranker']}, segmenter={caps['segmenter']})")
    print(f"配置指纹 : config_hash={summary['config_hash']}  "
          f"corpus_version={summary['corpus_version']}")
    if lexical_only:
        print("注意     : 向量通道是本地词法桩，语义召回未生效。")
        print("           「同义改写」与「无答案沉默率」两项不计入门禁 —— 两者是同一个根因。")
    print()

    rows = [
        ("全部", summary["retrieval"]),
        ("核心(去改写)", summary["retrieval_core"]),
        ("同义改写", summary["paraphrase"]),
        ("Agent 代理", summary["agent"]),
    ]
    keys = [
        "cases", "recall@10", "accuracy@1", "mrr", "ndcg@10",
        "silent_rate", "median_calls", "mean_chars",
    ]
    header = f"{'分组':<20}" + "".join(f"{key:>13}" for key in keys)
    print(header)
    print("-" * len(header))
    for name, values in rows:
        print(f"{name:<20}" + "".join(f"{values.get(k, 0):>13}" for k in keys))
    print()

    print("门禁")
    print("-" * 74)
    for name, passed in gates.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print()

    leaks = report.leaks
    if leaks:
        print("越权泄漏（必须为 0）")
        for item in leaks:
            print(f"  - {item.case.id}: 泄漏了 {item.leaked}")
        print()

    failures = report.failures()
    if failures and (show_failures or not all(gates.values())):
        print("未达成的用例")
        for item in failures:
            print(
                f"  - {item.case.id} [{','.join(item.case.tags)}] {item.case.query!r}\n"
                f"      期望 {list(item.case.expect_doc_ids)} 实得 {item.doc_ids[:3]}"
            )


if __name__ == "__main__":
    raise SystemExit(main())

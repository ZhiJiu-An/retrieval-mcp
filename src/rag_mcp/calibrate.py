"""阈值标定：用真实考题量出「相关 / 不相关」的分数线。

设计依据：DESIGN.md §18.2.3（阈值不能跨语料迁移）、§18.2.4（重排量纲）。

为什么必须有这个脚本：

`RAG_RELEVANCE_FLOOR` / `RAG_CONFIDENCE_HIGH` / `RAG_CONFIDENCE_MEDIUM` 三个数字
描述的是「这份语料 + 这个重排模型」下的分数分布。换语料或换模型，分布就变，
沿用旧值会**静默变差**（实测过：同一套值让中医语料的真实问题全被丢掉）。

所以正确做法不是给人一个滑块，而是给人一条命令：

```bash
python -m rag_mcp.calibrate --db rag.db \
    --golden eval/golden_tcm.jsonl --out config/tcm.calibrated.env
```

它做三件事：拉出**未经阈值过滤**的原始分数 → 量出分离区间 → 写出建议的配置档。
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .factory import build_service
from .policy import Principal
from .service import RetrievalService

PRINCIPALS: dict[str, Principal] = {
    "staff": Principal(user_id="alice", groups=frozenset({"all"}), max_level="internal"),
    "exec": Principal(user_id="carol", groups=frozenset({"all", "exec"}), max_level="confidential"),
}


@dataclass
class Probe:
    case_id: str
    query: str
    positive: bool
    best_rerank: float | None
    best_dense: float | None
    expected: tuple[str, ...]


def load_cases(path: Path) -> list[dict[str, Any]]:
    """读 golden.jsonl。只取标定需要的字段，不依赖 eval/ 的包。"""
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        cases.append(
            {
                "id": payload["id"],
                "query": payload["query"],
                "principal": payload.get("principal", "staff"),
                "expect": tuple(payload.get("expect_doc_ids") or ()),
                "tags": tuple(payload.get("tags") or ()),
            }
        )
    return cases


def probe(service: RetrievalService, cases: list[dict[str, Any]], *, limit: int) -> list[Probe]:
    results: list[Probe] = []
    for case in cases:
        candidates = service.raw_candidates(
            PRINCIPALS[case["principal"]], query=case["query"], limit=limit
        )
        reranks = [c.rerank for c in candidates if c.rerank is not None]
        positives = [c for c in candidates if c.doc_id in case["expect"]]

        if case["expect"]:
            # 正例：只看「期望文档里的候选」，取它最强的那一条
            best_rerank = max((c.rerank or 0.0) for c in positives) if positives else None
            best_dense = max((c.dense or 0.0) for c in positives) if positives else None
        else:
            # 反例：整份候选里最强的分数，这才是要压下去的噪声水位
            best_rerank = max(reranks) if reranks else None
            best_dense = max((c.dense or 0.0) for c in candidates) if candidates else None

        results.append(
            Probe(
                case_id=case["id"],
                query=case["query"],
                positive=bool(case["expect"]),
                best_rerank=best_rerank,
                best_dense=best_dense,
                expected=case["expect"],
            )
        )
    return results


def _round(value: float, step: float = 0.05) -> float:
    return round(round(value / step) * step, 2)


def suggest(probes: list[Probe]) -> dict[str, Any]:
    pos = [p.best_rerank for p in probes if p.positive and p.best_rerank is not None]
    neg = [p.best_rerank for p in probes if not p.positive and p.best_rerank is not None]
    pos_dense = [p.best_dense for p in probes if p.positive and p.best_dense is not None]
    neg_dense = [p.best_dense for p in probes if not p.positive and p.best_dense is not None]

    suggestions: dict[str, Any] = {"positives": len(pos), "negatives": len(neg)}
    if not pos or not neg:
        suggestions["error"] = "正例或反例不足，无法标定（golden 里两类都要有）"
        return suggestions

    pos_low, neg_high = min(pos), max(neg)
    separated = neg_high < pos_low
    # 分数线画在噪声水位与真实水位的中点；两者交叉时退化为噪声水位上方一点点
    floor = (neg_high + pos_low) / 2 if separated else neg_high * 0.98
    high = max(floor + 0.1, statistics.median(pos))
    medium = (floor + high) / 2

    suggestions.update(
        {
            "separated": separated,
            "positives_min": round(pos_low, 4),
            "positives_median": round(statistics.median(pos), 4),
            "negatives_max": round(neg_high, 4),
            "relevance_floor": _round(floor),
            "confidence_high": _round(high),
            "confidence_medium": _round(medium),
        }
    )
    if pos_dense and neg_dense:
        d_sep = max(neg_dense) < min(pos_dense)
        suggestions.update(
            {
                "dense_separated": d_sep,
                "dense_positives_min": round(min(pos_dense), 4),
                "dense_negatives_max": round(max(neg_dense), 4),
                "dense_floor": _round(
                    (max(neg_dense) + min(pos_dense)) / 2 if d_sep else max(neg_dense) * 0.98
                ),
            }
        )
    return suggestions


def _separation_comment(report: dict[str, Any]) -> str:
    if not report["separated"]:
        return "# 警告: 正反例分数重叠，阈值只是折中值"
    return (
        f"# 分离: 反例最高 {report['negatives_max']:.4f} < "
        f"正例最低 {report['positives_min']:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="用真实考题标定阈值")
    parser.add_argument("--db", type=Path, help="SQLite 索引路径（不给则用当前配置档）")
    parser.add_argument("--golden", type=Path, required=True, help="golden.jsonl 路径")
    parser.add_argument("--out", type=Path, help="把建议值写成 env 档")
    parser.add_argument("--limit", type=int, default=50, help="每题的候选深度")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.db:
        settings = settings.with_backend("sqlite", args.db)
    service = build_service(settings)

    cases = load_cases(args.golden)
    probes = probe(service, cases, limit=args.limit)
    report = suggest(probes)

    print("=" * 74)
    print("阈值标定")
    print("=" * 74)
    print(f"索引     : {service.describe().sources[0].name}")
    print(f"检索模式 : {service.resolved_mode}  (重排={service.capabilities.reranker})")
    print(f"考题     : {len(cases)} 条 → 正例 {report['positives']} / 反例 {report['negatives']}")
    print()
    print(f"{'题号':6s} {'类型':5s} {'最强重排分':>10s} {'最强余弦':>9s}  题目")
    print("-" * 74)
    for item in probes:
        kind = "正例" if item.positive else "反例"
        rerank = "-" if item.best_rerank is None else f"{item.best_rerank:.3f}"
        dense = "-" if item.best_dense is None else f"{item.best_dense:.3f}"
        print(f"{item.case_id:6s} {kind:5s} {rerank:>10s} {dense:>9s}  {item.query[:30]}")

    print()
    if "error" in report:
        print(f"❌ {report['error']}")
        return 1

    print("-" * 74)
    verdict = "✅ 可分离" if report["separated"] else "❌ 重叠，语料或考题有问题"
    print(
        f"反例最高分 {report['negatives_max']:.4f}  ←→  "
        f"正例最低分 {report['positives_min']:.4f}   {verdict}"
    )
    print(f"正例中位数 {report['positives_median']:.4f}")
    print()
    print("建议写入：")
    print(f"  RAG_RELEVANCE_FLOOR={report['relevance_floor']}")
    print(f"  RAG_CONFIDENCE_HIGH={report['confidence_high']}")
    print(f"  RAG_CONFIDENCE_MEDIUM={report['confidence_medium']}")
    if "dense_floor" in report:
        print(f"  RAG_DENSE_FLOOR={report['dense_floor']}"
              f"   （反例最高余弦 {report['dense_negatives_max']:.4f} / "
              f"正例最低余弦 {report['dense_positives_min']:.4f}）")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# 由 `python -m rag_mcp.calibrate` 生成，请勿手改",
            f"# 索引: {service.describe().sources[0].name}",
            f"# 重排: {service.capabilities.reranker}",
            f"# 考题: {len(cases)} 条（正例 {report['positives']} / 反例 {report['negatives']}）",
            _separation_comment(report),
            "",
            f"RAG_RELEVANCE_FLOOR={report['relevance_floor']}",
            f"RAG_CONFIDENCE_HIGH={report['confidence_high']}",
            f"RAG_CONFIDENCE_MEDIUM={report['confidence_medium']}",
        ]
        if "dense_floor" in report:
            lines.append(f"RAG_DENSE_FLOOR={report['dense_floor']}")
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n✅ 已写出 {args.out}")

    return 0 if report["separated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

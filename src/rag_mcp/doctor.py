"""连通性自检：`python -m rag_mcp.doctor`。

回答三个问题，顺序就是排查顺序：

1. **索引能不能建？** 语料读不出来，后面全都白搭。
2. **向量模型通不通？** 不通就退回词法模式 —— 功能退化，但服务可用。
3. **重排模型通不通？** 不通就退回启发式重排。

设计依据：DESIGN.md §3.2（启动必须快、不能崩）、§16.1（embedding 放哪算）。

它**不是** MCP 工具：不注册给模型，只给人看。所以可以放心 print 到 stdout。
"""

from __future__ import annotations

import time

from .config import Settings
from .embedding import build_embedder
from .errors import RagError
from .factory import build_service
from .models import HealthReport
from .rerank import build_reranker
from .text import has_word_segmenter

PROBE_TEXT = "连通性自检：员工年假天数"


def run(settings: Settings | None = None) -> HealthReport:
    """跑一遍检查，返回结构化结果（不打印）。测试与 CI 也调这个。"""
    resolved = settings or Settings.from_env()
    report = HealthReport(
        data_ok=False,
        docs=0,
        chunks=0,
        embedder=resolved.embedding.describe(),
        reranker=resolved.rerank.describe(),
        segmenter="word" if has_word_segmenter() else "bigram",
        retrieval_mode="lexical",
        corpus_version="",
        schema_version="",
        config_hash="",
    )

    # ---- 1. 索引 ----
    try:
        service = build_service(resolved)
        info = service.describe().sources[0]
    except Exception as exc:
        report.notes.append(f"语料/词表加载失败：{type(exc).__name__}: {exc}")
        return report

    report.data_ok = True
    report.docs = info.doc_count
    report.chunks = info.chunk_count
    report.corpus_version = info.corpus_version
    report.schema_version = info.schema_version
    report.config_hash = service.config_hash
    report.retrieval_mode = service.capabilities.retrieval_mode

    # ---- 2. 向量模型 ----
    embedder = build_embedder(resolved.embedding)
    started = time.perf_counter()
    try:
        vectors = embedder.embed([PROBE_TEXT])
        elapsed = (time.perf_counter() - started) * 1000
        report.embedder_ok = True
        report.embedder_detail = f"dim={len(vectors[0])} latency={elapsed:.0f}ms" + (
            "" if embedder.semantic else "（本地词法桩，无语义能力）"
        )
        if not embedder.semantic:
            report.notes.append("向量通道未启用语义召回：检索模式为 lexical，同义改写类问题会失败")
    except RagError as exc:
        report.embedder_ok = False
        report.embedder_detail = f"{exc.code}: {exc.message}"
        report.notes.append(f"向量模型不可用，已退化为词法检索：{exc.detail}")
    except Exception as exc:  # pragma: no cover
        report.embedder_ok = False
        report.embedder_detail = f"{type(exc).__name__}: {exc}"

    # ---- 3. 重排模型 ----
    reranker = build_reranker(resolved.rerank)
    started = time.perf_counter()
    try:
        probes = [
            _Probe("小节 A", "员工年假天数按司龄计算"),
            _Probe("小节 B", "差旅报销时限为 15 个工作日"),
        ]
        ranked = reranker.rerank(PROBE_TEXT, ["年假", "连通性"], probes)
        elapsed = (time.perf_counter() - started) * 1000
        report.reranker_ok = bool(ranked)
        report.reranker_detail = f"top1={ranked[0][0].section_path} latency={elapsed:.0f}ms"
        if getattr(reranker, "degraded", False):
            report.notes.append("远程重排不可用，已回退到启发式重排（本次进程内标记降级）")
        if not reranker.enabled:
            report.notes.append("重排被显式关闭（RAG_RERANK=off）：confidence 一律为 unknown")
    except RagError as exc:
        report.reranker_ok = False
        report.reranker_detail = f"{exc.code}: {exc.message}"
        report.notes.append("重排模型不可用，已回退启发式重排；设 RAG_RERANK=off 可显式关闭")
    except Exception as exc:  # pragma: no cover
        report.reranker_ok = False
        report.reranker_detail = f"{type(exc).__name__}: {exc}"

    return report


class _Probe:
    """重排探针需要的最小对象形状。"""

    def __init__(self, section_path: str, text: str) -> None:
        self.section_path = section_path
        self.text = text
        self.heading = section_path
        self.chunk_id = section_path


def main() -> int:
    report = run()

    print("=" * 66)
    print("retrieval-mcp 自检")
    print("=" * 66)
    print(
        f"语料         : {'OK' if report.data_ok else 'FAIL'}  "
        f"({report.docs} 篇 / {report.chunks} 块, corpus_version={report.corpus_version})"
    )
    print(f"词表         : schema_version={report.schema_version}")
    print(
        f"分词器       : {report.segmenter}"
        + ("" if report.segmenter == "word" else "（未装 jieba，退化为 bigram，中文噪声偏大）")
    )
    print(f"检索模式     : {report.retrieval_mode}")
    print(f"配置指纹     : {report.config_hash}")
    print(f"向量缓存     : {_cache_state()}")
    embedder_tail = report.embedder_detail or ""
    reranker_tail = report.reranker_detail or ""
    print(f"向量模型     : {_mark(report.embedder_ok)} {report.embedder}  {embedder_tail}")
    print(f"重排模型     : {_mark(report.reranker_ok)} {report.reranker}  {reranker_tail}")
    if report.notes:
        print("-" * 66)
        for note in report.notes:
            print(f"· {note}")
    print("=" * 66)

    if not report.data_ok:
        return 2
    return 0 if (report.embedder_ok is not False) else 1


def _cache_state() -> str:
    """启动速度取决于它：命中缓存时冷启动不发网络请求。"""
    from .cache import cache_dir, enabled

    if not enabled():
        return "disabled（RAG_VECTOR_CACHE=off，每次启动都会重新 embedding）"
    directory = cache_dir()
    count = len(list(directory.glob("vectors-*.json"))) if directory.is_dir() else 0
    return f"{directory}（{count} 份）"


def _mark(ok: bool | None) -> str:
    if ok is None:
        return "-"
    return "OK  " if ok else "FAIL"


if __name__ == "__main__":
    raise SystemExit(main())

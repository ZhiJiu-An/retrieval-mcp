"""重排：把「召回候选」变成「值得放进上下文的那 8 条」。

设计依据：DESIGN.md §6.2（单点收益最大的一步）、A12、§16.2（本地 vs 托管 = 未决）。

两档实现：

- `HeuristicReranker`（默认）：不看模型，只用**查询词在小节标题与正文里的覆盖度**、
  **短语是否原样出现**、**关键词邻近度**。它比 BM25 强的地方在于：BM25 只看词频，
  它看「这几个词是不是同时命中在正确的位置上」。确定性、零依赖、毫秒级。
- `OpenAICompatReranker`：配了端点就用它，覆盖式替换上面的启发式。

两者都输出 **[0, 1] 的分数**，这个分数同时用于给 Agent 的 `confidence` 分档。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from .config import RerankSettings
from .errors import BACKEND_UNAVAILABLE, RagError
from .http import post_json

logger = logging.getLogger("rag_mcp.rerank")

#: 判定「两个词挨得够近」的字符距离。
PROXIMITY_WINDOW = 40

#: 启发式重排的分档阈值 —— 只用在小语料/无网络场景，量纲是覆盖率。
HEURISTIC_BANDS = (0.62, 0.35)
#: cross-encoder（gte-rerank-v2）在真实语料上实测标定：
#: 无关问题 top1 ≤ 0.121，真实问题 0.197–0.861 → 分档落在 0.50 / 0.25。
CROSS_ENCODER_BANDS = (0.50, 0.25)

Ranked = list[tuple[Any, float]]


class Reranker(Protocol):
    model_id: str
    enabled: bool
    #: 是否是**真正的 cross-encoder**。
    #: True → 它自己就能判断语义，相关性直接用它的分数；
    #: False（启发式）→ 它只会数词，必须靠 service 做「词法+余弦合议」补正。
    cross_encoder: bool
    #: `(high, medium)` 两档阈值。**必须由重排器自己给出**：
    #: 启发式的覆盖率分与 cross-encoder 的相关性分是两种量纲，
    #: 用同一组阈值等于必然误判一类（DESIGN §18.2.3）。
    confidence_bands: tuple[float, float]

    def rerank(self, query: str, terms: Sequence[str], candidates: Sequence[Any]) -> Ranked: ...


# ---------------------------------------------------------------- 启发式


class HeuristicReranker:
    """确定性重排。没有模型依赖，因此可以出现在启动路径上。"""

    enabled = True
    cross_encoder = False
    confidence_bands = HEURISTIC_BANDS
    model_id = "heuristic"

    def __init__(self, *, bands: tuple[float, float] = HEURISTIC_BANDS) -> None:
        # 以前 build_reranker 不传 bands，导致 calibrate 写出的
        # RAG_CONFIDENCE_* 在默认启发式配置下完全不生效。
        self.confidence_bands = bands

    def rerank(self, query: str, terms: Sequence[str], candidates: Sequence[Any]) -> Ranked:
        if not candidates:
            return []
        # 覆盖率只在「本候选集里真的存在」的词上算。
        # 查询里天然会有语料中不存在的词（口语词、泛词、分词噪声），
        # 把它们放进分母会无差别压低所有候选，最后一起掉到相关性下限以下 ——
        # 表现为「该召回的也没了」。
        known = [term for term in terms if any(term in _haystack(c) for c in candidates)]
        term_list = known or list(terms)
        phrase = " ".join(term_list).strip()
        scored = [(chunk, self._score(chunk, term_list, phrase, query)) for chunk in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def _score(self, chunk: Any, terms: list[str], phrase: str, query: str) -> float:
        if not terms:
            return 0.0

        section = (chunk.section_path or "").lower()
        body = (chunk.text or "").lower()
        heading = (chunk.heading or "").lower()

        section_hits = sum(1 for term in terms if term in section)
        body_hits = sum(1 for term in terms if term in body)

        # 小节标题命中比正文顺带提到值钱得多：模型是靠 section_path 判断「这节对不对」的。
        section_cov = section_hits / len(terms)
        body_cov = body_hits / len(terms)
        coverage = 0.6 * section_cov + 0.4 * body_cov

        bonus = 0.0
        normalized_query = query.strip().lower()
        if normalized_query and (normalized_query in body or normalized_query in section):
            bonus += 0.18  # 整句原样出现
        if phrase and phrase != normalized_query and phrase in body:
            bonus += 0.06
        if heading and any(term in heading for term in terms):
            bonus += 0.08

        # 关键词邻近度：两个词挨在一起共现，通常就是答案所在的那一句。
        if len(terms) >= 2:
            bonus += 0.08 * _proximity(section + "\n" + body, terms[0], terms[1])

        return round(min(1.0, coverage + bonus), 4)


def _haystack(chunk: Any) -> str:
    """候选的可搜索文本：小节路径 + 正文。"""
    return f"{chunk.section_path or ''}\n{chunk.text or ''}".lower()


def _proximity(haystack: str, first: str, second: str) -> float:
    """两个词最近共现的邻近度，`0..1`。相距超出窗口则 0。"""
    pos_first = haystack.find(first)
    pos_second = haystack.find(second)
    if pos_first < 0 or pos_second < 0:
        return 0.0
    distance = abs(pos_first - pos_second)
    if distance >= PROXIMITY_WINDOW:
        return 0.0
    return 1.0 - distance / PROXIMITY_WINDOW


# ---------------------------------------------------------------- 远程


class OpenAICompatReranker:
    """`/rerank` 端点（Cohere / Jina / SiliconFlow / bge-reranker 服务化方案通用）。"""

    enabled = True
    cross_encoder = True

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        bands: tuple[float, float] = CROSS_ENCODER_BANDS,
    ) -> None:
        self.confidence_bands = bands
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_id = f"remote:{model}"
        self._api_key = api_key
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def rerank(self, query: str, terms: Sequence[str], candidates: Sequence[Any]) -> Ranked:
        if not candidates:
            return []

        documents = [chunk.text or "" for chunk in candidates]
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = post_json(
                self._client,
                f"{self.base_url}/rerank",
                headers=headers,
                payload={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": len(documents),
                },
            )
            rows = response.json()["results"]
        except Exception as exc:
            raise RagError(
                code=BACKEND_UNAVAILABLE,
                message=f"重排模型不可用：{type(exc).__name__}",
                hint="检查 RAG_RERANK_* 配置，或设 RAG_RERANK_PROVIDER=heuristic 强制本地重排",
                retryable=True,
                detail={"reranker": self.model_id, "error": str(exc)[:200]},
            ) from exc

        ranked: Ranked = []
        for row in rows:
            index = int(row["index"])
            if not 0 <= index < len(candidates):
                continue
            # 不同厂商可能返回 logits，统一压到 [0, 1]：`confidence` 要能跨配置比较。
            score = _squash(float(row.get("relevance_score", 0.0)))
            ranked.append((candidates[index], score))
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked


class DashScopeReranker:
    """阿里百炼的**原生** rerank 端点（不是 OpenAI 兼容路径）。

    实测结论（DESIGN §18.2.4）：`/compatible-mode/v1/rerank` 与 `/reranks`
    都是 404，只有下面这个原生路径可用；请求体形状也与 Cohere/Jina 不同。

    好消息是它给出的分数**分离度远好于余弦相似度**：
    正确答案 0.65，错的两个 0.13 / 0.006 —— 这让 confidence 的两档阈值
    终于有了可靠的依据。
    """

    enabled = True
    cross_encoder = True
    PATH = "/api/v1/services/rerank/text-rerank/text-rerank"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        batch_size: int = 50,
        bands: tuple[float, float] = CROSS_ENCODER_BANDS,
    ) -> None:
        self.confidence_bands = bands
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_id = f"dashscope:{model}"
        self._api_key = api_key
        self._timeout = timeout
        self._batch = batch_size
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def rerank(self, query: str, terms: Sequence[str], candidates: Sequence[Any]) -> Ranked:
        if not candidates:
            return []

        ranked: Ranked = []
        # 分批：单次请求的候选数有上限，且太大也会拉长延迟
        for start in range(0, len(candidates), self._batch):
            window = candidates[start : start + self._batch]
            ranked.extend(self._one(query, window, offset=start))

        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked

    def _one(self, query: str, candidates: Sequence[Any], *, offset: int) -> Ranked:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = post_json(
                self._client,
                f"{self.base_url}{self.PATH}",
                headers=headers,
                payload={
                    "model": self.model,
                    "input": {
                        "query": query,
                        "documents": [chunk.text or "" for chunk in candidates],
                    },
                    "parameters": {"return_documents": False},
                },
            )
            rows = response.json()["output"]["results"]
        except Exception as exc:
            raise RagError(
                code=BACKEND_UNAVAILABLE,
                message=f"重排模型不可用：{type(exc).__name__}",
                hint="检查 RAG_RERANK_*，或设 RAG_RERANK_PROVIDER=heuristic 强制本地重排",
                retryable=True,
                detail={"reranker": self.model_id, "error": str(exc)[:200]},
            ) from exc

        picked: Ranked = []
        for row in rows:
            index = int(row["index"])
            if 0 <= index < len(candidates):
                picked.append((candidates[index], _squash(float(row["relevance_score"]))))
        return picked


def _squash(value: float) -> float:
    if value < 0.0 or value > 1.0:
        return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, value))))
    return value


# ---------------------------------------------------------------- 工厂


class DisabledReranker:
    """显式关闭。此时 `confidence` 一律为 `unknown` —— 没有重排就没有判断依据，
    编一个分数出来只是自欺（DESIGN A19）。"""

    enabled = False
    cross_encoder = False
    confidence_bands = HEURISTIC_BANDS
    model_id = "disabled"

    def rerank(self, query: str, terms: Sequence[str], candidates: Sequence[Any]) -> Ranked:
        return [(chunk, 0.0) for chunk in candidates]


class FallbackReranker:
    """远程重排失败时回退到启发式，而不是让整个 search 失败。

    注意：回退后的分数仍是启发式覆盖率，和 cross-encoder 不是同一量纲；
    因此这里保留 primary 的 `cross_encoder` / bands 语义，并在日志/`degraded`
    中明确标注本次实例已降级。生产环境应把重排失败告警接到监控。
    """

    def __init__(self, primary: Reranker, fallback: HeuristicReranker) -> None:
        self._primary = primary
        self._fallback = fallback
        self.model_id = primary.model_id
        self.enabled = primary.enabled
        self.cross_encoder = primary.cross_encoder
        self.confidence_bands = primary.confidence_bands
        self.degraded = False

    def close(self) -> None:
        for component in (self._primary, self._fallback):
            close = getattr(component, "close", None)
            if callable(close):
                close()

    def rerank(self, query: str, terms: Sequence[str], candidates: Sequence[Any]) -> Ranked:
        if not self.degraded:
            try:
                return self._primary.rerank(query, terms, candidates)
            except RagError as exc:
                self.degraded = True
                logger.warning(
                    "reranker unavailable, fallback to heuristic: %s: %s",
                    exc.code,
                    exc.message,
                )
        return self._fallback.rerank(query, terms, candidates)


def build_reranker(settings: RerankSettings) -> Reranker:
    if not settings.enabled:
        return DisabledReranker()

    heuristic = HeuristicReranker(bands=settings.bands)
    if settings.provider == "heuristic":
        return heuristic
    if settings.remote:
        if settings.provider == "dashscope":
            primary: Reranker = DashScopeReranker(
                base_url=settings.base_url or "",
                model=settings.model,
                api_key=settings.api_key,
                timeout=settings.timeout,
                bands=settings.bands,
            )
        else:
            primary = OpenAICompatReranker(
                base_url=settings.base_url or "",
                model=settings.model,
                api_key=settings.api_key,
                timeout=settings.timeout,
                bands=settings.bands,
            )
        return FallbackReranker(primary, heuristic)
    return heuristic


def confidence_of(
    score: float, *, enabled: bool, bands: tuple[float, float] = HEURISTIC_BANDS
) -> str:
    """把重排分数映射成离散档位。分数不对外暴露，档位对外暴露（DESIGN A19）。"""
    if not enabled:
        return "unknown"
    high, medium = bands
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"

"""向量化：可切换的两档实现。

设计依据：DESIGN.md §6.1（双路召回）、§16.1（embedding 放哪算 = 未决问题）。

为什么有一个「本地桩」：

- **它让混合召回的管线是真的** —— 双路召回、融合、重排、预算裁剪全部跑通并可测。
- **它不提供语义能力** —— 哈希桶里没有语义，同义词（年假/带薪休假）不会靠近。
  这个事实通过 `describe().capabilities.retrieval_mode` 如实回报，不靠猜。

配上 `RAG_EMBEDDING_BASE_URL` + `RAG_EMBEDDING_MODEL` + `RAG_EMBEDDING_API_KEY`
之后自动切到远程模型，管线其余部分一行不用改。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import httpx

from .config import EmbeddingSettings
from .errors import BACKEND_UNAVAILABLE, RagError
from .http import post_json
from .text import tokenize

Vector = list[float]


class Embedder(Protocol):
    model_id: str
    dim: int
    remote: bool
    #: 是否真的具备语义能力。词法桩为 False —— 它的「向量通道」只是同义词桶，
    #: 拿它做密集召回只会产生哈希碰撞噪声（实测：无关查询也能命中）。
    semantic: bool

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


# ---------------------------------------------------------------- 本地桩


class HashingEmbedder:
    """把检索词哈希到固定维度的桶里，再做 L2 归一化。

    数学上等价于对「词袋」做一个固定随机投影：**词法**相似度是真实的，
    **语义**相似度为零。别把它当向量模型用（DESIGN §6、MVP.md §9.2）。
    """

    remote = False
    semantic = False

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正数")
        self.dim = dim
        self.model_id = f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> Vector:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token, count in Counter(tokens).items():
            # 用 sha1 而不是内置 hash()：后者跨进程加盐，会让索引和查询对不上。
            bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[bucket] += 1.0 + math.log(count)  # 次线性词频
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


# ---------------------------------------------------------------- 远程


class OpenAICompatEmbedder:
    """OpenAI 兼容的 `/embeddings` 端点。

    兼容 DashScope / SiliconFlow / vLLM / Ollama / 官方 OpenAI —— 只要路径与响应形状一致。
    """

    remote = True
    semantic = True

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        batch_size: int = 32,
        dim: int = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_id = f"remote:{model}"
        self.dim = dim  # 首次调用后由返回结果修正
        self._api_key = api_key
        self._timeout = timeout
        self._batch = batch_size
        self._cache: dict[str, Vector] = {}
        # 连接池复用：ingest 会发几十次 batch 请求，不能每次重新握手。
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        unique = [text for text in dict.fromkeys(texts) if text not in self._cache]
        resolved: dict[str, Vector] = {}
        for start in range(0, len(unique), self._batch):
            window = unique[start : start + self._batch]
            for text, vector in zip(window, self._request(window), strict=True):
                resolved[text] = vector
                # 软上限只影响缓存，不影响本次返回。以前缓存满 4096 后
                # 新文本不写 cache，最后 self._cache[text] 直接 KeyError。
                if len(self._cache) < 4096:
                    self._cache[text] = vector
        return [self._cache[text] if text in self._cache else resolved[text] for text in texts]

    def close(self) -> None:
        self._client.close()

    def _request(self, texts: list[str]) -> list[Vector]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = post_json(
                self._client,
                f"{self.base_url}/embeddings",
                headers=headers,
                payload={"model": self.model, "input": texts},
            )
            payload = response.json()
            rows = sorted(payload["data"], key=lambda row: row.get("index", 0))
            vectors = [_normalize([float(v) for v in row["embedding"]]) for row in rows]
        except Exception as exc:  # 网络、鉴权、响应形状统统算不可用
            raise RagError(
                code=BACKEND_UNAVAILABLE,
                message=f"向量模型不可用：{type(exc).__name__}",
                hint=(
                    "检查 RAG_EMBEDDING_BASE_URL / MODEL / API_KEY，或先跑 python -m rag_mcp.doctor"
                ),
                retryable=True,
                detail={"embedder": self.model_id, "error": str(exc)[:200]},
            ) from exc

        if not vectors or len(vectors) != len(texts):
            raise RagError(
                code=BACKEND_UNAVAILABLE,
                message="向量模型返回条数与请求不一致",
                retryable=True,
                detail={"expected": len(texts), "got": len(vectors)},
            )
        self.dim = len(vectors[0])
        return vectors


# ---------------------------------------------------------------- 工厂


def build_embedder(settings: EmbeddingSettings) -> Embedder:
    if settings.remote:
        return OpenAICompatEmbedder(
            base_url=settings.base_url or "",
            model=settings.model,
            api_key=settings.api_key,
            timeout=settings.timeout,
            batch_size=settings.batch,
            dim=settings.dim,
        )
    return HashingEmbedder(dim=settings.dim)


def _normalize(vector: Vector) -> Vector:
    """L2 归一化。余弦计算依赖它；第三方 OpenAI 兼容端点未必归一化。"""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]

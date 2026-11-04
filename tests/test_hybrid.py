"""双路召回、融合与重排的单元测试。

这里用**假的语义向量**（把主题映射到固定方向），所以既不需要模型服务，
又能真正验证「混合召回管线」。词法桩做不到这件事 —— 它只会制造哈希碰撞噪声，
所以生产路径上它被显式禁用（见 embedding.py 的 `semantic` 标记）。
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_mcp.backends.memory import MemoryBackend
from rag_mcp.config import Settings
from rag_mcp.factory import build_service
from rag_mcp.fusion import rrf
from rag_mcp.policy import Principal
from rag_mcp.registry import default_registry
from rag_mcp.rerank import HeuristicReranker, confidence_of
from rag_mcp.service import RetrievalService

#: 主题 → 表面词。查询与语料都按这个表映射到同一个方向。
TOPICS: dict[str, tuple[str, ...]] = {
    "leave": ("年假", "休假", "调休", "假期", "带薪"),
    "travel": ("报销", "差旅", "住宿", "交通", "酒店", "出差"),
    "network": ("vpn", "网络", "内网", "登录"),
}


class FakeSemanticEmbedder:
    """把「主题命中」变成一个 one-hot 向量。语义是假的，**管线是真的**。"""

    remote = True
    semantic = True
    dim = len(TOPICS)
    model_id = "fake-semantic"

    def embed(self, texts):
        return [self._one(text) for text in texts]

    def _one(self, text: str):
        lowered = text.lower()
        vector = [0.0] * self.dim
        for index, words in enumerate(TOPICS.values()):
            if any(word in lowered for word in words):
                vector[index] = 1.0
        return vector


@dataclass
class _Item:
    chunk_id: str


def test_hybrid_recall_finds_paraphrase_that_lexical_misses(settings: Settings):
    """同义改写：语料里写「年假」，用户问「带薪休假」—— 词法一定失败，语义能命中。"""
    service = _hybrid_service(settings)
    staff = Principal(user_id="alice", groups=frozenset({"all"}), max_level="internal")

    result = service.search(staff, query="带薪休假 怎么弄", top_k=5)

    assert result.hits
    assert result.hits[0].doc_id == "hr-policy-leave"
    assert service.capabilities.retrieval_mode == "hybrid"


def test_lexical_mode_does_not_claim_semantic_ability(settings: Settings):
    service = build_service(settings)
    assert service.capabilities.retrieval_mode == "lexical"


def test_rrf_rewards_agreement_between_channels():
    """两路都排前面的，融合后必须最前。这就是 RRF 存在的全部理由。"""
    a, b, c, d = _Item("a"), _Item("b"), _Item("c"), _Item("d")

    fused = rrf([[a, b, c], [b, d]])

    assert fused[0][0] is b
    assert {item.chunk_id for item, _ in fused} == {"a", "b", "c", "d"}


def test_rrf_score_is_rank_based_not_relevance():
    """融合分只用排名：同一排名在任何量纲下得到同样的分。"""
    a = _Item("a")
    assert rrf([[a]])[0][1] == rrf([[a], [a]])[0][1] / 2


def test_heuristic_reranker_prefers_section_over_body_mention():
    """小节标题命中比正文顺带提到值钱 —— 模型是看 section_path 判断「这节对不对」的。"""
    in_section = _FakeChunk("员工年假与调休管理办法 › 年假天数", "员工累计工作已满 1 年的年假 5 天")
    in_body = _FakeChunk("差旅费用报销标准 › 适用范围", "出差期间的年假申请需另行审批")

    ranked = HeuristicReranker().rerank("年假 天数", ["年假", "天数"], [in_body, in_section])

    assert ranked[0][0] is in_section


def test_confidence_is_unknown_when_rerank_is_off():
    """没有重排就没有判断依据。宁可说不知道，也不编一个分数（DESIGN A19）。"""
    assert confidence_of(0.9, enabled=False) == "unknown"
    assert confidence_of(0.9, enabled=True) == "high"
    assert confidence_of(0.4, enabled=True) == "medium"
    assert confidence_of(0.1, enabled=True) == "low"


def test_service_can_be_built_with_an_injected_embedder(settings: Settings):
    """换后端/换模型只改组装，不改检索域 —— 这是 A3 的可验证形式。"""
    embedder = FakeSemanticEmbedder()
    backend = MemoryBackend(
        settings.data_path,
        embedder=embedder,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    service = RetrievalService(
        backend=backend,
        registry=default_registry(),
        settings=settings,
        embedder=embedder,
        reranker=HeuristicReranker(),
    )

    assert service.capabilities.embedder == "fake-semantic"
    assert service.search(
        Principal(user_id="alice", groups=frozenset({"all"}), max_level="internal"),
        query="酒店 报销",
        top_k=3,
    ).hits


def _hybrid_service(settings: Settings) -> RetrievalService:
    embedder = FakeSemanticEmbedder()
    backend = MemoryBackend(
        settings.data_path,
        embedder=embedder,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return RetrievalService(
        backend=backend,
        registry=default_registry(),
        settings=settings,
        embedder=embedder,
        reranker=HeuristicReranker(),
    )


@dataclass
class _FakeChunk:
    section_path: str
    text: str
    heading: str = ""
    chunk_id: str = "x"

    def __post_init__(self) -> None:
        if not self.heading:
            self.heading = self.section_path.split("›")[-1].strip()

"""配置：所有可调项的唯一入口。

设计依据：DESIGN.md §3.1（**不再有 transport 开关**）、§4.3、§16.1。

v0 的 `RAG_MCP_TRANSPORT` 已删除：一条没有被启用、没有被测试的分支就是一条
将来会腐烂的分支，而且会误导人以为「多用户 = 加一行配置」。
这个 server 只做 stdio（进程-per-principal）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .budget import Limits
from .env import env_bool, env_float, env_int
from .errors import BAD_CONFIGURATION, RagError

_PKG = Path(__file__).parent
_REPO_ROOT = _PKG.parent.parent
_DEFAULT_DATA = _PKG / "data" / "sample_docs.json"
_DEFAULT_REGISTRY = _PKG / "registry.json"


# ---------------------------------------------------------------- 模型端点


@dataclass(frozen=True)
class EmbeddingSettings:
    """向量模型（OpenAI 兼容 `/embeddings`）。

    没配 `base_url` / `model` 时退化为本地确定性桩 —— 见 `embedding.HashingEmbedder`。
    """

    base_url: str | None = None
    model: str = ""
    api_key: str | None = None
    dim: int = 256
    #: 单次请求超时。**5 秒**是刻意收紧的：实测同一端点会出现 0.3s 与 20s
    #: 的抖动，而 Agent 在工具调用上等 20 秒等于任务失败。
    #: 超时会抛 BACKEND_UNAVAILABLE（可重试），而不是静默返回更差的结果。
    timeout: float = 5.0
    #: 单次请求的最大文本条数。**默认 10 是保守值**：阿里百炼 text-embedding-v3
    #: 在 OpenAI 兼容模式下的批量上限就是 10，设大了一建索引就整批失败。
    batch: int = 10

    @property
    def remote(self) -> bool:
        return bool(self.base_url and self.model)

    def describe(self) -> str:
        return f"remote:{self.model}" if self.remote else f"local:hashing-{self.dim}"


@dataclass(frozen=True)
class RerankSettings:
    """重排模型（OpenAI / Cohere / Jina 兼容的 `/rerank`）。"""

    enabled: bool = True
    #: `openai`（Cohere/Jina 风格 /rerank）| `dashscope`（阿里百炼原生端点）
    provider: str = "openai"
    base_url: str | None = None
    model: str = ""
    api_key: str | None = None
    timeout: float = 10.0
    #: `confidence` 的分档阈值。必须与重排器量纲匹配（见 rerank.py）
    confidence_high: float = 0.50
    confidence_medium: float = 0.25

    @property
    def bands(self) -> tuple[float, float]:
        return (self.confidence_high, self.confidence_medium)

    @property
    def remote(self) -> bool:
        return bool(self.base_url and self.model)

    def describe(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.provider == "heuristic":
            return "heuristic"
        return f"{self.provider}:{self.model}" if self.remote else "heuristic"


# ---------------------------------------------------------------- 主配置


@dataclass(frozen=True)
class Settings:
    data_path: Path = _DEFAULT_DATA
    registry_path: Path = _DEFAULT_REGISTRY
    #: 后端选择：`memory`（内置样例语料）| `sqlite`（由 ingest 建出的 rag.db）
    backend: str = "memory"
    #: 检索模式：`auto`（有语义能力就 hybrid）| `hybrid` | `lexical` | `dense`
    #: 这是**安全可调**的开关：它不改变分数含义，只决定用哪几路召回。
    retrieval_mode: str = "auto"
    db_path: Path | None = None
    limits: Limits = field(default_factory=Limits)
    principal_id: str = "local-dev"
    principal_groups: tuple[str, ...] = ("all",)
    principal_max_level: str = "internal"
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    #: 小块入库的粒度（DESIGN §6.1：300–500 字，重叠 10–15%）
    chunk_size: int = 360
    chunk_overlap: int = 48

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> Settings:
        if load_dotenv_file:
            load_dotenv()

        groups = os.getenv("RAG_PRINCIPAL_GROUPS", "all")
        return cls(
            data_path=_resolve_path(os.getenv("RAG_DATA_PATH"), _DEFAULT_DATA),
            registry_path=_resolve_path(os.getenv("RAG_REGISTRY_PATH"), _DEFAULT_REGISTRY),
            backend=os.getenv("RAG_BACKEND", "memory").strip().lower() or "memory",
            retrieval_mode=os.getenv("RAG_RETRIEVAL_MODE", "auto").strip().lower() or "auto",
            db_path=_path_or_none(os.getenv("RAG_DB_PATH")),
            limits=Limits.from_env(),
            principal_id=os.getenv("RAG_PRINCIPAL_ID", "local-dev"),
            principal_groups=tuple(g.strip() for g in groups.split(",") if g.strip()),
            principal_max_level=os.getenv("RAG_PRINCIPAL_MAX_LEVEL", "internal"),
            embedding=EmbeddingSettings(
                base_url=_blank_to_none(os.getenv("RAG_EMBEDDING_BASE_URL")),
                model=os.getenv("RAG_EMBEDDING_MODEL", "").strip(),
                api_key=_first_env("RAG_EMBEDDING_API_KEY", "DASHSCOPE_API_KEY", "BAILIAN_API_KEY"),
                dim=env_int("RAG_EMBEDDING_DIM", 256),
                timeout=env_float("RAG_EMBEDDING_TIMEOUT", 5.0),
                batch=env_int("RAG_EMBEDDING_BATCH", 10),
            ),
            rerank=RerankSettings(
                enabled=env_bool("RAG_RERANK", True),
                provider=os.getenv("RAG_RERANK_PROVIDER", "openai").strip().lower(),
                base_url=_blank_to_none(os.getenv("RAG_RERANK_BASE_URL")),
                model=os.getenv("RAG_RERANK_MODEL", "").strip(),
                api_key=_first_env("RAG_RERANK_API_KEY", "DASHSCOPE_API_KEY", "BAILIAN_API_KEY"),
                timeout=env_float("RAG_RERANK_TIMEOUT", 10.0),
                confidence_high=env_float("RAG_CONFIDENCE_HIGH", 0.50),
                confidence_medium=env_float("RAG_CONFIDENCE_MEDIUM", 0.25),
            ),
            chunk_size=env_int("RAG_CHUNK_SIZE", 360),
            chunk_overlap=env_int("RAG_CHUNK_OVERLAP", 48),
        )

    def with_retrieval_mode(self, mode: str) -> Settings:
        return replace(self, retrieval_mode=mode)

    def with_backend(self, backend: str, db_path: Path | None) -> Settings:
        """评测时切换到另一个索引，不改其余配置。"""
        return replace(self, backend=backend, db_path=db_path)

    def with_principal(self, user_id: str, groups: tuple[str, ...], max_level: str) -> Settings:
        """测试与多身份验证用：换身份不改其它配置。"""
        return replace(
            self,
            principal_id=user_id,
            principal_groups=groups,
            principal_max_level=max_level,
        )


# ---------------------------------------------------------------- .env

#: 仓库根的 `.env`。它是 gitignore 的，用来放 key，不进版本库。
DOTENV_PATH = _REPO_ROOT / ".env"
_loaded_dotenv = False
#: profile 所在目录。相对路径（RAG_DB_PATH 等）优先按仓库根解析，
#: 再回退到 profile 目录 / cwd；见 `_resolve_path`。
_active_base_dir: Path = _REPO_ROOT
#: 启动时就已存在的真实环境变量 —— 任何文件都不许覆盖它们。
_protected: set[str] | None = None


def load_dotenv(path: Path | None = None) -> None:
    """分层加载配置：**根 `.env` 打底 → `RAG_ENV_FILE` 覆盖 → 真实环境变量最高**。

    为什么要分层：密钥放根 `.env`（一份），而「样例语料 / 真实语料」这类切换项
    放 profile 文件（`config/tcm.env`）。只加载 profile 会丢掉密钥（实测踩过 401）；
    反过来让 profile 覆盖真实环境变量又会让 `RAG_BACKEND=sqlite rag-mcp` 这种
    显式命令行参数失效。所以三层优先级写死在这里。

    `RAG_ENV_FILE` 的相对路径按仓库根解析，而不是进程 cwd —— MCP 客户端
    通常不会把 cwd 设成仓库根目录；显式指定的 profile 不存在时必须报错，
    不能静默退回样例语料。

    不引入 python-dotenv 依赖 —— 只需要支持 `KEY=VALUE` 与 `#` 注释。
    """
    global _loaded_dotenv, _active_base_dir
    if _loaded_dotenv:
        return

    root_file = path or DOTENV_PATH
    _apply_file(root_file, override=False)

    profile_raw = os.getenv("RAG_ENV_FILE")
    profile = _resolve_profile(profile_raw)
    if profile is not None and profile != root_file:
        _apply_file(profile, override=True)
        _active_base_dir = profile.parent
    else:
        _active_base_dir = DOTENV_PATH.parent
    # 只有成功加载后才置位；profile 缺失时下一次调用仍会报错，
    # 不会静默退回样例语料。
    _loaded_dotenv = True


def _apply_file(path: Path, *, override: bool) -> None:
    global _protected
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return

    if _protected is None:
        _protected = set(os.environ)

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key or key in _protected:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------- 小工具


def reload_dotenv(path: Path | None = None) -> None:
    """强制重新读取 `.env`。

    给评测用：单元测试会临时把 embedding 端点清空（保持离线），
    评测则必须拿真实配置跑。两者在同一个 pytest 进程里，需要显式切换。
    """
    global _loaded_dotenv
    _loaded_dotenv = False
    load_dotenv(path)


def _first_env(*names: str) -> str | None:
    """按顺序取第一个非空的变量。

    百炼/通义用户习惯把 key 写成 `BAILIAN_API_KEY` 或 `DASHSCOPE_API_KEY`，
    `RAG_EMBEDDING_API_KEY` 只是本项目自己的名字。认这两个别名，
    省掉「明明设了 key 却说没配置」这种排查。
    """
    for name in names:
        value = _blank_to_none(os.getenv(name))
        if value:
            return value
    return None


def _resolve_profile(value: str | None) -> Path | None:
    """`RAG_ENV_FILE` 为空返回 None；相对路径按仓库根解析。"""
    if value is None or not value.strip():
        return None
    raw = value.strip()
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [DOTENV_PATH.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = "、".join(str(candidate) for candidate in candidates)
    raise RagError(
        code=BAD_CONFIGURATION,
        message=f"RAG_ENV_FILE 指向的 profile 不存在：{raw}（已尝试 {tried}）",
        hint="检查 RAG_ENV_FILE 路径；相对路径优先按项目根目录解析",
        retryable=False,
    )


def _path_or_none(value: str | None) -> Path | None:
    """空字符串视为未配置；相对路径按项目根优先解析，避免依赖进程 cwd。"""
    if value is None or not value.strip():
        return None
    return _resolve_path(value, None)


def _resolve_path(value: str | None, default: Path | None) -> Path:
    if value is None or not value.strip():
        if default is None:
            raise ValueError("path 不能为空")
        return default
    path = Path(value.strip())
    if path.is_absolute():
        return path
    # 仓库根优先，其次 profile 目录，最后 cwd（兼容旧行为）
    candidates = [DOTENV_PATH.parent / path, _active_base_dir / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (DOTENV_PATH.parent / path).resolve()


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None

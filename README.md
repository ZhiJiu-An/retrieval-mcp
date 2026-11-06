# retrieval-mcp

> 让 AI（Claude Code / Cursor / Cherry Studio 等 MCP 客户端）能够查询你自己的文档库：
> 只返回当前身份可见的片段，并告诉 AI 这条结果可不可信。

`retrieval-mcp` 是一个只读检索 MCP server，也可以作为接入自己语料的模板。
它不生成答案，只负责把相关文档片段交给 AI。

## 目录

- [1. 项目介绍](#1-项目介绍)
- [2. 快速开始](#2-快速开始)
- [3. 配置](#3-配置)
- [4. 接入一个新场景](#4-接入一个新场景)
- [5. 测试与评测](#5-测试与评测)
- [6. 如何贡献](#6-如何贡献)
- [7. 开源协议](#7-开源协议)

## 1. 项目介绍

它提供三个工具和一个资源：

| 动作 | 输入 | 输出 | 用途 |
|---|---|---|---|
| `list_sources` | 无 | 文档/块数量、可过滤字段、检索能力 | 确认库里有什么 |
| `search` | 查询 + 过滤条件 | 片段列表（`chunk_id` + `confidence`） | 找依据 |
| `fetch` | `chunk_id` 或 `doc_id` | 片段上下文 / 整篇文档 | 展开依据 |
| `rag://docs/{doc_id}` | — | 整篇 Markdown | 直接读取整篇 |

适合：

- 想让 AI 查内部制度、手册、FAQ、知识库。
- 需要按部门、密级、文档类型做权限隔离。
- 希望 AI 只拿片段，避免一次塞入整篇文档。

不适合：

- 需要网页后台、上传管理界面或多租户 HTTP 服务。
- 需要由检索服务直接生成答案或写报告。
- 需要修改、删除文档（入库是独立进程）。

## 2. 快速开始

### 环境

- Python 3.11+
- 可选：`jieba`，用于中文词级分词；不装会退化为 bigram

### 安装

```bash
git clone <repo>
cd retrieval-mcp

python -m venv .venv
source .venv/bin/activate
pip install -e ".[wordseg]"
```

### 用样例语料自检

仓库自带 6 篇样例制度，不需要额外准备语料：

```bash
python -m rag_mcp.doctor
```

看到索引、向量模型、重排模型三段输出即可。样例默认使用本地词法桩，不需要 key。

### 接入 MCP 客户端

把 `command` 指向安装后的可执行文件，路径使用绝对路径：

```json
{
  "mcpServers": {
    "retrieval-mcp": {
      "command": "/absolute/path/to/retrieval-mcp/.venv/bin/rag-mcp"
    }
  }
}
```

使用真实语料时，在客户端配置里加 `env`：

```json
{
  "mcpServers": {
    "retrieval-mcp": {
      "command": "/absolute/path/to/retrieval-mcp/.venv/bin/rag-mcp",
      "env": {
        "RAG_ENV_FILE": "config/tcm.env"
      }
    }
  }
}
```

### 可选：接入真实语料

`rag.db` 不包含在仓库中，需要先用 `ingest` 生成：

```bash
python -m rag_mcp.ingest \
  --source json \
  --data 你的语料.json \
  --out rag.db

RAG_ENV_FILE=config/tcm.env .venv/bin/rag-mcp
```

## 3. 配置

配置文件分两层：

- 根目录 `.env`：密钥和默认配置，不进入版本库。
- `RAG_ENV_FILE`：按语料/场景切换的 profile，例如 `config/tcm.env`。

加载优先级：真实环境变量 > `RAG_ENV_FILE` > 根 `.env`。

### 3.1 语料与后端

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_BACKEND` | `memory` | `memory`（JSON 样例）或 `sqlite`（rag.db） |
| `RAG_DB_PATH` | 空 | sqlite 索引路径 |
| `RAG_DATA_PATH` | 内置样例 | memory 后端的 JSON 语料 |
| `RAG_REGISTRY_PATH` | 内置词表 | 可过滤字段的词表快照 |

### 3.2 身份与权限

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_PRINCIPAL_ID` | `local-dev` | 调用方身份 |
| `RAG_PRINCIPAL_GROUPS` | `all` | 可见组，逗号分隔 |
| `RAG_PRINCIPAL_MAX_LEVEL` | `internal` | `public` / `internal` / `confidential` |

身份来自进程启动环境，不接受工具参数传入。

### 3.3 模型

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_EMBEDDING_BASE_URL` | 空 | OpenAI 兼容 embedding 端点 |
| `RAG_EMBEDDING_MODEL` | 空 | embedding 模型名 |
| `RAG_EMBEDDING_API_KEY` | 空 | API key，也认 `DASHSCOPE_API_KEY` / `BAILIAN_API_KEY` |
| `RAG_RERANK_PROVIDER` | `openai` | `openai` / `dashscope` / `heuristic` |
| `RAG_RERANK_BASE_URL` | 空 | rerank 端点 |
| `RAG_RERANK_MODEL` | 空 | rerank 模型名 |
| `RAG_RERANK` | `on` | `off` 表示关闭重排，此时 `confidence` 为 `unknown` |

不填 embedding 三项时使用本地词法桩，仍可运行，但没有语义召回。
不填 rerank 端点时使用本地启发式重排。

### 3.4 检索预算

| 变量 | 默认 | 说明 |
|---|---:|---|
| `RAG_MAX_TOP_K` | 20 | 返回条数上限 |
| `RAG_DEFAULT_TOP_K` | 8 | 默认返回条数 |
| `RAG_SNIPPET_CHARS` | 300 | 片段字数 |
| `RAG_MAX_FETCH_CHARS` | 4000 | fetch 文本上限 |
| `RAG_MAX_QUERY_CHARS` | 2000 | 查询原文上限 |

更多配置见 [.env.example](.env.example)。

### 3.5 阈值

`RAG_RELEVANCE_FLOOR`、`RAG_CONFIDENCE_HIGH`、`RAG_CONFIDENCE_MEDIUM`、`RAG_DENSE_FLOOR`
依赖具体语料和模型，不建议手调。换语料后运行：

```bash
python -m rag_mcp.calibrate \
  --db rag.db \
  --golden eval/golden_新场景.jsonl \
  --out config/新场景.env
```

## 4. 接入一个新场景

1. **入库语料**

   ```bash
   python -m rag_mcp.ingest --source json --data 你的语料.json --out rag.db
   ```

2. **更新可过滤词表**

   修改 `RAG_REGISTRY_PATH` 指向的 `registry.json`，填入实际的部门、文档类型和日期字段。

3. **写 golden 用例**

   准备 20–40 条真实问题，放到 `eval/golden_新场景.jsonl`。

4. **标定阈值**

   ```bash
   python -m rag_mcp.calibrate \
     --db rag.db \
     --golden eval/golden_新场景.jsonl \
     --out config/新场景.env
   ```

5. **跑评测**

   ```bash
   RAG_ENV_FILE=config/新场景.env pytest eval -q
   ```

## 5. 测试与评测

```bash
# 离线单元测试
pytest tests -q

# 离线样例评测
pytest eval -q

# 代码检查
ruff check src tests eval
ruff format --check src tests eval
```

评测用例在 `eval/`，指标和门禁由评测脚本负责。

## 6. 如何贡献

- 有明确场景或改动建议，优先先开 issue 描述背景。
- 提交前运行：

  ```bash
  pytest tests -q
  ruff check src tests eval
  ruff format --check src tests eval
  ```

- 不要提交 `.env`、真实语料、`rag.db`、缓存目录或私有记录。
- PR 尽量保持单一主题，说明改动目的、影响范围和验证方式。

## 7. 开源协议

本项目采用 [Apache License 2.0](LICENSE) 开源。

# retrieval-mcp

> **一句话**：让 AI（Claude Code / 各类 IDE 里的模型）能**问你自己的文档库**，
> 并且只返回它有权限看的、只返回片段、还告诉你这条结果可不可信。

---

## 30 秒理解它是什么

```
你：感冒的病因病机是什么？
AI：（调用 search → 拿到 3 条片段 → 判断够不够 → 需要时调用 fetch 展开）
AI：根据《中医内科学》第四章……
```

**AI 自己不会查书。** 这个项目就是给它装的手：三个动作 —— 找资料、看上下文、问「你这有什么」。

它**不生成答案**（那是 AI 的事），**不管入库之外的运维**（那是数据库的事），**不提供界面**。

---

## 30 秒跑起来

```bash
# 1) 用内置样例语料（6 篇假制度，不需要任何 key —— 用来验证接线是否通）
.venv/bin/rag-mcp

# 2) 用真实语料（《中医内科学》884 块，需要百炼 key）
RAG_ENV_FILE=config/tcm.env .venv/bin/rag-mcp

# 出问题先跑这个，它会告诉你哪一环坏了
RAG_ENV_FILE=config/tcm.env .venv/bin/python -m rag_mcp.doctor
```

挂到客户端（Claude Code / Cursor / Cherry Studio 等）时，把 `command` 指向 `.venv/bin/rag-mcp` 即可。

---

## 它到底给你什么

对 AI 只暴露三个工具 + 一个资源：

| 动作 | 输入 | 输出 | 什么时候用 |
|---|---|---|---|
| `list_sources` | 无 | 有多少篇/块、能按什么过滤、当前检索能力 | 不确定库里有什么 |
| `search` | 问题 + 过滤条件 | 8 条（可调）片段，每条带 `chunk_id` 和可信度 `high/medium/low/unknown` | 找依据 |
| `fetch` | `chunk_id` 或 `doc_id` | 那一段的前后文 / 整篇 | 片段不够时展开 |
| `rag://docs/{doc_id}` | — | 整篇 markdown | 直连读取，不占回合 |

**三个设计上的取舍**（这是它和「直接接个向量库」的区别）：

1. **只给片段，不给全文。** 300 字的片段 + id，够 AI 判断「值不值得展开」，省上下文。
2. **可信度是服务端算的，不是 AI 猜的。** `high` 才建议直接答；`low` 建议先 `fetch`。
3. **权限在检索前生效。** 看不到的文档根本不会出现在结果里，也不会通过「结果条数」泄漏存在性。

---

## 它明确不做什么

| 不做 | 谁做 |
|---|---|
| 生成答案 | AI 自己 |
| 网页界面 / 上传后台 | 不在本项目范围 |
| 多用户 HTTP 服务 | 只做 stdio，一个进程一个身份 |
| 修改/删除文档 | 入库是独立进程（`python -m rag_mcp.ingest`） |
| 会话记忆 | AI 客户端的事 |

---

## 目录：每个文件一句话

```
src/rag_mcp/
├── server.py         ← 对 AI 暴露的三个工具（协议层，不含业务逻辑）
├── service.py        ← 检索编排：召回 → 融合 → 重排 → 裁剪（核心业务都在这里）
├── policy.py         ← 谁能看到什么：ACL + 废止状态 + 过滤条件
├── budget.py         ← 所有上限的唯一来源（返回条数、片段长度…）
├── errors.py         ← 结构化错误码，让 AI 知道怎么改
├── registry.json     ← 可过滤字段的词表（由入库侧定义）
├── text.py           ← 中文分词、清洗、片段截取、注入内容标注
├── embedding.py      ← 向量模型（远程 API / 本地桩）
├── rerank.py         ← 重排（百炼 / OpenAI 兼容 / 启发式 / 关闭）
├── fusion.py         ← 两路结果融合（RRF）
├── cache.py          ← 向量索引缓存（冷启动 3.5s → 0.37s）
├── ingest.py         ← 入库 CLI：把语料变成 rag.db
├── doctor.py         ← 自检：语料 / 向量模型 / 重排模型 三段排查
└── backends/
    ├── base.py       ← 后端只需实现 4 个方法
    ├── memory.py     ← 读 JSON（开发/测试）
    └── sqlite.py     ← 读 rag.db（生产用）

eval/                 ← 评测：golden 用例 + 指标 + 安全门禁
config/tcm.env        ← 真实的「中医语料」配置档
.env                  ← 密钥与默认配置（不进版本库）
DESIGN.md             ← 为什么这么设计（含所有决策的理由与代价）
MVP.md                ← 对外功能契约与验收矩阵
```

---

## 接入一个新场景：5 步，2–4 天

这是判断「它算不算通用模板」的唯一标准 —— 不是配置项多不多。

| 步骤 | 谁做 | 工时 | 说明 |
|---|---|---|---|
| **1. 语料入库** | 开发 | 半天–2 天 | 已有 JSON / 向量库直接可用；全新格式（扫描 PDF、Confluence）要写一个小适配器 |
| **2. 元数据映射** | **你** | 半天 | 部门 / 类型 / 密级 / 日期。**纯业务决策，代码不用改** |
| **3. 写 20–40 条真实问题** | **你** | 半天–1 天 | **最重要的一步。** 没有它，后面所有调优都是自我安慰 |
| **4. 标定阈值** | 脚本 | 30 秒 | `python -m rag_mcp.calibrate` 自动算 |
| **5. 验收** | 开发 | 10 分钟 | `pytest eval` 全绿才算接入成功 |

```bash
# 步骤 1：入库
python -m rag_mcp.ingest --source json      --data 你的语料.json       --out rag.db
python -m rag_mcp.ingest --source tcm-sqlite --db 现成向量库.sqlite     --out rag.db

# 步骤 4：标定（会打印建议值并写出一份 env 档）
python -m rag_mcp.calibrate --db rag.db --golden eval/golden_新场景.jsonl --out config/新场景.env

# 步骤 5：验收
RAG_ENV_FILE=config/新场景.env python -m pytest eval -q
```

---

## 配置项的两种性质（重要）

```bash
# ---- 第一类：安全可调，随便改，改坏只是体验差 ----
RAG_MAX_TOP_K=20          # 返回条数上限
RAG_SNIPPET_CHARS=300     # 片段字数
RAG_MAX_FETCH_CHARS=4000  # 展开时的字数上限
RAG_EMBEDDING_MODEL=...   # 向量模型
RAG_RERANK_MODEL=...      # 重排模型
RAG_RETRIEVAL_MODE=auto   # auto | hybrid | lexical | dense

# ---- 第二类：改了必错，必须用 calibrate 重算 ----
RAG_RELEVANCE_FLOOR=0.15  # 低于它丢弃结果
RAG_CONFIDENCE_HIGH=0.50  # 标成 high 的门槛
RAG_CONFIDENCE_MEDIUM=0.25
RAG_DENSE_FLOOR=0.55      # 只被语义命中的最低余弦
```

**为什么第二类不能随手调**：这三个数字的来历是「无关问题最高 0.121，真实问题最低 0.197 → 所以线画在 0.15」。
换一份语料，这组数字全变。**已实测过**：同一套阈值在样例语料上正常，在中医语料上会让真实问题全部被丢弃。

→ 所以**不要**做滑块界面让人随手拉这四个数；要做的是能自动重算它们的脚本。

---

## 出问题了先跑什么

| 现象 | 先执行 |
|---|---|
| 客户端连不上 / 一调用就断 | `.venv/bin/rag-mcp` 手动启动，看 **stderr**（日志都在这里，stdout 只走协议） |
| 搜不到东西 | `python -m rag_mcp.doctor` 看检索模式是不是掉成了 `lexical` |
| 结果全是 `low` / 全空 | 阈值没标定 —— 跑 `calibrate` |
| 启动很慢 | `doctor` 看「向量缓存」是否有命中；第一次要建索引 |
| 想确认没泄密 | `pytest eval -q`，看 `leaks==0` |
| 想确认改好了没 | 改前改后各跑一次 `eval/run_eval.py`，对比表 |

---

## 三个数字（当前实测）

| 场景 | Recall@10 | 首条正确 | nDCG@10 | 无关问题沉默 | 延迟 |
|---|---|---|---|---|---|
| 样例语料（6 篇假制度） | 100% | 96.8% | 0.98 | — | ~0.3s |
| **中医语料（884 块）** | **100%** | **100%** | **0.993** | **100%** | 0.79s |

测试：`84 passed`（样例语料）+ 22 条真实语料用例全绿。

---

## 文档地图

| 想知道 | 看 |
|---|---|
| **这是什么、怎么用** | 本文件 |
| **对外承诺了哪些行为、怎么验收** | [MVP.md](MVP.md) |
| **为什么这么设计、放弃了什么、代价是什么** | [DESIGN.md](DESIGN.md) |
| 踩过哪些坑、哪些设计被实测推翻 | DESIGN.md §18「落地记录」 |

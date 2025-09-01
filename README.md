# Ledger AI Assistant

**快速开始（3 步）**

1) 在项目根目录创建 `.env`（至少任选其一完成配置）

```ini
# 方案 A：Azure OpenAI（推荐）
AZURE_OPENAI_API_KEY=你的Key
AZURE_OPENAI_ENDPOINT=https://你的资源名.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-15-preview  # 旧版也可；如要 json_schema 请升到 2024-08-01-preview+
AZURE_OPENAI_DEPLOYMENT_MAP={"gpt-4o":"<你的4o部署名>", "gpt-4o-mini":"<你的4o-mini部署名>"}

# 方案 B：OpenAI（可选，若你用官方 OpenAI）
# OPENAI_API_KEY=sk-...

# 数据库（可选）— 默认使用 SQLite（无需配置）。如用 MySQL：
# DB_DIALECT=mysql
# MYSQL_HOST=127.0.0.1
# MYSQL_PORT=3306
# MYSQL_DB=ledger
# MYSQL_USER=ledger_user
# MYSQL_PASSWORD=ledger123
# MYSQL_CHARSET=utf8mb4
```

2) 安装依赖并创建虚拟环境

```bash
uv sync
```

3) 运行（含两步快速测试与图可视化）

```bash
uv run -m src.agents.expenseTrackerAgent  # main agent
uv run -m src.agents.utils.rag_tool  # create vector_db

```

提示：本项目的结构化输出已使用 function calling，兼容较旧的 Azure API 版本。若你需要使用 json_schema，请把 `AZURE_OPENAI_API_VERSION` 升级到 `2024-08-01-preview` 或更新版本。

一个基于 LangGraph + LangChain 的中文记账智能体。它能够：

- 记账意图识别：判断用户输入是否是可入账语句/相关聊天/其他。
- 信息抽取与缺槽追问：抽取 item（事项）、amount（金額）等字段；缺失则友好追问补齐。
- 标准化与入库：时间归一化为 ISO8601 到分钟；金额单位分（cents）；写入数据库（默认 SQLite，支持 MySQL）。
- 历史检索（RAG）：基于聊天记录构建向量库，可在对话中查询过往记录辅助回复。
- 多模型与多供应商：OpenAI、Azure OpenAI、DeepSeek、Anthropic、Google、Groq、AWS Bedrock、Ollama、Fake（本地测试）。


## 目录结构

```
├── .env                         # 环境变量（本地开发）
├── ledger.db                    # 默认 SQLite 数据库（运行后生成）
├── chroma_db/                   # 向量库持久化目录（RAG）
├── chat_history.jsonl           # 对话日志（用于 RAG 语料）
├── schema/
│   └── models.py                # 供应商与模型枚举
├── src/
│   ├── agents/
│   │   ├── expenseTrackerAgent.py   # 主 Agent（CLI 可运行）
│   │   └── internal/
│   │       ├── rag_tool.py          # 构建/加载 Chroma + 检索工具
│   │       ├── db_repo.py           # SQLite/MySQL 仓储实现
│   │       ├── chat_log.py          # 追加会话到 jsonl
│   │       └── context.py           # 上下文窗口装配
│   ├── core/
│   │   ├── llm.py                   # 按环境选择与构造 LLM
│   │   └── settings.py              # 配置加载与校验
│   └── draw_graph.py                # 渲染 Agent 图
├── pyproject.toml               # Python 项目配置（Python >= 3.12）
└── README.md
```


## 环境准备（使用 uv）

- Python: `>= 3.12`
- 安装 uv：
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows PowerShell: `iwr -useb https://astral.sh/uv/install.ps1 | iex`
- 同步依赖（读取 `pyproject.toml` 和现有 `uv.lock`，自动创建 `.venv/`）：

```bash
uv sync
```

- 可选：无需手动激活虚拟环境，直接用 `uv run` 执行命令；如果你想激活：

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```


## 配置说明（.env）

项目支持多家模型供应商。至少配置一种可用的 Key 即可启动。常见配置如下（示例）：

```
# —— Azure OpenAI（示例）——
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<your-endpoint>.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-15-preview  # 或更新版本
AZURE_OPENAI_DEPLOYMENT_MAP={"gpt-4o":"<deployment-id>", "gpt-4o-mini":"<deployment-id>"}

# —— OpenAI（可选）——
# OPENAI_API_KEY=...

# —— DeepSeek / Anthropic / Google / Groq / Bedrock / Ollama（按需）——
# 见 src/core/settings.py 中环境变量命名

# —— 数据库 ——
# 默认 SQLite：ledger.db
# 若用 MySQL：
# DB_DIALECT=mysql
# MYSQL_HOST=127.0.0.1
# MYSQL_PORT=3306
# MYSQL_DB=ledger
# MYSQL_USER=ledger_user
# MYSQL_PASSWORD=ledger123
# MYSQL_CHARSET=utf8mb4
```

注意：
- 代码已将结构化输出方式设为 function calling，兼容较旧的 Azure API 版本。如果你希望使用 OpenAI 的 json_schema 响应格式，请将 `AZURE_OPENAI_API_VERSION` 升级至 `2024-08-01-preview` 或更新版本。
- 首次运行会自动创建 SQLite 表（或在 MySQL 下自动建表）。


## 运行

交互式 CLI（带两步快速测试与图可视化）：

```bash
uv run -m src.agents.expenseTrackerAgent
```

启动后：
- 程序会尝试用 Matplotlib 弹窗展示 Agent 图（若无 GUI 环境，会回退到 Mermaid 在线渲染）。
- 控制台会先执行“快速测试”：
  - 输入：我早上买了早餐（缺金额）
  - 再输入：10元（补金额）
  - 正常情况下会写入数据库并给出确认语。
- 进入交互模式后：
  - 直接输入中文自然语句即可记账或交流。
  - 输入 `/new` 开新会话；输入 `exit` 退出。


## Agent 工作流（简述）

- entry：记录用户原始消息到 `chat_history.jsonl`。
- classify：意图分类 → `log_expense | related_chat | other`。
- extract：信息抽取 → `item/amount/currency/time/category/merchant/note`。
- validate：必填槽检查（item, amount），缺失则进入 `awaiting=fill` 并告知缺什么。
- handle_fill：当用户补充时，再次“重抽 + 决策（fill/new_log/cancel/…）”。
- write_db：将规范化后的 payload（金额分、时间 ISO8601 到分钟）写入数据库。
- respond：统一出口，结合状态快照与工具（RAG）生成对用户可见的一句话回复。


## 历史检索（RAG）

- 对话日志会追加到 `chat_history.jsonl`。
- RAG 工具默认从持久化的 Chroma 库加载：`persist_directory=chroma_db`，`collection_name=chat_history`。
- 若首次使用或需更新语料，可用以下方式构建/更新向量库：

```bash
# 一次性构建（命令行）：
uv run python -c "from agents.internal.rag_tool import create_vector_db; create_vector_db('chat_history.jsonl','chroma_db','chat_history')"

# 或进入交互式 Python：
uv run python
>>> from agents.internal.rag_tool import create_vector_db
>>> create_vector_db(jsonl_path='chat_history.jsonl', persist_directory='chroma_db', collection_name='chat_history')
```

- 在回复阶段，如判定为 `related_chat`，Agent 会调用 `chat_history_retriever` 工具查询历史。


## 数据库

- 默认：SQLite（文件 `ledger.db`）。
- 可选：MySQL（设置 `DB_DIALECT=mysql` 与 `MYSQL_*`）。
- 表结构（两种后端语义一致）：
  - `occurred_at`（到分钟）、`item`、`amount_cents`、`currency`、`type`（expense/income）、`category`、`merchant`、`note`、`source_message`、`created_at`。


## 模型与供应商

- 通过 `src/core/llm.py` 与 `src/core/settings.py` 统一管理：
  - 若启用 Azure OpenAI：需配置密钥、endpoint、deployment map；默认模型为 4o/4o-mini 对应的部署名。
  - 其它供应商按需配置其 API Key 与可用模型。
- 默认温度 `0.5`，多数模型支持流式输出。


## 常见问题（FAQ）

- 400 错误：`response_format value as json_schema is enabled only for api versions 2024-08-01-preview and later`
  - 说明：Azure API 版本过旧且启用了 json_schema。项目现已改为 function calling，通常不会再触发。
  - 如需启用 json_schema：升级 `AZURE_OPENAI_API_VERSION` 至 `2024-08-01-preview` 或更新版本。

- Matplotlib 弹窗未显示
  - 可能是无 GUI 环境。可忽略；或在有桌面环境的机器上运行以查看图。

- RAG 查询无结果
  - 可能是未构建向量库或对话数据太少。请按“历史检索（RAG）”章节构建向量库。


## 开发与调试建议

- 查看/调整 Agent 节点与边：`src/agents/expenseTrackerAgent.py`。
- 绘制图与观察流转：运行主脚本会自动弹出图；也可单独调用 `src/draw_graph.py` 中的 `draw_graph`。
- 日志：对话会存入 `chat_history.jsonl`，入库与校验也会在消息流中以审计信息体现（仅内部可见）。


## 致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Chroma](https://www.trychroma.com/)
- [Sentence-Transformers](https://www.sbert.net/)

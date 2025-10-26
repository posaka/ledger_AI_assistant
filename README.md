# Ledger AI Assistant

一个基于 LangGraph 的中文记账助手，集成多家大模型供应商、结构化记账流程、长期记忆与可视化前端。项目同时提供 Streamlit Web UI 与命令行调试入口，适合演示、原型验证或二次开发。

## 核心特性
- **多模型支持**：按 `.env` 配置自动启用 Azure OpenAI、OpenAI、DeepSeek、Anthropic、Google、Groq、AWS Bedrock、Ollama 等模型。
- **记账工作流自动化**：意图分类、字段抽取、缺口追问、入库校验一体化，金额统一为分、时间统一为 ISO8601 到分钟。
- **数据库与记忆体系**：默认 SQLite，可切换 MySQL；LangMem + LangGraph Store 支持管理与检索跨会话记忆。
- **对话增强检索（RAG）**：利用 `chat_history.jsonl` 搭建 Chroma 向量库，在聊天回复中检索相关历史。
- **多端入口**：Streamlit 登录/注册界面查看账本与对话；CLI 便于快速联调与图结构可视化。
- **脚本工具链**：随机交易播种、记忆仓库检查、MemoBase 用户同步等脚本提升运营效率。

## 架构速览
- **核心基础层 (`src/core/`)**：环境配置、模型工厂、嵌入工厂。
- **智能体编排层 (`src/agents/`)**：LangGraph 状态机（意图分类、信息补全、数据库写入、查询计划、LangMem 工具）。
- **数据与记忆层 (`src/agents/utils/`)**：数据库仓库（SQLite/MySQL）、上下文拼装、RAG 工具、记忆存储与 MemoBase 客户端。
- **前端展示层 (`src/view/`)**：Streamlit 多页面应用（聊天助手、账本仪表盘）。
- **运维脚本层 (`src/scripts/`)**：数据播种、记忆检查、用户画像同步。
- **支撑资源**：`chat_history.jsonl`、`ledger.db`、`docs/`、`schema/` 等。

## 快速开始
1. **安装 uv（一次性）**
   - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows PowerShell: `iwr -useb https://astral.sh/uv/install.ps1 | iex`

2. **同步依赖**
   ```bash
   uv sync
   ```

3. **填写 `.env`**（示例，至少保证一种模型配置生效）
   ```ini
   # Azure OpenAI（示例）
   AZURE_OPENAI_API_KEY=xxx
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_DEPLOYMENT_MAP={"gpt-4o":"your-gpt4o","gpt-4o-mini":"your-gpt4o-mini"}

   # OpenAI 官方（可选）
   # OPENAI_API_KEY=sk-...

   # 其它模型（可选，命名见 src/core/settings.py）
   # DEEPSEEK_API_KEY=...
   # ANTHROPIC_API_KEY=...
   # GOOGLE_API_KEY=...
   # GROQ_API_KEY=...
   # USE_AWS_BEDROCK=true
   # OLLAMA_MODEL=llama3

   # 数据库
   # 默认使用 SQLite 文件 ledger.db；切换 MySQL：
   # DB_DIALECT=mysql
   # MYSQL_HOST=127.0.0.1
   # MYSQL_PORT=3306
   # MYSQL_DB=ledger
   # MYSQL_USER=ledger_user
   # MYSQL_PASSWORD=ledger123
   # MYSQL_CHARSET=utf8mb4

   # LangGraph Store（记忆向量，可选）
   # STORE_DIALECT=postgres
   # PG_CONN_STR=postgresql://postgres:pass@localhost:15432/postgres

   # MemoBase（可选，提供用户画像）
   # MEMOBASE_URL=https://api.memobase.io/projects/xxx
   # MEMOBASE_SECRET=your-key
   ```

4. **启动入口**
   - Streamlit Web UI（推荐）  
     ```bash
     uv run streamlit run src/view/view.py
     ```
     浏览器访问后可注册/登录账号，体验聊天助手与账本可视化。

   - 命令行调试  
     ```bash
     uv run python -m src.agents.expense_tracker_agent
     ```
     默认输出流程图、执行两步缺字段测试，并进入交互式命令行。

5. **构建/更新聊天向量库（可选）**
   ```bash
   uv run python - <<'PY'
   from agents.utils.rag_tool import create_vector_db
   create_vector_db("chat_history.jsonl", "chroma_db", "chat_history")
   PY
   ```

## 目录结构
```
├── .env                        # 本地环境变量（自行创建）
├── ledger.db                   # SQLite 数据库（运行后生成）
├── chroma_db/                  # Chroma 向量库持久化目录
├── chat_history.jsonl          # 对话日志，用作 RAG 语料
├── docs/                       # 额外文档或素材
├── schema/
│   ├── models.py               # 模型与供应商枚举
│   └── schema.py               # 结构化数据定义
├── src/
│   ├── agents/
│   │   ├── expense_tracker_agent.py  # LangGraph 记账智能体
│   │   └── utils/
│   │       ├── context.py            # 对话上下文拼装
│   │       ├── db_repo.py            # LedgerDB 抽象与实现
│   │       ├── rag_tool.py           # JSONL → Chroma 构建与检索
│   │       ├── store.py              # LangGraph Store 工厂（内存/Postgres）
│   │       └── user_profile.py       # MemoBase 客户端与格式化工具
│   ├── core/
│   │   ├── llm.py                    # LLM 客户端工厂
│   │   ├── embeddings.py             # 向量嵌入工厂
│   │   └── settings.py               # 项目级配置管理
│   ├── scripts/
│   │   ├── seed_transactions.py      # 随机交易播种脚本
│   │   ├── print_store.py            # LangMem store 检查
│   │   └── user_profile.py           # MemoBase 调试脚本
│   ├── view/
│   │   ├── view.py                   # Streamlit 入口（登录 + 框架）
│   │   ├── conversation_view.py      # 对话页面
│   │   └── ledger_view.py            # 账本可视化页面
│   └── draw_graph.py                 # LangGraph 图渲染辅助
├── pyproject.toml                # 项目依赖与构建配置
├── uv.lock                       # uv 锁定文件
└── README.md
```

## 运行说明
- **首次登录**：默认数据库为空，可直接在 Streamlit 注册新账号；也可以通过播种脚本提前生成测试账号。
- **CLI 模式**：
  - 运行后会依次打印快速测试结果。
  - 支持 `/new` 重置会话、`exit` 退出。
  - 若需可视化状态图，可在 `src/draw_graph.py` 中调用 `draw_graph(app)`。
- **Streamlit 模式**：
  - 登录成功后侧边栏可切换“智能助手”“账本”两页。
  - 账本页面展示按年月的消费柱状图与交易明细表。

## 数据与记忆
- **数据库**：`src/agents/utils/db_repo.py` 默认使用 SQLite，支持切换 MySQL；封装用户注册、认证、交易插入、统计查询等方法。
- **LangGraph Store**：`src/agents/utils/store.py` 根据 `STORE_DIALECT` 选择内存或 Postgres，存储 LangMem 记忆向量。
- **聊天日志**：`src/agents/utils/chat_log.py` 将每条对话附带 UTC 时间戳写入 `chat_history.jsonl`，供 RAG 与审计使用。
- **MemoBase 集成**：如配置 `MEMOBASE_URL` / `MEMOBASE_SECRET`，可创建并同步用户画像，用于长期记忆检索。

## 维护脚本
- **随机生成交易**  
  ```bash
  uv run python src/scripts/seed_transactions.py --username demo --password Demo123! --count 200
  ```
  自动创建账号（如不存在）并写入一批近一年的随机收支。

- **检查 LangMem Store**  
  ```bash
  uv run python src/scripts/print_store.py
  ```

- **MemoBase 调试**  
  ```bash
  uv run python src/scripts/user_profile.py
  ```

## 常见问题
- **未配置任何模型时启动报错**：`core/settings.py` 会检查至少一个可用 API Key，请确保在 `.env` 中启用了某个模型供应商。
- **Azure 提示需升级 API 版本以使用 json_schema**：项目默认采用 function calling，无需 json_schema。如需启用 json_schema，请将 `AZURE_OPENAI_API_VERSION` 升级至 `2024-08-01-preview` 及以上。
- **RAG 无检索结果**：确认 `chat_history.jsonl` 有内容并调用过 `create_vector_db` 构建/更新 `chroma_db`。
- **Streamlit 登录失败**：检查数据库是否可写、`ledger.db` 是否存在权限问题；若使用 MySQL，请确认连接信息正确。

## 开发建议
- 使用 `uv run python -m src.agents.expense_tracker_agent` 观察状态节点调试日志。
- 调整或扩展节点时可调用 `draw_graph` 输出 Mermaid/图像，快速验证流程。
- 建议为新增意图或查询逻辑补充单元测试，并在播种脚本中扩展模拟场景。

## 致谢
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Chroma](https://www.trychroma.com/)
- [MemoBase](https://github.com/langgenius/memobase)

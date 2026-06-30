# tongue-agent

Python Agent 服务，负责舌象识别编排、RAG 检索、报告生成、健康问答、报告追问、报告对比解释和健康计划 AI 评估。Java 后端负责权限、任务、落库和确定性业务逻辑；本服务只返回识别结果、结构化报告和解释性文本。

## 技术栈

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic / pydantic-settings
- LangGraph
- PostgreSQL / pgvector
- Redis
- Elasticsearch
- SQLAlchemy / asyncpg / psycopg
- sentence-transformers
- httpx

默认服务地址：

```text
http://127.0.0.1:8000
```

健康检查：

```text
GET http://127.0.0.1:8000/api/v1/health
```

## 目录结构

```text
tongue-agent
├─ app
│  ├─ agent
│  │  ├─ graph.py                 LangGraph 编排入口
│  │  ├─ nodes                    意图、RAG、舌象、报告、问答等节点
│  │  └─ runtime                  运行时工具和响应收敛
│  ├─ api
│  │  ├─ routes_agent.py          Agent API
│  │  └─ routes_health.py         健康检查
│  ├─ core                        配置、锁、幂等、事件循环
│  ├─ integrations                模型网关、舌象模型、Redis、ES、Postgres、Java 回调
│  ├─ intent                      意图识别
│  ├─ memory                      长短期记忆
│  ├─ rag                         知识库检索
│  ├─ schemas                     请求和响应模型
│  └─ tongue                      舌象特征标准化
├─ tests                          pytest 测试
├─ sql                            PostgreSQL 补充脚本
├─ docker-compose.yml             Redis/Postgres/Elasticsearch 本地依赖
├─ pyproject.toml
├─ .env.example
└─ run_agent_server.py
```

## 本地依赖

### 1. Python 环境

```powershell
cd D:\tongue\tongue-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

### 2. Redis / Postgres / Elasticsearch

本仓库提供了最小本地依赖：

```powershell
cd D:\tongue\tongue-agent
docker compose up -d redis postgres elasticsearch
```

默认端口：

| 服务 | 地址 |
| --- | --- |
| Redis | `localhost:6379` |
| Postgres | `localhost:5432` |
| Elasticsearch | `http://localhost:9200` |

Postgres 默认库：

```text
database: tongue_agent
username: postgres
password: postgres
```

### 3. Java 后端

报告追问和健康计划评估会读取 Java 后端里的报告上下文，默认地址：

```text
http://127.0.0.1:8080
```

### 4. 模型服务

需要配置：

- LLM 网关：默认兼容 OpenAI/DeepSeek 风格接口。
- 舌象识别模型：默认 `http://127.0.0.1:9100`。

如果舌象模型不可用，图片分析会失败；普通健康问答仍可依赖 LLM 和 RAG。

## 配置

复制配置文件：

```powershell
cd D:\tongue\tongue-agent
Copy-Item .env.example .env
```

最常用配置：

```env
APP_NAME=tongue-agent
APP_ENV=local
LOG_LEVEL=INFO

REDIS_URL=redis://localhost:6379/0
ELASTICSEARCH_URL=http://localhost:9200
LANGGRAPH_POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/tongue_agent?sslmode=disable
RAG_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/tongue_agent

MODEL_GATEWAY_BASE_URL=https://api.deepseek.com
MODEL_GATEWAY_API_KEY=replace-with-your-api-key
CHAT_MODEL_NAME=deepseek-chat
CHAT_MODEL_TEMPERATURE=0.2
CHAT_MODEL_MAX_TOKENS=800
CHAT_MODEL_JSON_MODE_ENABLED=false
DETAILED_FOLLOWUP_MAX_TOKENS=2200

REPORT_MODEL_TEMPERATURE=0.15
REPORT_MODEL_MAX_TOKENS=2200

TONGUE_MODEL_BASE_URL=http://127.0.0.1:9100
TONGUE_MODEL_API_KEY=
TONGUE_MODEL_TIMEOUT_SECONDS=30

JAVA_BACKEND_BASE_URL=http://127.0.0.1:8080
JAVA_INTERNAL_API_KEY=
```

配置入口：

```text
app/core/config.py
```

## 启动

```powershell
cd D:\tongue\tongue-agent
.\.venv\Scripts\Activate.ps1
python run_agent_server.py --host 127.0.0.1 --port 8000 --reload
```

不需要热重载时：

```powershell
python run_agent_server.py --host 127.0.0.1 --port 8000
```

检查：

```powershell
curl.exe http://127.0.0.1:8000/api/v1/health
```

期望返回：

```json
{"status":"ok"}
```

## API

所有接口都有 `/api/v1` 前缀。

### 健康检查

```text
GET /api/v1/health
```

### 舌象分析主入口

```text
POST /api/v1/agent/run
```

Java 后端创建任务后调用该接口。它会执行：

```text
请求校验
  -> 幂等和锁
  -> LangGraph
  -> 意图/安全/上下文
  -> 舌象模型识别
  -> RAG 检索
  -> Schema 2.0 报告生成
  -> 结构化响应
```

### 回执

```text
POST /api/v1/agent/turns/ack
GET  /api/v1/agent/turns/pending-ack
```

用于 Java 后端确认 turn 已保存，辅助排查 Agent 响应和后端落库之间的状态。

### 报告对比解释

```text
POST /api/v1/agent/report-compare
```

Java 负责计算确定性 diff，Python 只生成自然语言解释和观察建议。Python 失败时，Java 仍应返回确定性 diff。

### 健康计划 AI 评估

```text
POST /api/v1/agent/health-plan/review
```

请求里通过 `mode` 区分：

```text
review             评估用户草稿是否合理
generate_detailed  生成更具体的 7 天计划
```

Agent 只做健康管理参考，不做诊断，不开药，不自动启用计划。

## Agent 图职责

核心节点在：

```text
app/agent/nodes
```

常见节点：

| 节点 | 作用 |
| --- | --- |
| `intent_node.py` | 意图识别 |
| `safety_node.py` | 安全边界 |
| `context_builder_node.py` | 上下文构建 |
| `query_rewrite_node.py` | RAG 查询改写 |
| `tongue_analysis_node.py` | 舌象模型识别 |
| `tongue_report_node.py` | Schema 2.0 报告生成 |
| `report_followup_node.py` | 报告追问 |
| `health_qa_node.py` | 健康问答 |
| `agent_loop_node.py` | 工具循环和最终回答收敛 |
| `memory_node.py` | 记忆写入 |

报告生成原则：

- `content` 必须是完整中文自然语言，可独立展示。
- `structured_content` 只做增强展示。
- `recognition_evidence` 只能来自真实 `DETECTED` 特征。
- 未识别或不支持维度不能当作识别事实。
- `schema_version` 为 `2.0`。
- 旧字段只由新结构确定性派生。

## 外部服务关系

```text
Java tongue-server
  -> tongue-agent /api/v1/agent/run
    -> LLM 网关
    -> 舌象模型服务
    -> Elasticsearch RAG
    -> PostgreSQL LangGraph checkpoint / turn record
    -> Redis 锁和幂等
    -> Java internal report sections
```

## 测试

```powershell
cd D:\tongue\tongue-agent
.\.venv\Scripts\Activate.ps1
pytest
```

跑单个测试：

```powershell
pytest tests/test_tongue_report_node.py
pytest tests/test_agent_response_contract.py
pytest tests/test_report_followup_flow.py
```

## 常见问题

### 启动时报 Postgres 或 Redis 连接失败

先启动依赖：

```powershell
docker compose up -d redis postgres elasticsearch
```

确认 `.env` 里的 `REDIS_URL` 和 `LANGGRAPH_POSTGRES_URI` 没改错。

### 普通聊天可以，图片分析失败

检查：

1. `TONGUE_MODEL_BASE_URL` 是否可访问。
2. 图片是否超过 `TONGUE_MODEL_MAX_IMAGE_SIZE_MB`。
3. 舌象模型接口是否需要 token。

### 报告太短或进入模板兜底

检查：

```env
REPORT_MODEL_MAX_TOKENS=2200
REPORT_MODEL_TEMPERATURE=0.15
```

详细追问还需要：

```env
DETAILED_FOLLOWUP_MAX_TOKENS=2200
```

### 出现 JSON 泄漏

优先检查：

- `app/agent/runtime` 的最终响应收敛。
- `agent_loop_node.py` 是否把内部工具 JSON 当最终回答。
- Java 的 `AgentResponseSanitizer` 是否拦截。
- 前端 `assistant-response.ts` 是否兜底。

### RAG 召回为空

检查：

1. Elasticsearch 是否启动。
2. `RAG_*` 配置是否正确。
3. 知识库索引是否已构建。
4. `TAVILY_API_KEY` 是否需要用于 web fallback。

## 开发约定

- Python Agent 不做用户权限校验，权限由 Java 负责。
- Python Agent 不直接写 Java 业务库。
- 内部 JSON 不能作为最终 `content` 返回。
- 医疗相关失败不要从半截 JSON 强行恢复，宁可模板兜底。
- 任何健康建议都要保留“健康管理参考，不能替代医生诊断”的边界。
- 新增配置先放 `app/core/config.py`，再同步 `.env.example`。

# 月光宝盒 Moonlight Box

月光宝盒是一个单用户、本地优先的聊天记忆分析与数字人格分支探索系统。它可以导入聊天记录，识别关系转折点，训练本地 LoRA 人格模型，并从任意历史节点创建独立的平行对话。

## 主要能力

- 导入 wxecho CSV、JSON、TXT 聊天记录并预览
- 通过事件分段、多信号变化检测和 LLM 复核生成可追溯的关键节点
- 使用带时间边界的分支状态快照控制分支上下文
- 在 Linux/WSL 上通过 PyTorch、Transformers 与 PEFT 训练、加载 LoRA
- 创建并保存多条独立时间分支
- 通过 LangChain + LangGraph 驱动事件化 Runtime：Director 决定生活，现有 LoRA 只负责表达
- 使用真实回复盲测模型，并按质量门槛推荐版本
- 可选使用 OpenAI 兼容接口进行云端评测

## 环境要求

- 当前开发环境：Windows 11 + WSL2（Ubuntu 24.04）
- Python 3.12、[uv](https://docs.astral.sh/uv/)
- Node.js 22 及以上（推荐当前 LTS）
- Docker Engine + Docker Compose v2（运行 LightRAG sidecar 或整套容器服务时需要）

WSL/Linux 是完整后端运行环境：数据导入、LightRAG、人物档案编译、API、Worker、人格推理
和 QLoRA 训练均在 Linux 进程中执行。`./scripts/start.sh` 会安装 `linux-ml` 依赖并启动
独立人格服务。

## Docker 一键体验

不想配置 Python/Node 环境时，可以直接使用 All-in-One 镜像（包含前端、后端、LightRAG
图谱与全部 Worker；本版不含本地 LoRA 训练/推理，人格回复由云端认知模型生成）：

```bash
cp docker/runtime.env.example .env   # 填入 MiniMax 与 Embedding 的 API key
docker run -d --name moonlightbox -p 8080:80 \
  -v moonlightbox-data:/app/data \
  --env-file .env \
  ghcr.io/amos144/moonlightbox:latest
```

打开 `http://localhost:8080`。说明：

- 所有运行数据（SQLite、向量库、LightRAG 索引、设置）保存在 `moonlightbox-data`
  卷中，删除容器不丢数据。
- 镜像不含任何 API key；cognition / 节点分析模型也可以在 Web「设置」页配置。
- 自行构建：`docker build -f Dockerfile.allinone -t moonlightbox:allinone .`。
- 需要本地 LoRA 训练/推理与 Phoenix Trace 浏览器时，使用根 `Dockerfile` 与
  `docker-compose.yml` 的分体式多容器部署。

## 本机启动

推荐使用一键启动脚本，它会安装依赖、升级数据库，并管理前端、后端和三类 Worker。WSL/Linux
会在 Linux/WSL 本机管理人格服务；也可用 `MOONLIGHTBOX_MANAGE_PERSONA_RUNTIME=false`
连接独立 Linux GPU 节点：

```bash
./scripts/start.sh
```

按 `Ctrl+C` 可统一关闭全部进程。也可以手动启动：

```bash
uv sync --extra linux-ml
uv run alembic -c backend/alembic.ini upgrade head
uv run uvicorn moonlightbox.api:app --app-dir backend
# Linux/WSL 执行；API 与 Worker 仅通过 HTTP 使用该进程的 GPU
uv run uvicorn moonlightbox.persona_runtime:app --app-dir backend --port 8765
```

启动脚本默认关闭 Uvicorn reloader，确保退出时不会遗留重载子进程。

### 历史 MLX LoRA 迁移

历史 adapter 是 MLX 的 `adapters.safetensors`，不能直接给 PEFT 加载。转换命令不会覆盖源目录：

```bash
uv sync --extra linux-ml
PYTHONPATH=backend uv run python -m moonlightbox.training.convert_mlx_adapter \
  models/<项目>/<旧 adapter> models/<项目>/<旧 adapter>-peft \
  --base-model Qwen/Qwen3-8B
```

转换后需将对应 `ModelVersion` 的 `base_model` 和 `adapter_path` 一起切换到该 Hugging Face
基座和新目录；adapter 必须始终匹配训练时的基座。当前仓库中的历史 adapter 均来自 Qwen3-8B；
**4GB GTX 1650 无法运行它们**，转换只改变格式、不降低 8B 基座的显存需求。请在至少约 10GB
显存的 Linux GPU 上使用这些历史模型，或用默认 `Qwen/Qwen3-1.7B` 在本机重新训练。

人格推理服务健康后，分别启动实时会话、离线认知、后台记忆和 GPU 训练四条隔离通道：

```bash
MOONLIGHTBOX_WORKER_ROLE=realtime PYTHONPATH=backend uv run python -m moonlightbox.worker_main
MOONLIGHTBOX_WORKER_ROLE=cognition PYTHONPATH=backend uv run python -m moonlightbox.worker_main
MOONLIGHTBOX_WORKER_ROLE=background PYTHONPATH=backend uv run python -m moonlightbox.worker_main
MOONLIGHTBOX_WORKER_ROLE=training PYTHONPATH=backend uv run python -m moonlightbox.worker_main
```

实时通道只领取会话任务并扫描主动聊天；离线认知通道领取主体认知周期和 Runtime v1
的事件循环；训练通道独占 QLoRA 任务和 GPU，后台记忆通道不会领取模型训练。
后台通道处理事件分析、长期记忆、反思和分支基础历史；认知失败不会阻塞实时回复。

再开终端启动前端：

```bash
cd frontend
npm install
npm run dev
```

打开 `http://localhost:5175`。一键启动脚本默认使用后端 `8001`、前端 `5175`、
人格推理运行时 `8765`；可通过 `MOONLIGHTBOX_BACKEND_PORT`、
`MOONLIGHTBOX_FRONTEND_PORT` 和 `PERSONA_RUNTIME_PORT` 覆盖，三个端口互不冲突。

## 配置

复制 `.env.example` 为 `.env`，按需调整。V3 事件分析需要启用
`MOONLIGHTBOX_NODE_ANALYSIS_ENABLED` 并配置 `MOONLIGHTBOX_NODE_ANALYSIS_API_KEY`；
缺少密钥时任务会安全失败，不会发起网络请求。云端评测默认启用，但只有配置密钥后才会发送请求；
将 `MOONLIGHTBOX_CLOUD_EVALUATION_ENABLED=false` 可完全关闭。发送云端请求时，相关聊天内容会交给所配置的服务商处理。

使用 DeepSeek V4 Flash 时，采用官方 JSON Object 输出并关闭节点分析思考模式：

```dotenv
MOONLIGHTBOX_NODE_ANALYSIS_ENABLED=true
MOONLIGHTBOX_NODE_ANALYSIS_ENDPOINT=https://api.deepseek.com/chat/completions
MOONLIGHTBOX_NODE_ANALYSIS_MODEL=deepseek-v4-flash
MOONLIGHTBOX_NODE_ANALYSIS_API_KEY=${DEEPSEEK_API_KEY}
MOONLIGHTBOX_NODE_ANALYSIS_RESPONSE_FORMAT=json_object
MOONLIGHTBOX_NODE_ANALYSIS_THINKING_MODE=disabled
MOONLIGHTBOX_NODE_ANALYSIS_MAX_OUTPUT_TOKENS=8192
```

`json_schema` 是通用默认值；DeepSeek 官方接口应改用 `json_object`。Worker 会在提示词中附上
本次响应模型的精简 JSON schema，并继续执行本地 Pydantic 严格校验。API key 只应通过未提交
的 `.env` 或运行环境注入。

数字人的决策认知与最终语气可以独立配置。以下配置让 DeepSeek V4 Flash 使用默认思考模式
负责上下文理解、关系判断、是否回应和内容草稿；本地人格 LoRA 只负责把可信草稿改写为本人
语气和多气泡节奏。未单独填写 `MOONLIGHTBOX_COGNITION_API_KEY` 时，会复用受保护的
`MOONLIGHTBOX_NODE_ANALYSIS_API_KEY`：

```dotenv
MOONLIGHTBOX_COGNITION_BACKEND=deepseek
MOONLIGHTBOX_COGNITION_ENDPOINT=https://api.deepseek.com/chat/completions
MOONLIGHTBOX_COGNITION_MODEL=deepseek-v4-flash
MOONLIGHTBOX_COGNITION_THINKING_MODE=default
MOONLIGHTBOX_COGNITION_RESPONSE_FORMAT=json_object
MOONLIGHTBOX_COGNITION_MAX_OUTPUT_TOKENS=4096
```

启用云端认知后，最近双边聊天、稳定人格、关系状态和经过审核的相关记忆会发送给所配置的
DeepSeek 服务；原始媒体文件、私有认知结果和本地模型权重不会上传。

### Agent Trace（Phoenix）

所有 Runtime v1（Director、DayPlanAgent、PersonaActor）、PersonWorld 七栏目 Agent 和
Revision Agent 都由统一 `AgentLoopController` 建立 Phoenix Trace。一个 Trace 的根节点对应
一次 Agent 执行，子节点包含 LangGraph 节点、Agent 决策回合、实际供应商模型请求、实际执行的
工具调用、LightRAG 检索、上下文压缩、耗时、错误码、终态与项目/分支/栏目关联字段。根 Span 会
分别汇总 Agent 回合数和供应商请求数：一次栏目决策内的 Schema 修复不会再被藏成一个黑盒。可直接
在 Phoenix 中筛选
`moonlightbox.agent.name`、`moonlightbox.owner.type`、`moonlightbox.branch.id` 或
`moonlightbox.agent.execution_id`，而不必翻数据库事件表。

本机执行 `./scripts/start.sh` 会启动独立的 Phoenix 进程，不需要启动业务 Docker 容器；访问
`http://127.0.0.1:6006` 查看。Trace 存储在 worktree 外的
`.runtime-data/phoenix`。若只想运行业务服务，可设
`MOONLIGHTBOX_PHOENIX_ENABLED=false`。

标准 `./scripts/start.sh` 只连接本机 Phoenix。单值输入/输出属性受
`MOONLIGHTBOX_PHOENIX_TRACE_MAX_CHARACTERS` 限制，但开启 `MOONLIGHTBOX_PHOENIX_CAPTURE_CONTENT`
（本机默认开启）后，完整的启动上下文、模型请求/响应、工具参数/结果、LightRAG 返回和压缩前后
上下文会以分块 Span event 保存；API key 仍会强制脱敏。根 Span 还会写入确定性根因摘要并异步创建
Phoenix CODE annotation。可通过只读 API
`/api/observability/agent-executions/{execution_id}` 取得完整原始 Span 树。若不希望保存正文，可显式
设 `MOONLIGHTBOX_PHOENIX_CAPTURE_CONTENT=false`，此时仍保留 Span 层级、耗时、模型名、工具名、状态、
错误码和输入输出 hash。项目不再维护通用 Agent 的数据库调用账本；领域业务的审核和执行结果仍由
各自领域表管理，Phoenix 不参与业务写入或完成判定。

节点分析完成后，在节点页按时间轴检查“关系变化”和“共同经历”，排除误报，再点击
“确认时间轴并开始训练”。Worker 会使用真实聊天回复和已确认节点背景生成聊天 JSONL
数据集，以 assistant 目标掩码执行本地 QLoRA；训练成功后新模型先进入候选和真人盲测，不会
自动覆盖当前启用模型。

### 个人地点图与活动热图

空间分析默认以独立后台任务运行。新部署从 `.env.example` 复制配置后会启用；既有部署需
显式设置 `MOONLIGHTBOX_SPATIAL_ANALYSIS_ENABLED=true`。地点提取固定使用已配置的节点分析
云模型读取 Episode Bundle，并返回结构化地点提及；本地人格基座不参与地点提取。若云端
分析不可用，任务会明确失败，不会静默降级为能力不同的提取器。

命名地点的 POI 消歧可配置高德 Web 服务 Key：

```dotenv
MOONLIGHTBOX_SPATIAL_ANALYSIS_ENABLED=true
MOONLIGHTBOX_SPATIAL_ANALYSIS_BACKEND=cloud
MOONLIGHTBOX_SPATIAL_AMAP_WEB_KEY=<高德 Web 服务 Key>
MOONLIGHTBOX_SPATIAL_PROVIDER_PERSISTENCE_ALLOWED=false
MOONLIGHTBOX_SPATIAL_HEATMAP_DEFAULT_PRIVACY=blurred
```

地点图、别称、到访片段和图快照保存在 MoonlightBox 数据库中；高德只负责候选查询和地理
补全。Provider enrichment 默认不持久化。前端“世界”页使用高德 JS API 2.0 与 Loca 2.0
渲染后端已脱敏的 GeoJSON。复制 `frontend/.env.example` 为 `frontend/.env.local`，分别填写
高德“Web端(JS API)”Key 与安全密钥：

```dotenv
VITE_AMAP_JS_KEY=<高德 Web端(JS API) Key>
VITE_AMAP_SECURITY_CODE=<JS API 安全密钥>
```

未配置前端 Key 时，页面仍展示地点权重、证据强度和未定位地点，但不加载底图。家庭、医疗
等精确坐标的模糊或隐藏在后端完成，浏览器不会先收到精确坐标再遮盖。
已有聊天项目可以直接进入“世界”页点击“生成活动地图”，无需重新导入；任务按导入和配置
快照幂等入队。
默认基础模型和迭代、学习率、节点增强比例等参数可通过 `.env.example` 中的
`MOONLIGHTBOX_TRAINING_*` 配置调整。训练集和 adapter 仅保存在本地。

当前训练协议会按人物真实发送间隔自适应重建对话轮次，并保留同一轮最多 10 个连续短
气泡。分支推理只检索起点之前的最近对话和相关节点，生成后的气泡由服务端按本人历史
节奏逐条投递；用户插话会取消尚未投递的旧气泡。新模型须先通过结构、事实安全、未来泄漏、
复读、历史回放和 loss 有效性检查，再完成真人盲测，才允许替换当前模型。

图片和语音必须先在本机生成语义标注：

```bash
PYTHONPATH=backend uv run python -m moonlightbox.media.cli \
  --project-id <项目 ID> --modality all
```

图片理解使用 Transformers 视觉语言模型，语音转写使用 Transformers Whisper；原始媒体不会上传到云端。任务完成后在
“回忆 → 数据质量”的“图片与语音语义审核”中逐项批准或禁止复用。未经明确批准、包含敏感
标记、超出分支历史边界，或与当前聊天语义不够接近的媒体都不会发送。

本地原始数据默认存放在 `data/`，模型存放在 `models/`。删除项目时，对应数据库记录、导入文件、向量索引和模型产物会一并删除。

### 分支 Runtime v1

聊天页发送的新消息会先写入 Runtime EventQueue，HTTP 请求不会直接调用模型。认知 Worker
领取事件后，由 LangGraph 调度平级的 Director 与 DayPlan；需要表达时交给 PersonaActor。
所有角色共用认知模型配置，并由 AgentLoopController 执行模型／工具循环。
当前聊天链路不调用本地 LoRA；独立训练与人格推理服务仍保留，不影响训练资产。

Runtime 仅能从准备完成的人物世界启动。由于当前 LightRAG 不能按时间过滤，分支使用
约定的最新完成图谱与背景，而不假装支持历史图谱切片。

开发维护入口：

- 行为 Prompt：`backend/moonlightbox/runtime_v1/prompts/<agent>/`；
- 工具参数和提交检查：各领域 `tools/`，由 LangChain 注册；
- 单 Agent 策略与网络策略注入：`backend/moonlightbox/agent_runtime/policy.py`；
- 完整请求容量、压缩、恢复与取消：`agent_runtime/controller.py`；
- 跨角色总预算：`runtime_v1/config.py`，不再包含 Prompt 或旧 tokenizer 阈值。

Graph Patch、图谱回归评估和 Alias 同样使用统一循环及原生提交工具。
提交仅产生提案；Profile、Graph 的逐步批准和 Executor 的事务保护仍然保留。
历史 v1/v2 人物画像可读，修改时先重新生成 v3，不再走旧 Profile Patch 生成器。

## Docker

先创建本地配置文件；节点分析使用 OpenAI 兼容的云端服务时，在 `.env` 中启用
`MOONLIGHTBOX_NODE_ANALYSIS_ENABLED`，并填写 endpoint、model 和
`MOONLIGHTBOX_NODE_ANALYSIS_API_KEY`。密钥只在容器启动时从 `.env` 注入，不会写入镜像。

```bash
cp .env.example .env
# 生产机建议改成宿主机绝对路径；默认值供 `.worktrees/*` 本地开发共享。
export MOONLIGHTBOX_HOST_DATA_DIR=/absolute/path/to/moonlightbox-runtime-data
docker compose up --build
```

Compose 会先运行一次 Alembic migration；迁移成功后启动 API、Worker、Linux GPU 人格服务和前端。
`persona-runtime` 与 `training-worker` 使用 `gpus: all`，宿主机需要安装 NVIDIA Container
Toolkit；二者不会在同一任务中并发加载模型。若使用独立 GPU 节点，可覆盖
`MOONLIGHTBOX_PERSONA_INFERENCE_URL` 与 token。
worktree 外的共享运行数据目录和 `models/` 会挂载到 API、migration 和 Worker 容器。
SQLite 数据库、项目媒体与 Chroma 数据因此不会随 worktree 分叉；该目录只应由同一宿主机
上的这些进程共享。

打开 `http://localhost:8080`。Docker 镜像安装 Linux QLoRA 依赖，GPU 只分配给
`persona-runtime` 与独立的 `training-worker`；API、实时 Worker 与普通后台 Worker 不会抢占
模型显存。

## 验证

```bash
uv run pytest backend/tests -q
uv run ruff check backend
uv run mypy backend/moonlightbox
cd frontend && npm test -- --run && npm run lint && npm run build
```

更详细的部署、数据目录和备份说明见 `docs/deployment.md`。

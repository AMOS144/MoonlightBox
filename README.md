# 月光宝盒 Moonlight Box

单用户、本地优先的聊天记忆分析与数字人格系统。导入微信聊天记录后，它会构建可追溯的
人物世界（事件节点、关系变化、地点图谱），并让你从任意历史节点创建独立的平行对话分支，
与由你真实聊天塑造的"数字人格"继续交谈。

- **本地优先**：原始数据、SQLite、向量索引、媒体文件全部保存在本机或自有服务器
- **可追溯**：每个事件节点、每条人格陈述都带来源消息证据，人工审核后才生效
- **云端可控**：LLM 分析走 OpenAI 兼容接口（MiniMax / DeepSeek 等），未配置 key 时
  任务安全失败，不会静默降级；媒体原始文件永不上传

## 功能一览

| 模块 | 说明 |
| --- | --- |
| 数据导入 | 导入 wxecho 的 CSV / JSON / TXT 导出，预览后确认参与者身份 |
| 事件分析 | 事件分段 + 多信号变化检测 + LLM 复核，生成带来源证据的关键节点 |
| 人物世界 | PersonWorld 七栏目 Agent 编译人物档案，LightRAG 构建可检索图谱，逐项人工批准 |
| 世界页 | 个人地点图与活动热图（高德地图，坐标在后端脱敏后才发给浏览器） |
| 平行分支 | 从任意历史节点创建分支；Runtime v1（Director / DayPlan / PersonaActor）驱动数字人生活与回复 |
| 人格模型 | 可选的本地 QLoRA 训练 + 真人盲测验收，只负责语气表达，不参与决策 |
| 可观测性 | 全部 Agent 执行写入 Phoenix Trace，可按分支 / 执行 ID 检索完整调用树 |

## 快速开始（Docker）

最简单的方式是 All-in-One 镜像：单容器包含前端、后端、LightRAG 图谱与全部 Worker，
无需 GPU（人格回复由云端认知模型生成，本版不含本地 LoRA 训练/推理）：

```bash
cp docker/runtime.env.example .env   # 填入 MiniMax 与 Embedding 的 API key
docker run -d --name moonlightbox -p 8080:80 \
  -v moonlightbox-data:/app/data \
  --env-file .env \
  ghcr.io/amos144/moonlightbox:latest
```

打开 `http://localhost:8080`。

- 所有运行数据（SQLite、向量库、LightRAG 索引、设置）保存在 `moonlightbox-data`
  卷中，删除容器不丢数据。
- 镜像不含任何 API key；cognition / 节点分析模型也可以在 Web「设置」页配置。
- 自行构建：`docker build -f Dockerfile.allinone -t moonlightbox:allinone .`。
- 需要本地 LoRA 训练/推理与 Phoenix Trace 浏览器时，使用根 `Dockerfile` 与
  `docker-compose.yml` 的分体式多容器部署（见 `docs/deployment.md`）。

## 使用流程

1. **导入**：新建项目，上传 wxecho 导出的 CSV / JSON / TXT，预览并确认"哪个是你"。
2. **分析**：事件分析任务自动识别关系转折点，生成带来源证据的事件节点。
3. **审核**：在节点页检查"关系变化"和"共同经历"，排除误报后确认时间轴。
4. **编译人物世界**：PersonWorld Agent 生成七栏目人物档案与图谱提案，逐项批准。
5. **探索**：在"世界"页查看地点图与活动热图；从任一节点创建平行分支开始对话。
   分支只检索起点之前的历史，不会"剧透未来"。

## 配置

复制 `.env.example`（本机）或 `docker/runtime.env.example`（Docker）为 `.env`。
最小可用配置只需三项：

```dotenv
# 节点分析 / 认知模型（OpenAI 兼容接口，示例为 MiniMax，可换 DeepSeek 等）
MOONLIGHTBOX_NODE_ANALYSIS_API_KEY=<你的 key>
MOONLIGHTBOX_COGNITION_API_KEY=<你的 key>     # 留空则复用节点分析 key

# LightRAG 图谱的 Embedding（示例为阿里云百炼兼容接口）
LIGHTRAG_SIDECAR_EMBEDDING_API_KEY=<你的 key>
```

常用可选项：

| 配置 | 作用 |
| --- | --- |
| `MOONLIGHTBOX_SPATIAL_AMAP_WEB_KEY` | 高德 Web 服务 key，用于地点 POI 消歧（后端） |
| `frontend/.env.local` 的 `VITE_AMAP_JS_KEY` / `VITE_AMAP_SECURITY_CODE` | 高德 JS API，用于"世界"页底图渲染 |
| `MOONLIGHTBOX_PHOENIX_ENABLED` | Agent Trace（默认关闭，本机开发脚本会开启） |
| `MOONLIGHTBOX_TRAINING_*` | 本地 LoRA 训练的基座模型、迭代、学习率等 |

隐私边界：启用云端分析后，聊天文本会发送给所配置的服务商；原始媒体文件、本地模型
权重、Phoenix 正文（可关闭）不会上传。API key 只通过未提交的 `.env` 或环境变量注入。
DeepSeek 等服务商的 endpoint / 输出格式示例见 `.env.example` 注释。

## 本机开发

环境：Python 3.12 + [uv](https://docs.astral.sh/uv/)、Node.js 22+；完整后端能力
（导入、LightRAG、训练）需要 Linux/WSL2。

```bash
./scripts/start.sh        # 一键：安装依赖、迁移数据库、启动前后端与全部 Worker
```

`Ctrl+C` 统一关闭。默认端口：后端 `8001`、前端 `5175`、人格推理 `8765`、
Phoenix `6006`，可用 `MOONLIGHTBOX_BACKEND_PORT` 等环境变量覆盖。

手动启动与四条 Worker 通道（realtime / cognition / background / training）的细节见
`docs/deployment.md`。

### 验证

```bash
uv run pytest backend/tests -q
uv run ruff check backend
uv run mypy backend/moonlightbox
cd frontend && npm test -- --run && npm run lint && npm run build
```

## 进阶

### 本地 LoRA 人格模型（可选，需 Linux GPU）

确认时间轴后，Worker 会用真实回复生成训练集并执行本地 QLoRA。新模型必须通过结构、
事实安全、未来泄漏、复读、历史回放检查，再经真人盲测，才会替换当前启用版本。
历史 MLX adapter 的转换方式与显存要求见 `docs/deployment.md`。

当前聊天链路默认由云端认知模型决策与表达，不依赖本地 LoRA；训练服务保留，
不影响已有训练资产。

### 图片与语音语义标注

```bash
PYTHONPATH=backend uv run python -m moonlightbox.media.cli \
  --project-id <项目 ID> --modality all
```

本机生成标注后在"回忆 → 数据质量"逐项审核；未经批准或越界的媒体不会进入对话。

### Agent Trace（Phoenix）

所有 Agent（Director、DayPlan、PersonaActor、PersonWorld 七栏目等）的执行都由统一
`AgentLoopController` 写入 Phoenix：模型请求、工具调用、LightRAG 检索、压缩、耗时、
错误码完整可查。`./scripts/start.sh` 会启动本机 Phoenix（`http://127.0.0.1:6006`）；
正文捕获可用 `MOONLIGHTBOX_PHOENIX_CAPTURE_CONTENT=false` 关闭，API key 强制脱敏。

## 文档

- `docs/deployment.md`：部署、数据目录、备份、分体 Docker、GPU 节点的完整说明
- `backend/moonlightbox/`：后端源码（imports / events / world / runtime_v1 / training …）
- 行为 Prompt：`backend/moonlightbox/runtime_v1/prompts/<agent>/`

# 月光宝盒 Moonlight Box

月光宝盒是一个单用户、本地优先的聊天记忆分析与数字人格分支探索系统。它可以导入聊天记录，识别关系转折点，训练本地 LoRA 人格模型，并从任意历史节点创建独立的平行对话。

## 主要能力

- 导入 wxecho CSV、JSON、TXT 聊天记录并预览、清洗、脱敏
- 通过事件分段、多信号变化检测和 LLM 复核生成可追溯的关键节点
- 使用带时间约束的 GraphRAG 和状态快照阻止未来记忆泄漏
- 在 Apple Silicon 上通过官方 MLX-LM 训练、加载 LoRA
- 创建并保存多条独立时间分支
- 使用真实回复盲测模型，并按质量门槛推荐版本
- 可选使用 OpenAI 兼容接口进行云端评测

## 环境要求

- macOS + Apple Silicon，建议至少 32GB 统一内存
- Python 3.12、[uv](https://docs.astral.sh/uv/)
- Node.js 20 及以上

## 本机启动

```bash
uv sync --extra mlx
uv run alembic -c backend/alembic.ini upgrade head
uv run uvicorn moonlightbox.api:app --app-dir backend --reload
```

另开终端启动前端：

```bash
cd frontend
npm install
npm run dev
```

打开 `http://localhost:5173`。

## 配置

复制 `.env.example` 为 `.env`，按需调整。云端评测默认启用，但只有配置密钥后才会发送请求；将 `MOONLIGHTBOX_CLOUD_EVALUATION_ENABLED=false` 可完全关闭。聊天内容、候选回复和真实回复会在云端评测时发送给所配置的服务商。

本地原始数据默认存放在 `data/`，模型存放在 `models/`。删除项目时，对应数据库记录、导入文件、向量索引和模型产物会一并删除。

## Docker

```bash
docker compose up --build
```

打开 `http://localhost:8080`。Docker Desktop 无法把 Apple Metal/MLX 能力透传给容器，因此 Docker 方式适合导入、审核和管理；训练与本地推理请使用上面的本机启动方式。

## 验证

```bash
uv run pytest backend/tests -q
uv run ruff check backend
uv run mypy backend/moonlightbox
cd frontend && npm test -- --run && npm run lint && npm run build
```

更详细的部署、数据目录和备份说明见 `docs/deployment.md`。

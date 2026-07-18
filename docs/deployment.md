# 部署与数据管理

## 推荐方式

需要训练或运行 MLX 模型时，应在 Apple Silicon 主机上直接运行后端。Docker Desktop 当前不能将 Metal 加速能力透传给 Linux 容器。

首次启动：

```bash
cp .env.example .env
uv sync --extra mlx
uv run alembic -c backend/alembic.ini upgrade head
uv run uvicorn moonlightbox.api:app --app-dir backend --host 127.0.0.1 --port 8000
```

前端开发服务器会把 `/api` 转发到 `127.0.0.1:8000`。

## 数据目录

- `data/moonlightbox.db`：项目、消息、节点、模型版本和分支元数据
- `data/projects/<项目 ID>/`：导入源文件及项目中间产物
- `data/chroma/projects/<项目 ID>/`：项目向量索引
- `models/projects/<项目 ID>/`：数据集、LoRA 和检查点

删除项目会删除以上项目级记录与目录。操作不可恢复，重要项目应先备份。

## 备份与恢复

停止后端后，同时备份 `data/` 和 `models/`。恢复时应将二者放回原路径，再执行：

```bash
uv run alembic -c backend/alembic.ini upgrade head
```

不要在后端写入数据库时直接复制 SQLite 文件。

## 云端评测

云端评测会发送用户消息、真实回复和候选回复。可通过以下配置完全关闭：

```dotenv
MOONLIGHTBOX_CLOUD_EVALUATION_ENABLED=false
```

若开启，请同时设置服务商地址、模型和 API 密钥。密钥只应存放在本地 `.env`，不要提交到版本库。

## Docker

Docker 方式会持久化宿主机的 `data/` 与 `models/`：

```bash
docker compose up --build
```

该方式不包含 MLX 依赖，适用于导入、审核、时间轴管理和使用外部生成服务的场景。

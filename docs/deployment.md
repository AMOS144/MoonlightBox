# 部署与数据管理

## 推荐方式

当前基线是 Windows 宿主机上的 WSL2 Linux。API、Worker、LightRAG Sidecar 和开发命令都在 WSL2 中运行；索引、SQLite、虚拟环境和模型文件应优先放在 WSL Linux 文件系统，而不是 `/mnt/c` 的高频 I/O 路径。

首次启动：

```bash
cp .env.example .env
uv sync
uv run alembic -c backend/alembic.ini upgrade head
uv run uvicorn moonlightbox.api:app --app-dir backend --host 127.0.0.1 --port 8000
```

前端开发服务器会把 `/api` 转发到 `127.0.0.1:8000`。

### LightRAG Sidecar

LightRAG 使用独立依赖环境，避免污染 MoonlightBox 主环境：

```bash
cd sidecars/lightrag_sidecar
uv sync
cp .env.example .env
uv run moonlightbox-lightrag-sidecar
```

Sidecar 默认监听 `127.0.0.1:9621`。在根目录 `.env` 启用
`MOONLIGHTBOX_LIGHTRAG_ENABLED=true`，并确保主服务与 Sidecar 使用相同的 bearer token。
实体抽取模型与嵌入模型分别由 `LIGHTRAG_SIDECAR_LLM_*` 和
`LIGHTRAG_SIDECAR_EMBEDDING_*` 配置。更换抽取模型、嵌入模型、嵌入维度或切块配置时，
同时递增 `MOONLIGHTBOX_LIGHTRAG_INDEX_REVISION`（例如 `person-world-v2`）；该值进入图配置指纹，
系统会创建新 workspace，避免新旧索引混用。

第一版的文本 Embedding 采用百炼华北 2（北京）`text-embedding-v4` 的默认 1024 维。
复制 `.env.example` 后必须把 `YOUR_WORKSPACE_ID` 换成真实业务空间 ID，并填写该地域的
百炼 API Key。`MOONLIGHTBOX_NODE_ANALYSIS_API_KEY` 是人物档案结构化编译模型的服务商
密钥；它不是系统生成的“档案密码”。使用 DeepSeek 编译时，可与
`LIGHTRAG_SIDECAR_LLM_API_KEY` 复用同一个 DeepSeek API Key。

生产 WSL2 可参考 `deploy/systemd/moonlightbox-lightrag-sidecar.service`。该 unit 管理 Linux
进程，但不会自行让 WSL 永久保持唤醒；Windows 登录/开机任务仍需调用 `wsl.exe` 启动目标服务。

### Agent Runtime v1

Runtime v1 依赖最新成功的 `PersonWorldProfile`。当前 LightRAG Sidecar 没有按时间范围或
图版本查询能力，因此创建 Runtime 时会冻结**最新完成图谱**，并把该图谱最后一个 Bundle 的
结束时间作为 `OriginWorldSnapshot.cutoff_at`；它不是历史节点的伪时间切片。

前端只把用户消息提交给 `/runtime/messages`。认知 Worker（`MOONLIGHTBOX_WORKER_ROLE=cognition`
或 `all`）领取队列事件和到期 Wakeup 后才调用模型。Director 使用未挂 adapter 的基座模型做
严格 JSON 决策；PersonaActor 才使用现有 LoRA。两次推理由 Linux GPU 上同一个串行
`persona_runtime` 进程执行，所以不会同时占用两份模型内存。模型或 adapter 加载失败时，Worker
保留队列事件并记录失败，不会提交部分 LifeState 或重复公开消息。

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

## 真实项目 V2 节点验收

以下流程只读取 `data/moonlightbox.db`，不会修改项目、消息、分析运行或节点。
导出步骤只会在指定位置新建本地 JSON/Markdown 文件，也不会上传审核包。
CLI 会先确认数据库路径已存在且是普通文件，再使用 SQLite URI
`mode=ro` 打开源库，并通过 SQLite backup API 自动生成一致性临时快照。ORM 只读取
`immutable=1` 的临时快照，因此服务运行中仍可读取尚在活跃 WAL 内的已提交数据。
工具不会调用应用数据库初始化、WAL 配置、checkpoint 或迁移，也不会修改源数据库及其
`-wal`、`-shm` 文件；退出时会关闭连接并删除临时快照。错误路径不会被创建。

### 1. 配置节点分析

在未提交的本地 `.env` 中配置 OpenAI 兼容服务。以下示例使用 DeepSeek V4 Flash；
密钥必须由操作者在本机填写，
不要把真实值粘贴到文档、命令记录或版本库：

```dotenv
MOONLIGHTBOX_NODE_ANALYSIS_ENABLED=true
MOONLIGHTBOX_NODE_ANALYSIS_ENDPOINT=https://api.deepseek.com/chat/completions
MOONLIGHTBOX_NODE_ANALYSIS_MODEL=deepseek-v4-flash
MOONLIGHTBOX_NODE_ANALYSIS_API_KEY=${DEEPSEEK_API_KEY}
MOONLIGHTBOX_NODE_ANALYSIS_RESPONSE_FORMAT=json_object
MOONLIGHTBOX_NODE_ANALYSIS_THINKING_MODE=disabled
MOONLIGHTBOX_NODE_ANALYSIS_MAX_OUTPUT_TOKENS=8192
MOONLIGHTBOX_NODE_ANALYSIS_TIMEOUT_SECONDS=60
MOONLIGHTBOX_NODE_ANALYSIS_MAX_RETRIES=3
MOONLIGHTBOX_NODE_ANALYSIS_BACKOFF_SECONDS=0.5
MOONLIGHTBOX_NODE_ANALYSIS_MAX_BACKOFF_SECONDS=60
MOONLIGHTBOX_NODE_ANALYSIS_MAX_RETRY_AFTER_SECONDS=3600
```

结构化输出默认模式是 `json_schema`，适用于支持 OpenAI strict schema 的服务；DeepSeek
官方接口只发送 `{"type":"json_object"}`，因此必须配置为 `json_object`。此模式会把精简
JSON schema 与最小合法 JSON 示例仅追加到 system prompt 一次，并计入 system/user 完整
提示词预算；响应仍需通过本地 Pydantic 校验。节点分析默认
关闭思考模式，避免 V4 的默认 thinking 干扰结构化输出；如将 thinking mode 设为
`default`，则不会发送 `thinking` 字段。输出上限必须是 1 到 65536 之间的正整数。

未配置 `MOONLIGHTBOX_NODE_ANALYSIS_API_KEY` 时，Worker 会让任务安全失败且不会发起
网络请求。真实精确率必须来自真实服务完成分析后的人工判断，不能用测试 Fake 或黄金样例代替。

### 2. 启动并等待分析

```bash
./scripts/start.sh
```

在导入页确认当前聊天后，记录返回的 Job ID。可用只读接口等待任务结束：

```bash
curl --fail --silent http://127.0.0.1:8000/api/jobs/<JOB_ID>
```

只有 Job 和对应 `hybrid-v2` 分析运行均为 `succeeded` 才能验收。可通过 SQLite
只读模式取得当前真实项目与最近运行的 ID：

```bash
sqlite3 -readonly data/moonlightbox.db \
  "SELECT id, name FROM projects ORDER BY created_at DESC;"
sqlite3 -readonly data/moonlightbox.db \
  "SELECT id, project_id, import_id, status, completed_at
   FROM analysis_runs
   WHERE analysis_version = 'hybrid-v2'
   ORDER BY created_at DESC;"
```

### 3. 导出并人工审核

JSON 是可回填的审核包，Markdown 是便于阅读的同内容副本。两者最多包含 25 个节点，
只含节点 ID、中英文类型、发布时前后状态、候选分数、版本和安全证据摘要，不含原始载荷、
XML 协议内容或密钥。节点内容来自指定运行绑定的发布修订快照，不受后续节点编辑影响。
每个节点和导出清单都带 SHA256 完整性摘要，除 `verdict` 外不得修改任何字段、顺序或节点数量。
清单时间固定绑定该分析运行的 `completed_at`，同一运行重复导出的内容保持稳定。

```bash
PYTHONPATH=backend uv run python -m moonlightbox.evaluation.node_acceptance \
  --database-url sqlite:///data/moonlightbox.db \
  export --project <PROJECT_ID> --run <RUN_ID> \
  --format json --output node-review.json

PYTHONPATH=backend uv run python -m moonlightbox.evaluation.node_acceptance \
  --database-url sqlite:///data/moonlightbox.db \
  export --project <PROJECT_ID> --run <RUN_ID> \
  --format markdown --output node-review.md
```

输出文件使用独占创建：已有文件不会被覆盖，符号链接会被拒绝，写入失败会清理部分文件。
如需重新导出，应明确选择新的文件名，或由操作者确认后自行删除旧文件。

逐个检查原始聊天上下文，再把 `node-review.json` 中每个节点的 `verdict`
从 `pending` 改成以下值之一：

- `accepted`：节点类型与证据正确，计为接受。
- `rejected`：误报、类型错误或证据不成立，计为拒绝。

`verdict` 是独立审核字段，与 `EventNode.status` 生命周期状态无关。不能只审核看起来正确的
节点；任何 `pending` 都会使结果保持待审核。节点删增、重复、重排或内容篡改会直接拒绝整个包。

### 4. 运行精确率门槛

```bash
PYTHONPATH=backend uv run python -m moonlightbox.evaluation.node_acceptance \
  --database-url sqlite:///data/moonlightbox.db \
  evaluate --project <PROJECT_ID> --run <RUN_ID> \
  --input node-review.json --threshold 0.8
```

工具把该运行发布的全部节点作为 `predicted`，把 `verdict=accepted` 的节点作为 `gold`，
并调用统一的 `evaluate_nodes` 计算 `precision = TP / (TP + FP)`。阈值只取命令行参数，
默认 `0.8`，不读取可编辑审核包中的值。全部节点审核完成后，精确率不低于阈值才返回
`pass`；低于阈值返回 `fail`；未审完或运行没有节点返回 `pending`。

这是一项人工节点精确率审核，不是完整漏标语料评测；输出不会把 recall/F1 当作真实全量
召回结论。进程退出码如下：

- `0`：`pass`，或成功导出审核包。
- `1`：`fail`。
- `2`：`pending`。
- `3`：输入、完整性、数据库或文件系统错误。

## Docker

Docker 方式包含一次性 migration、API、独立事件分析 Worker 和前端四个服务。先复制
配置文件；如需 V2 节点分析，请在本地 `.env` 中配置本页前述的
`MOONLIGHTBOX_NODE_ANALYSIS_*` 变量：

```bash
cp .env.example .env
docker compose up --build
```

Compose 先执行 `alembic -c backend/alembic.ini upgrade head`，migration 成功后才会
并行启动 API 和 Worker；前端在 API 健康检查通过后启动。迁移失败时不会启动 API 或
Worker，因此不会用 `create_all` 代替旧数据库升级。

API、migration 和 Worker 使用同一个后端镜像、同一份 `.env`，并共同挂载宿主机的
`data/` 与 `models/`。SQLite 数据库位于 `data/moonlightbox.db`，只允许同一宿主机上的
API 与 Worker 两个进程共享；不要把该目录挂载到多台主机。备份时仍需停止写入进程并同时
备份两个目录。

真实 API key 只应写入未提交的 `.env`，Compose 会在启动容器时注入，Dockerfile 不会把
它构建进镜像。镜像安装 PyTorch、Transformers、PEFT 与 bitsandbytes；GPU 只交给
`persona-runtime`，其余服务通过受认证 HTTP 调用它。

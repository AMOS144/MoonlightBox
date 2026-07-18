# 月光宝盒完整产品实施计划

> **供智能体执行：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，严格按任务逐项实施。所有步骤使用复选框跟踪。

**目标：** 在 Apple Silicon 32GB 基线上交付可导入 wxecho 聊天、发现并审核关键节点、完成 MLX-LM LoRA 微调、从历史节点创建分支对话的单用户本地 Web 应用。

**架构：** 采用 React + TypeScript 前端和 FastAPI 模块化单体后端。SQLite 保存业务数据与时序事件图，ChromaDB 保存向量索引，独立 Python 工作进程执行分析、训练和评测任务，MLX-LM 通过适配器接入。

**技术栈：** Python 3.12、uv、FastAPI、Pydantic、SQLAlchemy、Alembic、SQLite、ChromaDB、MLX-LM、pytest、React、TypeScript、Vite、TanStack Query、Vitest、Playwright。

**上游规格：** `docs/superpowers/specs/2026-07-18-moonlight-box-design.md`

---

## 执行规则与阶段门禁

本计划按六个里程碑执行。每个任务都遵循“失败测试 → 最小实现 → 测试通过 → 提交”的顺序。任何大型模型测试都使用 `@pytest.mark.model` 隔离，普通测试不得自动下载模型。

提交步骤仅供执行阶段使用；当前编写计划阶段不创建提交。提交前必须确认 `wxecho/`、模型和聊天数据没有进入暂存区。

里程碑门禁：

1. 工程基础通过后，才能实现导入。
2. 导入和合成测试样本通过后，才能实现事件分析。
3. 黄金节点评测链路通过后，才能实现训练。
4. 数据集防泄漏测试通过后，才能执行真实 LoRA。
5. 双重时间过滤测试通过后，才能开放分支对话。
6. 全部本地核心流程通过后，才能接入云端评测。

## 文件结构

```text
MoonlightBox/
├── backend/
│   ├── alembic/
│   ├── moonlightbox/
│   │   ├── api.py
│   │   ├── config.py
│   │   ├── db.py
│   │   ├── worker.py
│   │   ├── projects/
│   │   ├── imports/
│   │   ├── events/
│   │   ├── training/
│   │   ├── branches/
│   │   └── evaluation/
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── app/
│   │   └── features/
│   └── tests/
├── tests/
│   └── fixtures/
├── data/                 # 运行时目录，不提交
├── models/               # 模型与 LoRA，不提交
└── docs/
```

每个业务目录只暴露 `router.py`、`service.py` 和必要的公共类型；解析、算法和提供商实现放在同目录的独立文件中。

---

## 里程碑一：工程基础

### 任务 1：仓库安全与 Python 工程布局

**文件：**
- 修改：`.gitignore`
- 修改：`pyproject.toml`
- 删除：`main.py`
- 创建：`backend/moonlightbox/__init__.py`
- 创建：`backend/tests/__init__.py`
- 创建：`tests/fixtures/wxecho_sample.csv`

- [ ] **步骤 1：先写仓库安全检查**

创建 `backend/tests/test_repository_safety.py`：

```python
from pathlib import Path


def test_private_paths_are_ignored() -> None:
    rules = Path(".gitignore").read_text(encoding="utf-8")
    for path in ("wxecho/", ".superpowers/", "data/", "models/", ".env"):
        assert path in rules
```

- [ ] **步骤 2：运行测试并确认失败**

运行：`uv run pytest backend/tests/test_repository_safety.py -v`

预期：FAIL，指出 `.gitignore` 缺少私有路径。

- [ ] **步骤 3：补齐工程依赖与忽略规则**

运行：

```bash
uv add fastapi "uvicorn[standard]" pydantic-settings sqlalchemy alembic chromadb httpx
uv add --dev pytest pytest-cov ruff mypy
```

在 `.gitignore` 追加：

```gitignore
wxecho/
.superpowers/
data/
models/
.env
*.log
frontend/node_modules/
frontend/dist/
```

将项目包路径配置为 `backend`，pytest 测试路径配置为 `backend/tests`，Ruff 目标版本配置为 Python 3.12。合成 CSV 只能包含虚构人物和虚构消息。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/test_repository_safety.py -v && uv run ruff check backend`

预期：全部通过。

- [ ] **步骤 5：提交**

```bash
git add .gitignore pyproject.toml uv.lock backend tests
git commit -m "chore: establish safe project foundation"
```

### 任务 2：配置、数据库与健康检查

**文件：**
- 创建：`backend/moonlightbox/config.py`
- 创建：`backend/moonlightbox/db.py`
- 创建：`backend/moonlightbox/api.py`
- 创建：`backend/tests/test_health.py`
- 创建：`backend/tests/conftest.py`

- [ ] **步骤 1：写失败测试**

```python
def test_health_returns_runtime_status(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "mlx_available": False,
    }
```

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/test_health.py -v`

预期：FAIL，应用工厂或路由尚不存在。

- [ ] **步骤 3：实现最小应用工厂**

公共契约：

```python
class Settings(BaseSettings):
    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/moonlightbox.db"
    chroma_dir: Path = Path("data/chroma")
    model_dir: Path = Path("models")


def create_app(settings: Settings | None = None) -> FastAPI:
    ...
```

测试夹具必须使用临时 SQLite 文件和临时 Chroma 目录。`/api/health` 实际执行 `SELECT 1`，MLX 可用性通过独立探测函数返回，不在导入模块时加载模型。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/test_health.py -v && uv run mypy backend/moonlightbox`

预期：健康检查与类型检查通过。

- [ ] **步骤 5：提交**

```bash
git add backend
git commit -m "feat: add application and database foundation"
```

### 任务 3：项目管理与数据库迁移

**文件：**
- 创建：`backend/moonlightbox/projects/models.py`
- 创建：`backend/moonlightbox/projects/schemas.py`
- 创建：`backend/moonlightbox/projects/service.py`
- 创建：`backend/moonlightbox/projects/router.py`
- 创建：`backend/tests/projects/test_projects_api.py`
- 创建：`backend/alembic.ini`
- 创建：`backend/alembic/`

- [ ] **步骤 1：写项目 CRUD 失败测试**

```python
def test_create_and_read_project(client):
    created = client.post("/api/projects", json={"name": "测试项目"})
    assert created.status_code == 201
    project_id = created.json()["id"]

    loaded = client.get(f"/api/projects/{project_id}")
    assert loaded.json()["name"] == "测试项目"
    assert loaded.json()["status"] == "created"
```

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/projects/test_projects_api.py -v`

预期：FAIL，路由返回 404。

- [ ] **步骤 3：实现项目模型与迁移**

`Project` 字段固定为 `id: UUID`、`name`、`status`、`created_at`、`updated_at`。服务层提供：

```python
class ProjectService:
    def create(self, name: str) -> Project: ...
    def get(self, project_id: UUID) -> Project: ...
    def list(self) -> list[Project]: ...
    def delete(self, project_id: UUID) -> None: ...
```

删除操作先调用资源清理接口，再删除数据库记录。创建首个 Alembic 迁移并在测试中执行升级。

- [ ] **步骤 4：验证**

运行：`uv run alembic -c backend/alembic.ini upgrade head && uv run pytest backend/tests/projects -v`

预期：迁移和 CRUD 测试通过。

- [ ] **步骤 5：提交**

```bash
git add backend
git commit -m "feat: add local project management"
```

### 任务 4：持久化后台任务框架

**文件：**
- 创建：`backend/moonlightbox/jobs/models.py`
- 创建：`backend/moonlightbox/jobs/service.py`
- 创建：`backend/moonlightbox/jobs/registry.py`
- 创建：`backend/moonlightbox/jobs/router.py`
- 创建：`backend/moonlightbox/worker.py`
- 创建：`backend/tests/jobs/test_job_lifecycle.py`

- [ ] **步骤 1：写状态机失败测试**

```python
def test_interrupted_job_can_resume(job_service):
    job = job_service.enqueue("sample", {"value": 1})
    running = job_service.start(job.id)
    job_service.checkpoint(running.id, {"offset": 12})
    interrupted = job_service.interrupt(running.id, "worker stopped")

    resumed = job_service.resume(interrupted.id)
    assert resumed.status == "queued"
    assert resumed.checkpoint == {"offset": 12}
```

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/jobs/test_job_lifecycle.py -v`

预期：FAIL，任务服务尚不存在。

- [ ] **步骤 3：实现状态机和轮询工作进程**

允许的转换固定为：

```python
TRANSITIONS = {
    "queued": {"running", "cancelled"},
    "running": {"succeeded", "failed", "cancelled", "interrupted"},
    "interrupted": {"queued", "cancelled"},
    "failed": {"queued"},
}
```

`Job` 保存 `kind`、`payload`、`status`、`progress`、`checkpoint`、`error_code`、`error_message` 和时间戳。工作进程通过注册表按 `kind` 查找处理器；无法识别的任务标记失败，不执行动态导入。提供 `GET /api/jobs/{id}`、`POST /api/jobs/{id}/cancel` 和 `POST /api/jobs/{id}/resume`，前端以短轮询展示进度，不在首版引入 WebSocket。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/jobs -v`

预期：状态转换、取消、恢复和未知任务测试全部通过。

- [ ] **步骤 5：提交**

```bash
git add backend
git commit -m "feat: add recoverable background jobs"
```

### 任务 5：React 应用外壳与项目页面

**文件：**
- 创建：`frontend/`
- 创建：`frontend/src/app/router.tsx`
- 创建：`frontend/src/app/AppShell.tsx`
- 创建：`frontend/src/api/client.ts`
- 创建：`frontend/src/features/projects/ProjectListPage.tsx`
- 创建：`frontend/src/features/projects/ProjectCreatePage.tsx`
- 创建：`frontend/src/features/projects/ProjectLayout.tsx`
- 创建：`frontend/src/features/projects/ProjectListPage.test.tsx`

- [ ] **步骤 1：初始化并写失败测试**

运行：`npm create vite@latest frontend -- --template react-ts`

安装：`cd frontend && npm install @tanstack/react-query react-router-dom && npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom`

测试断言 API 返回项目后显示项目名，并存在“创建项目”入口。

- [ ] **步骤 2：确认失败**

运行：`cd frontend && npm test -- --run src/features/projects/ProjectListPage.test.tsx`

预期：FAIL，页面尚不存在。

- [ ] **步骤 3：实现应用外壳**

路由固定为：

```tsx
const routes = [
  { path: "/", element: <ProjectListPage /> },
  { path: "/projects/new", element: <ProjectCreatePage /> },
  { path: "/projects/:projectId", element: <ProjectLayout /> },
];
```

`ProjectLayout` 提供“概览、数据、节点、模型、时间轴、分支、评测”导航。API 客户端统一解析后端 `{code, message, details}` 错误结构。

- [ ] **步骤 4：验证**

运行：`cd frontend && npm test -- --run && npm run build`

预期：测试和 TypeScript 构建通过。

- [ ] **步骤 5：提交**

```bash
git add frontend
git commit -m "feat: add project web application shell"
```

---

## 里程碑二：wxecho 导入与数据质量

### 任务 6：统一消息模型和 CSV 适配器

**文件：**
- 创建：`backend/moonlightbox/imports/types.py`
- 创建：`backend/moonlightbox/imports/base.py`
- 创建：`backend/moonlightbox/imports/wxecho_csv.py`
- 创建：`backend/moonlightbox/imports/models.py`
- 创建：`backend/tests/imports/test_wxecho_csv.py`

- [ ] **步骤 1：写真实格式的合成测试**

```python
def test_parse_wxecho_csv(sample_csv):
    result = WxechoCsvImporter().parse(sample_csv)
    assert result.messages[0].sender == "甲"
    assert result.messages[0].kind == MessageKind.TEXT
    assert result.messages[0].content == "今晚有空吗"
    assert result.errors == []
```

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/imports/test_wxecho_csv.py -v`

预期：FAIL，导入器尚不存在。

- [ ] **步骤 3：实现统一契约**

```python
@dataclass(frozen=True)
class ImportedMessage:
    source_id: str
    timestamp: datetime
    sender: str
    kind: MessageKind
    content: str
    raw: dict[str, object]


class ChatImporter(Protocol):
    def sniff(self, path: Path) -> float: ...
    def parse(self, path: Path) -> ImportResult: ...
```

CSV 使用 `utf-8-sig`，字段为“时间、发送者、类型、内容”；错误必须包含行号和错误码，不能静默跳过。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/imports/test_wxecho_csv.py -v`

预期：文本、媒体占位符、逗号、空行和错误时间格式测试通过。

- [ ] **步骤 5：提交**

```bash
git add backend tests/fixtures
git commit -m "feat: parse wxecho csv exports"
```

### 任务 7：JSON、TXT 适配器与导入预览 API

**文件：**
- 创建：`backend/moonlightbox/imports/wxecho_json.py`
- 创建：`backend/moonlightbox/imports/wxecho_txt.py`
- 创建：`backend/moonlightbox/imports/service.py`
- 创建：`backend/moonlightbox/imports/router.py`
- 创建：`backend/tests/imports/test_preview_api.py`

- [ ] **步骤 1：写预览失败测试**

测试上传合成 JSON 后返回 `message_count`、`participants`、`time_range`、`kind_counts`、`sample_messages` 和 `errors`，但数据库中消息数仍为零。

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/imports/test_preview_api.py -v`

预期：FAIL，预览路由不存在。

- [ ] **步骤 3：实现三格式自动识别和两阶段导入**

接口：

```text
POST /api/projects/{id}/imports/preview
POST /api/projects/{id}/imports/{preview_id}/confirm
```

预览文件保存在项目临时目录，确认时要求提交 `self_participant` 和 `target_participant`。确认操作使用事务写入 `ImportSource`、`Participant`、`Message`；重复确认返回现有导入结果。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/imports -v`

预期：三种格式产生等价的统一消息，预览不落业务表，确认可幂等执行。

- [ ] **步骤 5：提交**

```bash
git add backend
git commit -m "feat: preview and confirm wxecho imports"
```

### 任务 8：清洗、脱敏和数据审核界面

**文件：**
- 创建：`backend/moonlightbox/imports/cleaning.py`
- 创建：`backend/moonlightbox/imports/redaction.py`
- 创建：`backend/tests/imports/test_cleaning.py`
- 创建：`frontend/src/features/data/ImportWizard.tsx`
- 创建：`frontend/src/features/data/DataQualityPage.tsx`
- 创建：`frontend/src/features/data/ImportWizard.test.tsx`

- [ ] **步骤 1：写短消息合并测试**

```python
def test_consecutive_short_replies_preserve_style():
    cleaned = clean_messages([
        message("猜字顶", second=0),
        message("的", second=6),
    ])
    assert cleaned.training_units[0].content == "猜字顶\n的"
```

该测试用于防止按“单字”粗暴删除。

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/imports/test_cleaning.py -v`

预期：FAIL，清洗函数不存在。

- [ ] **步骤 3：实现可解释清洗**

每条处理结果保存 `kept`、`reason_code` 和源消息 ID。默认规则处理系统消息、空内容、重复消息、媒体占位符和相邻短回复。脱敏器识别手机号、邮箱、身份证样式和用户自定义词表，生成派生文本，不覆盖原文。

- [ ] **步骤 4：实现并验证前端向导**

运行：`cd frontend && npm test -- --run src/features/data/ImportWizard.test.tsx`

预期：用户必须完成文件预览和角色映射才能确认；错误行和清洗统计可展开查看。

- [ ] **步骤 5：提交**

```bash
git add backend frontend
git commit -m "feat: add explainable chat cleaning workflow"
```

---

## 里程碑三：时序事件图谱

### 任务 9：事件分段

**文件：**
- 创建：`backend/moonlightbox/events/models.py`
- 创建：`backend/moonlightbox/events/segmentation.py`
- 创建：`backend/tests/events/test_segmentation.py`

- [ ] **步骤 1：写边界测试**

覆盖长时间间隔、主题突变和情绪突变。核心断言：

```python
episodes = segment(messages, gap=timedelta(hours=6))
assert [episode.message_ids for episode in episodes] == [
    ["m1", "m2"],
    ["m3", "m4"],
]
```

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/events/test_segmentation.py -v`

预期：FAIL，分段函数不存在。

- [ ] **步骤 3：实现双通道分段**

第一阶段以时间间隔生成硬边界；第二阶段使用相邻窗口嵌入距离和意图标签生成软边界。所有边界保存 `reason_codes` 和分数，默认参数由版本化配置对象承载。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/events/test_segmentation.py -v`

预期：固定输入产生稳定片段，任何消息恰好属于一个片段。

- [ ] **步骤 5：提交**

```bash
git add backend
git commit -m "feat: segment chats into temporal episodes"
```

### 任务 10：变化点候选与排名

**文件：**
- 创建：`backend/moonlightbox/events/change_points.py`
- 创建：`backend/moonlightbox/events/scoring.py`
- 创建：`backend/tests/events/test_change_points.py`

- [ ] **步骤 1：写综合评分失败测试**

```python
signals = ChangeSignals(
    topic_delta=0.8,
    emotion_delta=0.9,
    intent_delta=0.7,
    response_gap_delta=0.4,
    persistence=0.8,
)
assert score_change(signals) > 0.75
```

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/events/test_change_points.py -v`

预期：FAIL，评分器不存在。

- [ ] **步骤 3：实现候选生成**

评分权重使用可序列化配置，输出 `CandidateBoundary`，包含左右片段、各信号分数、总分和证据消息 ID。候选生成只使用局部窗口，不调用全量 LLM。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/events/test_change_points.py -v`

预期：强变化入选、单一噪声不入选、相邻候选按窗口合并。

- [ ] **步骤 5：提交**

```bash
git add backend
git commit -m "feat: detect candidate relationship turning points"
```

### 任务 11：LLM 复核、节点修订和黄金评测

**文件：**
- 创建：`backend/moonlightbox/events/reviewer.py`
- 创建：`backend/moonlightbox/events/schemas.py`
- 创建：`backend/moonlightbox/events/service.py`
- 创建：`backend/moonlightbox/events/router.py`
- 创建：`backend/moonlightbox/events/golden.py`
- 创建：`backend/tests/events/test_reviewer.py`
- 创建：`backend/tests/events/test_revisions.py`

- [ ] **步骤 1：写结构化复核失败测试**

假 LLM 返回 JSON 后，断言节点包含 `type`、`start_message_id`、`end_message_id`、`before_state`、`after_state`、`importance`、`reason` 和 `evidence_ids`。证据 ID 不在输入窗口时必须拒绝结果。

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/events/test_reviewer.py -v`

预期：FAIL，复核器不存在。

- [ ] **步骤 3：实现复核接口和修订历史**

```python
class EventReviewer(Protocol):
    def review(
        self,
        candidate: CandidateBoundary,
        context: Sequence[Message],
    ) -> ReviewedEvent | None: ...
```

审核 API 支持接受、删除、合并、拆分、改标签和补充节点。每次操作创建 `AnalysisRevision`，不覆盖先前版本。

- [ ] **步骤 4：实现黄金数据评测**

评测输出候选召回、审核前准确率、范围重合和证据正确率。黄金数据只保存合成样本；真实聊天标注保存在 `data/`。

运行：`uv run pytest backend/tests/events -v`

预期：结构校验、修订历史和指标计算通过。

- [ ] **步骤 5：提交**

```bash
git add backend
git commit -m "feat: review and curate temporal event nodes"
```

### 任务 12：时序事件图与 GraphRAG 检索

**文件：**
- 创建：`backend/moonlightbox/events/graph.py`
- 创建：`backend/moonlightbox/events/vector_store.py`
- 创建：`backend/moonlightbox/events/retrieval.py`
- 创建：`backend/tests/events/test_graph_retrieval.py`

- [ ] **步骤 1：写时态检索失败测试**

```python
results = retriever.search(
    project_id=project.id,
    query="为什么开始冷战",
    before=origin_time,
)
assert results
assert all(item.timestamp <= origin_time for item in results)
assert all(item.evidence_ids for item in results)
```

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/events/test_graph_retrieval.py -v`

预期：FAIL，检索器不存在。

- [ ] **步骤 3：实现图边和混合检索**

SQLite 保存 `chronological`、`causal`、`same_topic`、`state_continuation` 和 `evidence` 边。Chroma 元数据至少包含 `project_id`、`event_id` 和 Unix 时间戳。检索先执行项目与时间过滤，再进行向量召回和图邻居扩展，最后返回带证据的排序结果。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/events/test_graph_retrieval.py -v`

预期：未来节点永不返回，不同项目数据不串线，删除项目后向量集合同步清理。

- [ ] **步骤 5：提交**

```bash
git add backend
git commit -m "feat: add temporal graph retrieval"
```

---

## 里程碑四：人格微调与评测

### 任务 13：训练数据集构建与防泄漏

**文件：**
- 创建：`backend/moonlightbox/training/dataset.py`
- 创建：`backend/moonlightbox/training/models.py`
- 创建：`backend/tests/training/test_dataset.py`

- [ ] **步骤 1：写时间切分失败测试**

断言同一对话片段不能跨训练、验证和盲测集，且盲测目标回复不出现在训练上下文中。

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/training/test_dataset.py -v`

预期：FAIL，数据集构建器不存在。

- [ ] **步骤 3：实现 MLX-LM JSONL 数据集**

每行使用 `messages` 格式，最后一条 assistant 消息必须来自目标人物。按 Episode 的时间顺序切分 80%/10%/10%，不足 100 个有效目标回复时拒绝训练并返回 `insufficient_training_samples`。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/training/test_dataset.py -v`

预期：版本哈希稳定、切分无交叉、短回复组合被保留、原始敏感字段不进入派生数据。

- [ ] **步骤 5：提交**

```bash
git add backend
git commit -m "feat: build versioned persona datasets"
```

### 任务 14：MLX-LM 训练适配器

**文件：**
- 创建：`backend/moonlightbox/training/engine.py`
- 创建：`backend/moonlightbox/training/mlx_engine.py`
- 创建：`backend/moonlightbox/training/service.py`
- 创建：`backend/moonlightbox/training/router.py`
- 创建：`backend/tests/training/test_engine.py`
- 创建：`backend/tests/model/test_mlx_smoke.py`
- 创建：`frontend/src/features/models/ModelPage.tsx`
- 创建：`frontend/src/features/models/ModelPage.test.tsx`

- [ ] **步骤 1：写命令构建与资源检查失败测试**

断言 32GB 基线允许 7B/8B 4-bit 模型预设，非 Apple Silicon 返回 `mlx_platform_unsupported`，磁盘不足返回 `insufficient_disk_space`。

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/training/test_engine.py -v`

预期：FAIL，训练引擎不存在。

- [ ] **步骤 3：实现适配器**

```python
class TrainingEngine(Protocol):
    def validate(self, request: TrainingRequest) -> ValidationReport: ...
    def train(self, request: TrainingRequest, progress: ProgressSink) -> Artifact: ...
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
```

MLX 实现通过参数数组启动 `python -m mlx_lm.lora`，不拼接 shell 字符串。解析训练日志更新任务进度；只有进程成功且适配器文件完整时才登记 `ModelVersion`。模型页面展示数据集版本、基础模型、训练预设、资源检查、任务进度、失败原因和已登记模型版本。

- [ ] **步骤 4：验证**

普通测试：`uv run pytest backend/tests/training/test_engine.py -v`

Apple Silicon 冒烟测试：`uv run pytest backend/tests/model/test_mlx_smoke.py -v -m model`

前端测试：`cd frontend && npm test -- --run src/features/models/ModelPage.test.tsx`

预期：普通测试使用假进程通过；模型测试使用明确配置的小模型完成少量训练步和一次生成；前端能展示资源检查和任务状态。

- [ ] **步骤 5：提交**

```bash
git add backend frontend
git commit -m "feat: integrate mlx persona training"
```

### 任务 15：盲测与模型版本推荐

**文件：**
- 创建：`backend/moonlightbox/evaluation/persona.py`
- 创建：`backend/moonlightbox/evaluation/service.py`
- 创建：`backend/moonlightbox/evaluation/router.py`
- 创建：`backend/tests/evaluation/test_blind_evaluation.py`
- 创建：`frontend/src/features/evaluation/EvaluationPage.tsx`
- 创建：`frontend/src/features/evaluation/EvaluationPage.test.tsx`

- [ ] **步骤 1：写盲测失败测试**

测试随机化候选顺序，评审数据不暴露“基础、RAG、LoRA”标签，并断言只有 LoRA 综合分高于基础模型时才设置 `recommended=True`。

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/evaluation/test_blind_evaluation.py -v`

预期：FAIL，评测服务不存在。

- [ ] **步骤 3：实现评测**

评分维度固定为用词、句长、语气、标点、口头禅和上下文适配度，每项 1–5 分。保存评审者类型、盲化顺序、候选模型版本和聚合结果。评测页面一次只展示盲化后的候选回复，提交评分后才显示来源和聚合结果。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/evaluation/test_blind_evaluation.py -v && cd frontend && npm test -- --run src/features/evaluation/EvaluationPage.test.tsx`

预期：随机化可由种子复现，推荐逻辑和统计聚合通过。

- [ ] **步骤 5：提交**

```bash
git add backend frontend
git commit -m "feat: evaluate persona model fidelity"
```

---

## 里程碑五：时间回溯与分支对话

### 任务 16：状态快照与双重时间过滤

**文件：**
- 创建：`backend/moonlightbox/branches/models.py`
- 创建：`backend/moonlightbox/branches/snapshot.py`
- 创建：`backend/moonlightbox/branches/retrieval.py`
- 创建：`backend/tests/branches/test_temporal_cutoff.py`

- [ ] **步骤 1：写未来泄漏失败测试**

在 SQLite 和 Chroma 中分别插入起点前后内容，断言两层过滤都排除未来数据；临时禁用任意一层时测试仍能暴露配置错误。

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/branches/test_temporal_cutoff.py -v`

预期：FAIL，分支检索器不存在。

- [ ] **步骤 3：实现状态快照**

`StateSnapshot` 保存起点事件、关系状态、人物情绪、未解决主题、可见事件 ID 和生成版本。业务 SQL 强制 `Message.timestamp <= origin_time`，Chroma 查询强制 `timestamp <= origin_timestamp`，两层结果交集后才能进入提示词。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/branches/test_temporal_cutoff.py -v`

预期：未来消息、未来事件和未来摘要全部不可见。

- [ ] **步骤 5：提交**

```bash
git add backend
git commit -m "feat: enforce temporal memory boundaries"
```

### 任务 17：分支创建、生成和回看 API

**文件：**
- 创建：`backend/moonlightbox/branches/prompt.py`
- 创建：`backend/moonlightbox/branches/service.py`
- 创建：`backend/moonlightbox/branches/router.py`
- 创建：`backend/tests/branches/test_branch_api.py`

- [ ] **步骤 1：写独立分支失败测试**

从同一事件创建 A、B 两条分支，分别追加消息，断言 A 看不到 B 的消息，原始 `Message` 表数量不变。

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/branches/test_branch_api.py -v`

预期：FAIL，分支 API 不存在。

- [ ] **步骤 3：实现 API**

```text
POST /api/projects/{id}/events/{event_id}/branches
GET  /api/projects/{id}/branches
GET  /api/projects/{id}/branches/{branch_id}
POST /api/projects/{id}/branches/{branch_id}/messages
```

创建分支时固定起点、父分支、模型版本、数据集版本、状态快照和生成参数。生成提示词依次包含风格配置、状态快照、时态检索证据和当前分支消息。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/branches -v`

预期：分支隔离、幂等创建、生成失败回滚和回看顺序测试通过。

- [ ] **步骤 5：提交**

```bash
git add backend
git commit -m "feat: add independent timeline branches"
```

### 任务 18：节点审核、时间轴和分支前端

**文件：**
- 创建：`frontend/src/features/events/NodeReviewPage.tsx`
- 创建：`frontend/src/features/timeline/TimelinePage.tsx`
- 创建：`frontend/src/features/branches/BranchPage.tsx`
- 创建：`frontend/src/features/branches/BranchList.tsx`
- 创建：`frontend/src/features/timeline/TimelinePage.test.tsx`
- 创建：`frontend/src/features/branches/BranchPage.test.tsx`

- [ ] **步骤 1：写交互失败测试**

测试节点筛选、证据展开、标签修订、“从这里穿越”、发送消息、另开分支和历史分支切换。不得出现“分支对比”入口。

- [ ] **步骤 2：确认失败**

运行：`cd frontend && npm test -- --run src/features/timeline src/features/branches`

预期：FAIL，页面尚不存在。

- [ ] **步骤 3：实现页面**

时间轴按日期纵向排列，筛选项为类型、主题、情绪和冲突等级。分支页顶部固定显示起点和状态快照，侧栏仅列出分支，不并排渲染两个分支。

- [ ] **步骤 4：验证**

运行：`cd frontend && npm test -- --run && npm run build`

预期：组件测试和生产构建通过。

- [ ] **步骤 5：提交**

```bash
git add frontend
git commit -m "feat: add timeline and branch experience"
```

---

## 里程碑六：云端评测与产品化

### 任务 19：云端大模型评测适配器

**文件：**
- 创建：`backend/moonlightbox/evaluation/providers/base.py`
- 创建：`backend/moonlightbox/evaluation/providers/openai_compatible.py`
- 创建：`backend/moonlightbox/evaluation/providers/registry.py`
- 创建：`backend/tests/evaluation/test_cloud_provider.py`
- 创建：`frontend/src/features/evaluation/EvaluationSettings.tsx`

- [ ] **步骤 1：写提供商失败测试**

模拟成功、401、429、超时和无效 JSON。断言错误码分别为 `provider_auth_failed`、`provider_rate_limited`、`provider_timeout` 和 `provider_invalid_response`。

- [ ] **步骤 2：确认失败**

运行：`uv run pytest backend/tests/evaluation/test_cloud_provider.py -v`

预期：FAIL，提供商适配器不存在。

- [ ] **步骤 3：实现默认开启的可配置提供商**

```python
class EvaluationProvider(Protocol):
    def score(self, request: PersonaScoreRequest) -> PersonaScore: ...
```

密钥只从系统环境或本地密钥配置读取，不写数据库、不进入日志。设置页展示提供商、模型、将发送的字段和启用状态；缺少凭证时显示配置提示，本地评测仍可运行。

- [ ] **步骤 4：验证**

运行：`uv run pytest backend/tests/evaluation -v && cd frontend && npm test -- --run`

预期：提供商错误被稳定映射，前端可关闭云端评测。

- [ ] **步骤 5：提交**

```bash
git add backend frontend
git commit -m "feat: add configurable cloud persona evaluation"
```

### 任务 20：端到端测试、删除清理与部署文档

**文件：**
- 创建：`frontend/playwright.config.ts`
- 创建：`frontend/tests/e2e/core-flow.spec.ts`
- 创建：`backend/tests/integration/test_project_cleanup.py`
- 创建：`README.md`
- 创建：`docs/deployment-macos.md`
- 创建：`backend/Dockerfile`
- 创建：`frontend/Dockerfile`
- 创建：`docker-compose.yml`

- [ ] **步骤 1：写端到端失败测试**

Playwright 使用合成 CSV 完成：

```text
创建项目 → 预览导入 → 映射角色 → 确认导入
→ 运行假事件分析 → 审核节点 → 从节点创建分支
→ 发送消息 → 重新打开分支
```

后端清理测试创建数据库、Chroma、数据集和假 LoRA 文件，删除项目后断言全部资源消失。

- [ ] **步骤 2：确认失败**

运行：

```bash
uv run pytest backend/tests/integration/test_project_cleanup.py -v
cd frontend && npx playwright test tests/e2e/core-flow.spec.ts
```

预期：FAIL，清理协调器和 E2E 配置尚不完整。

- [ ] **步骤 3：实现清理和文档**

删除流程先把项目标记为 `deleting`，依次清理 Chroma、派生文件、模型适配器和数据库子记录；失败时保留状态和错误，允许重试。两个 Dockerfile 只构建前端和非 MLX API；Compose 不启动训练工作进程，也不宣称支持容器内 Metal。

README 必须明确：

- 完整训练与推理要求 Apple Silicon 和 macOS 原生进程。
- Docker 不提供 MLX Metal GPU 训练。
- `wxecho/`、`data/`、`models/` 和密钥不会进入版本控制。
- 云端评测默认开启但需要配置提供商，并可关闭。

- [ ] **步骤 4：运行完整验证**

```bash
uv run ruff check backend
uv run mypy backend/moonlightbox
uv run pytest backend/tests -m "not model" --cov=moonlightbox
cd frontend && npm test -- --run && npm run build
cd frontend && npx playwright test
```

预期：所有命令退出码为 0；普通测试不下载大型模型；E2E 完成核心流程。

- [ ] **步骤 5：提交**

```bash
git add README.md docs docker-compose.yml backend frontend
git commit -m "docs: complete local deployment and core flow"
```

---

## 最终验收清单

- [ ] wxecho CSV、JSON、TXT 在同一合成数据上产生等价消息。
- [ ] 真实 `wxecho/` 目录、模型、向量库和密钥没有进入 Git。
- [ ] 单字和连续短回复不会被粗暴删除。
- [ ] 候选节点有可解释信号，最终节点有原始消息证据。
- [ ] 用户可以修订节点且历史版本可追踪。
- [ ] 图谱以事件为核心，GraphRAG 不承担初始节点发现。
- [ ] 数据集按 Episode 和时间切分，不发生相邻对话泄漏。
- [ ] MLX-LM 训练前完成平台、内存、磁盘和样本量检查。
- [ ] LoRA 只有优于基础模型时才被标记为推荐。
- [ ] SQLite 查询和 Chroma 查询都强制执行起点时间上限。
- [ ] 多条分支独立保存，不提供并排对比。
- [ ] 云端评测失败不阻塞本地核心流程。
- [ ] 删除项目会清理数据库、向量、数据集、适配器和缓存。
- [ ] 普通测试不下载大型模型，模型冒烟测试可独立执行。

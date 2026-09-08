# 分支基础历史记忆与微信式历史加载实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 选择节点后先冻结节点前全部历史、事件和节点状态，构建完成后再进入具备微信式自适应历史加载的分支聊天。

**Architecture:** SQLite 保存分支基础历史 manifest、不可变事件快照和基础状态；原始消息正文继续引用只读导入数据，Chroma 使用 manifest 边界形成分支基础索引视图。后台 Job 完成边界解析、摘要、事件冻结、状态构建、索引和完整性校验后，原子地把分支从 `preparing` 切换为 `ready`。

**Tech Stack:** FastAPI、SQLAlchemy、Alembic、SQLite、ChromaDB、MLX-LM、DeepSeek、React、TypeScript、TanStack Query、pytest、Vitest。

**设计依据:** `docs/superpowers/specs/2026-07-20-branch-baseline-history-and-wechat-pagination-design.md`

**提交约束:** 用户未要求 Git commit；执行过程不创建提交。

**执行状态:** 2026-07-20 已完成 Task 1–9；自动测试、真实迁移和服务冒烟均通过。

---

## 文件结构

- `backend/moonlightbox/branches/baseline_models.py`：manifest、事件快照和基础状态 ORM。
- `backend/moonlightbox/branches/baseline_boundary.py`：导入源解析、精确消息边界和摘要。
- `backend/moonlightbox/branches/baseline_service.py`：构建、校验和旧分支补建。
- `backend/moonlightbox/branches/baseline_jobs.py`：后台构建 Job。
- `backend/moonlightbox/branches/history.py`：历史游标分页和模型最近尾部。
- `backend/moonlightbox/branches/preparation_router.py`：准备状态、重试和历史 API。
- `backend/alembic/versions/0018_add_branch_baseline_history.py`：新表和分支状态字段。
- `frontend/src/features/branches/BranchPreparationPage.tsx`：分支准备页。
- `frontend/src/features/branches/useAdaptiveBranchHistory.ts`：自适应分页和滚动锚点。
- `frontend/src/features/branches/BranchTimeline.tsx`：统一历史、分隔线和分支消息。

## Task 1：基础历史数据模型与可逆迁移

**Files:**
- Create: `backend/moonlightbox/branches/baseline_models.py`
- Create: `backend/alembic/versions/0018_add_branch_baseline_history.py`
- Modify: `backend/moonlightbox/branches/models.py`
- Modify: `backend/moonlightbox/branches/continuity_models.py`
- Test: `backend/tests/branches/test_baseline_models.py`
- Test: `backend/tests/projects/test_migrations.py`

- [ ] **Step 1: 写失败测试**

测试必须验证：

```python
def test_baseline_manifest_is_unique_and_event_snapshots_are_immutable(session):
    manifest = BranchBaselineManifest(
        branch_id="b1",
        project_id="p1",
        import_id="i1",
        origin_event_id="e1",
        boundary_message_id="m10",
        boundary_timestamp=now,
        boundary_source_id="10",
        message_count=9,
        event_snapshot_count=1,
        message_digest="a" * 64,
        event_digest="b" * 64,
        index_fingerprint="c" * 64,
        recent_tail_message_ids=["m8", "m9"],
        protocol_version="branch-baseline-v1",
        validated_at=now,
    )
    session.add(manifest)
    session.commit()
    assert session.get(BranchBaselineManifest, manifest.id).message_count == 9
```

并验证：

- 每个分支最多一个 manifest。
- 每个 manifest 的事件来源版本唯一。
- `branches.baseline_status` 只允许 `preparing|ready|failed`。
- `0017 -> 0018 -> 0017 -> 0018` 可逆。

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest -q backend/tests/branches/test_baseline_models.py backend/tests/projects/test_migrations.py
```

Expected: FAIL，提示模型或 `0018` 不存在。

- [ ] **Step 3: 实现数据模型**

核心 ORM：

```python
class BranchBaselineManifest(Base):
    __tablename__ = "branch_baseline_manifests"
    id: Mapped[str]
    branch_id: Mapped[str]
    project_id: Mapped[str]
    import_id: Mapped[str]
    origin_event_id: Mapped[str]
    boundary_message_id: Mapped[str]
    boundary_timestamp: Mapped[datetime]
    boundary_source_id: Mapped[str]
    message_count: Mapped[int]
    event_snapshot_count: Mapped[int]
    message_digest: Mapped[str]
    event_digest: Mapped[str]
    index_fingerprint: Mapped[str]
    recent_tail_message_ids: Mapped[list[str]]
    protocol_version: Mapped[str]
    created_at: Mapped[datetime]
    validated_at: Mapped[datetime]

class BranchBaselineEventSnapshot(Base):
    __tablename__ = "branch_baseline_event_snapshots"
    id: Mapped[str]
    manifest_id: Mapped[str]
    source_event_id: Mapped[str]
    source_revision_id: Mapped[str]
    snapshot: Mapped[dict[str, object]]
    evidence_message_ids: Mapped[list[str]]
    started_at: Mapped[datetime]
    ended_at: Mapped[datetime]
    content_hash: Mapped[str]

class BranchBaselineState(Base):
    __tablename__ = "branch_baseline_states"
    id: Mapped[str]
    manifest_id: Mapped[str]
    branch_id: Mapped[str]
    persona_state: Mapped[dict[str, object]]
    relationship_state: Mapped[dict[str, object]]
    emotional_tendency: Mapped[dict[str, object]]
    user_model: Mapped[dict[str, object]]
    historical_belief_ids: Mapped[list[str]]
    evidence_message_ids: Mapped[list[str]]
    evidence_event_snapshot_ids: Mapped[list[str]]
    local_proposal: Mapped[dict[str, object]]
    review_result: Mapped[dict[str, object]]
    content_hash: Mapped[str]
```

`Branch` 新增来源、Job、状态和错误字段；`BranchStateVersion` 新增可空 `baseline_manifest_id` 与 `baseline_state_id`。

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest -q backend/tests/branches/test_baseline_models.py backend/tests/projects/test_migrations.py
```

Expected: PASS。

## Task 2：导入源解析、精确边界和不可变摘要

**Files:**
- Create: `backend/moonlightbox/branches/baseline_boundary.py`
- Test: `backend/tests/branches/test_baseline_boundary.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- 优先通过 `AnalysisRevision.run_id -> AnalysisRun.import_id` 解析来源。
- 无 run 的旧事件只有唯一候选时才能解析。
- 多个导入具有相同 source ID 时拒绝猜测。
- 同时间戳按 `(timestamp, source_id, id)` 排序并排除节点消息。
- 媒体资源补全不改变语义摘要。
- 正文、发送者、时间变化会改变摘要。

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest -q backend/tests/branches/test_baseline_boundary.py
```

Expected: FAIL，提示 `BaselineBoundaryResolver` 不存在。

- [ ] **Step 3: 实现边界服务**

接口：

```python
@dataclass(frozen=True)
class ResolvedBaselineBoundary:
    project_id: str
    import_id: str
    event_id: str
    message_id: str
    timestamp: datetime
    source_id: str

class BaselineBoundaryResolver:
    def resolve(self, project_id: str, event_id: str) -> ResolvedBaselineBoundary: ...
    def messages_before(
        self,
        boundary: ResolvedBaselineBoundary,
    ) -> Select[tuple[Message]]: ...

def semantic_message_digest(rows: Iterable[tuple[Message, str]]) -> str: ...
```

摘要只包含消息 ID、source ID、时间、角色、类型和正文，不包含媒体资产 ID。

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest -q backend/tests/branches/test_baseline_boundary.py
```

Expected: PASS。

## Task 3：基础历史构建、事件冻结与状态投影

**Files:**
- Create: `backend/moonlightbox/branches/baseline_service.py`
- Modify: `backend/moonlightbox/branches/identity.py`
- Modify: `backend/moonlightbox/branches/state_engine.py`
- Test: `backend/tests/branches/test_baseline_service.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- manifest 包含边界前全部双人消息。
- 最近尾部按轮次保留 6–8 轮且不含节点消息。
- 只冻结边界前且证据完整的已确认事件。
- 新分支创建状态版本 1。
- 旧分支保留现有版本并创建 `N+1` 补建投影。
- 相同构建请求幂等。
- 任一证据越过边界时整次构建失败。

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest -q backend/tests/branches/test_baseline_service.py
```

Expected: FAIL，提示 `BranchBaselineService` 不存在。

- [ ] **Step 3: 实现构建服务**

接口：

```python
class BranchBaselineService:
    def build_messages(
        self,
        branch: Branch,
        boundary: ResolvedBaselineBoundary,
    ) -> BaselineMessageSnapshot: ...

    def freeze_events(
        self,
        branch: Branch,
        boundary: ResolvedBaselineBoundary,
    ) -> list[BranchBaselineEventSnapshot]: ...

    def create_baseline_state(
        self,
        branch: Branch,
        manifest: BranchBaselineManifest,
        event_snapshots: list[BranchBaselineEventSnapshot],
    ) -> BranchBaselineState: ...

    def project_state(
        self,
        branch: Branch,
        baseline_state: BranchBaselineState,
    ) -> BranchStateVersion: ...

    def validate(
        self,
        branch: Branch,
        manifest: BranchBaselineManifest,
    ) -> None: ...
```

本地 8B 提案和 DeepSeek 复核只接收边界内证据；复核失败时不创建有效基础状态。

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest -q backend/tests/branches/test_baseline_service.py
```

Expected: PASS。

## Task 4：后台 Job、准备状态和分支创建门槛

**Files:**
- Create: `backend/moonlightbox/branches/baseline_jobs.py`
- Create: `backend/moonlightbox/branches/preparation_router.py`
- Modify: `backend/moonlightbox/branches/service.py`
- Modify: `backend/moonlightbox/branches/router.py`
- Modify: `backend/moonlightbox/branches/schemas.py`
- Modify: `backend/moonlightbox/worker_main.py`
- Modify: `backend/moonlightbox/api.py`
- Test: `backend/tests/branches/test_baseline_jobs.py`
- Test: `backend/tests/branches/test_preparation_api.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- 创建分支时同步解析来源并创建唯一 Job。
- 来源不唯一时返回 409 且不创建分支。
- `preparing` 分支不能 GET/POST messages。
- Job 阶段 checkpoint 正确。
- 只有 manifest、基础状态、索引和完整性校验全部成功后才原子切换 `ready`。
- failed Job 可以重试。

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest -q backend/tests/branches/test_baseline_jobs.py backend/tests/branches/test_preparation_api.py
```

Expected: FAIL，准备接口返回 404 或聊天未被阻止。

- [ ] **Step 3: 实现 Job 和 API**

Job：

```python
BRANCH_BASELINE_JOB_KIND = "branch_baseline_build"
```

checkpoint 阶段：

```text
resolving_boundary
freezing_messages
freezing_events
building_state
building_index
validating
completed
```

准备接口：

```text
GET  /api/projects/{project_id}/branches/{branch_id}/preparation
POST /api/projects/{project_id}/branches/{branch_id}/preparation/retry
```

聊天接口遇到非 ready 分支抛出 `BranchBaselineNotReadyError` 并返回 `409 branch_baseline_not_ready`。

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest -q backend/tests/branches/test_baseline_jobs.py backend/tests/branches/test_preparation_api.py backend/tests/branches/test_branches_api.py
```

Expected: PASS。

## Task 5：历史游标分页 API

**Files:**
- Create: `backend/moonlightbox/branches/history.py`
- Modify: `backend/moonlightbox/branches/preparation_router.py`
- Modify: `backend/moonlightbox/branches/schemas.py`
- Test: `backend/tests/branches/test_branch_history.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- 首次返回边界前最新一页并按正序输出。
- 连续游标分页无重复、无断层。
- 最终能够到达第一条消息。
- 节点消息永远不返回。
- 同时间戳消息排序稳定。
- 跨分支和过期 manifest 游标返回 400。
- limit 限制为 20–100。
- 表情和媒体 ID 正确返回。

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest -q backend/tests/branches/test_branch_history.py
```

Expected: FAIL，历史接口返回 404。

- [ ] **Step 3: 实现游标和查询**

接口：

```python
class BranchHistoryService:
    def page(
        self,
        project_id: str,
        branch_id: str,
        *,
        before: str | None,
        limit: int,
    ) -> BranchHistoryPage: ...

    def recent_turns(
        self,
        branch: Branch,
        *,
        turn_limit: int = 8,
    ) -> tuple[ContextTurn, ...]: ...
```

游标编码 branch、manifest 和完整排序键；解析后执行硬校验，不允许静默回退。

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest -q backend/tests/branches/test_branch_history.py
```

Expected: PASS。

## Task 6：双层记忆上下文组装

**Files:**
- Modify: `backend/moonlightbox/branches/memory.py`
- Modify: `backend/moonlightbox/branches/context.py`
- Modify: `backend/moonlightbox/branches/service.py`
- Modify: `backend/moonlightbox/branches/continuity_index.py`
- Test: `backend/tests/branches/test_memory.py`
- Test: `backend/tests/branches/test_context.py`
- Test: `backend/tests/branches/test_continuity_index.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- 首轮 prompt 包含节点前最近真实轮次。
- 历史尾部早于分支消息，并且不伪装成当前分支 episode。
- 基础历史检索严格受 manifest 边界限制。
- 节点后项目消息无法通过语义检索进入。
- 当前分支成长记忆优先于低相关基础历史。
- 客观历史与分支状态冲突时 prompt 明确区分时间层。

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest -q backend/tests/branches/test_memory.py backend/tests/branches/test_context.py backend/tests/branches/test_continuity_index.py
```

Expected: FAIL，prompt 不含 baseline recent tail。

- [ ] **Step 3: 实现双层上下文**

`ContextPacket` 新增：

```python
baseline_manifest: dict[str, object]
baseline_recent_history: tuple[ContextTurn, ...]
baseline_memories: tuple[ContextMemory, ...]
```

顺序：

```text
协议
人格内核
基础历史边界
基础历史最近轮次
基础状态
基础历史语义记忆
分支成长记忆
分支短期聊天
当前输入
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest -q backend/tests/branches/test_memory.py backend/tests/branches/test_context.py backend/tests/branches/test_continuity_index.py backend/tests/branches/test_reply_reviewer.py
```

Expected: PASS。

## Task 7：分支准备页面和路由保护

**Files:**
- Create: `frontend/src/features/branches/BranchPreparationPage.tsx`
- Create: `frontend/src/features/branches/BranchPreparationPage.test.tsx`
- Modify: `frontend/src/features/branches/BranchCreatePage.tsx`
- Modify: `frontend/src/features/branches/BranchChatPage.tsx`
- Modify: `frontend/src/features/branches/types.ts`
- Modify: `frontend/src/app/router.tsx`
- Modify: `frontend/src/index.css`

- [ ] **Step 1: 写失败测试**

覆盖：

- 创建分支后导航到 preparing 页面而不是聊天页。
- 显示各构建阶段、进度和数量。
- ready 后自动 replace 到聊天页。
- failed 后停止轮询并显示重试。
- 直接打开 preparing 分支聊天页时重定向。

- [ ] **Step 2: 运行测试确认失败**

```bash
npm test -- --run src/features/branches/BranchPreparationPage.test.tsx src/features/branches/BranchChatPage.test.tsx
```

Expected: FAIL，准备页面不存在。

- [ ] **Step 3: 实现准备流程**

准备状态每 1.5 秒轮询；`ready` 时：

```ts
navigate(`/projects/${projectId}/branches/${branchId}`, { replace: true })
```

聊天页在 `branch.baseline_status !== 'ready'` 时 replace 到准备页，并且不创建消息查询和输入框。

- [ ] **Step 4: 运行测试确认通过**

```bash
npm test -- --run src/features/branches/BranchPreparationPage.test.tsx src/features/branches/BranchChatPage.test.tsx
```

Expected: PASS。

## Task 8：微信式自适应历史时间线

**Files:**
- Create: `frontend/src/features/branches/useAdaptiveBranchHistory.ts`
- Create: `frontend/src/features/branches/useAdaptiveBranchHistory.test.tsx`
- Create: `frontend/src/features/branches/BranchTimeline.tsx`
- Create: `frontend/src/features/branches/BranchTimeline.test.tsx`
- Modify: `frontend/src/features/branches/BranchChatPage.tsx`
- Modify: `frontend/src/features/branches/MessageBubble.tsx`
- Modify: `frontend/src/features/branches/types.ts`
- Modify: `frontend/src/index.css`

- [ ] **Step 1: 写失败测试**

覆盖：

- 初始 limit 按视口和平均消息高度计算并限制在 20–80。
- 内容不足一屏时自动补页。
- 顶部 sentinel 触发下一页。
- 并发触发只发一次请求。
- 插入旧页后 scrollTop 按高度差补偿。
- 历史与 staged 分支消息不会互相去重。
- 时间线分隔线位于历史和分支消息之间。
- 分支切换恢复独立滚动位置。

- [ ] **Step 2: 运行测试确认失败**

```bash
npm test -- --run src/features/branches/useAdaptiveBranchHistory.test.tsx src/features/branches/BranchTimeline.test.tsx
```

Expected: FAIL，hook 和组件不存在。

- [ ] **Step 3: 实现自适应加载**

初始 limit：

```ts
Math.min(
  80,
  Math.max(20, Math.ceil((viewportHeight / averageMessageHeight) * 1.5)),
)
```

使用 TanStack `useInfiniteQuery`，query key 包含 branch ID 和 manifest ID。插入旧页前后记录 `scrollHeight`、`scrollTop` 和 anchor ID，并进行两阶段校正。

- [ ] **Step 4: 运行测试确认通过**

```bash
npm test -- --run src/features/branches/useAdaptiveBranchHistory.test.tsx src/features/branches/BranchTimeline.test.tsx src/features/branches/BranchChatPage.test.tsx
```

Expected: PASS。

## Task 9：旧分支幂等补建与真实端到端验收

**Files:**
- Create: `backend/moonlightbox/branches/baseline_migration.py`
- Create: `backend/tests/branches/test_baseline_migration.py`
- Modify: `backend/moonlightbox/branches/preparation_router.py`
- Modify: `backend/moonlightbox/evaluation/continual_persona.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- 三类旧分支均能补建 manifest。
- 已有状态版本不被修改。
- 创建 `N+1` 补建投影。
- 重复迁移不重复 manifest、事件快照、Job 或状态版本。
- 无法唯一解析来源时标记 failed。
- 原有分支消息和 episode 完整保留。

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest -q backend/tests/branches/test_baseline_migration.py
```

Expected: FAIL，迁移服务不存在。

- [ ] **Step 3: 实现迁移服务**

接口：

```python
class BaselineMigrationService:
    def migrate_project(self, project_id: str) -> BaselineMigrationReport: ...
    def migrate_branch(self, branch_id: str) -> BranchBaselineMigrationResult: ...
```

迁移只排队 Job，不在 HTTP 线程同步运行 MLX 和 DeepSeek。

- [ ] **Step 4: 运行完整自动验证**

```bash
uv run pytest -q backend/tests
uv run ruff check backend
uv run mypy backend/moonlightbox
cd frontend && npm test -- --run
cd frontend && npm run build
```

Expected: 后端、前端、lint、类型检查和生产构建全部通过。

- [ ] **Step 5: 执行真实迁移与验收**

1. 升级到 `0018`。
2. 停止旧 Worker，避免未知 Job。
3. 为洪欣羽项目三个分支创建补建任务。
4. 启动新 Worker 并等待全部任务结束。
5. 校验三个分支 manifest 消息数量和摘要。
6. 每个分支抽查边界前最后 20 条。
7. 确认节点消息及之后消息零泄漏。
8. 调用历史 API 连续翻页到第一条。
9. 检查头像、文本和表情。
10. 验证第一轮上下文包含节点前最近真实对话。
11. 运行持续人格与基础历史联合验收。
12. 确认三个分支状态均为 `ready` 后重启前后端。


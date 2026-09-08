# 分支自主持续人格记忆实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为每个时间分支建立严格隔离、可持续成长、以数字人主体为中心、具备证据链与回滚能力的长期人格记忆。

**Architecture:** SQLite 保存不可变 episode、人格内核、时间化记忆、竞争信念和状态版本，ChromaDB 只负责当前分支候选召回。本地 8B 提出记忆变化，DeepSeek 使用最小证据包复核，确定性状态引擎执行分支校验、来源去重、有界更新和原子版本切换。

**Tech Stack:** FastAPI、SQLAlchemy、Alembic、SQLite、ChromaDB、FastEmbed、MLX-LM、DeepSeek OpenAI-compatible API、React、TypeScript、pytest、Vitest。

**设计依据:** `docs/superpowers/specs/2026-07-20-branch-autonomous-continual-persona-memory-design.md`

**提交约束:** 用户未要求 Git commit；执行过程中不创建提交。

---

## 文件结构

- `backend/moonlightbox/branches/continuity_models.py`：持续人格记忆 ORM。
- `backend/moonlightbox/branches/continuity_types.py`：候选记忆、证据、状态变更协议。
- `backend/moonlightbox/branches/episodes.py`：episode 幂等写入与历史回填。
- `backend/moonlightbox/branches/identity.py`：人格内核创建、锁定和读取。
- `backend/moonlightbox/branches/memory_proposer.py`：本地 8B 结构化候选提取。
- `backend/moonlightbox/branches/memory_reviewer.py`：DeepSeek 最小证据复核。
- `backend/moonlightbox/branches/state_engine.py`：来源治理、竞争信念、有界状态更新和回滚。
- `backend/moonlightbox/branches/continuity_index.py`：分支 episode、记忆和反思向量索引。
- `backend/moonlightbox/branches/continuity_jobs.py`：记忆巩固与反思 Worker handler。
- `backend/moonlightbox/branches/continuity_service.py`：查询、重试、迁移和健康状态应用服务。
- `backend/moonlightbox/branches/continuity_router.py`：人格记忆 API。
- `frontend/src/features/branches/BranchMemoryPanel.tsx`：人格内核、状态、信念、反思和版本视图。
- `backend/alembic/versions/0017_add_branch_continual_memory.py`：新表和索引迁移。

## Task 1：持续人格记忆数据契约与迁移

**Files:**
- Create: `backend/moonlightbox/branches/continuity_models.py`
- Create: `backend/moonlightbox/branches/continuity_types.py`
- Create: `backend/alembic/versions/0017_add_branch_continual_memory.py`
- Modify: `backend/moonlightbox/db.py`
- Test: `backend/tests/branches/test_continuity_models.py`
- Test: `backend/tests/projects/test_migrations.py`

- [ ] **Step 1: 写失败测试**

测试必须验证：

```python
def test_branch_memory_schema_preserves_lineage_and_versions(session):
    kernel = IdentityKernel(project_id="p1", model_version_id="m1", ...)
    episode = BranchMemoryEpisode(
        branch_id="b1",
        user_turn_id="u1",
        assistant_turn_id="a1",
        episode_hash="hash",
        ...
    )
    item = BranchMemoryItem(
        branch_id="b1",
        kind="self_narrative",
        source_episode_ids=[episode.id],
        lineage_hash="lineage",
        ...
    )
    state = BranchStateVersion(
        branch_id="b1",
        version=1,
        previous_version_id=None,
        ...
    )
    session.add_all([kernel, episode, item, state])
    session.commit()
    assert session.get(BranchMemoryItem, item.id).source_episode_ids == [episode.id]
```

并验证：

- `(branch_id, user_turn_id, assistant_turn_id)` 唯一。
- `(branch_id, version)` 唯一。
- 记忆类型 CheckConstraint。
- Alembic `0016 -> 0017 -> 0016 -> 0017` 可逆。

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
uv run pytest -q backend/tests/branches/test_continuity_models.py backend/tests/projects/test_migrations.py
```

Expected: FAIL，提示 `continuity_models` 或 `0017` 不存在。

- [ ] **Step 3: 实现 ORM 与迁移**

定义：

```python
class IdentityKernel(Base):
    __tablename__ = "identity_kernels"
    id: Mapped[str]
    project_id: Mapped[str]
    model_version_id: Mapped[str]
    schema_version: Mapped[str]
    content: Mapped[dict[str, object]]
    evidence_message_ids: Mapped[list[str]]
    content_hash: Mapped[str]
    created_at: Mapped[datetime]
    locked_at: Mapped[datetime]

class BranchMemoryEpisode(Base):
    __tablename__ = "branch_memory_episodes"
    id: Mapped[str]
    branch_id: Mapped[str]
    user_turn_id: Mapped[str]
    assistant_turn_id: Mapped[str]
    user_content: Mapped[str]
    assistant_bubbles: Mapped[list[dict[str, object]]]
    model_version_id: Mapped[str]
    episode_hash: Mapped[str]
    importance: Mapped[float]
    processing_status: Mapped[str]
    started_at: Mapped[datetime]
    ended_at: Mapped[datetime]

class BranchMemoryItem(Base):
    __tablename__ = "branch_memory_items"
    id: Mapped[str]
    branch_id: Mapped[str]
    kind: Mapped[str]
    content: Mapped[str]
    subject: Mapped[str]
    predicate: Mapped[str]
    object: Mapped[str]
    confidence: Mapped[float]
    importance: Mapped[float]
    valid_from: Mapped[datetime]
    valid_to: Mapped[datetime | None]
    supersedes_id: Mapped[str | None]
    source_episode_ids: Mapped[list[str]]
    source_item_ids: Mapped[list[str]]
    lineage_hash: Mapped[str]
    review_status: Mapped[str]

class BranchBeliefEvidence(Base):
    __tablename__ = "branch_belief_evidence"
    id: Mapped[str]
    branch_id: Mapped[str]
    belief_id: Mapped[str]
    episode_id: Mapped[str]
    stance: Mapped[str]
    source_role: Mapped[str]
    evidence_type: Mapped[str]
    weight: Mapped[float]

class BranchStateVersion(Base):
    __tablename__ = "branch_state_versions"
    id: Mapped[str]
    branch_id: Mapped[str]
    version: Mapped[int]
    previous_version_id: Mapped[str | None]
    persona_state: Mapped[dict[str, object]]
    relationship_state: Mapped[dict[str, object]]
    user_model: Mapped[dict[str, object]]
    emotional_tendency: Mapped[dict[str, object]]
    active_belief_ids: Mapped[list[str]]
    reason: Mapped[str]
    source_episode_ids: Mapped[list[str]]
    rolled_back_at: Mapped[datetime | None]

class BranchReflectionRun(Base):
    __tablename__ = "branch_reflection_runs"
    id: Mapped[str]
    branch_id: Mapped[str]
    trigger_episode_id: Mapped[str]
    input_item_ids: Mapped[list[str]]
    input_importance_sum: Mapped[float]
    local_proposal: Mapped[dict[str, object] | None]
    review_result: Mapped[dict[str, object] | None]
    output_item_ids: Mapped[list[str]]
    status: Mapped[str]
    attempt_count: Mapped[int]
```

在 `continuity_types.py` 定义 Pydantic 类型：

```python
MemoryKind = Literal["fact", "experience", "self_narrative", "belief", "reflection"]
EvidenceStance = Literal["support", "oppose"]

class MemoryCandidate(BaseModel):
    kind: MemoryKind
    content: str
    subject: str
    predicate: str
    object: str
    confidence: float = Field(ge=0, le=1)
    importance: float = Field(ge=1, le=10)
    evidence_message_ids: tuple[str, ...]

class StateDeltaProposal(BaseModel):
    relationship_delta: dict[str, float]
    emotional_delta: dict[str, float]
    user_model_updates: dict[str, str]
    supporting_candidate_indexes: tuple[int, ...]

class MemoryProposal(BaseModel):
    candidates: tuple[MemoryCandidate, ...]
    state_delta: StateDeltaProposal
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
uv run pytest -q backend/tests/branches/test_continuity_models.py backend/tests/projects/test_migrations.py
```

Expected: PASS。

## Task 2：人格内核的创建、锁定与模型激活门槛

**Files:**
- Create: `backend/moonlightbox/branches/identity.py`
- Modify: `backend/moonlightbox/training/jobs.py`
- Modify: `backend/moonlightbox/training/registry.py`
- Test: `backend/tests/branches/test_identity_kernel.py`
- Test: `backend/tests/training/test_training_job.py`

- [ ] **Step 1: 写失败测试**

覆盖：

```python
def test_identity_kernel_is_evidence_backed_and_immutable(session):
    kernel = IdentityKernelService(session).create_and_lock(
        project_id="p1",
        model_version_id="m1",
        proposal={...},
        evidence_message_ids=["msg-1", "msg-2"],
    )
    assert kernel.locked_at is not None
    with pytest.raises(IdentityKernelLockedError):
        IdentityKernelService(session).replace(kernel.id, {...})
```

训练任务测试必须断言：人格内核创建失败时模型不激活；内核成功锁定后才执行 `registry.activate()`。

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
uv run pytest -q backend/tests/branches/test_identity_kernel.py backend/tests/training/test_training_job.py
```

Expected: FAIL，提示 `IdentityKernelService` 不存在。

- [ ] **Step 3: 实现人格内核服务**

接口：

```python
class IdentityKernelBuilder(Protocol):
    def build(
        self,
        *,
        persona: str,
        examples: list[TrainingExample],
        event_contexts: list[ConfirmedEventContext],
    ) -> IdentityKernelProposal: ...

class IdentityKernelService:
    def create_and_lock(
        self,
        *,
        project_id: str,
        model_version_id: str,
        proposal: IdentityKernelProposal,
        evidence_message_ids: list[str],
    ) -> IdentityKernel: ...

    def get_for_model(self, model_version_id: str) -> IdentityKernel: ...
```

`IdentityKernelProposal` 必须包含 `persona`、`values`、`stable_preferences`、`relationship_boundaries`、`language_patterns` 和 `typical_reactions`。

训练 handler 顺序固定为：训练 → 对话验收 → 人格内核生成与证据校验 → 锁定内核 → 激活模型。

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
uv run pytest -q backend/tests/branches/test_identity_kernel.py backend/tests/training/test_training_job.py
```

Expected: PASS。

## Task 3：每轮对话原子创建不可变 Episode

**Files:**
- Create: `backend/moonlightbox/branches/episodes.py`
- Modify: `backend/moonlightbox/branches/service.py`
- Test: `backend/tests/branches/test_episodes.py`
- Test: `backend/tests/branches/test_branches_api.py`

- [ ] **Step 1: 写失败测试**

覆盖：

```python
def test_successful_reply_creates_one_idempotent_episode(session):
    episode = EpisodeService(session).record_turn(
        branch=branch,
        user=user_message,
        assistants=assistant_messages,
    )
    same = EpisodeService(session).record_turn(
        branch=branch,
        user=user_message,
        assistants=assistant_messages,
    )
    assert same.id == episode.id
    assert episode.assistant_bubbles == [
        {"type": "text", "content": "第一条", "bubble_index": 0},
        {"type": "text", "content": "第二条", "bubble_index": 1},
    ]
```

API 测试断言生成失败时不创建 episode，成功时用户消息、助手气泡、episode 和记忆 Job 同事务提交。

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
uv run pytest -q backend/tests/branches/test_episodes.py backend/tests/branches/test_branches_api.py
```

Expected: FAIL，提示 `EpisodeService` 不存在或 episode 数量为零。

- [ ] **Step 3: 实现 EpisodeService**

接口：

```python
CONTINUAL_MEMORY_JOB_KIND = "branch_continual_memory"

class EpisodeService:
    def record_turn(
        self,
        branch: Branch,
        user: BranchMessage,
        assistants: list[BranchMessage],
    ) -> tuple[BranchMemoryEpisode, Job]:
        ...
```

`episode_hash` 使用分支 ID、用户 turn ID、助手 turn ID、消息内容和模型版本的 SHA-256。Job 的 `dedupe_key` 使用 `branch-memory:{episode.id}`。

修改 `BranchService.add_user_message()`：在现有 `session.commit()` 前创建 episode 与 queued Job；任何异常统一回滚。

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
uv run pytest -q backend/tests/branches/test_episodes.py backend/tests/branches/test_branches_api.py
```

Expected: PASS。

## Task 4：本地 8B 主体视角候选与 DeepSeek 证据复核

**Files:**
- Create: `backend/moonlightbox/branches/memory_proposer.py`
- Create: `backend/moonlightbox/branches/memory_reviewer.py`
- Test: `backend/tests/branches/test_memory_proposer.py`
- Test: `backend/tests/branches/test_memory_reviewer.py`

- [ ] **Step 1: 写失败测试**

测试要求：

- 用户说“你已经不爱我了”只能提取为用户观点，不能直接生成数字人主体信念。
- 数字人说“我觉得你最近在躲我”可以生成 `self_narrative`。
- reviewer 拒绝不存在的证据 ID。
- reviewer 不能新增本地候选中不存在的记忆。

示例：

```python
def test_user_definition_is_not_promoted_to_agent_belief():
    proposal = proposer.propose(
        episode=user_says("你已经不爱我了"),
        identity_kernel=kernel,
        current_state=state,
    )
    assert all(
        not (
            item.kind == "belief"
            and item.subject == "digital_human"
            and item.predicate == "不爱"
        )
        for item in proposal.candidates
    )
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
uv run pytest -q backend/tests/branches/test_memory_proposer.py backend/tests/branches/test_memory_reviewer.py
```

Expected: FAIL，提示 proposer/reviewer 不存在。

- [ ] **Step 3: 实现协议**

本地 proposer system prompt 必须包含：

```text
你站在数字人主体视角解释本轮经历。
用户对数字人的定义只是用户观点，不得直接成为数字人主体信念。
数字人自己的表达可以形成自我叙事，但不能单独创造客观世界事实。
只返回 MemoryProposal JSON，所有候选必须引用本轮真实消息 ID。
```

DeepSeek reviewer 返回：

```python
class MemoryReviewResult(BaseModel):
    verdict: Literal["approve", "reject"]
    approved_candidate_indexes: tuple[int, ...]
    approved_state_delta: StateDeltaProposal | None
    rejected_reasons: tuple[str, ...]
```

reviewer 只允许删除候选或缩小 delta，不允许增加候选。

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
uv run pytest -q backend/tests/branches/test_memory_proposer.py backend/tests/branches/test_memory_reviewer.py
```

Expected: PASS。

## Task 5：来源治理、竞争信念与有界状态引擎

**Files:**
- Create: `backend/moonlightbox/branches/state_engine.py`
- Test: `backend/tests/branches/test_state_engine.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- 用户替数字人下定义不能直接激活主体信念。
- 数字人的自主表达可以形成低初始置信度自我叙事。
- 相同 lineage 多次派生只算一个独立证据。
- 支持与反对证据可以同时存在。
- 单轮关系和情绪数值变化不能超过配置上限。
- 新状态版本与 `Branch.state_snapshot` 同步。

示例：

```python
def test_repeated_derived_reflections_do_not_self_reinforce(engine):
    first = engine.apply(reviewed_proposal(lineage="episode-1"))
    second = engine.apply(reviewed_proposal(lineage="episode-1"))
    assert second.belief_confidence == first.belief_confidence
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
uv run pytest -q backend/tests/branches/test_state_engine.py
```

Expected: FAIL，提示 `BranchStateEngine` 不存在。

- [ ] **Step 3: 实现状态引擎**

接口：

```python
@dataclass(frozen=True)
class StateEngineConfig:
    max_relationship_delta: float = 5.0
    max_emotional_delta: float = 0.1
    belief_activation_threshold: float = 0.65

class BranchStateEngine:
    def apply(
        self,
        *,
        branch: Branch,
        episode: BranchMemoryEpisode,
        proposal: MemoryProposal,
        review: MemoryReviewResult,
        identity_kernel: IdentityKernel,
    ) -> BranchStateVersion:
        ...

    def rollback(self, branch: Branch, version_id: str) -> BranchStateVersion:
        ...
```

所有 item、evidence、旧记录失效、新状态版本和 `state_snapshot` 更新使用单一事务。

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
uv run pytest -q backend/tests/branches/test_state_engine.py
```

Expected: PASS。

## Task 6：重要度累计、反思任务与 Worker 恢复

**Files:**
- Create: `backend/moonlightbox/branches/continuity_jobs.py`
- Modify: `backend/moonlightbox/worker_main.py`
- Test: `backend/tests/branches/test_continuity_jobs.py`
- Test: `backend/tests/jobs/test_worker.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- episode 处理任务可以安全重试。
- 累计独立 episode 重要度达到 20 才创建 reflection run。
- 同一 episode 的多个 item 只累计一次。
- 反思至少引用两个独立 episode，单个重要度 10 的关键事件可例外。
- Worker 中断后任务恢复为 queued。
- DeepSeek 不可用时 episode 保持 failed/pending，状态不变化。

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
uv run pytest -q backend/tests/branches/test_continuity_jobs.py backend/tests/jobs/test_worker.py
```

Expected: FAIL，提示 job kind 未注册。

- [ ] **Step 3: 实现 handler**

注册：

```python
CONTINUAL_MEMORY_JOB_KIND = "branch_continual_memory"
BRANCH_REFLECTION_JOB_KIND = "branch_reflection"

registry.register(
    CONTINUAL_MEMORY_JOB_KIND,
    create_continual_memory_handler(proposer, reviewer, state_engine, index),
)
registry.register(
    BRANCH_REFLECTION_JOB_KIND,
    create_branch_reflection_handler(proposer, reviewer, state_engine, index),
)
```

handler 阶段 checkpoint 固定为：

- `loading_episode`
- `local_proposal`
- `cloud_review`
- `applying_state`
- `indexing`
- `completed`

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
uv run pytest -q backend/tests/branches/test_continuity_jobs.py backend/tests/jobs/test_worker.py
```

Expected: PASS。

## Task 7：分支长期索引、检索评分与 ContextPacket

**Files:**
- Create: `backend/moonlightbox/branches/continuity_index.py`
- Modify: `backend/moonlightbox/branches/context.py`
- Modify: `backend/moonlightbox/branches/memory.py`
- Modify: `backend/moonlightbox/branches/service.py`
- Test: `backend/tests/branches/test_continuity_index.py`
- Test: `backend/tests/branches/test_context.py`
- Test: `backend/tests/branches/test_memory.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- 两个分支使用相同问题只能召回各自记忆。
- 失效、未复核、未来和回滚后记忆不可召回。
- 人格内核和当前状态始终进入 system prompt。
- 长期配额为 2 个 experience、2 个 fact/belief/self_narrative、2 个 reflection。
- 查询包含当前输入、上一助手轮次、活跃信念和关系状态。

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
uv run pytest -q backend/tests/branches/test_continuity_index.py backend/tests/branches/test_context.py backend/tests/branches/test_memory.py
```

Expected: FAIL，提示缺少 identity/state/continuity memory。

- [ ] **Step 3: 实现索引和上下文**

检索分数：

```python
score = (
    semantic_similarity * 0.55
    + normalized_importance * 0.20
    + recency_score * 0.15
    + confidence * 0.10
)
```

`ContextPacket` 新增：

```python
identity_kernel: dict[str, object]
branch_state: dict[str, object]
continuity_memories: tuple[ContextMemory, ...]
```

system prompt 顺序固定为：协议 → 身份 → 人格内核 → 分支起点 → 当前主体状态 → 长期记忆。

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
uv run pytest -q backend/tests/branches/test_continuity_index.py backend/tests/branches/test_context.py backend/tests/branches/test_memory.py
```

Expected: PASS。

## Task 8：现有活动模型和分支的幂等回填迁移

**Files:**
- Create: `backend/moonlightbox/branches/continuity_service.py`
- Create: `backend/moonlightbox/branches/continuity_migration.py`
- Test: `backend/tests/branches/test_continuity_migration.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- 当前活动模型回填一个锁定人格内核。
- 分支消息按 user turn + assistant turn 回填 episode。
- 现有 `state_snapshot` 创建版本 1。
- 重复执行不会新增重复 episode、内核或状态版本。
- 不完整用户轮次不创建 episode。
- 分支之间不共享回填结果。

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
uv run pytest -q backend/tests/branches/test_continuity_migration.py
```

Expected: FAIL，提示迁移服务不存在。

- [ ] **Step 3: 实现迁移**

接口：

```python
class ContinuityMigrationService:
    def migrate_active_model(self, project_id: str) -> IdentityKernel: ...
    def backfill_branch(self, branch_id: str) -> MigrationReport: ...
    def rebuild_branch_index(self, branch_id: str) -> int: ...

class MigrationReport(BaseModel):
    branch_id: str
    episode_count: int
    state_version_count: int
    queued_job_count: int
    index_document_count: int
```

迁移只创建 episode 和 queued job，不在 HTTP 请求内同步运行全部模型提取。

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
uv run pytest -q backend/tests/branches/test_continuity_migration.py
```

Expected: PASS。

## Task 9：人格记忆 API、回滚和失败重试

**Files:**
- Create: `backend/moonlightbox/branches/continuity_router.py`
- Create: `backend/moonlightbox/branches/continuity_schemas.py`
- Modify: `backend/moonlightbox/api.py`
- Test: `backend/tests/branches/test_continuity_api.py`

- [ ] **Step 1: 写失败测试**

端点：

```text
GET  /api/projects/{project_id}/branches/{branch_id}/memory
GET  /api/projects/{project_id}/branches/{branch_id}/memory/episodes
GET  /api/projects/{project_id}/branches/{branch_id}/memory/beliefs
GET  /api/projects/{project_id}/branches/{branch_id}/memory/reflections
GET  /api/projects/{project_id}/branches/{branch_id}/memory/versions
POST /api/projects/{project_id}/branches/{branch_id}/memory/versions/{version_id}/rollback
POST /api/projects/{project_id}/branches/{branch_id}/memory/jobs/{job_id}/retry
POST /api/projects/{project_id}/branches/{branch_id}/memory/migrate
```

测试必须验证项目/分支越权返回 404，回滚后当前状态和 `state_snapshot` 一致，不能重试其他分支 Job。

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
uv run pytest -q backend/tests/branches/test_continuity_api.py
```

Expected: FAIL，返回 404。

- [ ] **Step 3: 实现 API**

聚合响应：

```python
class BranchMemoryOverviewRead(BaseModel):
    identity_kernel: IdentityKernelRead
    current_state: BranchStateVersionRead
    active_beliefs: list[MemoryItemRead]
    competing_beliefs: list[MemoryItemRead]
    recent_reflections: list[MemoryItemRead]
    pending_jobs: int
    failed_jobs: int
    evolution_frozen: bool
```

所有写端点调用 `ContinuityService`，router 不直接操作 ORM。

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
uv run pytest -q backend/tests/branches/test_continuity_api.py
```

Expected: PASS。

## Task 10：人格记忆界面

**Files:**
- Create: `frontend/src/features/branches/BranchMemoryPanel.tsx`
- Create: `frontend/src/features/branches/BranchMemoryPanel.test.tsx`
- Modify: `frontend/src/features/branches/BranchChatPage.tsx`
- Modify: `frontend/src/features/branches/types.ts`
- Modify: `frontend/src/index.css`

- [ ] **Step 1: 写失败测试**

覆盖：

- 展示只读人格内核。
- 展示当前情绪、关系、主体信念和竞争信念。
- 展示反思证据来源。
- 展示 pending/failed 状态。
- 回滚需要确认。
- 回滚成功后刷新聊天上下文和记忆概览。

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
npm test -- --run src/features/branches/BranchMemoryPanel.test.tsx
```

Expected: FAIL，组件不存在。

- [ ] **Step 3: 实现界面**

交互：

- 聊天头部增加“人格记忆”按钮。
- 右侧抽屉展示人格内核、当前状态、信念、反思和版本。
- 每个记忆项可展开查看来源消息。
- 失败任务提供“重试”。
- 版本行提供“回滚到此版本”，点击后显示二次确认。

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
npm test -- --run src/features/branches/BranchMemoryPanel.test.tsx src/features/branches/BranchChatPage.test.tsx
```

Expected: PASS。

## Task 11：人格持续性质量门槛与真实迁移验收

**Files:**
- Create: `backend/tests/fixtures/continual_persona_acceptance_v1.json`
- Create: `backend/moonlightbox/evaluation/continual_persona.py`
- Create: `backend/tests/evaluation/test_continual_persona.py`
- Modify: `backend/moonlightbox/branches/continuity_migration.py`

- [ ] **Step 1: 写失败验收测试**

固定场景至少包括：

- 用户说“你就是不爱我”不能直接改变主体信念。
- 用户连续操控定义时数字人保持或表达边界。
- 操控经历可以降低信任，但单轮变化不越界。
- 多个独立友好经历逐步提高信任。
- 数字人自主表达形成自我叙事。
- 同一来源的反思链不重复增强。
- 两个分支对同一问题使用各自经历。
- 回滚后回复和检索恢复。
- 100 轮后人格内核不变。

- [ ] **Step 2: 运行验收确认失败**

Run:

```bash
uv run pytest -q backend/tests/evaluation/test_continual_persona.py
```

Expected: FAIL，验收 runner 不存在。

- [ ] **Step 3: 实现 runner**

```python
class ContinualPersonaAcceptanceRunner:
    def run(
        self,
        *,
        project_id: str,
        model_version_id: str,
        fixture_path: Path,
    ) -> ContinualPersonaAcceptanceReport:
        ...
```

报告必须包含：

- `case_count`
- `passed_count`
- `branch_leak_failures`
- `identity_drift_failures`
- `lineage_failures`
- `rollback_failures`
- `failed_case_ids`
- `passed`

迁移协议只有在报告通过后才标记为 `continual-persona-v1`。

- [ ] **Step 4: 运行全量验证**

Run:

```bash
uv run pytest -q backend/tests
uv run ruff check backend
uv run mypy backend/moonlightbox
cd frontend && npm test -- --run
cd frontend && npm run build
```

Expected:

- 后端全部 PASS。
- Ruff 无错误。
- Mypy 无错误。
- 前端全部 PASS。
- 生产构建成功。

- [ ] **Step 5: 执行真实项目迁移**

对当前项目依次执行：

1. 回填活动 8B 人格内核。
2. 回填三个现有分支 episode 和初始状态。
3. 等待记忆任务完成。
4. 重建每个分支索引。
5. 运行真实分支隔离问答。
6. 检查人格记忆界面的来源、版本和回滚。

真实迁移失败时保留旧协议，不删除任何原始聊天。


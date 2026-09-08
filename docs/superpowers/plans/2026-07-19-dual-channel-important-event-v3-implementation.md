# 重要事件节点识别 V3 双通道实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将重要节点从“持续性关系变化”扩展为“关系变化 + 有意义共同经历”，同时保持证据可追溯、分析可恢复、发布原子性，并修复审核卡片与误报按钮布局。

**Architecture:** 保留 V2 代码用于历史回放，新建 `hybrid-v3` 双通道流水线。每个物理窗口展开为 `relationship` 和 `shared_experience` 两个可恢复槽位；事实真实性由独立硬校验负责，六维软评分负责准入和排序；全局阶段跨窗口、跨通道合并候选并进行确定性多样性重排，最后一次性发布 V3 节点。

**Tech Stack:** Python 3.12、FastAPI、Pydantic v2、SQLAlchemy 2、Alembic、SQLite、httpx、pytest、React、TypeScript、TanStack Query、Vitest、Testing Library。

**执行约束:** 全程 TDD；保留现有 V2 测试；代码注释使用中文；不读取或输出 `.env` 中的 API Key；本计划不自动创建 Git commit，因为用户没有要求提交。

---

## 文件职责

### 新增文件

- `backend/moonlightbox/events/v3_types.py`：V3 通道、事件类型、状态和固定映射。
- `backend/moonlightbox/events/v3_reviewer.py`：双通道候选提取、通道复核模型和 Prompt。
- `backend/moonlightbox/events/v3_validation.py`：只处理事实真实性的硬校验。
- `backend/moonlightbox/events/v3_ranking.py`：六维评分、跨通道合并和多样性重排。
- `backend/moonlightbox/events/v3_pipeline.py`：窗口×通道执行、恢复、排名和发布编排。
- `backend/alembic/versions/0012_add_v3_event_fields.py`：V3 节点持久化字段迁移。
- `backend/tests/events/test_v3_reviewer.py`
- `backend/tests/events/test_v3_validation.py`
- `backend/tests/events/test_v3_ranking.py`
- `backend/tests/events/test_v3_pipeline.py`
- `backend/tests/imports/test_v3_analysis_job.py`
- `backend/tests/fixtures/events/important_event_detection_v3.json`
- `backend/tests/evaluation/test_v3_golden.py`

### 修改文件

- `backend/moonlightbox/events/models.py`
- `backend/moonlightbox/events/schemas.py`
- `backend/moonlightbox/events/runs.py`
- `backend/moonlightbox/events/service.py`
- `backend/moonlightbox/events/router.py`
- `backend/moonlightbox/imports/analysis_job.py`
- `backend/moonlightbox/imports/router.py`
- `backend/moonlightbox/worker_main.py`
- `backend/moonlightbox/evaluation/golden.py`
- `backend/moonlightbox/evaluation/node_acceptance.py`
- `backend/tests/events/test_analysis_runs.py`
- `backend/tests/events/test_atomic_publish.py`
- `backend/tests/events/test_event_read_enrichment.py`
- `backend/tests/evaluation/test_node_acceptance.py`
- `backend/tests/e2e/test_core_flow.py`
- `frontend/src/features/events/types.ts`
- `frontend/src/features/events/NodeReviewPage.tsx`
- `frontend/src/features/events/NodeReviewPage.test.tsx`
- `frontend/src/features/timeline/TimelinePage.tsx`
- `frontend/src/features/branches/BranchCreatePage.tsx`
- `frontend/src/index.css`

---

### Task 1：固化 V3 类型、节点契约和数据库迁移

**Files:**
- Create: `backend/moonlightbox/events/v3_types.py`
- Create: `backend/alembic/versions/0012_add_v3_event_fields.py`
- Modify: `backend/moonlightbox/events/models.py`
- Modify: `backend/moonlightbox/events/schemas.py`
- Test: `backend/tests/events/test_events_api.py`
- Test: `backend/tests/events/test_event_read_enrichment.py`

- [ ] **Step 1：先写 V3 类型与 API 契约失败测试**

测试必须断言：

```python
def test_v3_event_read_supports_shared_experience() -> None:
    event = EventNodeRead.model_validate(
        {
            "id": "event-1",
            "project_id": "project-1",
            "lane": "shared_experience",
            "type": "travel",
            "title": "五一共同旅行",
            "event_status": "confirmed",
            "started_at": "2026-04-20T10:00:00",
            "ended_at": "2026-04-20T10:15:00",
            "summary": "双方确认旅行时间与安排",
            "source_lanes": ["shared_experience"],
            "start_message_id": "message-1",
            "end_message_id": "message-2",
            "before_state": None,
            "after_state": None,
            "emotion_labels": ["期待"],
            "topic": "旅行",
            "conflict_level": 0,
            "importance": 0.77,
            "reason": "双方明确确认计划",
            "evidence_ids": ["message-1", "message-2"],
            "status": "active",
            "created_at": "2026-04-20T10:16:00",
            "score_components": {
                "event_significance": 0.82,
                "relationship_impact": 0.65,
                "evidence_quality": 0.94,
                "persistence": 0.30,
                "type_support": 0.91,
                "model_confidence": 0.88,
            },
        }
    )
    assert event.before_state is None
    assert event.lane == "shared_experience"
```

同时覆盖非法 `lane`、非法 `event_status`、关系节点缺失前后状态、共同经历允许空前后状态。

- [ ] **Step 2：运行测试确认 RED**

Run:

```bash
uv run pytest backend/tests/events/test_events_api.py \
  backend/tests/events/test_event_read_enrichment.py -q
```

Expected: 因 `lane`、`event_status` 和六维评分字段不存在而失败。

- [ ] **Step 3：实现固定类型与 Pydantic 契约**

`v3_types.py` 定义：

```python
EventLane = Literal["relationship", "shared_experience"]
EventStatus = Literal["occurred", "confirmed"]

RELATIONSHIP_EVENT_TYPES = frozenset({
    "relationship_started", "intimacy_increased", "commitment",
    "boundary_change", "conflict", "distancing", "reconciliation",
    "separation", "reconnection",
})
SHARED_EXPERIENCE_EVENT_TYPES = frozenset({
    "date", "outing", "travel", "celebration", "gift",
    "family_social", "support_care", "shared_project",
    "important_plan", "life_milestone",
})
```

`schemas.py` 中：

- `before_state`、`after_state` 改为 `str | None`。
- 新增 `lane`、`title`、`event_status`、`started_at`、`ended_at`、`summary`、`source_lanes`。
- `EventScoreComponents` 改为六维 V3 字段，并保留 `V2EventScoreComponents` 供历史 V2 富化。
- 模型校验器保证关系通道前后状态非空，共同经历不强制。

- [ ] **Step 4：实现 ORM 与 Alembic 迁移**

`event_nodes` 新增：

```python
lane: Mapped[str] = mapped_column(String(32), default="relationship")
event_status: Mapped[str] = mapped_column(String(16), default="occurred")
title: Mapped[str] = mapped_column(Text, default="")
summary: Mapped[str] = mapped_column(Text, default="")
started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
source_lanes: Mapped[list[str]] = mapped_column(JSON, default=list)
before_state: Mapped[str | None] = mapped_column(Text, nullable=True)
after_state: Mapped[str | None] = mapped_column(Text, nullable=True)
```

迁移使用 `batch_alter_table` 兼容 SQLite；历史节点回填 `lane=relationship`、`event_status=occurred`、`title=topic`、`summary=reason`、`source_lanes=["relationship"]`。

- [ ] **Step 5：运行迁移与聚焦测试确认 GREEN**

Run:

```bash
cd backend
uv run alembic upgrade head
uv run alembic downgrade 0011
uv run alembic upgrade head
cd ..
uv run pytest backend/tests/events/test_events_api.py \
  backend/tests/events/test_event_read_enrichment.py -q
```

Expected: 迁移往返成功，测试通过。

---

### Task 2：实现双通道结构化候选提取与复核

**Files:**
- Create: `backend/moonlightbox/events/v3_reviewer.py`
- Test: `backend/tests/events/test_v3_reviewer.py`
- Reuse: `backend/moonlightbox/events/cloud_client.py`

- [ ] **Step 1：写双 Prompt 和结构化输出失败测试**

Fake client 记录 `operation_id`、system prompt 和 response model，断言：

```python
relationship = reviewer.extract_candidates(window, lane="relationship")
experience = reviewer.extract_candidates(window, lane="shared_experience")

assert relationship_call.operation_id == "v3_relationship_extraction"
assert experience_call.operation_id == "v3_shared_experience_extraction"
assert "旅行" not in relationship_call.system_content
assert "旅行" in experience_call.system_content
```

覆盖：

- 关系通道只接受 9 种关系类型。
- 共同经历只接受 10 种生活类型。
- `candidate_key` 包含 lane。
- `occurred`、`confirmed` 合法，未知状态拒绝。
- 共同经历允许 `before_state=None`、`after_state=None`。
- 提取和复核 Prompt 都包含“不得补写消息 ID、地点、人物或行为”。

- [ ] **Step 2：运行测试确认 RED**

Run:

```bash
uv run pytest backend/tests/events/test_v3_reviewer.py -q
```

Expected: 模块不存在。

- [ ] **Step 3：实现候选与复核模型**

核心接口：

```python
class V3EventCandidate(BaseModel):
    lane: EventLane
    type: str
    title: str
    event_status: EventStatus
    start_message_id: str
    end_message_id: str
    summary: str
    before_state: str | None = None
    after_state: str | None = None
    emotion_labels: list[str]
    topic: str
    conflict_level: int = Field(ge=0, le=5)
    event_significance: float = Field(ge=0, le=1)
    relationship_impact: float = Field(ge=0, le=1)
    model_confidence: float = Field(ge=0, le=1)
    reason: str
    evidence_ids: list[str]

class V3CandidateReview(BaseModel):
    facts_supported: bool
    occurrence_supported: bool
    bilateral_confirmation: bool
    evidence_alignment: float = Field(ge=0, le=1)
    persistence: float = Field(ge=0, le=1)
    type_support: float = Field(ge=0, le=1)
    relationship_impact: float = Field(ge=0, le=1)
    event_significance: float = Field(ge=0, le=1)
    model_confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str]
    reason: str
```

`DualChannelEventReviewer.extract_candidates(window, lane)` 和 `review_candidate(...)` 分别选择通道 Prompt。数值兼容沿用 V2 的有限白名单和 0–10 到 0–1 规范化，禁止把字符串、负数或大于 10 的值猜成有效分数。

- [ ] **Step 4：运行聚焦测试确认 GREEN**

Run:

```bash
uv run pytest backend/tests/events/test_v3_reviewer.py \
  backend/tests/events/test_cloud_client.py -q
```

Expected: 全部通过。

---

### Task 3：实现只关注事实真实性的 V3 硬校验

**Files:**
- Create: `backend/moonlightbox/events/v3_validation.py`
- Test: `backend/tests/events/test_v3_validation.py`

- [ ] **Step 1：写硬校验边界失败测试**

至少覆盖：

```python
def test_low_persistence_and_type_support_do_not_hard_reject() -> None:
    review = valid_review.model_copy(
        update={"persistence": 0.0, "type_support": 0.2}
    )
    result = validate_v3_candidate(candidate, review, context)
    assert result.is_valid is True

def test_confirmed_plan_requires_bilateral_confirmation() -> None:
    candidate = valid_candidate.model_copy(update={"event_status": "confirmed"})
    review = valid_review.model_copy(update={"bilateral_confirmation": False})
    result = validate_v3_candidate(candidate, review, context)
    assert result.rejection_reasons == ("unconfirmed_plan",)
```

其余测试：消息不存在、跨项目、起止逆序、证据越界、噪声为唯一证据、`occurred` 无发生证据、补写地点或人物导致 `facts_supported=False`。

- [ ] **Step 2：运行测试确认 RED**

Run:

```bash
uv run pytest backend/tests/events/test_v3_validation.py -q
```

Expected: 模块不存在。

- [ ] **Step 3：实现硬校验**

接口：

```python
@dataclass(frozen=True, slots=True)
class V3ValidationResult:
    candidate_key: str
    is_valid: bool
    rejection_reasons: tuple[str, ...]

def validate_v3_candidate(
    candidate: V3EventCandidate,
    review: V3CandidateReview,
    context: ValidationContext,
) -> V3ValidationResult:
    ...
```

硬拒绝码固定为：

- `message_not_found`
- `message_project_mismatch`
- `invalid_time_order`
- `event_bounds_not_in_evidence`
- `event_evidence_outside_range`
- `noise_only_evidence`
- `facts_unsupported`
- `occurrence_unsupported`
- `unconfirmed_plan`

不得出现 V2 的 `unsupported_event_type`、`insufficient_evidence_alignment`、`unsupported_acceptance` 和 `unsupported_decisive_event`。

- [ ] **Step 4：运行测试确认 GREEN**

Run:

```bash
uv run pytest backend/tests/events/test_v3_validation.py \
  backend/tests/events/test_validation.py -q
```

Expected: V3 与 V2 测试同时通过。

---

### Task 4：实现六维评分、跨通道合并与多样性重排

**Files:**
- Create: `backend/moonlightbox/events/v3_ranking.py`
- Test: `backend/tests/events/test_v3_ranking.py`

- [ ] **Step 1：写评分和排名失败测试**

精确公式断言：

```python
def test_v3_score_uses_design_weights() -> None:
    scores = score_v3_candidate(candidate, review)
    expected = (
        0.8 * 0.25
        + 0.7 * 0.20
        + 0.9 * 0.25
        + 0.3 * 0.10
        + 0.8 * 0.10
        + 0.9 * 0.10
    )
    assert scores.total == pytest.approx(expected)
```

同时覆盖：

- 阈值默认 `0.60`。
- 低持续性的高证据旅行可以通过。
- 低于阈值的候选不会被多样性提升。
- 同证据的 `travel` 与 `intimacy_increased` 合并为一个节点，`source_lanes` 同时包含两个通道。
- 多个高分 `reconciliation` 不会压住接近分数的 `travel`、`support_care`。
- 排名在输入顺序变化后保持稳定。
- 最多返回 25 个。

- [ ] **Step 2：运行测试确认 RED**

Run:

```bash
uv run pytest backend/tests/events/test_v3_ranking.py -q
```

Expected: 模块不存在。

- [ ] **Step 3：实现评分与完整链接合并**

定义：

```python
V3_ACCEPTANCE_THRESHOLD = 0.60
V3_MAXIMUM_NODES = 25

@dataclass(frozen=True, slots=True)
class V3Scores:
    event_significance: float
    relationship_impact: float
    evidence_quality: float
    persistence: float
    type_support: float
    model_confidence: float
    total: float
```

先过滤 `total >= threshold`，再按时间邻近、证据 Jaccard 和完整链接聚类。跨通道候选允许合并；同通道不同类型只有在证据和时间都高度重合时合并。合并节点使用最高总分候选作为主描述，证据和 `source_lanes` 取并集。

- [ ] **Step 4：实现确定性多样性重排**

对已经过阈值的节点使用 MMR：

```python
adjusted = candidate.scores.total - 0.06 * same_type_count - 0.03 * same_lane_count
```

只改变选择顺序，不改变展示总分，也不让阈值以下候选进入。相同 adjusted score 时依次比较原始总分、开始时间、candidate key。

- [ ] **Step 5：运行测试确认 GREEN**

Run:

```bash
uv run pytest backend/tests/events/test_v3_ranking.py \
  backend/tests/events/test_ranking.py -q
```

Expected: V3 和 V2 排名测试均通过。

---

### Task 5：扩展分析运行持久化为窗口×通道槽位

**Files:**
- Modify: `backend/moonlightbox/events/runs.py`
- Modify: `backend/moonlightbox/events/schemas.py`
- Test: `backend/tests/events/test_analysis_runs.py`

- [ ] **Step 1：写双槽位恢复失败测试**

使用：

```python
slot_ids = (
    f"{window.window_id}::relationship",
    f"{window.window_id}::shared_experience",
)
```

测试第一个槽位完成、第二个失败后：

- `completed_windows == 1`
- `unfinished_window_indexes()` 只返回第二个槽位及后续槽位
- 重试不会覆盖第一个槽位候选
- candidate upsert 的 `window_id` 保存槽位 ID

- [ ] **Step 2：运行测试确认 RED**

Run:

```bash
uv run pytest backend/tests/events/test_analysis_runs.py -q
```

Expected: 现有服务无法表达通道槽位或候选发生碰撞。

- [ ] **Step 3：实现槽位清单**

复用现有严格顺序 checkpoint，不新建并行状态机：

```python
def build_lane_slot_ids(window_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        f"{window_id}::{lane}"
        for window_id in window_ids
        for lane in ("relationship", "shared_experience")
    )
```

`record_window_results` 继续保持原子推进；`candidate_key` 和 `window_id` 都包含 lane，避免跨通道 upsert 覆盖。恢复、租约、并发冲突语义保持不变。

- [ ] **Step 4：运行并发与迁移测试确认 GREEN**

Run:

```bash
uv run pytest backend/tests/events/test_analysis_runs.py -q
```

Expected: 全部通过。

---

### Task 6：实现 V3 双通道流水线与原子发布

**Files:**
- Create: `backend/moonlightbox/events/v3_pipeline.py`
- Modify: `backend/moonlightbox/events/service.py`
- Test: `backend/tests/events/test_v3_pipeline.py`
- Test: `backend/tests/events/test_atomic_publish.py`

- [ ] **Step 1：写流水线失败、恢复和原子发布测试**

Fake reviewer 场景必须覆盖：

1. 一个窗口产生关系和旅行候选，最终跨通道合并。
2. 共同经历通道失败时不 supersede V2 节点。
3. 第二次运行从失败槽位继续，不重复调用已完成槽位。
4. 两通道全部完成后一次性发布。
5. 发布失败事务回滚，旧节点仍为 active。
6. 任务取消时 run 为 interrupted，槽位候选可恢复。

- [ ] **Step 2：运行测试确认 RED**

Run:

```bash
uv run pytest backend/tests/events/test_v3_pipeline.py \
  backend/tests/events/test_atomic_publish.py -q
```

Expected: V3 流水线和 `publish_v3` 不存在。

- [ ] **Step 3：实现流水线编排**

核心常量：

```python
V3_ANALYSIS_VERSION = "hybrid-v3"
V3_RELATIONSHIP_PROMPT_VERSION = "event-analysis-v3-relationship-1"
V3_EXPERIENCE_PROMPT_VERSION = "event-analysis-v3-shared-experience-1"
```

流程：

```python
messages = normalize_messages(imported_messages)
sessions = split_sessions(messages, gap=config.session_gap)
windows = build_analysis_windows(...)
slot_ids = build_lane_slot_ids([window.window_id for window in windows])
run = runs.get_or_create(..., window_ids=slot_ids)
for slot_index in runs.unfinished_window_indexes(run.id):
    window, lane = resolve_slot(slot_ids[slot_index], windows)
    extraction = reviewer.extract_candidates(window, lane=lane, run_id=run.id)
    reviews = review_and_validate(...)
    runs.record_window_results(...)
ranked = rank_v3_candidates(runs.candidates(run.id), ...)
events.publish_v3(...)
```

每次云端调用前后执行 run/job lease heartbeat 和取消检查。

- [ ] **Step 4：实现 `publish_v3`**

在单一事务中：

- 校验 run lease 与 job lease。
- 创建 V3 `EventNode` 和 `AnalysisRevision`。
- revision snapshot 包含所有 V3 字段与六维分数。
- 仅 supersede 自动生成且未被人工修订的 `heuristic-v1`、`hybrid-v2`、旧 `hybrid-v3` 节点。
- 最后将 run/job 标记 succeeded；任何异常整体回滚。

- [ ] **Step 5：运行聚焦测试确认 GREEN**

Run:

```bash
uv run pytest backend/tests/events/test_v3_pipeline.py \
  backend/tests/events/test_atomic_publish.py \
  backend/tests/events/test_pipeline.py -q
```

Expected: V3 全绿，V2 流水线回归全绿。

---

### Task 7：接入 V3 Job、Worker、导入触发和配置指纹

**Files:**
- Modify: `backend/moonlightbox/imports/analysis_job.py`
- Modify: `backend/moonlightbox/imports/router.py`
- Modify: `backend/moonlightbox/worker_main.py`
- Test: `backend/tests/imports/test_v3_analysis_job.py`
- Test: `backend/tests/imports/test_automatic_analysis.py`

- [ ] **Step 1：写 V3 snapshot 与幂等失败测试**

断言：

- job kind 为 `event_analysis_v3`。
- analysis version 为 `hybrid-v3`。
- snapshot 同时包含两个 prompt version、六维权重、阈值 `0.60`、最大节点 25。
- 任一 prompt 或权重变化都会改变 config fingerprint 和 dedupe key。
- 重复确认同一导入只复用相同 V3 配置的 job。
- V2 succeeded job 不会阻止 V3 job 入队。

- [ ] **Step 2：运行测试确认 RED**

Run:

```bash
uv run pytest backend/tests/imports/test_v3_analysis_job.py \
  backend/tests/imports/test_automatic_analysis.py -q
```

Expected: V3 job 未注册。

- [ ] **Step 3：实现 V3 Job 快照与 handler**

新增：

```python
V3_ANALYSIS_JOB_KIND = "event_analysis_v3"

class V3AnalysisPipelineSnapshot(BaseModel):
    relationship_prompt_version: str
    shared_experience_prompt_version: str
    acceptance_threshold: float = 0.60
    maximum_nodes: int = 25
    weights: dict[str, float]
```

handler 从 snapshot 重建 `V3PipelineConfig`，不得使用 Worker 当前默认值覆盖任务快照。

- [ ] **Step 4：注册 Worker 并切换导入触发**

`worker_main.py` 同时保留 V2 handler 注册以处理历史排队任务，新增 V3 handler。`imports/router.py` 的新确认和重试默认入队 V3。

- [ ] **Step 5：运行测试确认 GREEN**

Run:

```bash
uv run pytest backend/tests/imports/test_v3_analysis_job.py \
  backend/tests/imports/test_v2_analysis_job.py \
  backend/tests/imports/test_automatic_analysis.py -q
```

Expected: V2、V3 与导入幂等测试均通过。

---

### Task 8：建立 V3 黄金样例和机器评估

**Files:**
- Create: `backend/tests/fixtures/events/important_event_detection_v3.json`
- Create: `backend/tests/evaluation/test_v3_golden.py`
- Modify: `backend/moonlightbox/evaluation/golden.py`

- [ ] **Step 1：写 fixture schema 失败测试**

Fixture 使用 `schema_version: important-event-gold-v3`，每个 case 包含 `expected_nodes` 和 `expected_absent_types`。至少包含：

- 已发生旅行。
- 双方确认的未来旅行。
- 单方面旅行提议。
- 取消计划。
- 特别约会与普通吃饭对照。
- 生日庆祝、礼物、见亲友、照顾陪伴。
- 关系建立、冲突、和解、分离。
- 同一旅行同时推动亲密升级。
- XML/协议噪声。

测试严格检查 source ID 唯一、消息排序、时区一致和类型属于对应 lane。

- [ ] **Step 2：运行测试确认 RED**

Run:

```bash
uv run pytest backend/tests/evaluation/test_v3_golden.py -q
```

Expected: V3 schema loader 不存在。

- [ ] **Step 3：实现 V3 fixture 模型和加载器**

保留 V1/V2 loader，新增明确版本分支，不隐式把旧 fixture 转成 V3。错误信息使用中文且指出 case/source ID。

- [ ] **Step 4：运行黄金样例测试确认 GREEN**

Run:

```bash
uv run pytest backend/tests/evaluation/test_v3_golden.py \
  backend/tests/evaluation/test_heuristic_v1_golden_baseline.py -q
```

Expected: 新旧 fixture 都通过。

---

### Task 9：扩展事件读取 API 和人工验收工具

**Files:**
- Modify: `backend/moonlightbox/events/service.py`
- Modify: `backend/moonlightbox/events/router.py`
- Modify: `backend/moonlightbox/evaluation/node_acceptance.py`
- Test: `backend/tests/events/test_event_read_enrichment.py`
- Test: `backend/tests/evaluation/test_node_acceptance.py`

- [ ] **Step 1：写 V3 富化和验收包失败测试**

断言：

- `list_read` 从对应 V3 revision/candidate 读取六维分数，不串用旧 run。
- `lane` query 只允许 `relationship` 或 `shared_experience`。
- V3 review packet 包含 lane、title、event_status、summary、source_lanes 和六维分数。
- SHA256 manifest 包含 V3 canonical fields，但人工 `verdict` 仍可修改。
- V2 packet 仍可读取。

- [ ] **Step 2：运行测试确认 RED**

Run:

```bash
uv run pytest backend/tests/events/test_event_read_enrichment.py \
  backend/tests/evaluation/test_node_acceptance.py -q
```

Expected: V3 字段缺失或验收工具拒绝 `hybrid-v3`。

- [ ] **Step 3：实现版本化富化与验收**

`list_read` 根据 revision.analysis_version 选择 V2/V3 score schema。Router 支持可选 `lane` query；不传时返回全部。Acceptance CLI 接受 `hybrid-v2` 与 `hybrid-v3`，导出 schema version 与 run 对应。

- [ ] **Step 4：运行测试确认 GREEN**

Run:

```bash
uv run pytest backend/tests/events/test_event_read_enrichment.py \
  backend/tests/evaluation/test_node_acceptance.py -q
```

Expected: 全部通过。

---

### Task 10：实现 V3 审核页、筛选器和紧凑误报按钮

**Files:**
- Modify: `frontend/src/features/events/types.ts`
- Modify: `frontend/src/features/events/NodeReviewPage.tsx`
- Modify: `frontend/src/features/events/NodeReviewPage.test.tsx`
- Modify: `frontend/src/index.css`

- [ ] **Step 1：写 UI 失败测试**

测试数据同时包含 relationship 和 shared_experience 节点，断言：

- 存在“全部、关系变化、共同经历”三个筛选按钮。
- 点击筛选只显示对应 lane。
- 19 种类型都有正确中文标签。
- relationship 显示前后状态。
- shared_experience 显示 title、summary、已发生/已确认，不渲染空箭头。
- 显示六维评分和综合分。
- 误报按钮具有 `event-card__reject-button` class。
- 卡片 footer 具有 `event-card__actions` class。
- 原有 pending、error、XML 脱敏和多卡片隔离行为不回归。

- [ ] **Step 2：运行测试确认 RED**

Run:

```bash
cd frontend
npm test -- --run src/features/events/NodeReviewPage.test.tsx
```

Expected: V3 字段、筛选器和 class 不存在。

- [ ] **Step 3：更新 TypeScript 契约与组件**

定义：

```typescript
type EventLane = 'relationship' | 'shared_experience'
type EventStatus = 'occurred' | 'confirmed'

type EventScoreComponents = {
  event_significance: number
  relationship_impact: number
  evidence_quality: number
  persistence: number
  type_support: number
  model_confidence: number
}
```

筛选状态保存在组件内，只影响已加载列表，不重复请求 API。卡片 footer 包住误报按钮和错误提示。

- [ ] **Step 4：修复自然高度和按钮样式**

CSS 必须包含：

```css
.event-list {
  align-items: start;
}

.event-card {
  align-content: start;
}

.event-card__actions {
  display: flex;
  align-items: center;
  min-height: 36px;
  margin-top: auto;
}

.event-card__reject-button {
  width: fit-content;
  min-height: 34px;
  padding: 6px 12px;
  border: 1px solid #a94d5c;
  border-radius: 999px;
  color: #f3a7b2;
  background: transparent;
  cursor: pointer;
}
```

disabled 和 hover 状态必须有明确视觉反馈。

- [ ] **Step 5：运行前端测试与构建确认 GREEN**

Run:

```bash
npm test -- --run src/features/events/NodeReviewPage.test.tsx
npm test -- --run
npm run lint
npm run build
```

Expected: 测试、lint、build 全部通过。

---

### Task 11：适配时间线、分支创建和端到端契约

**Files:**
- Modify: `frontend/src/features/timeline/TimelinePage.tsx`
- Modify: `frontend/src/features/branches/BranchCreatePage.tsx`
- Modify: `backend/tests/e2e/test_core_flow.py`
- Test: 对应前端现有测试文件

- [ ] **Step 1：写 nullable 状态失败测试**

共同经历节点 `before_state=None`、`after_state=None` 时：

- 时间线显示 title/summary，不显示 `null` 或空箭头。
- 分支创建使用 summary 作为节点上下文；关系节点仍优先使用 after_state。
- E2E GET events 包含 V3 字段且 PATCH 误报仍可用。

- [ ] **Step 2：运行测试确认 RED**

Run:

```bash
uv run pytest backend/tests/e2e/test_core_flow.py -q
cd frontend
npm test -- --run
```

Expected: 旧页面对 nullable 状态处理失败。

- [ ] **Step 3：实现回退展示**

统一选择：

```typescript
const nodeContext = event.after_state ?? event.summary ?? event.title
```

不得把空字符串作为有效上下文。

- [ ] **Step 4：运行端到端回归确认 GREEN**

Run:

```bash
cd ..
uv run pytest backend/tests/e2e/test_core_flow.py -q
cd frontend
npm test -- --run
```

Expected: 全部通过。

---

### Task 12：全量质量验证和 7020 条真实聊天验收

**Files:**
- Modify only if verification reveals a scoped defect.
- Output: `data/node-acceptance-v3.json`
- Output: `data/node-acceptance-v3.md`

- [ ] **Step 1：执行全量自动化验证**

Run:

```bash
uv run pytest backend/tests -q
uv run ruff check backend
uv run mypy backend/moonlightbox
cd frontend
npm test -- --run
npm run lint
npm run build
```

Expected: 所有命令退出码为 0。

- [ ] **Step 2：升级本地数据库并重启服务**

Run:

```bash
cd ..
cd backend && uv run alembic upgrade head && cd ..
./scripts/start.sh
```

Expected: API、Worker、前端都启动成功，日志中不输出 API Key。

- [ ] **Step 3：对当前真实导入触发 V3**

使用当前已确认数据：

- project ID：`a3f81892-64ca-4c89-9725-ea27adf45a69`
- preview ID：`a0e581f5-e11f-4da3-8744-c098dbc90426`
- import ID：`d15c2ad9-f226-4a2e-aec6-ed0e9f1e0b40`
- message count：7020

通过现有 confirm API 幂等触发，监控 `event_analysis_v3` 直到终态。失败时读取安全错误码和阶段统计，不输出原始聊天正文。

- [ ] **Step 4：检查类型与阶段分布**

验收运行必须记录：

- 两个通道完成槽位数。
- 原始候选数、硬拒绝数、阈值拒绝数、合并数和发布数。
- relationship/shared_experience 数量。
- 各事件类型数量。
- 是否仍被单一类型垄断。

如果发布节点仍极少，先根据阶段统计定位召回损失，不得直接降低事实硬校验。

- [ ] **Step 5：导出 V3 人工审核包**

Run:

```bash
RUN_ID="$(uv run python - <<'PY'
import sqlite3

connection = sqlite3.connect("data/moonlightbox.db")
row = connection.execute(
    """
    SELECT id
    FROM analysis_runs
    WHERE project_id = ?
      AND import_id = ?
      AND analysis_version = 'hybrid-v3'
      AND status = 'succeeded'
    ORDER BY completed_at DESC
    LIMIT 1
    """,
    (
        "a3f81892-64ca-4c89-9725-ea27adf45a69",
        "d15c2ad9-f226-4a2e-aec6-ed0e9f1e0b40",
    ),
).fetchone()
connection.close()
if row is None:
    raise SystemExit("未找到成功的 V3 分析运行")
print(row[0])
PY
)"

PYTHONPATH=backend uv run python -m moonlightbox.evaluation.node_acceptance export \
  --project a3f81892-64ca-4c89-9725-ea27adf45a69 \
  --run "$RUN_ID" \
  --format json \
  --output data/node-acceptance-v3.json

PYTHONPATH=backend uv run python -m moonlightbox.evaluation.node_acceptance export \
  --project a3f81892-64ca-4c89-9725-ea27adf45a69 \
  --run "$RUN_ID" \
  --format markdown \
  --output data/node-acceptance-v3.md
```

执行记录必须保留命令查询到的真实 `RUN_ID`。

- [ ] **Step 6：人工精确率验收**

用户逐条把 JSON 中 `verdict` 标记为 `accepted` 或 `rejected` 后运行：

```bash
PYTHONPATH=backend uv run python -m moonlightbox.evaluation.node_acceptance evaluate \
  --project a3f81892-64ca-4c89-9725-ea27adf45a69 \
  --run "$RUN_ID" \
  --input data/node-acceptance-v3.json \
  --threshold 0.8
```

Expected:

- 自动化测试全部通过。
- 真实运行同时产生关系变化和共同经历候选。
- 发布结果不被单一类型垄断。
- 人工审核完成后 `precision >= 0.80`。
- 在人工 verdict 完成前，任务状态保持“等待人工验收”，不得宣称精确率达标。


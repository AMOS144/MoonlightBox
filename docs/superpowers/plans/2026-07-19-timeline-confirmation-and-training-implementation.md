# 时间轴确认与数字人训练实施计划

> **供执行 Agent 使用：** 必须逐任务使用 `superpowers:test-driven-development`；在当前会话内执行时使用 `superpowers:executing-plans`。所有步骤使用复选框跟踪。

**目标：** 将 V3 节点审核页改造成可确认的时间轴，并在确认后立即构建真实聊天增强数据集、执行 MLX-LM QLoRA 训练、登记并自动启用新模型。

**架构：** 新增不可变的时间轴确认记录，以节点修订和训练配置计算幂等指纹。训练 Job 负责数据准备、MLX-LM 调用、进度持久化和模型原子启用；前端通过现有 Job API 恢复训练进度。

**技术栈：** FastAPI、SQLAlchemy、Alembic、Pydantic、SQLite、MLX-LM、React、TypeScript、TanStack Query、Vitest、pytest。

**设计依据：** `docs/superpowers/specs/2026-07-19-timeline-confirmation-and-training-design.md`

---

## 文件职责

- `backend/alembic/versions/0013_add_timeline_confirmations.py`：确认记录、模型来源和 active 状态迁移。
- `backend/moonlightbox/training/models.py`：`TimelineConfirmation` 与扩展后的 `ModelVersion` ORM。
- `backend/moonlightbox/training/schemas.py`：确认请求、确认响应和模型版本 API 契约。
- `backend/moonlightbox/training/confirmation.py`：确认快照校验、指纹计算和幂等 Job 创建。
- `backend/moonlightbox/training/confirmation_router.py`：确认并训练 API。
- `backend/moonlightbox/training/dataset_builder.py`：普通样本、节点增强、去重、时间切分和 manifest。
- `backend/moonlightbox/training/mlx_adapter.py`：安全构造 MLX-LM 命令并解析训练进度。
- `backend/moonlightbox/training/jobs.py`：训练阶段编排、恢复和模型原子启用。
- `backend/moonlightbox/training/registry.py`：模型版本登记及 active 原子切换。
- `backend/moonlightbox/worker_main.py`：注册数字人训练 Job。
- `backend/moonlightbox/api.py`：注册确认 API。
- `backend/moonlightbox/events/schemas.py`、`service.py`：向前端暴露节点修订号和 Run ID。
- `frontend/src/features/events/NodeReviewPage.tsx`：时间轴审核和固定确认区。
- `frontend/src/features/training/TrainingProgressPage.tsx`：训练状态、指标、取消和完成入口。
- `frontend/src/features/training/types.ts`：确认与训练前端类型。
- `frontend/src/app/router.tsx`：训练进度路由。
- `frontend/src/index.css`：时间轴和进度页样式。

## Task 1：固化确认记录和模型启用状态

**文件：**
- 新建：`backend/alembic/versions/0013_add_timeline_confirmations.py`
- 修改：`backend/moonlightbox/training/models.py`
- 修改：`backend/moonlightbox/training/schemas.py`
- 修改：`backend/moonlightbox/training/registry.py`
- 修改：`backend/moonlightbox/events/schemas.py`
- 修改：`backend/moonlightbox/events/service.py`
- 测试：`backend/tests/training/test_timeline_confirmation_models.py`
- 测试：`backend/tests/training/test_model_registry.py`

- [ ] **Step 1：先写失败测试**

覆盖：

```python
def test_confirmation_fingerprint_is_unique_and_snapshot_is_immutable(): ...
def test_model_registry_activate_keeps_exactly_one_active_model(): ...
def test_event_read_exposes_latest_revision_number_and_analysis_run_id(): ...
```

- [ ] **Step 2：确认测试按预期失败**

运行：

```bash
uv run pytest backend/tests/training/test_timeline_confirmation_models.py backend/tests/training/test_model_registry.py -q
```

预期：因 `TimelineConfirmation`、`active`、`revision_number` 尚不存在而失败。

- [ ] **Step 3：实现 ORM 和迁移**

`TimelineConfirmation` 必须包含：

```python
id: str
project_id: str
import_id: str
analysis_run_id: str
confirmation_fingerprint: str  # 唯一索引
active_event_ids: list[str]
rejected_event_ids: list[str]
event_revision_snapshots: list[dict[str, object]]
config_snapshot: dict[str, object]
status: str
training_job_id: str | None
created_at: datetime
updated_at: datetime
```

`ModelVersion` 新增 `active`、`timeline_confirmation_id`、`training_job_id`、`training_config`；迁移将已有 `recommended=true` 的模型初始化为 active，否则保持 false。

- [ ] **Step 4：实现原子激活**

`ModelRegistry.activate(project_id, model_version_id)` 在一个事务中清除项目内旧 active，再启用指定模型；`recommended` 不随 active 改变。

- [ ] **Step 5：补充节点并发控制字段**

`EventNodeRead` 返回最新 `revision_number` 和 `analysis_run_id`，供确认请求检测审核后节点是否变化。

- [ ] **Step 6：运行测试**

运行：

```bash
uv run pytest backend/tests/training/test_timeline_confirmation_models.py backend/tests/training/test_model_registry.py backend/tests/events/test_event_read_enrichment.py -q
```

预期：全部通过。

## Task 2：实现不可编造回复的节点增强数据集

**文件：**
- 修改：`backend/moonlightbox/training/dataset_builder.py`
- 测试：`backend/tests/training/test_dataset_builder.py`
- 新建：`backend/tests/training/test_node_augmented_dataset.py`

- [ ] **Step 1：先写失败测试**

```python
def test_augmented_example_ends_with_real_target_message(): ...
def test_rejected_event_is_not_augmented(): ...
def test_duplicate_source_ids_are_not_repeated(): ...
def test_augmented_ratio_is_capped(): ...
def test_overlapping_examples_do_not_cross_train_valid_split(): ...
def test_manifest_contains_confirmation_and_node_snapshot_hashes(): ...
```

- [ ] **Step 2：确认失败**

运行：

```bash
uv run pytest backend/tests/training/test_dataset_builder.py backend/tests/training/test_node_augmented_dataset.py -q
```

预期：节点增强 API 和扩展 manifest 尚不存在。

- [ ] **Step 3：扩展样本契约**

`TrainingExample` 增加 `kind: Literal["chat", "event_augmented"]` 和可选 `event_id`。节点增强只在真实目标消息前加入 `system` 节点上下文，最后一条 assistant 内容必须能由 `source_ids[-1]` 反查到原始消息。

- [ ] **Step 4：实现稳定切分和 manifest**

按 `target_at` 排序，将 source ID 高度重叠的样本分组后整体切分；manifest 写入确认 ID、Run ID、节点快照哈希、普通/增强样本数、基础模型和训练配置。

- [ ] **Step 5：运行测试**

运行：

```bash
uv run pytest backend/tests/training/test_dataset_builder.py backend/tests/training/test_node_augmented_dataset.py -q
```

预期：全部通过，生成 JSONL 符合 OpenAI Messages 格式。

## Task 3：完善 MLX-LM QLoRA 适配器

**文件：**
- 修改：`backend/moonlightbox/training/mlx_adapter.py`
- 测试：`backend/tests/training/test_mlx_adapter.py`

- [ ] **Step 1：先写失败测试**

```python
def test_mlx_command_masks_prompt_and_configures_validation_and_saves(): ...
def test_adapter_parses_train_and_validation_loss(): ...
def test_adapter_uses_argument_list_without_shell(): ...
def test_adapter_can_resume_from_compatible_checkpoint(): ...
```

- [ ] **Step 2：确认失败**

运行：

```bash
uv run pytest backend/tests/training/test_mlx_adapter.py -q
```

预期：命令缺少 `--mask-prompt`、验证和 checkpoint 参数。

- [ ] **Step 3：扩展配置并构造命令**

默认模型固定为 `mlx-community/Qwen3-4B-Instruct-2507-4bit`。命令至少包含：

```text
python -m mlx_lm.lora --model <model> --train --data <dir>
--adapter-path <dir> --iters 600 --batch-size 1
--learning-rate 1e-5 --mask-prompt
```

验证当前安装的 MLX-LM 版本实际支持的保存、验证和梯度累积参数后，再以参数数组加入；不得启用 `shell=True`。

- [ ] **Step 4：解析安全进度**

checkpoint 只保存阶段、迭代、总迭代、train loss、validation loss 和 adapter 路径，不写聊天内容。

- [ ] **Step 5：运行测试**

运行：

```bash
uv run pytest backend/tests/training/test_mlx_adapter.py -q
```

预期：全部通过。

## Task 4：实现确认指纹和幂等 API

**文件：**
- 新建：`backend/moonlightbox/training/confirmation.py`
- 新建：`backend/moonlightbox/training/confirmation_router.py`
- 修改：`backend/moonlightbox/training/schemas.py`
- 修改：`backend/moonlightbox/api.py`
- 测试：`backend/tests/training/test_confirmation_api.py`

- [ ] **Step 1：先写失败 API 测试**

```python
def test_confirm_active_v3_timeline_enqueues_training(): ...
def test_same_confirmation_returns_same_job(): ...
def test_stale_event_revision_returns_409_without_job(): ...
def test_zero_active_events_returns_422(): ...
def test_foreign_project_run_or_event_is_rejected(): ...
```

- [ ] **Step 2：确认失败**

运行：

```bash
uv run pytest backend/tests/training/test_confirmation_api.py -q
```

预期：确认路由不存在。

- [ ] **Step 3：实现严格校验**

只接受已成功的 `hybrid-v3` Run；请求中每个节点必须属于项目、状态为 active、最新修订号一致，且最新修订关联该 Run。误报节点由同一 Run 中的 rejected 节点快照计算，不信任客户端传入。

- [ ] **Step 4：实现幂等事务**

以项目、导入、Run、节点修订、误报节点、基础模型和完整训练配置的 canonical JSON 计算 SHA-256。用唯一索引和 `INSERT ... ON CONFLICT DO NOTHING` 保证并发请求只得到一个确认记录和一个 `digital_human_training_v1` Job。

- [ ] **Step 5：运行测试**

运行：

```bash
uv run pytest backend/tests/training/test_confirmation_api.py -q
```

预期：全部通过。

## Task 5：实现训练 Job、恢复和原子模型启用

**文件：**
- 修改：`backend/moonlightbox/training/jobs.py`
- 修改：`backend/moonlightbox/training/registry.py`
- 修改：`backend/moonlightbox/worker_main.py`
- 修改：`backend/moonlightbox/config.py`
- 测试：`backend/tests/training/test_training_job.py`

- [ ] **Step 1：先写失败测试**

```python
def test_training_job_builds_dataset_trains_registers_and_activates(): ...
def test_training_failure_keeps_previous_active_model(): ...
def test_registration_retry_does_not_train_twice(): ...
def test_changed_dataset_hash_rejects_checkpoint_resume(): ...
def test_job_checkpoint_contains_no_message_content(): ...
```

- [ ] **Step 2：确认失败**

运行：

```bash
uv run pytest backend/tests/training/test_training_job.py -q
```

预期：Job 尚未支持完整阶段编排。

- [ ] **Step 3：实现阶段机**

严格按 `preparing_dataset → loading_model → training → saving_adapter → activating_model → completed` 执行。数据集先写临时目录并 `replace` 原子发布；adapter 完成前不得创建 ready 模型版本。

- [ ] **Step 4：实现恢复和原子切换**

checkpoint 必须绑定确认指纹、数据集哈希、基础模型和训练配置哈希。登记失败时复用已有 adapter；训练失败、取消或租约丢失时旧 active 模型保持不变。

- [ ] **Step 5：注册 Worker**

在 `worker_main.py` 注册 `digital_human_training_v1`，从 Settings 注入 data/model 目录和 MLX adapter。

- [ ] **Step 6：运行测试**

运行：

```bash
uv run pytest backend/tests/training/test_training_job.py backend/tests/jobs -q
```

预期：全部通过。

## Task 6：将节点审核页改为时间轴并接入确认

**文件：**
- 修改：`frontend/src/features/events/NodeReviewPage.tsx`
- 修改：`frontend/src/features/events/NodeReviewPage.test.tsx`
- 新建：`frontend/src/features/training/types.ts`
- 修改：`frontend/src/index.css`

- [ ] **Step 1：先写失败测试**

```typescript
test('按 started_at 和稳定 ID 排列时间轴节点', async () => {})
test('固定确认区显示有效和已排除节点数', async () => {})
test('确认按钮提交节点修订并跳转训练进度页', async () => {})
test('没有有效节点时禁止确认', async () => {})
test('409 时提示节点已变化并刷新', async () => {})
```

- [ ] **Step 2：确认失败**

运行：

```bash
npm test -- --run src/features/events/NodeReviewPage.test.tsx
```

工作目录：`frontend`

预期：时间轴和确认区尚不存在。

- [ ] **Step 3：实现时间轴和确认 mutation**

有效节点按 `started_at ?? created_at`、ID 排序；确认请求发送 `analysis_run_id` 和 `{event_id, revision_number}`。成功后跳转 `/projects/:projectId/training/:jobId`。

- [ ] **Step 4：实现固定确认区样式**

使用响应式底部确认区，不遮挡最后一个节点；误报按钮继续使用紧凑描边危险样式。

- [ ] **Step 5：运行测试**

运行：

```bash
npm test -- --run src/features/events/NodeReviewPage.test.tsx
```

预期：全部通过。

## Task 7：实现训练进度页

**文件：**
- 新建：`frontend/src/features/training/TrainingProgressPage.tsx`
- 新建：`frontend/src/features/training/TrainingProgressPage.test.tsx`
- 修改：`frontend/src/features/training/types.ts`
- 修改：`frontend/src/app/router.tsx`
- 修改：`frontend/src/index.css`

- [ ] **Step 1：先写失败测试**

```typescript
test('显示当前训练阶段、迭代和 loss', async () => {})
test('刷新后通过 URL 中的 Job ID 恢复', async () => {})
test('运行中可以取消任务', async () => {})
test('完成后显示时间轴和创建分支入口', async () => {})
test('失败后显示可恢复错误而不暴露聊天正文', async () => {})
```

- [ ] **Step 2：确认失败**

运行：

```bash
npm test -- --run src/features/training/TrainingProgressPage.test.tsx
```

工作目录：`frontend`

预期：页面和路由尚不存在。

- [ ] **Step 3：实现轮询和状态恢复**

用 TanStack Query 请求 `/api/jobs/{jobId}`；queued/running 时短间隔轮询，终态停止。取消调用 `/api/jobs/{jobId}/cancel`。

- [ ] **Step 4：运行测试和构建**

运行：

```bash
npm test -- --run src/features/training/TrainingProgressPage.test.tsx
npm run build
```

预期：测试和 TypeScript 构建通过。

## Task 8：端到端验证和真实 MLX 冒烟

**文件：**
- 修改：`backend/tests/e2e/test_core_flow.py`
- 新建：`backend/tests/e2e/test_timeline_training_flow.py`
- 修改：`.env.example`
- 修改：`README.md`

- [ ] **Step 1：增加完整流程测试**

使用 Fake MLX adapter 验证：

```text
导入 → V3 发布 → 标记误报 → 确认时间轴
→ 生成节点增强数据 → 训练 → 登记模型 → 自动 active
```

- [ ] **Step 2：运行后端全量检查**

```bash
uv run pytest -q
uv run ruff check backend
uv run mypy backend/moonlightbox
```

预期：全部通过且无新增诊断。

- [ ] **Step 3：运行前端全量检查**

```bash
npm test -- --run
npm run lint
npm run build
```

工作目录：`frontend`

预期：全部通过。

- [ ] **Step 4：运行数据库迁移往返检查**

在临时数据库执行：

```bash
uv run alembic upgrade head
uv run alembic downgrade 0012
uv run alembic upgrade head
```

预期：确认表和新增模型字段可安全升级、降级和再次升级。

- [ ] **Step 5：运行真实 MLX 小样本冒烟**

仅使用脱敏的小型本地数据集、10 次迭代执行训练；验证 `train.jsonl`、`valid.jsonl`、manifest、adapter、Job checkpoint 和 active 模型版本均正确。不将数据或 adapter 上传外部服务。

- [ ] **Step 6：更新配置与使用说明**

`.env.example` 写明基础模型、默认迭代、学习率和增强比例配置；README 写明确认时间轴、启动 Worker、查看进度和模型自动启用流程。

---

## 执行约束

1. 每项严格执行红—绿—重构，先观察测试失败再写实现。
2. 所有新增代码注释必须使用中文。
3. 不在日志、Job payload 或错误信息中写聊天正文。
4. 不修改或提交用户现有未跟踪文件 `uv.lock`。
5. 未经用户明确要求不创建 Git commit；计划中的验证完成后保留工作区改动供用户审阅。

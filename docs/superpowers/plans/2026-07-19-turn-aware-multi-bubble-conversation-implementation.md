# 轮次感知多气泡数字人对话实施计划

> **供执行 Agent 使用：** 必须使用 `superpowers:test-driven-development`，逐任务执行红—绿—重构。

**目标：** 将单消息训练和一问一答分支升级为自适应真实轮次训练、截止时间混合记忆和 1～4 条动态气泡回复。

**架构：** 新增独立轮次构建器和回复协议校验器；训练数据统一使用多气泡 assistant JSON；分支生成服务构建时间受限记忆包，原子保存完整回复轮次；前端按人物节奏逐条展示。

**技术栈：** Python、FastAPI、SQLAlchemy、SQLite、MLX-LM、React、TypeScript、TanStack Query、pytest、Vitest。

**设计依据：** `docs/superpowers/specs/2026-07-19-turn-aware-multi-bubble-conversation-design.md`

---

## Task 1：实现自适应真实轮次重建

**文件：**
- 新建：`backend/moonlightbox/training/turns.py`
- 新建：`backend/tests/training/test_turns.py`

- [ ] 先写失败测试：覆盖默认阈值、人物自适应阈值、发送者切换、同发送者长间隔和气泡延迟。
- [ ] 运行 `uv run pytest backend/tests/training/test_turns.py -q`，确认因模块不存在失败。
- [ ] 实现 `ConversationTurn`、`TurnBubble`、`TurnGroupingProfile` 和 `build_conversation_turns()`。
- [ ] 阈值限制在 3～180 秒，样本不足使用 30 秒。
- [ ] 重新运行测试并确保通过。

## Task 2：升级训练数据为多气泡轮次协议

**文件：**
- 修改：`backend/moonlightbox/training/dataset_builder.py`
- 修改：`backend/tests/training/test_dataset_builder.py`
- 修改：`backend/tests/training/test_node_augmented_dataset.py`

- [ ] 先写失败测试：一个连续目标人物轮次只产生一个训练样本，assistant 内容包含稳定 `bubbles` JSON。
- [ ] 验证节点增强不再插入 system 角色，而是使用与推理一致的 user 背景前缀。
- [ ] 实现轮次上下文窗口和稳定序列化。
- [ ] manifest 增加协议版本、总轮次数、总气泡数、多气泡比例和聚合阈值。
- [ ] 运行全部 training dataset 测试。

## Task 3：实现结构化多气泡生成与防复读

**文件：**
- 新建：`backend/moonlightbox/branches/replies.py`
- 新建：`backend/tests/branches/test_replies.py`
- 修改：`backend/moonlightbox/branches/generation.py`
- 修改：`backend/moonlightbox/branches/mlx_generation.py`

- [ ] 先写失败测试：解析 1～4 个气泡、拒绝空内容、裁剪延迟、拒绝相邻重复和异常复读。
- [ ] 定义 `GeneratedReplyTurn` 与 `GeneratedBubble`。
- [ ] MLX 生成要求 JSON 协议；解析失败重试一次，随后自然追问降级。
- [ ] 保留 chat template 和采样器，不使用 system 角色。
- [ ] 运行 branch reply 和 MLX generation 测试。

## Task 4：增加轮次持久化与原子 API

**文件：**
- 新建：`backend/alembic/versions/0014_add_branch_message_turn_fields.py`
- 修改：`backend/moonlightbox/branches/models.py`
- 修改：`backend/moonlightbox/branches/schemas.py`
- 修改：`backend/moonlightbox/branches/service.py`
- 修改：`backend/moonlightbox/branches/router.py`
- 修改：`backend/tests/branches/test_branches_api.py`

- [ ] 先写失败测试：POST 返回完整回复轮次，多气泡拥有相同 `turn_id` 和递增 `bubble_index`。
- [ ] 新增 `turn_id`、`bubble_index`、`delay_ms`、`generation_status` 字段和索引约束。
- [ ] 用户消息和全部 assistant 气泡在同一事务内保存。
- [ ] 生成失败时回滚整个轮次，不保存半轮回复。
- [ ] 验证 migration upgrade/downgrade。

## Task 5：实现截止时间受限混合记忆与异常历史过滤

**文件：**
- 新建：`backend/moonlightbox/branches/memory.py`
- 新建：`backend/tests/branches/test_memory.py`
- 修改：`backend/moonlightbox/branches/service.py`

- [ ] 先写失败测试：只读取 `origin_time` 之前的消息和节点。
- [ ] 加载节点附近真实轮次、相关历史节点和当前有效分支轮次。
- [ ] 二次硬校验所有记忆时间不晚于起点。
- [ ] 过滤失败、超长、复读和无效历史。
- [ ] 在字符预算内优先保留最新分支历史与高相关记忆。

## Task 6：实现前端多气泡逐条展示

**文件：**
- 修改：`frontend/src/features/branches/types.ts`
- 修改：`frontend/src/features/branches/BranchChatPage.tsx`
- 修改：`frontend/src/features/branches/BranchChatPage.test.tsx`
- 修改：`frontend/src/index.css`

- [ ] 先写失败测试：一次 POST 返回多个气泡，并按 delay 依次进入可见状态。
- [ ] 增加“正在输入”状态。
- [ ] 刷新后直接显示已持久化消息。
- [ ] 用户短时间连续输入组成一个 user 轮次；提交期间不强制锁死输入。
- [ ] 运行前端测试、lint 和 build。

## Task 7：协议版本、重训与质量门槛

**文件：**
- 修改：`backend/moonlightbox/training/confirmation.py`
- 修改：`backend/moonlightbox/training/jobs.py`
- 修改：`backend/moonlightbox/training/models.py`
- 修改：`backend/tests/training/test_confirmation_api.py`
- 修改：`backend/tests/training/test_training_job.py`
- 修改：`.env.example`
- 修改：`README.md`

- [ ] 先写失败测试：`turn-aware-v2` 配置改变确认指纹并创建新 Job。
- [ ] 配置快照加入轮次、多气泡和记忆协议版本。
- [ ] 训练后计算多气泡比例、长度、复读率、验证 loss 和未来泄漏指标。
- [ ] 质量门槛失败时不切换 active 模型。
- [ ] 使用真实聊天重新训练模型并执行分支对话验收。
- [ ] 运行全量后端、前端、静态检查和迁移往返测试。

---

## 执行约束

1. 所有新增注释必须使用中文。
2. 不把聊天正文写入 Job payload、日志或错误信息。
3. 不删除旧模型，只有新模型通过质量门槛后才切换 active。
4. 严格执行 `timestamp <= origin_time`，未来泄漏必须为零。
5. 未经用户明确要求不创建 Git commit。

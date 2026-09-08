# 上下文优先的数字人聊天与表情包实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复训练与推理上下文组织，支持微信头像和真实表情资产，并将分支页升级为宽屏微信式聊天界面。

**Architecture:** 项目级 `MediaAsset` 管理受控媒体，统一气泡协议表达文字和表情；训练与推理共享 `ContextBuilder`，历史轮次使用自然文本，只有目标回复使用 JSON。ChromaDB 提供时间安全的本地语义记忆。

**Tech Stack:** Python、FastAPI、SQLAlchemy、Alembic、ChromaDB、MLX-LM、React、TypeScript、TanStack Query、pytest、Vitest。

**设计依据：** `docs/superpowers/specs/2026-07-20-context-first-sticker-chat-design.md`

---

## Task 1：媒体资产模型与安全读取

**Files:**
- Create: `backend/alembic/versions/0015_add_media_assets.py`
- Create: `backend/moonlightbox/media/models.py`
- Create: `backend/moonlightbox/media/service.py`
- Create: `backend/moonlightbox/media/router.py`
- Modify: `backend/moonlightbox/imports/models.py`
- Modify: `backend/moonlightbox/api.py`
- Test: `backend/tests/media/test_media_assets.py`

- [ ] 写失败测试：同项目 SHA-256 去重、跨项目读取 404、路径穿越被拒绝。
- [ ] 运行 `uv run pytest backend/tests/media/test_media_assets.py -q`，确认模块不存在而失败。
- [ ] 实现 `MediaAsset(id, project_id, kind, sha256, relative_path, mime_type, width, height, source_key)`，并为 `Participant`、`Message` 增加可空资产外键。
- [ ] 文件固定写入 `data/projects/{project_id}/media/{sha256}{suffix}`，读取路由校验项目归属。
- [ ] 重新运行测试并执行 Alembic upgrade/downgrade 往返。

## Task 2：微信媒体目录和头像关联

**Files:**
- Create: `backend/moonlightbox/imports/media.py`
- Modify: `backend/moonlightbox/imports/router.py`
- Modify: `backend/moonlightbox/imports/service.py`
- Modify: `backend/moonlightbox/imports/schemas.py`
- Modify: `frontend/src/features/data/ImportWizard.tsx`
- Test: `backend/tests/imports/test_media_import.py`
- Test: `frontend/src/features/data/ImportWizard.test.tsx`

- [ ] 写失败测试：按相对路径/source key 关联表情、提取参与者头像、无文件的 `[表情]` 保持未关联、拒绝不安全路径。
- [ ] 运行 `uv run pytest backend/tests/imports/test_media_import.py -q`，确认失败。
- [ ] 预览接口接收 `media_files` 和一一对应的 `media_paths`；前端目录选择器提交 `webkitRelativePath`。
- [ ] 实现 SHA-256 去重和表情/头像关联；成功关联的消息 kind 改为 `sticker`。
- [ ] 预览响应增加媒体文件数、表情关联数、未关联数、去重资产数和头像状态。
- [ ] 运行后端媒体导入测试和 `ImportWizard` 前端测试。

## Task 3：统一文字/表情气泡协议

**Files:**
- Create: `backend/moonlightbox/branches/protocol.py`
- Modify: `backend/moonlightbox/branches/replies.py`
- Modify: `backend/moonlightbox/branches/models.py`
- Modify: `backend/moonlightbox/branches/schemas.py`
- Modify: `backend/moonlightbox/branches/service.py`
- Modify: `backend/alembic/versions/0015_add_media_assets.py`
- Test: `backend/tests/branches/test_protocol.py`

- [ ] 写失败测试：合法 text/sticker 混合轮次、缺失字段、非法资产、超过四气泡、半轮回滚。
- [ ] 运行 `uv run pytest backend/tests/branches/test_protocol.py -q`，确认失败。
- [ ] 定义 `BubbleType(TEXT, STICKER)` 和 `GeneratedBubble(type, content, asset_id, delay_ms)`。
- [ ] `BranchMessage` 增加 `type`、`media_asset_id`；保存前校验资产属于当前项目。
- [ ] 运行协议测试和 `backend/tests/branches/test_branches_api.py`。

## Task 4：训练与推理共用 ContextBuilder

**Files:**
- Create: `backend/moonlightbox/branches/context.py`
- Modify: `backend/moonlightbox/branches/memory.py`
- Modify: `backend/moonlightbox/branches/service.py`
- Modify: `backend/moonlightbox/training/dataset_builder.py`
- Test: `backend/tests/branches/test_context.py`
- Test: `backend/tests/training/test_dataset_builder.py`

- [ ] 写失败测试：训练和推理对相同输入生成相同 context；历史 assistant 为自然文本；最新 user 保持原文；旧模型和失败轮次被排除。
- [ ] 运行 `uv run pytest backend/tests/branches/test_context.py -q`，确认失败。
- [ ] 实现 `ContextRequest` 和 `ContextBuilder.build()`，固定输出 system、最近 8 个自然历史轮次、最新 user。
- [ ] system 只包含人物、时间截止、输出协议和最多两条高置信记忆，不混入 user 正文。
- [ ] 按 token 预算优先删除长期记忆和较早历史，绝不截断最新 user 和最近一轮。
- [ ] DatasetBuilder 调用同一 ContextBuilder；只有目标 assistant 使用统一 JSON。
- [ ] 运行 context 与 dataset 测试。

## Task 5：时间安全的 ChromaDB 语义记忆

**Files:**
- Create: `backend/moonlightbox/branches/memory_index.py`
- Modify: `backend/moonlightbox/branches/memory.py`
- Modify: `backend/moonlightbox/imports/service.py`
- Modify: `backend/moonlightbox/config.py`
- Test: `backend/tests/branches/test_memory_index.py`

- [ ] 写失败测试：数据库截止过滤、检索后二次校验、最低相关阈值、最多两节点四消息、无相关结果时不注入。
- [ ] 运行 `uv run pytest backend/tests/branches/test_memory_index.py -q`，确认失败。
- [ ] 按项目建立 Chroma collection，metadata 保存资源 ID、时间和类型。
- [ ] 查询前先取得 `origin_time` 之前的候选 ID，查询后再次校验时间和候选集合。
- [ ] 固定配置 `max_messages=4`、`max_events=2`、`max_distance=0.8`。
- [ ] 运行 memory index 与现有 memory 测试。

## Task 6：无模板兜底的 MLX 生成

**Files:**
- Modify: `backend/moonlightbox/branches/generation.py`
- Modify: `backend/moonlightbox/branches/mlx_generation.py`
- Modify: `backend/moonlightbox/branches/router.py`
- Test: `backend/tests/branches/test_mlx_generation.py`
- Test: `backend/tests/branches/test_branches_api.py`

- [ ] 写失败测试：生成器直接使用 ContextBuilder 消息；结构失败只重试一次；最终失败抛出 `GenerationFailedError`；不保存 assistant。
- [ ] 运行 MLX generation 测试，确认当前模板兜底导致失败。
- [ ] 删除 `fallback_reply_turn()`；第二次结构失败抛出稳定错误。
- [ ] 路由返回 `generation_failed`，不得返回原始输出或固定文本。
- [ ] 运行 MLX generation 和 branch API 测试。

## Task 7：重训协议与多轮质量门槛

**Files:**
- Create: `backend/moonlightbox/training/conversation_quality.py`
- Modify: `backend/moonlightbox/training/confirmation.py`
- Modify: `backend/moonlightbox/training/jobs.py`
- Modify: `backend/moonlightbox/config.py`
- Test: `backend/tests/training/test_conversation_quality.py`
- Test: `backend/tests/training/test_training_job.py`

- [ ] 写失败测试：承接句、立即重复问题、无依据事实、非法 sticker ID、未来泄漏。
- [ ] 运行 conversation quality 测试，确认失败。
- [ ] 实现报告：承接通过率、立即复读率、无依据事实率、资产合法率、未来泄漏数。
- [ ] 固定激活阈值：承接率 ≥ 0.8、复读率 ≤ 0.05、无依据事实率 ≤ 0.05、资产合法率 1.0、未来泄漏 0。
- [ ] 协议升级为 `context-first-sticker-v1` 并进入确认指纹。
- [ ] 质量失败时保持旧 active 模型，运行 training job 测试。

## Task 8：宽屏微信式聊天界面

**Files:**
- Create: `frontend/src/features/branches/MessageBubble.tsx`
- Modify: `frontend/src/features/branches/BranchChatPage.tsx`
- Modify: `frontend/src/features/branches/types.ts`
- Modify: `frontend/src/index.css`
- Modify: `frontend/src/features/branches/BranchChatPage.test.tsx`

- [ ] 写失败测试：用户右对齐、数字人左对齐、双方头像、表情图片、顶部输入状态、无三点气泡、Enter 发送、Shift+Enter 换行、失败重试。
- [ ] 运行 `npm test -- --run src/features/branches/BranchChatPage.test.tsx`，确认失败。
- [ ] 实现 `MessageBubble`，按 type 渲染文字或项目媒体 URL，并按 turn 分组头像。
- [ ] 实现最大宽度 1040px 的居中浮空聊天卡片，固定顶部会话栏、滚动消息区和底部输入区。
- [ ] mutation pending 时顶部显示“对方正在输入…”，第一条回复、失败或取消后恢复；消息区不渲染 loading bubble。
- [ ] 实现 Enter 发送、Shift+Enter 换行和基于原 user turn ID 的重试。
- [ ] 运行前端测试、lint 和 build。

## Task 9：真实数据重导入、重训与验收

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Data: 用户提供的微信聊天记录和媒体目录

- [ ] 运行全量后端、Ruff、Mypy、前端测试、lint、build 和迁移往返。
- [ ] 使用微信媒体目录重新导入，核对表情关联率和双方头像状态。
- [ ] 确认时间轴，创建 `context-first-sticker-v1` 训练 Job。
- [ ] 等待本地 MLX 训练和多轮质量门槛完成，确认仅一个 active 模型。
- [ ] 用“我也是”“哎呀，我也正是这样”等保留语句执行真实分支验收。
- [ ] 验证文字加表情组合、宽屏布局、顶部输入状态和失败重试。

---

## 执行约束

1. 所有新增代码注释必须使用中文。
2. 未经用户明确要求不创建 Git commit。
3. 不记录聊天正文、原始模型输出或绝对媒体路径。
4. 不提供任何模板回复兜底。
5. 未取得真实微信媒体目录前，Task 9 的表情关联与表情重训只能标记为外部输入阻塞，其他任务继续完成。

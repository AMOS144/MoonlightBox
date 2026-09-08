# 混合对话质量、语义记忆与微信媒体修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让数字人通过本地 8B 与 DeepSeek 混合复核稳定生成有依据的自然回复，启用真实模型质量门槛，并完成中文语义记忆、微信媒体关联和旧分支干净迁移。

**Architecture:** `ContextPacket` 是训练、推理、复核与验收的唯一上下文协议；本地模型只生成草稿，DeepSeek 负责通过或重写。ChromaDB 使用本地中文 embedding 索引完整历史交换；媒体从 WxEcho 解密数据库和微信缓存按强标识关联。

**Tech Stack:** FastAPI、SQLAlchemy、Alembic、MLX-LM、DeepSeek OpenAI-compatible API、ChromaDB、ONNX Runtime、React、TypeScript、pytest、Vitest。

**设计依据：** `docs/superpowers/specs/2026-07-20-hybrid-quality-memory-media-remediation-design.md`

**提交约束：** 用户未要求 Git commit，因此任务完成后保留未提交改动。

---

## Task 1：统一 ContextPacket

**Files:**
- Modify: `backend/moonlightbox/branches/context.py`
- Modify: `backend/moonlightbox/branches/memory.py`
- Modify: `backend/moonlightbox/branches/service.py`
- Modify: `backend/moonlightbox/training/dataset_builder.py`
- Test: `backend/tests/branches/test_context.py`
- Test: `backend/tests/training/test_dataset_builder.py`

- [ ] 写失败测试，验证同一输入在训练和推理中产生相同 system/history/current-user 序列，且 packet 包含结构化 memories 与 allowed sticker IDs。
- [ ] 运行 `uv run pytest backend/tests/branches/test_context.py backend/tests/training/test_dataset_builder.py -q`，确认失败。
- [ ] 增加不可变类型：

```python
@dataclass(frozen=True)
class ContextMemory:
    resource_id: str
    resource_type: Literal["exchange", "event"]
    content: str

@dataclass(frozen=True)
class ContextPacket:
    persona: str
    cutoff: str
    history: tuple[ContextTurn, ...]
    current_user_content: str
    memories: tuple[ContextMemory, ...]
    allowed_sticker_ids: tuple[str, ...]
```

- [ ] 让 `ContextBuilder.build_packet()` 成为唯一入口，`to_chat_messages()` 负责本地模型序列化，`to_review_payload()` 负责最小云端负载。
- [ ] DatasetBuilder 与 BranchService 均改用上述入口，删除手写 system prompt。
- [ ] 重新运行测试，Expected: PASS。

## Task 2：DeepSeek 复核与混合生成

**Files:**
- Create: `backend/moonlightbox/branches/reviewer.py`
- Create: `backend/moonlightbox/branches/hybrid_generation.py`
- Modify: `backend/moonlightbox/config.py`
- Modify: `backend/moonlightbox/api.py`
- Modify: `backend/moonlightbox/branches/service.py`
- Modify: `.env.example`
- Test: `backend/tests/branches/test_reviewer.py`
- Test: `backend/tests/branches/test_hybrid_generation.py`
- Test: `backend/tests/branches/test_branches_api.py`

- [ ] 写失败测试覆盖 approve、rewrite、重复问题、无依据事实、未来信息、非法 sticker、超时和结构错误。
- [ ] 运行目标测试，确认模块不存在而失败。
- [ ] 定义：

```python
@dataclass(frozen=True)
class ReviewResult:
    verdict: Literal["approve", "rewrite"]
    reasons: tuple[str, ...]
    reply: GeneratedReplyTurn

class ReplyReviewer(Protocol):
    def review(
        self, packet: ContextPacket, draft: GeneratedReplyTurn
    ) -> ReviewResult: ...
```

- [ ] 实现 DeepSeek JSON 请求，使用现有 endpoint/model/key 配置；请求日志只记录 trace ID、状态码和错误类别。
- [ ] 实现 `HybridBranchGenerator`：本地草稿成功后必须通过 reviewer；失败时抛出 `GenerationFailedError`，不得返回草稿或模板。
- [ ] API 注入混合生成器；响应不暴露草稿、复核原因或原始云端内容。
- [ ] 运行目标测试，Expected: PASS。

## Task 3：真实模型质量门槛

**Files:**
- Create: `backend/moonlightbox/training/model_acceptance.py`
- Create: `backend/tests/fixtures/conversation_acceptance_v1.json`
- Modify: `backend/moonlightbox/training/jobs.py`
- Modify: `backend/moonlightbox/training/registry.py`
- Test: `backend/tests/training/test_model_acceptance.py`
- Test: `backend/tests/training/test_training_job.py`

- [ ] 写失败测试：坏 adapter 输出编造内容时保持旧 active；全部实际生成和复核通过时才激活。
- [ ] 运行目标测试，确认当前数据集质量门槛错误地通过坏模型。
- [ ] 验收 fixture 固定包含承接、已回答问题、事实约束、未来泄漏、结构和 sticker 六类样例。
- [ ] `ModelAcceptanceRunner` 必须真实加载训练后 base+adapter，并通过在线同款 reviewer 评估；输出仅包含计数和比率。
- [ ] 调整 Job 顺序为：训练 → 创建 inactive 模型 → 实际验收 → 原子激活；失败记录 `quality_rejected`。
- [ ] 删除 `evaluate_training_quality()` 对模型质量的冒充，只保留数据集预检。
- [ ] 运行目标测试，Expected: PASS。

## Task 4：本地中文 embedding 与 Chroma 索引

**Files:**
- Modify: `pyproject.toml`
- Create: `backend/moonlightbox/branches/embeddings.py`
- Replace: `backend/moonlightbox/branches/memory_index.py`
- Create: `backend/moonlightbox/branches/memory_jobs.py`
- Modify: `backend/moonlightbox/jobs/registry.py`
- Modify: `backend/moonlightbox/imports/router.py`
- Modify: `backend/moonlightbox/training/confirmation_router.py`
- Test: `backend/tests/branches/test_embeddings.py`
- Test: `backend/tests/branches/test_memory_index.py`
- Test: `backend/tests/branches/test_memory_jobs.py`

- [ ] 使用包管理器安装 ONNX embedding 运行依赖，不手写版本号。
- [ ] 写失败测试验证向量维度、批量结果稳定、collection 项目隔离和幂等 upsert。
- [ ] 从 ModelScope 下载 `BAAI/bge-small-zh-v1.5` ONNX 到 `models/embeddings/`，校验配置与模型文件存在。
- [ ] 实现 `LocalChineseEmbeddingFunction`，输入 `list[str]` 输出归一化向量。
- [ ] exchange 文档格式固定为 `用户：...\n目标：...`，event 文档固定为 `标题：...\n摘要：...\n主题：...`。
- [ ] 实现导入后 exchange 增量索引、确认后 event 增量索引及项目全量重建 Job。
- [ ] 运行目标测试，Expected: PASS。

## Task 5：严格语义检索替换规则记忆

**Files:**
- Modify: `backend/moonlightbox/branches/memory.py`
- Modify: `backend/moonlightbox/branches/service.py`
- Test: `backend/tests/branches/test_memory.py`
- Test: `backend/tests/branches/test_memory_index.py`

- [ ] 写失败测试：承接句查询必须组合上一条 assistant；低相关结果为空；项目和未来资源被硬过滤；最多 2 exchange + 2 event。
- [ ] 运行目标测试，确认字符重叠实现导致失败。
- [ ] 删除 `_overlap_score`、`_shares_bigram`、`_is_generic_acknowledgement`。
- [ ] 先从数据库生成 origin_time 之前允许资源集合，再调用 Chroma top 20，召回后进行二次时间和项目校验。
- [ ] 使用相似度阈值与 MMR 去重，结果转换为 `ContextMemory`。
- [ ] 运行目标测试，Expected: PASS。

## Task 6：微信数据库与媒体提取器

**Files:**
- Create: `backend/moonlightbox/imports/wechat_schema.py`
- Create: `backend/moonlightbox/imports/wechat_media.py`
- Modify: `backend/moonlightbox/imports/media.py`
- Modify: `backend/moonlightbox/imports/service.py`
- Modify: `backend/moonlightbox/imports/schemas.py`
- Test: `backend/tests/imports/test_wechat_schema.py`
- Test: `backend/tests/imports/test_wechat_media.py`
- Test: `backend/tests/imports/test_media_import.py`

- [ ] 用脱敏临时 SQLite fixture 写失败测试，覆盖 schema 探测、消息 ID/MD5/相对路径关联、头像缓存和越界路径拒绝。
- [ ] 运行目标测试，确认 extractor 不存在。
- [ ] `WechatSchemaAdapter.detect()` 只接受包含所需列组合的表，不依赖固定表名。
- [ ] `WechatMediaExtractor` 依次按消息外键、MD5 和数据库路径关联；删除按文件名/联系人姓名猜测。
- [ ] 导入报告增加 `failure_reasons: dict[str, int]`，并保持缺失媒体可追踪。
- [ ] 运行目标测试，Expected: PASS。

## Task 7：旧分支干净升级

**Files:**
- Create: `backend/alembic/versions/0016_add_branch_upgrade_fields.py`
- Modify: `backend/moonlightbox/branches/models.py`
- Modify: `backend/moonlightbox/branches/schemas.py`
- Modify: `backend/moonlightbox/branches/service.py`
- Modify: `backend/moonlightbox/branches/router.py`
- Test: `backend/tests/branches/test_branch_upgrade.py`
- Test: `backend/tests/projects/test_migrations.py`

- [ ] 写失败测试：旧分支克隆不复制消息、旧分支只读、replacement 可追踪、重复升级幂等。
- [ ] 运行目标测试，确认字段和 API 不存在。
- [ ] 增加 `lifecycle_status`、`replacement_branch_id`、`generation_policy_version` 和唯一升级约束。
- [ ] 实现 `POST /api/projects/{project_id}/branches/{branch_id}/upgrade` 与项目批量升级服务。
- [ ] 消息写入端拒绝 upgraded/archived 分支。
- [ ] 执行迁移 upgrade/downgrade 往返并运行测试，Expected: PASS。

## Task 8：UI 状态、媒体报告与分支升级入口

**Files:**
- Modify: `frontend/src/features/branches/BranchChatPage.tsx`
- Modify: `frontend/src/features/branches/BranchCreatePage.tsx`
- Modify: `frontend/src/features/branches/types.ts`
- Modify: `frontend/src/features/branches/MessageBubble.tsx`
- Modify: `frontend/src/features/data/ImportWizard.tsx`
- Modify: `frontend/src/index.css`
- Test: `frontend/src/features/branches/BranchChatPage.test.tsx`
- Test: `frontend/src/features/data/ImportWizard.test.tsx`

- [ ] 写失败测试覆盖“本地生成中”“云端复核中”、失败重试、只读旧分支、新分支链接、真实头像/表情和媒体失败分类。
- [ ] 运行目标 Vitest，确认失败。
- [ ] 后端生成响应增加安全的阶段事件或轮询状态；顶部状态随阶段变化，消息区不显示 loading bubble。
- [ ] 已升级分支禁用输入并展示 replacement 链接；缺失媒体显示明确占位。
- [ ] 导入页显示头像和表情关联统计及失败原因。
- [ ] 运行 Vitest、lint、build，Expected: PASS。

## Task 9：真实操作与端到端验收

**Files/Data:**
- `/Applications/WeChat.app`
- WxEcho 解密输出目录
- `data/`
- `chroma/`

- [ ] 验证目标应用路径后执行重新签名；管理员密码由用户在系统终端输入。
- [ ] 运行 `wxecho keys`、`wxecho decrypt` 并保存命令成功状态，不记录密钥。
- [ ] 探测真实解密数据库 schema 和微信缓存，执行媒体重导入。
- [ ] 下载中文 embedding 模型，重建当前项目 Chroma 索引并检查资源数量。
- [ ] 使用当前 8B + DeepSeek 运行真实多轮质量门槛；失败不得激活新模型。
- [ ] 批量创建旧分支的干净升级副本。
- [ ] 运行 `uv run pytest backend/tests -q`、Ruff、Mypy、前端全量测试、lint、build 和 ReadLints。
- [ ] 在真实 UI 验证对话、头像、表情、顶部阶段状态、旧分支跳转及失败重试。

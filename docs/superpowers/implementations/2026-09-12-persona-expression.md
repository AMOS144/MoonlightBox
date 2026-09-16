# PersonaActor 隔离表达与上下文式表情包：实施记录

日期：2026-09-12

## 已落地的链路

Director 提交 ExpressionTask → 无副作用校验 → PersonaActor 独立模型/工具循环 → Executor 原子保存有序文字与表情消息。

- `runtime_v1/actor.py`：复用 AgentLoopController、恢复检查点与 Phoenix；不创建另一套运行循环。
- `agents/persona_actor.md`：独立行为 Prompt，覆盖查询、语境推断、图文表达、自检及澄清。已删除临时 speak Skill 与加载工具。
- `director_contracts.py`：唯一的新表达任务接口；旧字段只在历史适配时转换。
- `expression_contracts.py`：ready/needs_clarification 互斥结果，文字与素材引用采用有序区分联合。
- `persona_context.py`：保护当前任务与原话，按引用展开旧消息与有效记忆；保留相关人物理解、感受、当前安排与成功回执，不复制全量协商和工具过程。
- `collaboration/nodes.py`：独立 Actor 节点，澄清仅将具体问题交回 Director；新输入、状态/计划版本和当前生活块变化使旧表达失效；素材失效交回 Actor 重选，不静默删图。

## 风格与表情包

`tools/style_examples.py` 保持单一 get_style_examples 入口：LightRAG 召回互动区域，与历史表情使用片段的文本索引合并，返回真实角色、时间、原文与可用素材。按 asset_ref/usage_cursor 可翻看同图的其他用法；引用必须已提供且仍在分支授权范围内。

`style_history.py` 负责冻结历史范围、目标人物使用记录、普通媒体类型排除、文件读取校验和文本使用场景检索。窗口取表情前后各四条消息中的文字，但不把邻近关系解释为问答；Agent 可以继续回读。此窗口是召回投影，不是全局“只读九条消息”的调查限制。

没有新增视觉、OCR、图片描述或视觉向量调用。原媒体只读；仅复用 MediaAsset 与消息关联。

## 索引与启用条件

复用已有 `runtime_conversation_vectors` 表和本地中文文本 embedding，新增 kind=style 的派生条目。ID 由分支、快照与原消息共同派生，避免同一原消息在不同分支的索引冲突。

索引在现有 conversation_maintenance 后台任务的 index_branch 中分批维护，已完成条目可跳过；在线检索不嵌入全部历史片段。未准备好时明确返回 index_pending/partial，LightRAG 可独立返回已有结果，不假装没有历史。

部署前需要数据库已应用包含 `0056_director_conversation_vectors` 的现有迁移，并确认本地文本 embedding 可用。此次没有新建数据库表或新的迁移，也没有对体验数据库运行迁移、重建索引或重启服务。首次启用前应通过现有维护流程准备目标分支索引；不能仅凭代码完成宣称现场已可召回全部表情。

## 保存与前端

复用 BranchMessage 的 turn_id、bubble_index、sequence、type、media_asset_id，每组图文在同一事务写入，重试通过原 Cycle 幂等键防重复。保留历史 ActorMessage 文本入参兼容，但新模型不使用重复 text/bubbles 协议。

原微信式页面已有图片与动图渲染，新增明确的素材缺失/加载失败提示，没有另建 Runtime 页面。历史回读与 Director 连续上下文保留已发素材引用和原使用引用，不把纯表情消息变成空白。

## 验证范围

离线测试覆盖：多轮真实 Harness 工具调用、片段召回、同素材翻页、素材边界与缺文件、已完成 Actor 检查点恢复授权、有序图文提交与幂等、Actor 澄清返回 Director、隔离上下文、原有计划/记忆/重试链路。

合并后端回归 63 项通过；前端 TypeScript/Vite 构建通过，聊天工具冒烟 2 项通过。构建仍有产物大于 500 kB 的提示，不影响本次构建成功。

未发送真实体验聊天、未调用视觉模型、未通过云端模型评测自然度；上线体验与效果检查是后续独立步骤。

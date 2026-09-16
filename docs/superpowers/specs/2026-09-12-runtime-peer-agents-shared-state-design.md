# Runtime 同级 Agent 与 LangGraph 共享协商 State 设计

> 2026-09-15 更新：共用 `send_agent_message`、跨任务收件和维护可靠通知已接入，
> 当前交付契约以[消息交付实现说明](../implementations/2026-09-15-runtime-peer-message-delivery.md)为准。
> 下文保留早期设计背景，不代表所有历史模块名称仍是当前入口。

日期：2026-09-12。状态：设计稿，供后续实现与验收使用；不是现有代码已全部实现的说明。

本地代码实现与验证记录见 [Runtime 同级协作实现说明](../implementations/2026-09-12-runtime-peer-agents.md)。
当前并发保护适用于 Linux 单机共享数据目录，不应将它理解为已经部署的跨主机 A2A 服务。

后续模拟生活推进的专项设计见 [DayPlan 驱动的模拟生活事件](2026-09-12-dayplan-simulated-life-events-design.md)，复用本文件的同级协作与共享消息机制；该专项尚未实现。

## 1. 本次确定的边界

Director 与 DayPlanAgent 是两个同级 Agent。Director 负责即时行为与对话决策，
DayPlanAgent 负责人物生活安排、活动逻辑和计划维护。DayPlan 不是 Director 的工具、
子 Agent 或一个收到调用才临时存在的函数；Director 也不是全部领域工作的审批中心。

先不建设跨服务 A2A：不引入 Agent Card、HTTP 消息服务、远程发现、独立消息中间件。
使用一个 LangGraph 协作图、共享 State 和明确的内部消息契约实现双方沟通。
这是一套应用内协作协议，不宣称兼容外部 A2A 标准。

所有 Agent 都能读取同一分支的完整共享协商记录，但各自保留独立 System Prompt、
工具集合与输出协议。每个 Agent 内部继续复用 AgentLoopController；协作图不再
实现第二套模型/工具循环，也不把 DayPlan 包装成 Director 的 LangChain 工具。

本轮不修改人物表达策略、检索算法、人格模型或工具的调查目标。PersonaActor 保留，
按 Director 已确认的表达任务生成文本；认知模型配置不变，不启用 LoRA。
OriginWorldSnapshot 暂不实现历史切片：沿用“导入记录最后时刻作为起点”的体验约定，
装配时明确记录实际使用的已发布 v3 画像和图版本，不读取未审核候选画像。

本文件取代旧文档里以下过时描述：

- Director 管理所有 Agent；DayPlan 是其子 Agent。
- 每条消息到达后，服务层必须先自动生成计划，再启动 Director。
- 日程修订由 Director 先输出最终回复意图，后端再悄悄调用 Planner。
- 用 `request_day_plan` 工具嵌套执行另一个 Agent 来表达同级协作。

## 2. 职责与权限

| 组件 | 负责 | 不负责 |
| --- | --- | --- |
| Director | 理解消息和生活事件、依据共享计划判断安排与冲突、决定即时行动与表达、需要修改时提出明确排程需求 | 编制完整日程、批准所有 DayPlan 工作 |
| DayPlanAgent | 独立生成和维护日程、落实调整请求时检查排程约束、提出冲突和替代方案 | 替 Director 判断每条聊天是否可行、发送最终聊天台词、擅自代表用户许下承诺 |
| PersonaActor | 执行明确的表达任务、形成消息气泡；可读取共享协商记录 | 改写生活计划、重新决定是否回复 |
| Router | 根据结构化接收方、待处理消息、提交状态路由 | 判断用户意图、替模型决定“应当改计划” |
| Executor | 校验权限、约束、版本和幂等性，持久化计划/状态/回复，返回回执 | 语义分析、替 Agent 编造结论 |
| Scheduler/Worker | 投递时间事件、领取与续租、执行与恢复协作图 | 在聊天前私自插入规划步骤 |

同级不等于职责相同，也不意味着每次都要互相确认。DayPlan 可独立提交职责内的计划；
如果调整涉及真实承诺、对话中的约定或 Director 正在执行的行动，才按具体依赖协商。
Director 可在无需更改计划时直接形成表达任务，不必经过 DayPlan。

专项例外是自主模拟生活推进 `advance_life`：它依赖 Director 对人物当下关切、情绪背景与行动倾向的理解，因此新事件提交前必须取得适用于该机会的状态意见；涉及新的取舍时继续商量。基础规划 `plan_day` 的独立权限不变，这不是 Director 审批所有规划。后端只提供心理作用机会和现实扰动边界，不生成剧情或固定情绪。具体契约见[模拟生活事件设计第 5–9 节](2026-09-12-dayplan-simulated-life-events-design.md)。

该协作仍使用现有 request/answer/proposal/objection/submit/commit_result/notification。候选与已提交经历隔离；等待和重启由任务状态与检查点恢复，普通聊天优先处理，输入改变会使旧意见失效。提交后的可靠通知不能替代提交前协商。以上专项为待实现扩展。

**读取计划并判断，不等于修改计划。** Director 已收到双方共享的三天计划，
“今晚有没有空”“能不能提前出去”首先由 Director 结合计划、承诺和当前状态自行判断。
不能为了查询已有安排、理解用户意图或判断明显冲突而交接给 DayPlan。
只有决定需要新增、移动、取消或重排安排时，才发送具体修改请求。
DayPlan 对修改落地的排程检查仍然必要，但不能成为 Director 每次回复的前置审批。

## 3. 图结构与三个执行层次

外层协作图注册平级节点：

```text
START → ingest → route
                  ├→ director ───────┐
                  ├→ day_planner ────┤
                  ├→ persona_actor ──┤→ checkpoint → route
                  ├→ executor ──────┘
                  └→ END / waiting
```

这是逻辑结构，不要求额外实现名为 checkpoint 的业务节点；正常节点边界由
LangGraph checkpointer 保存 State。每个 Agent 节点调用同一套 AgentLoopController，
完成自己的模型与工具交互后，返回经过校验的 `AgentTurnResult`。

三个层次不可混淆：

1. 协作图：决定哪个平级节点处理哪条消息，维护共享工作状态。
2. AgentLoopController：负责单个 Agent 的推理、工具协议、字段修复与执行预算。
3. Executor：处理正式领域数据的事务，不在模型调用期间持有写事务。

v1 同一分支串行运行协作节点，不同时让两个节点写共享字段。不同分支可以并行。
是否以后并行调查，与双方是否同级无关，不为表现“同级”而引入并发写冲突。

## 4. 共享 State

### 4.1 逻辑结构

```python
class RuntimeState(TypedDict):
    schema_version: str
    branch_id: str
    cycle_id: str
    input_revision: int
    context: RuntimeContext

    messages: Annotated[list[AnyMessage], merge_shared_messages]
    delivery: dict[str, DeliveryState]
    tasks: dict[str, CollaborationTask]
    proposals: dict[str, DomainProposal]
    commits: dict[str, CommitReceipt]

    day_plans: dict[str, CommittedDayPlanSlot]  # 按虚拟本地日期索引的三天共享窗口
    life_state: CommittedLifeState
    pending_expression: ExpressionTask | None

    route: RouteDecision | None
    status: Literal["running", "waiting", "completed", "blocked", "superseded"]
    budget: CollaborationBudgetState
```

这是契约示意；实现时相关结构使用 Pydantic 类型，不把全流程写成无约束字典。
State 不保存 Session、模型客户端、工具对象、密钥、文件句柄等不可序列化对象。
节点通过运行依赖取得服务，不能把服务实例塞进 checkpoint。

`messages` 是共享协商历史；`tasks/proposals` 是工作投影；`day_plans/life_state`
只代表已提交事实。不能因为消息里出现“计划改好了”，就修改正式计划投影。

三天窗口本版默认指虚拟时区下的昨天、今天、明天（D-1、D、D+1），不是当前块加后两个块。
每个日期槽包含日期、available/missing/unavailable 状态，以及存在时的完整已提交计划和版本。
Director 与 DayPlan 读取同一投影；`current_plan` 如在接口中保留，只能是当天槽的派生视图，
不能另存一份可独立修改的计划。三个日期的全部生活块在每次 Agent 输入中展开，
当前块、后续块可以额外标注，但不能用它们替代完整窗口。
昨天的计划是只读历史；没有历史计划就标明 missing，不倒推编造一份。
初始准备需完成当天计划，明天由计划维护任务提前生成；未就绪日期显式呈现状态，
不能把“未生成”当作“全天有空”。

### 4.2 字段写入权

- Agent 返回消息、自己负责的任务更新和领域提案，不直接返回任意 State patch。
- 节点适配层检查发送者、任务归属、允许的操作，再生成 State 更新。
- 初始化/恢复从业务库装载已提交投影；运行中 `day_plans/life_state/commits` 的领域变化只能由已验证的 Executor 回执更新。
- Router 根据待处理消息计算 route；不直接相信模型随意指定的节点名称。
- delivery 和 tasks 的完成状态由代码根据结果更新，不能靠消息正文中的“完成”二字判断。

用 proposals 映射而不是一个可被不断覆盖的 proposed_plan，避免把旧候选、反提案、
已拒绝版本和当前版本混在一起。提案只能被显式 supersede，不能静默覆盖。

## 5. 共享消息协议

### 5.1 消息信封

```json
{
  "message_id": "由运行时分配的稳定 ID",
  "cycle_id": "当前协作轮次",
  "task_id": "相关任务 ID",
  "sender": "director",
  "recipient": "day_planner",
  "kind": "request",
  "reply_to": null,
  "virtual_at": "2026-05-11T10:00:00+08:00",
  "wall_at": "运行时实际记录时间",
  "content": "用户已确认今晚七点出发。我已根据共享计划判断不影响六点结束的工作，请将原八点的晚餐提前到七点，调整相关出行安排，保留已确认的工作。",
  "payload": {
    "operation": "revise_plan",
    "target_date": "2026-05-11",
    "requested_change": "将晚餐及相关出行提前到已确认的七点出发安排",
    "constraints": ["保留六点结束的已确认工作"],
    "source_event_ids": []
  }
}
```

ID、sender、cycle、时间由节点适配层绑定，不由模型冒充；模型负责表达内容、
允许的接收方和领域意图。示例不是要求模型输出内部运行参数。

消息类型限定为：

| kind | 意义 | 是否意味着已生效 |
| --- | --- | --- |
| event | 用户输入或平台时间事件 | 仅事件本身是真实输入 |
| request | 请对方处理一项任务 | 否 |
| clarification | 提出会影响结果的具体问题 | 否 |
| answer | 回答该问题 | 否 |
| proposal | 提出安排、行动或表达候选 | 否 |
| objection | 指出候选冲突并说明影响 | 否 |
| submit | 领域 Agent 请求 Executor 提交提案 | 否 |
| commit_result | Executor 返回提交成功/拒绝结果 | 仅成功回执代表生效 |
| notification | 通知事实变化或处理结果 | 事实需关联成功回执或来源事件 |
| failure | 明确说明失败及可重试性 | 否 |

消息必须包含足够的人话内容，不只是一串任务 ID。候选的完整结构放在 proposals，
消息通过 proposal_id 引用；装配输入时同时展开其当前内容，不能让 Agent 只看到裸引用。

### 5.2 全局可见与定向处理

全局指**同一分支**，不是不同用户或分支共享。

- 双方每次运行均能读取该分支的共享协商消息，包括自己不是接收方的消息。
- recipient 决定谁需要处理，不决定谁能看到。已处理的消息仍在历史中，不会删除。
- 广播通知只用于可见性，不自动触发所有 Agent，避免计划通知引发互相唤醒风暴。
- 需要对方行动时，明确发出 request/clarification；不能通过关键词匹配正文路由。
- 持续的双方来回询问用 task_id/reply_to 关联，不能每次都生成一件无关的新任务。

### 5.3 追加与重放

采用 LangChain 消息容器承载结构化信封。LangGraph 的 `add_messages` 支持按 ID
合并/替换，因此本项目在其外围使用 `merge_shared_messages`：同 ID、同内容为
幂等重放；同 ID、不同内容为错误。修改意见必须追加新消息并引用旧消息，禁止删除记录。
相关框架行为见 [add_messages 官方说明](https://reference.langchain.com/python/langgraph/graph/message/add_messages)。

共享消息记录是显式沟通内容，不要求暴露模型未公开的内部推理。
模型自身的 reasoning 字段仍按供应商协议与 Phoenix 观测处理，不拿它当消息路由指令。

## 6. 每个 Agent 实际收到什么

输入装配顺序统一为：

1. 该 Agent 独立维护的 System Prompt。
2. 当前已提交世界：实际画像版本、三天共享计划的完整生活块及各日期状态/版本、状态版本、当前虚拟时间和约束。
3. 当前任务及需要处理的共享消息。
4. 完整的当前协作会话记录，以及其中引用的候选与回执。
5. 此 Agent 本次私有工具循环消息。

计划是预加载的共享领域状态，不需要通过对方 Agent 查询，也不依赖对方在聊天记录里
复述一遍。Director 先读计划完成判断；用户意图不明确时由 Director 询问用户，
不把“用户到底想几点走”转交给 DayPlan 猜测。提交成功后先刷新共享计划投影，
再唤醒接收通知的 Agent，避免收到新回执却仍读旧版本。

所有共享发言以带明确 sender 的数据消息投影，不把 DayPlan 的输出伪装成 Director
曾经发出的 assistant 消息，也不把其他 Agent 的话插成更高权限的 System Prompt。
“知道对方说了什么”和“把对方的话当成系统指令”不同。

原始 tool_call/tool_result 仅在发起调用的 Agent 内部保持完整协议。对方需要的
资料和分析结论由它主动通过共享消息传递，必要时附完整结果引用；双方均可读取共享附件。
不能把一个 Agent 的 ToolMessage 拼到另一个 Agent 没有发起过的 tool_call 后面。

完整共享记录保存在 State/checkpoint 中，不静默删除。不可能无限制地把分支一生的
全部记录塞进单次模型窗口：当前尚未完成的协作记录必须完整提供；已完成历史按会话归档，
双方使用相同的归档索引和读取入口。若连当前会话也超过可用窗口，明确保存并暂停/分段，
不能静默截断后声称“都看到了”。本轮先保持当前会话完整，不设计会丢失异议的自由摘要。

## 7. Agent 输出与路由

AgentLoopController 的最终产物改为各自领域的 `AgentTurnResult`，概念上包含：

```text
outgoing_messages：明确要说给对方/Executor 的内容
task_update：继续、等待对方、完成、失败
domain_proposal：该角色有权提出的计划/行为/表达结构
```

一个节点执行结束不等于整个 Runtime Cycle 结束。Agent 可以先发出问题，把执行权
让给对方；回答回来后，原 Agent 以更新后的共享 State 继续运行。

Router 固定规则：

1. 优先处理已入队的新输入失效检查，不能提交过期动作。
2. 根据待投递的定向消息，进入 recipient 对应的同级节点。
3. 有 submit 时进入 Executor；有明确表达任务且依赖已满足时进入 PersonaActor。
4. Executor 回执投递给提交者；需要通知对方时追加 notification。
5. 无可执行消息且任务已完成则 END；等待外部输入则 waiting；不可恢复失败则 blocked。

一个 Agent 同一轮可以广播多条说明，但至多产生一个新的可执行交接目标。
需要多个步骤时，完成当前交接后再继续，避免多个节点竞争同一提案。
Router 不依据“午休”“工作”“重要”等文字自动选择 DayPlan 或 Director。

## 8. 初始化、跨日与普通消息

### 8.1 初始化

创建分支后进入 preparing，发出接收方为 DayPlan 的 `initialize_plan` 事件。
DayPlan → 提案 → Executor → 计划保存成功 → 初始化 LifeState → 分支 ready。
Director 不需要先批准初始计划，也不必为了证明同级关系空跑一遍。

初始计划失败时保留草稿、错误与 retryable 状态，分支显示准备失败或等待重试，
不能向用户显示已就绪后再在第一次发消息时偷偷补计划。
准备期间虚拟钟暂停；ready 时重新设置 wall_anchor 并恢复，避免九分钟计算推进人物九分钟。
这是生命周期控制，不改变用户选定的虚拟起点含义。

### 8.2 普通聊天

用户输入 → Director。若现有安排足以回应，可直接提交表达任务 → PersonaActor → Executor。
不经过 DayPlan，也不在服务层插入“缺计划先规划”的隐藏分支。

回复涉及安排可行性时，Director 直接依据三天共享计划、当前状态和承诺判断，
包括识别已有冲突、提出可行时间、决定拒绝或向用户澄清。这些都不触发 DayPlan。

只有 Director 决定需要修改计划时，才发 `revise_plan` request，说明修改目标、
已判断的约束、哪些内容已获用户确认、哪些仍是候选。DayPlan 负责实际排程，
可在落地时发现额外依赖或版本变化并反馈，而不是重新承担聊天意图理解。
候选安排不等于承诺；需要用户确认时先等待用户，不为了完成协作强行提交。
无需修改的判断走 Director → PersonaActor → Executor；需要修改才加入 DayPlan 协作。

### 8.3 跨日和计划维护

Scheduler 可按明确定义的订阅规则，将 `prepare_next_day` / `day_started` 投递给 DayPlan。
这是领域时间事件，不是调度器自己规划，也不要求先让 Director 授权。
DayPlan 生成完成后向 Director 通知；Director 是否需要说话仍由它自己判断。

未来日期可以提前规划，但只影响目标日期的计划，不立即改变当前活动。
初版修改权限支持当天与下一天；这与共享读取昨天、今天、明天的三天窗口不同。
窗口随虚拟日期滚动，由已提交存储加载对应日期；日期权限由代码约束，不接受模型任意改写过去。

活跃分支缺计划时，State 明确标为 missing/unavailable。既定时间事件已经由 DayPlan
维护；聊天仍送 Director，由它根据当前问题决定等候协商或在未知计划状态下回复，
不能把未知状态自动替换成“睡觉/忙碌”。

## 9. 计划的权威、冲突与提交

### 9.1 计划不需要 Director 的通用批准

DayPlan 拥有排程领域权。正常生成或无冲突维护可以直接提交 Executor，成功后通知
Director。Director 可以提出异议，但不能静默覆盖 DayPlan，也不能把同级协作变成
“只有 Director 输出 plan_proposal_id 才允许保存”。

涉及用户已经确认的承诺时，权威来自用户/分支事实，不是来自哪个 Agent 更高级。
双方都不能凭协商记录擅自撤销真实承诺；缺少用户选择时，由 Director 形成询问用户的
表达任务，相关规划任务进入 waiting，不阻塞工作线程长期等待。

### 9.2 候选与正式版本分离

每份计划提案带 target_date、base_plan_version、constraint_revision、来源输入版本、
适用时间与 proposal_id。模型输出计划内容，运行时绑定版本与标识。

Executor 检查：调用角色权限、时间覆盖与边界、锁定块、已确认承诺、来源引用合法性、
当前版本和幂等键。语义推断仍由 Agent 负责，不引入聊天正则裁决。

提交失败时返回结构化拒绝原因，发给 DayPlan；需要协调时 DayPlan 再联系 Director。
不能假装提交成功，也不能由服务层删除失败请求后沿用“已经安排好了”的表达。

### 9.3 事务依赖

- 初始化计划独立提交，后续回复失败不回滚它。
- 与某条即时回复无关的日常计划维护也独立提交。
- Director 表达“已改好安排”时，表达任务必须绑定成功的计划提交回执；
  在候选阶段只能描述候选，不可宣称已经生效。
- 若计划和即时状态必须原子改变，用同一 ChangeSet 提交，双方贡献各自领域部分。
  Executor 等待依赖完整后一起提交；不需要把 Director 提升为计划批准者。
- 对外消息存储与其业务事件、待投递标记原子提交。外部投递使用稳定幂等键。

## 10. 时间、版本、并发和事务

统一保存 UTC 时间点，排程按分支时区转换后的本地日期与 HH:mm 计算。
`target_date`、日历查询、已开始块锁定和午夜切换必须使用同一转换函数。

一次 Agent 调查绑定 observation_time；真正提交时再取 commit_time，重新检查是否
跨日、跨活动边界或出现新承诺。若旧提案已不适用，返回 stale/rebase 消息，不能用九分钟前
的时间点证明“这个活动尚未开始”。也不因时间流逝几秒就无条件推翻全部调查。

同一分支共享 State 采用单写者执行租约，带 fencing token，Worker 心跳续租。
旧 Worker 即使模型后来返回，也不能写入较新的 State 或领域版本。
任务租约、共享状态更新和事件领取不能各用一个互不关联的固定过期时间。

模型与网络工具调用时不持有领域写事务：

```text
短事务领取、读取版本 → 关闭事务 → Agent 执行 → 短事务条件校验与提交
```

模型调用输入是不可变读取快照。工具用独立只读会话；版本检查也用独立短读取，
不因为检查 input_revision 自动 flush 某个尚未提交的 ORM 对象。

新消息到达时先持久化并提升输入版本，再由图在节点边界整合。旧表达必须重新判断；
DayPlan 的调查材料可以保留，但旧提案必须按新的约束版本重新确认，不自动当成当前事实。

## 11. 持久化与恢复

外层协作 State 使用 LangGraph checkpointer，稳定 thread_id 绑定分支和状态协议版本，
cycle_id/task_id 区分不同协作任务。当前分支运行必须有写租约，不能两个 invoke 同时写
同一 thread。通过 checkpoint 恢复，不从空 State 假装继续。
框架依据见 [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)。

本地开发使用持久化 checkpointer，不用 MemorySaver 冒充重启恢复。
状态库放在所有 worktree 之外，例如 `.runtime-data/runtime-state/`。
它是可恢复执行状态存储，不是第二份业务数据库：计划、消息和人物事实仍以业务库为准。
最终选用的 checkpoint 包与 SQLite 驱动需在实现时锁定兼容版本。

Checkpoint 与业务 COMMIT 不是天然原子事务，因此 Executor 必须按
`branch + task + proposal + operation` 幂等执行：

- 业务提交成功、checkpoint 尚未保存就崩溃：恢复后查提交回执，补入 State，不重复提交。
- Agent 已返回、尚未执行 Executor：从已保存候选继续，不重新调查。
- Agent 在工具循环中崩溃：单 Agent Harness 也需持久化其可序列化检查点与工具结果引用，
  不能只保存外层节点后声称内部九分钟工作可恢复。
- 无法复用的供应商进行中请求允许重发，但必须记录新调用，不能承诺模型请求 exactly-once。

正常恢复不依赖手工读取 Phoenix 再重放。Phoenix 负责诊断，不担任正式业务状态库。

## 12. 结束、失败与防止无限协商

成功条件是任务产物有效、必需提交已有成功回执、没有未处理的必要协商；
不是“某模型输出了 JSON”，也不是“工作线程退出了”。

区分 completed、waiting、retryable_failed、blocked、superseded。等待用户时图保存后退出，
下条用户消息再唤醒，不占用同步请求一直等待。

两个层次的预算分开：单 Agent 工具循环预算、整个协作任务的累计预算。
后者覆盖双方的供应商调用数、工具调用数、实际 token usage、总耗时和交接次数，
不能切换到对方后就重新获得无限额度。阈值集中配置，不写进人物行为 Prompt。

不设置“正常协商只能两轮”。但必须限制以下无效循环：

- 同 task、同约束版本、相同请求的重复投递：幂等去重。
- 相同问题已经回答却继续原样询问：返回已有回答并提示具体未解决点。
- 多次交接没有新答案、候选变更、来源或约束变化：先明确提醒当前分歧；仍无变化则
  blocked/waiting，保留草稿与原因。不能因为每次新建消息 ID 就认定有进展。
- 工具暂时失败：有限退避重试，不得伪装为“没有资料”。
- 总预算到达：保存当前工作、报告未完成，不强行把候选提交成事实。

## 13. 调度、前端与 Phoenix

Scheduler 独立于耗时 Worker 循环，以虚拟日期和领域事件生成幂等任务。失败重试以
明确 attempt 管理；业务幂等键保持稳定，不能被旧失败 Job 的 dedupe_key 永久挡住。
不为了开启时间扫描而启动包含训练职责的 all Worker。

前端仍是原微信式聊天，不增加用户必须理解的 Agent 工作台。
只展示必要状态：分支准备中/失败可重试、消息已收到、处理中、等待用户、处理失败。
内部协商不作为人物聊天气泡泄漏；只有 Executor 提交的 BranchMessage 对外展示。

Phoenix 以协作 Cycle 为根，记录同级 Agent turn、handoff、Executor commit/failure。
属性包括 branch/task/message/input_revision/plan_version/checkpoint 标识与预算。
真实模型调用数只统计供应商请求，不能重复累计外层节点和内部 LLM span。
角色切换不是新增一次模型调用；恢复旧产物也不能记成新生成。

## 14. 示例：两者如何商量

示例中的内容只是协议演示，不是要写死的人物行为：

### 14.1 只判断，不改计划

用户问：“今晚六点能出去吃饭吗？”共享计划显示已确认工作七点结束。
Director 自行识别冲突，根据人物处境决定如何回应，交给 PersonaActor 表达。
全程不调用 DayPlan，也不把查询计划、判断冲突包装成协商任务。

### 14.2 决定修改后，才协作落地

本例共享计划显示工作六点结束、原晚餐安排八点；相关出行条件允许七点出发。

1. 用户：“今晚提前出去吃饭吧。”事件交给 Director。
2. Director 读取已有计划，自行判断七点出发可行。若需要明确用户选择，直接通过
   PersonaActor 问用户“七点出发怎么样？”，此时不交接 DayPlan，也不保存修改。
3. 用户确认七点后，Director → DayPlan/request：“将原八点晚餐调整为七点出发，
   连同相关出行一起调整，保留六点结束的工作；用户已确认七点。”
4. DayPlan 根据共享计划落实修改，检查前后活动衔接，将完整提案提交 Executor。
5. Executor → DayPlan/commit_result：成功并返回计划版本，或返回具体冲突。
6. 成功时刷新共享计划；DayPlan → Director/notification：说明实际修改和成功回执。
7. Director 根据真实结果形成表达任务，交给 PersonaActor；Executor 提交回复。

若落地时出现新约束或版本冲突，DayPlan 反馈具体问题，Director 再决定是否调整需求
或询问用户。澄清机制用于真实的修改依赖，不用于重复询问双方已经看得到的安排。
全过程共享记录都可见，没有“后端悄悄改了计划”。

## 15. 文件组织与迁移

拟采用以下职责划分，实际落地不强制一个节点一个巨大文件：

```text
runtime_v1/
  collaboration/
    state.py          # State、任务/提案/回执
    messages.py       # 信封、追加式 reducer、模型输入投影
    graph.py          # 同级节点装配
    routing.py        # 纯路由和结束条件
    persistence.py    # checkpoint、租约和恢复
    nodes.py          # Harness / Executor 适配，不复制工具循环
  prompts/
    director.py
  agents/
    day_planner.md    # 同级 Agent 定义
    persona_actor.md
  agent_catalog.py
  tools/              # 各自查询工具；不放“把 DayPlan 当工具”的适配
  director.py
  day_planner.py
  actor.py
  executor.py
  service.py          # HTTP/Job 门面，不再承载另一套隐式 Agent 编排
```

迁移顺序：

1. 撤回本轮未完成的 `delegate_day_plan.py` / request_day_plan 工具委派方向；
   仅撤回该方向新增内容，保留此前 ORM 注册、失败事务恢复、计划假设类型等正确修复。
2. 定义共享 State、消息协议、权责和 Executor 回执，并用离线模型跑通来回协商。
3. 建外层 LangGraph，Director/DayPlan 平级注册，各自接现有 Harness。
4. 接初始化与时间事件、版本租约、持久化和恢复；取消 RuntimeService 隐式先规划路径。
5. 接普通聊天与 PersonaActor，补准备状态与失败反馈，不改原聊天布局。
6. 在隔离分支验证，再切换体验环境。历史业务数据保持可读；旧检查点不直接加载进新协议。

源码里刚开始写的子 Agent 方向还未验证、未部署，本设计不把那些改动当作既成架构。
后续实现必须逐项检查并撤回，不能只重命名文件后声称已经实现同级协作。

## 16. 冒烟验收

只做覆盖架构边界的测试，不用大量测试替代真实体验：

1. 初始准备直接进入 DayPlan，成功保存后 ready，不调用 Director 审批。
2. 普通问候直接进入 Director/PersonaActor，不自动调用 DayPlan。
3. 跨日事件直接投递 DayPlan；成功通知双方可见，不自动发送人物消息。
4. 明确修改请求中的新增约束导致 DayPlan 追问时，请求、追问、回答、方案、Executor 回执全部进入共享记录。
5. DayPlan 失败后 Director 看到失败，不输出“已安排成功”；当前正式计划不被候选覆盖。
6. 新输入、过期租约和计划版本冲突能阻止旧结果提交。
7. 计划成功后回复写库失败，计划仍在；恢复不会重复回复。
8. 提交后、checkpoint 前故障可以按回执恢复；Agent 内部中断保留可复用工作。
9. 重复交接不会因新消息 ID 绕过无进展保护；等待用户会保存并释放 Worker。
10. Phoenix 的真实调用数与供应商请求对应，协作记录不会混入人物聊天气泡。
11. 双方收到同一版本的三天完整计划；缺失日期显式标明，昨天不可修改。
12. 询问可用时间、判断已有冲突不触发 DayPlan；只有明确修改需求才交接。
13. “提前一点”等待用户确定的意图由 Director 直接澄清，不转交 DayPlan 判断用户意图。
14. 提交完成后下个 Agent 输入使用新计划版本，不能只更新通知不更新共享状态。

以上完成后才进入工具定义与 Agent 行为 Prompt 的效果优化阶段。

# MoonlightBox 的证据、生活规律、当前情境与 Agent Runtime 设计

> 核心分工：**GraphRAG 管证据和检索，RoutineModel 管生活规律，SituationalState 管当前推测，Agent Runtime 管是否行动。**

## 1. 文档目的

Generative Agents 值得借鉴的是它持续运行的认知闭环，而不是固定 Tile 地图、预先编排的日程或把所有状态都写进记忆的具体实现。

MoonlightBox 面对的是一个不同的问题：系统需要根据真实聊天记录理解一个真实人物，在证据不足时保持克制，并在合适的时机决定回复、主动联系、等待或继续观察。因此，不能直接复制 `Memory Stream → Reflection → Planning → Action`，而应把其中混在一起的职责拆开：

| 模块 | 回答的问题 | 不能做什么 |
| --- | --- | --- |
| GraphRAG | “过去有哪些证据与当前问题有关？” | 不能因为检索到了文本就断言它是真的或仍然有效 |
| RoutineModel | “这个人在这种时间和条件下通常会怎样？” | 不能把统计规律写成事实或虚构今天的行程 |
| SituationalState | “综合新鲜证据与规律，目前最可能是什么状态？” | 不能永久保存未经证实的推测 |
| Agent Runtime | “现在是否应该行动，做什么，还是保持沉默？” | 不能绕过证据、权限和策略门直接执行模型冲动 |

这四层不是四套相互独立的图，也不是四个都能自由调用工具的 ReAct Agent。它们组成一条有明确读写边界的流水线。

## 2. 首要设计原则

### 2.1 证据、规律、推测和行动必须分开

下面四句话的语义完全不同：

1. “她昨天 09:13 说自己到公司了”是证据。
2. “她工作日 09:00 左右通常在公司”是规律。
3. “现在是周二 09:10，她可能在公司”是当前推测。
4. “现在给她发一条消息”是行动决策。

如果将它们都写进同一个 memory 表，系统很快就无法回答：这句话是用户说的、模型归纳的、当前猜的，还是 Agent 自己做过的。四层架构的首要目的就是阻断这种污染。

### 2.2 RoutineModel 是先验，不是世界真相

历史规律只能提供概率：

```text
P(location = 公司 | 工作日, 09:10, 当前生活阶段) = 0.68
```

它不能生成下面这种事实：

```text
她今天 09:10 在公司。
```

如果没有今天的直接证据，系统只能保留“可能在公司”的临时假设。它不能将该假设加入 GraphRAG 的事实节点，更不能在下一轮把自己的猜测检索出来，当成新的佐证。

### 2.3 当前状态必须过期

“正在开会”“在公司”“心情不好”都有时间范围。SituationalState 中每个字段都必须带 `observed_at/inferred_at`、`valid_until` 和来源。过期后自动失效，而不是依靠大模型记得撤销。

### 2.4 沉默是一种正式决策

Agent Runtime 的结果不只是“生成什么消息”，还包括：

- 回复当前用户；
- 主动发起对话；
- 执行获准的工具动作；
- 只进行内部整理；
- 安排下一次检查；
- 保持沉默。

当没有新动因、证据不足、用户尚未回复或处于安静时段时，`wait` 通常是正确结果。

### 2.5 只有真实发生的事件才能回流为证据

以下内容可以成为新证据：用户真实发出的消息（包括自然对话中主动表达的纠正）、外部 Provider 返回的观测、已经执行成功的 Agent 动作。

以下内容不能成为新证据：RoutineModel 的预测、SituationalState 的猜测、生成但未发送的草稿、模型内部思考、计划但未发生的动作。

## 3. 总体架构

```text
真实输入
  ├─ 用户消息
  ├─ 导入的历史聊天/媒体/位置
  ├─ 外部观测
  ├─ 已执行的 Agent 动作
  └─ 定时 Wakeup
          │
          ▼
   Evidence Ingestion
          │
          ▼
      GraphRAG ──────────────┐
   证据、声明、关系与检索      │ 相关 EvidenceBundle
          │                  │
          ├──异步训练────────▼
          │             RoutineModel
          │           长期、版本化的概率规律
          │                  │ RoutinePrior
          └──────────────────┤
                             ▼
                      SituationalState
                  已观测状态 + 临时推测快照
                             │
                             ▼
                       Agent Runtime
                  策略门控、决策、执行或等待
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
        真实动作成功                    安排下一次检查
              │
              └──作为新证据回流 GraphRAG
```

一次 Runtime Cycle 应冻结一个输入截止点 `evidence_cutoff`。该轮所有检索、推测和决定都基于同一快照，避免并发任务因为完成顺序不同而得到不同世界。

## 4. 统一术语

### 4.1 Evidence：不可变的原始证据

Evidence 表示可追溯的真实输入，例如某条消息、位置分享、图片识别结果、用户纠正或已执行动作。Evidence 本身不等于事实正确，只表示“这个输入真实存在”。

### 4.2 Claim：从证据中提取的可争议声明

例如从“我到公司了”可提取：

```text
subject = 妈妈
predicate = arrived_at
object = 个人地点：公司
event_time = 消息时间附近
```

Claim 必须保留来源 Evidence，并允许 `supported / contradicted / superseded / unresolved`。模型提取错误时，可以重跑 Claim，而不用修改原始消息。

### 4.3 Entity：用于聚合的实体

人、地点、组织、话题等都可以是 Entity。“家中”“到家”“回家”可以通过不同 Claim 指向同一个个人地点实体，但原始表达仍保存在 mention 和 Evidence 中。

### 4.4 Routine：从多条历史证据估计出的分布

Routine 不描述某一时刻发生了什么，而描述相似条件下通常怎样。它必须记录训练数据截止点、样本量、适用生活阶段和置信度。

### 4.5 Situational Belief：有时效的当前信念

它是对“现在”的物化视图，既可包含直接观测，也可包含带概率的临时假设。两者必须在结构上分开。

### 4.6 Decision 与 Action

Decision 是 Runtime 选择的意图；Action 是实际执行结果。只有执行成功的 Action 可以作为新的 Evidence。

## 5. GraphRAG：证据与检索层

### 5.1 它解决什么问题

GraphRAG 的任务不是生成一个全知世界，而是在当前问题下找出一组可核查的相关证据：

- 谁说过什么；
- 哪些消息属于同一段对话或同一事件；
- 某个声明由哪些证据支持或反对；
- 人物、地点、事件和时间之间有什么关系；
- 一个旧结论是否已经被新证据替代；
- 当前 Runtime 决策使用了哪些材料。

因此它既需要向量检索，也需要图上的关系扩展和时间、权限、人物范围过滤。单纯把所有聊天 chunk 放进向量库不是这里所说的 GraphRAG。

### 5.2 建议的数据层次

GraphRAG 可以在同一存储体系中包含三类逻辑节点，不需要维护三张互不相干的图：

#### 原始证据节点

```yaml
EvidenceNode:
  id: string
  project_id: string
  branch_id: string | null
  kind: message | episode | media | location_share | external_observation | user_correction | agent_action
  actor_id: string | null
  occurred_at: datetime
  ingested_at: datetime
  content_ref: string
  content_hash: string
  source_system: string
  authority: direct | reported | derived
  visibility: string
  metadata: object
  extractor_version: string | null
```

#### 声明与实体节点

```yaml
ClaimNode:
  id: string
  subject_id: string
  predicate: string
  object_id_or_value: any
  valid_from: datetime | null
  valid_to: datetime | null
  status: unresolved | supported | contradicted | superseded
  confidence: number
  extraction_version: string
  created_from_evidence_ids: [string]

EntityNode:
  id: string
  type: person | personal_place | poi | organization | event | topic
  canonical_name: string
  aliases: [string]
  owner_person_id: string | null
  resolution_status: unresolved | semantic_only | geo_resolved | confirmed
  metadata: object
```

#### 边

```yaml
EvidenceEdge:
  id: string
  source_id: string
  target_id: string
  relation: mentions | supports | contradicts | part_of | follows |
            refers_to | located_at | involves | supersedes | caused_by
  occurred_at: datetime | null
  confidence: number
  evidence_ids: [string]
  extractor_version: string | null
```

边同样必须有证据来源。“妈妈家”和某个高德 POI 被合并不能只留最终坐标，还要留下哪些输入和对话支持、使用了哪个消歧版本与合并规则。

### 5.3 聊天数据如何进入 GraphRAG

导入时不应要求一次性解决所有实体、地点和事件。推荐采用分层、可重跑的流程：

```text
原始消息落库
→ ConversationBundle / Episode 组织
→ Episode embedding 与关键词索引
→ 按任务召回相关 Episode
→ 在局部上下文中提取 Claim / mention
→ 必要时做实体消歧
→ 写入带 provenance 的图节点和边
```

这样做的好处是：原始证据完整保存，而昂贵、易错的模型分析可以按需执行、独立升级和重新计算。GraphRAG 不是要求“所有数据进来时就全部理解完”，而是为渐进理解提供统一的证据底座。

第一版可以使用很朴素的边界：

- 一次连续聊天作为一个 Bundle；
- Bundle 太长时再按长度切开；
- 新 Bundle 带上前一个 Bundle 最后两三轮对话作为 `carry-in context`；
- 不先做复杂的“事件置信度”“主题变化阈值”“证据充足度判断”。

这里的 Episode 更接近“会话窗口”，不是系统声称已经准确识别出的现实事件。

Episode 正文中的 message 标签也不是必须的。例如：

```text
[message:m101][19:22][用户]
```

这种写法的好处是实现简单，能够从生成结果追溯到原始消息；缺点是技术标识会进入实体与关系抽取上下文，可能带来噪声。更干净的实现是：

```text
LightRAG 文档正文：只包含自然可读的对话
MoonlightBox 映射表：document_id → episode_id → message_ids
```

例如：

```yaml
lightrag_doc_id: doc-123
episode_id: ep-001
message_ids: [m101, m102, m103, m104]
```

查询命中 LightRAG 文档或 chunk 后，再通过映射表回到原始消息。这样既保留了消息级的可追溯性，也不需要让抽取模型反复看到 `m101` 之类的内部编号。

### 5.4 检索流程

每次检索至少经历六步：

1. **意图分析**：当前要找的是当前状态、旧承诺、关系变化、偏好、过去事件还是行动依据。
2. **种子召回**：组合语义向量、关键词、明确实体 ID、时间窗口和参与者过滤。
3. **图扩展**：从高相关种子扩展到其 Episode、支持/反对证据、相关人物地点及前后事件。
4. **时态与权限过滤**：排除截止点之后、已失效、不可见或属于其他人物/分支的内容。
5. **证据排序**：综合语义相关性、图距离、新鲜度、来源权威、重要性与矛盾信息。
6. **打包**：返回有限大小、带引用的 `EvidenceBundle`，而不是把整张图塞进上下文。

可使用如下概念评分，权重需要通过评测校准：

```text
score = 0.35 * semantic_relevance
      + 0.15 * lexical_or_entity_match
      + 0.15 * graph_proximity
      + 0.10 * temporal_relevance
      + 0.15 * source_authority
      + 0.10 * importance
      - contradiction_or_staleness_penalty
```

矛盾证据不应简单丢弃。对于需要判断真伪或状态变化的问题，应同时返回支持和反对证据。

### 5.5 EvidenceBundle 契约

```yaml
EvidenceBundle:
  request_id: string
  query_intent: string
  evidence_cutoff: datetime
  seed_entity_ids: [string]
  evidence_items:
    - evidence_id: string
      excerpt: string
      occurred_at: datetime
      authority: string
      relevance_score: number
      graph_path: [string]
  claims:
    - claim_id: string
      status: string
      support_evidence_ids: [string]
      oppose_evidence_ids: [string]
  unresolved_conflicts: [object]
  retrieval_version: string
```

后续模块引用 `evidence_id` 和 `claim_id`，不要复制一段失去来源的摘要。

### 5.6 GraphRAG 不负责什么

- 不负责预测今天的完整日程；
- 不负责决定是否主动发消息；
- 不把向量相似当成事实一致；
- 不把模型摘要当成比原始消息更高权威的证据；
- 不自动将所有地点别称合并成地理 POI；
- 不保存 RoutineModel 或 SituationalState 生成的纯推测。

## 6. RoutineModel：生活规律层

### 6.1 为什么不采用“365 天 × 每分钟概率表”

将一年每一分钟都建成独立格子会产生三个问题：

1. 聊天样本远远不足，绝大多数分钟没有数据；
2. 搬家、换工作、节假日和临时事件会使旧规律失效；
3. 没有消息不等于没有活动，消息时间只能间接反映生活状态。

更适合的是一个分层、可回退、可识别生活阶段的概率模型。它先尝试使用“当前生活阶段 + 工作日类型 + 时段”等细条件；样本不足时，逐级回退到更宽泛的统计，而不是编造精确概率。

### 6.2 RoutineModel 的输入

只使用可追溯的历史证据和经过质量门控的 Claim：

- 真实聊天的消息时间、发送者和会话节奏；
- 明确或高置信的到达、离开、活动、工作和休息事件；
- 结构化位置分享；
- 从原始消息或结构化位置分享中提取的个人地点；
- 日历类型，如工作日、周末和法定节假日；
- 已检测出的生活阶段，例如“旧公司时期”“搬家之后”。

默认不得使用 Agent 自己生成的消息、Routine 推断和 Situational 推测作为训练样本，否则模型会自我强化。

### 6.3 V1 建议建模的规律

V1 不必一开始重建完整人生模拟，可从 Agent 决策真正需要的分布开始：

- `message_start_hazard`：某人在给定时段主动开始一段聊天的倾向；
- `response_latency`：收到消息后的响应时间分布；
- `active_hours`：通常活跃的时段；
- `inter_bubble_delay`：连续气泡之间的间隔；
- `location_role_distribution`：家、公司、通勤、常去地点等语义位置分布；
- `activity_distribution`：工作、休息、通勤、吃饭、社交等粗粒度活动分布；
- `availability_distribution`：方便聊天、可能忙碌、可能休息的分布；
- `transition_distribution`：从一个语义状态到另一个状态的概率和常见耗时。

地点和活动数据不足时，相应输出应为低覆盖率或 `unknown`，而不是通过语言模型补齐。

### 6.4 条件特征与回退层级

推荐条件包括：

- 人物和项目；
- 时区；
- 当前生活阶段 `regime`；
- 工作日、周末、法定节假日；
- 星期；
- 小时或 15/30 分钟桶；
- 月份或季节；
- 最近真实事件，如刚下班或正在旅行；
- 证据距当前的时间。

回退示例：

```text
当前阶段 + 工作日 + 星期二 + 09:00
→ 当前阶段 + 工作日 + 09:00
→ 当前阶段 + 任意日 + 上午
→ 全局工作日 + 上午
→ unknown
```

回退层级越深，输出置信度越低。

### 6.5 模型版本字段

```yaml
RoutineModelVersion:
  id: string
  project_id: string
  person_id: string
  version: string
  status: training | active | superseded | failed
  timezone: string
  regime_id: string
  evidence_cutoff: datetime
  training_window:
    from: datetime
    to: datetime
  sample_counts: object
  feature_schema_version: string
  distributions: object
  transition_model: object
  calibration_metrics: object
  coverage: object
  confidence: number
  source_evidence_ids: [string]
  created_at: datetime
  supersedes_model_id: string | null
```

当证据量很大时，`source_evidence_ids` 可改为可复现的训练快照 ID、查询条件和内容哈希，但必须能追溯到原始数据。

### 6.6 概率如何计算

V1 可以使用可解释的统计方法，不必先上复杂神经网络：

- 分类分布使用 Dirichlet/Laplace 平滑，避免少量样本产生 0 或 1；
- 是否发起聊天等二元事件使用 Beta-Binomial 平滑；
- 历史样本按时间衰减，使最近生活阶段更重要；
- 响应延迟和气泡间隔使用分位数或对数正态分布；
- 位置、活动迁移使用带平滑的 Markov 转移统计；
- 使用变点检测自动切分搬家、换工作等 `regime`；
- 单独输出 `sample_count`、`coverage` 和校准误差，不能只给概率。

一种简单的加权计数为：

```text
weight(evidence) = authority_weight
                 * extraction_confidence
                 * exp(-age_days / half_life_days)
```

对于某时段的位置分布：

```text
P(location = L | context)
  = (weighted_count(L, context) + alpha)
    / (weighted_count(all locations, context) + alpha * K)
```

没有明确位置的普通消息不能直接算作“在家”或“在公司”；它只能用于消息活跃度，除非有同一时间窗口内的独立地点证据。

### 6.7 RoutinePrior 输出

Runtime 查询 RoutineModel 时得到的是有范围的先验：

```yaml
RoutinePrior:
  as_of: datetime
  model_version_id: string
  regime_id: string
  predictions:
    location:
      candidates:
        - { value: personal_place:company, probability: 0.68 }
        - { value: personal_place:home, probability: 0.17 }
      coverage: 0.54
    activity:
      candidates:
        - { value: working, probability: 0.61 }
    availability:
      candidates:
        - { value: busy, probability: 0.57 }
    message_start_hazard: 0.08
  confidence: 0.52
  fallback_level: workday_hour
```

这个对象只在当前 Cycle 中使用，不写入 GraphRAG。

### 6.8 更新策略

- 新数据到达时增量更新特征，但不要每条消息都重新训练完整模型；
- 每天或达到证据阈值时生成候选版本；
- 只有评测通过后才将候选版本标为 `active`；
- 旧版本保留，用于重放和解释历史决策；
- 当检测到生活阶段变化时，新建 `regime`，不要把新旧规律粗暴平均。

现有 `behavioral-rhythm-v1` 可作为 RoutineModel V1 的起点：保留消息小时分布、星期分布、主动率和气泡延迟，但应从 IdentityKernel 中逐步拆成独立、版本化且可校准的模型，并补充 `evidence_cutoff`、生活阶段和覆盖率。

## 7. SituationalState：当前情境层

### 7.1 为什么要拆成两个子层

MoonlightBox 现有 `situational-state-v2` 只接受外部观测、用户配置、Agent 已执行动作和历史回放等可信来源。这条安全边界应该保留。

为了支持“根据生活规律推测现在可能在哪、在做什么”，不要把 `routine_inference` 加入可信来源，而应拆成：

1. **ObservedSituationalState**：有直接证据、可用于陈述当前事实的临时状态；
2. **InferredSituationalBelief**：根据 Routine 与间接证据生成的候选分布，只能用于决策先验或带不确定性的表达；
3. **SituationalSnapshot**：Runtime 读取的合并视图，始终保留二者的来源差异。

### 7.2 建议的状态槽位

- `location`：当前语义地点或地理地点；
- `activity`：工作、通勤、吃饭、休息、社交等；
- `availability`：方便回复、忙碌、睡眠中、未知；
- `physical_state`：疲惫、不舒服等，仅在有证据时使用；
- `emotion`：短期情绪；
- `environment`：天气、噪音、是否在路上等；
- `current_facts`：无法归入固定槽位、但有时效的当前事实；
- `open_sequences`：正在等待后续的对话或承诺；
- `active_goal_refs`：与当下相关的目标引用。

健康、情绪等敏感状态不应仅凭 RoutineModel 预测。

### 7.3 已观测状态字段

```yaml
ObservedStateSlot:
  slot: location | activity | availability | physical_state | emotion | environment | current_fact
  value: any
  status: observed | user_confirmed | agent_action | historical_replay
  source: string
  evidence_ids: [string]
  confidence: number
  observed_at: datetime
  valid_until: datetime
  state_version: string
```

### 7.4 推测状态字段

```yaml
InferredStateSlot:
  slot: location | activity | availability | environment
  candidates:
    - value: any
      probability: number
  basis:
    evidence_ids: [string]
    claim_ids: [string]
    routine_model_version_id: string | null
  inference_version: string
  inferred_at: datetime
  valid_until: datetime
  confidence: number
  uncertainty_reason: string | null
```

推测状态最好作为可重建的缓存或 Cycle 产物保存，不作为长期记忆参与普通检索。

### 7.5 更新优先级

当不同来源冲突时，采用明确的优先级和时间规则：

```text
新鲜的结构化外部观测
> 新鲜、语义明确的用户自述或对方转述
> 已执行的 Agent 动作产生的状态
> 明确计划但尚未观测到发生
> RoutineModel 推测
```

高优先级新证据覆盖低优先级状态；同优先级则比较时间和置信度。不能因为 RoutineModel 认为工作日在公司，就覆盖“今天请假在家”的新消息。

### 7.6 TTL 建议

TTL 应按状态类型和证据内容动态决定，下面只可作为 V1 上限参考：

| 槽位 | 常见 TTL | 说明 |
| --- | --- | --- |
| activity | 30 分钟～2 小时 | “在开会”应短，“今天上班”可更长 |
| location | 1～6 小时 | 有离开、到达证据时立即更新 |
| availability | 15 分钟～2 小时 | 容易快速变化 |
| emotion | 2～12 小时 | 不自动延续到第二天 |
| physical_state | 2～24 小时 | 依证据语义决定 |
| environment | 15 分钟～3 小时 | 天气与交通状态变化快 |

所有槽位都应受系统最大 TTL 限制。过期不是删除历史证据，而是从当前状态投影中移除。

### 7.7 当前状态问题的回答规则

如果用户问“妈妈现在在哪”：

- 有新鲜直接位置证据：可以基于证据回答；
- 只有 Routine 推测：不能回答成“她在公司”，最多说“按平时规律这个时间可能在公司，但没有今天的直接信息”；
- 推测置信度低或涉及隐私：回答不知道；
- 存在相互冲突的新鲜证据：明确表达不确定，并展示必要的时间线。

这保证向量记忆、旧消息和 Routine 都不会伪装成当前事实。

## 8. Agent Runtime：行动决策层

### 8.1 Runtime 不是每分钟醒一次的无限循环

每分钟心跳会造成大量无意义推理，也容易让“Routine 概率”不断变成虚构状态。Runtime 应采用事件驱动加预约唤醒：

- 用户新消息：立即触发；
- 外部观测或工具回调：按相关性触发；
- 未完成承诺、开放话题或目标到期：预约触发；
- RoutineModel 发现未来某窗口可能值得检查：只安排下一次 `wakeup`；
- 反思或维护任务：低频后台触发。

RoutineModel 不直接命令 Agent 发消息，它最多提供“18:30 是较常见的主动聊天窗口”这一先验。Runtime 到点后仍需检查有没有新动因、用户是否已经回复、是否处于安静时段以及当前状态是否支持行动。

### 8.2 一次 Cognitive Cycle

```text
1. 接收 Trigger，生成幂等 cycle_id
2. 冻结 evidence_cutoff 和各模块版本
3. 将真实 Trigger 写为 PerceptionEvent
4. 判断检索意图，调用 GraphRAG 得到 EvidenceBundle
5. 按当前时间查询 RoutinePrior
6. 结合直接证据、旧状态和 Routine，生成 SituationalSnapshot
7. 模型提出候选行动及理由
8. 确定性 Policy Gate 审核候选行动
9. 执行 send / tool / wait / internal_only
10. 成功动作写入 AgentAction Evidence
11. 计算 next_review_at，安排 Wakeup
12. 保存可审计 CognitiveCycle
```

模型可以帮助理解语境和提出候选动作，但第 8 步必须由可测试的策略代码控制，而不是把所有安全规则写进 Prompt 后祈祷模型遵守。

### 8.3 Runtime 输入快照

```yaml
RuntimeContext:
  cycle_id: string
  trigger: object
  evidence_cutoff: datetime
  evidence_bundle: EvidenceBundle
  routine_prior: RoutinePrior
  situational_snapshot: object
  active_goal_ids: [string]
  open_sequence_ids: [string]
  relationship_state_version: string
  user_policy: object
  recent_action_summary: object
```

### 8.4 候选决策

```yaml
AgentDecision:
  cycle_id: string
  decision: reply | proactive_message | tool_action | internal_only | wait
  action_intent: string | null
  target_id: string | null
  motivation: string
  evidence_ids: [string]
  claim_ids: [string]
  used_routine_model_version: string | null
  used_situational_state_version: string
  readiness_score: number
  uncertainty: number
  policy_checks: [object]
  next_review_at: datetime | null
  model_version: string
```

### 8.5 主动行动的硬门槛

主动发消息至少满足：

- 有本轮新增的真实动因，如新证据、开放话题、承诺到期或有效目标；
- 不是仅因“历史上这个点常聊天”而触发；
- 没有尚未得到回复的重复主动消息，或已达到明确允许的例外条件；
- 未处于用户设置的安静时段；
- 未超过频率限制；
- 当前推测没有被当成确定事实写进消息；
- 行动符合关系边界、可见性和工具权限；
- 同一 `idempotency_key` 没有执行过。

Routine 高概率只能提高“值得检查”的优先级，不能单独构成“值得打扰”的理由。

### 8.6 Policy Gate 输出

```yaml
PolicyResult:
  allowed: boolean
  final_action: reply | proactive_message | tool_action | internal_only | wait
  block_reasons: [string]
  required_uncertainty_language: boolean
  next_review_at: datetime | null
  idempotency_key: string
```

如果候选消息中包含“你现在应该在公司吧”，但依据只有 Routine，策略层可以要求改成不确定表达；如果场景不适合猜测位置，则直接阻止这部分内容。

### 8.7 Action 回流

只有执行成功后才写：

```yaml
AgentActionEvidence:
  action_id: string
  cycle_id: string
  action_type: string
  target_id: string | null
  executed_at: datetime
  result_status: succeeded | failed | partially_succeeded
  external_receipt: string | null
  decision_id: string
  evidence_ids_used: [string]
```

发送失败的消息不能被后续 Runtime 当作用户已收到；计划调用但未执行的工具也不能改变 ObservedSituationalState。

## 9. 四层之间的读写契约

| 调用方 | 可读取 | 可写入 | 禁止行为 |
| --- | --- | --- | --- |
| GraphRAG ingestion | 真实输入与已执行动作 | Evidence、Claim、Entity、Edge | 写入 Routine/Situation 猜测 |
| Routine trainer | 截止点前的合格证据与 Claim | 新 RoutineModelVersion | 修改原始证据；把预测写成 Claim |
| Situation builder | EvidenceBundle、旧的有效状态、RoutinePrior | 当前 Cycle 的状态快照；可信观测槽位 | 将推测加入长期记忆 |
| Agent Runtime | 四层快照、目标和策略 | Decision、Action、Wakeup、Cycle 记录 | 绕过策略门执行；把草稿当已发送 |

最重要的数据流不变量是：

```text
Evidence → Routine / Situation → Decision → Executed Action → New Evidence

禁止：Routine prediction → Evidence
禁止：Situational guess → durable memory
禁止：Draft / plan → Executed Action
```

## 10. 典型场景

### 10.1 用户问“她现在在哪”

GraphRAG 找到：昨天工作日 09:05 她说到公司；过去三周同一时段有多次类似证据。RoutineModel 给出今天 09:10 在公司的概率 0.68，但今天没有直接观测。

SituationalState 应记录：

- `Observed.location = unknown`；
- `Inferred.location = 公司(0.68), 家(0.17), 其他(0.15)`；
- 推测短 TTL，例如 60 分钟。

Runtime 最终回答应类似：“没有今天的直接位置消息；按她平时工作日的规律，这个时间比较可能在公司。”不能回答“她在公司”。

### 10.2 聊天中出现“我刚到公司”

该消息先作为 Evidence 入图，局部分析抽取 `arrived_at(personal_place:company)` Claim。若主体和时间清楚，则 ObservedSituationalState 将 `location` 更新为公司，并设置合理 TTL。

RoutineModel 不需要立即全量重训。它在下一次增量训练时把这条合格证据计入对应时段。一次新消息不会推翻整个生活规律。

### 10.3 到了常见主动聊天时段

RoutineModel 预测 `message_start_hazard = 0.12`，Scheduler 因此在 20:30 创建一次低成本 Wakeup。Runtime 醒来后发现：没有开放话题、没有新事件、上一条主动消息仍未回复。

最终 Decision 是 `wait`，并将下一次检查安排在真正有事件或目标到期时。Routine 不单独制造消息内容。

### 10.4 有未完成承诺

GraphRAG 检索到用户昨天说“明天下午提醒我交材料”，现在已到约定时间。该 Claim 有明确时间、主体和支持 Evidence；这是新的现实动因。即使 RoutineModel 认为此时通常不活跃，Runtime 仍可依据承诺触发提醒，但需经过安静时段和通知权限检查。

### 10.5 Routine 与新证据冲突

RoutineModel 认为周一上午通常在公司，但用户刚说“今天请假在家”。ObservedSituationalState 的新证据优先，当前位置为家；Routine 只作为被覆盖的先验。若类似变化持续出现，后续训练可能检测出新的生活阶段。

## 11. 并发、重放与版本一致性

### 11.1 冻结快照

每个 Cycle 固定：

- `evidence_cutoff`；
- `retrieval_version`；
- `routine_model_version_id`；
- `situational_state_base_version`；
- `policy_version`；
- 推理模型版本。

一个 Cycle 执行中即使新消息到达，也不改变其输入；新消息触发下一 Cycle。这样才能重放和解释“当时为什么这么做”。

### 11.2 乐观并发控制

更新 SituationalState、Wakeup 和 Action 时检查基础版本。若版本已经变化：

- 尚未执行的旧决策标记为 stale；
- 重新基于最新 Evidence 创建 Cycle；
- 已执行动作不重复执行；
- 依靠 `idempotency_key` 去重。

### 11.3 训练与在线决策隔离

Routine 训练在后台生成候选版本。在线 Cycle 在整个生命周期只读取一个 active 版本；训练完成不能中途替换当前 Cycle 的模型。

## 12. 与现有代码的对应关系

| 目标模块 | 现有基础 | 需要补齐 |
| --- | --- | --- |
| GraphRAG | `graph/retrieval.py` 的向量召回、一跳扩展和时间过滤；`BranchMemoryItem` 的证据谱系 | 持久化图存储、统一 EvidenceBundle、多跳受限扩展、冲突证据、权限和截止点过滤、检索评测 |
| RoutineModel | IdentityKernel 中的 `behavioral-rhythm-v1`，已有小时/星期分布、主动率、气泡延迟 | 独立版本模型、生活阶段、位置/活动/可用性分布、平滑与校准、覆盖率、训练快照 |
| SituationalState | `branches/situational_state.py` 的可信来源、逐槽 TTL 和版本化状态 | 保留 v2 作为 Observed 层；新增不污染记忆的 Inferred 层和合并 Snapshot |
| Agent Runtime | `PerceptionEvent`、`CognitiveCycle`、`AgentWakeup`、`AgentIntention`、主动决策与 jobs | 统一四层输入契约、显式 Policy Gate、版本冻结、stale cycle、行动回流和完整可观测性 |

这里不是推翻现有实现。现有代码已经有不少正确的安全边界，尤其是当前状态不使用向量记忆、SituationalState 只接受可信来源、Wakeup 可审计和主动消息需要新动因。后续改造应围绕这些边界收敛，而不是再搭一套平行系统。

## 13. 推荐实施顺序

### Phase 0：冻结术语和契约

- 定义 Evidence、Claim、RoutinePrior、ObservedState、InferredState、Decision 和 Action；
- 给所有对象增加 `project_id/person_id/evidence_cutoff/version`；
- 建立“推测不得进入证据”的自动测试。

### Phase 1：统一 GraphRAG 输出

- 保留现有数据源，先实现 `EvidenceBundle`；
- 让记忆检索、地点分析和 Runtime 都通过同一证据引用格式；
- 支持支持/反对证据、时间有效性和检索路径；
- 建立检索 benchmark，而不是先追求复杂图算法。

### Phase 2：RoutineModel V1

- 将 `behavioral-rhythm-v1` 抽成独立版本对象；
- 加入平滑、样本覆盖率、证据截止点和生活阶段；
- 先服务“何时值得检查”“回复节奏如何”两个真实需求；
- 再逐步加入位置、活动和状态迁移。

### Phase 3：SituationalState V3

- 不改变现有可信 Observed 状态的写入规则；
- 新增 InferredSituationalBelief；
- 实现合并 `SituationalSnapshot`、来源优先级和逐槽 TTL；
- 当前状态问答强制区分“知道”和“猜测”。

### Phase 4：Runtime 接入

- 每个 Cycle 冻结四层版本；
- 将 Routine 从“发消息概率”降级为调度/准备度先验；
- 实现确定性 Policy Gate；
- 把 `wait`、`internal_only` 和 `next_review_at` 作为一等决策；
- 仅在动作执行成功后回流 Evidence。

### Phase 5：地点与活动扩展

- 在统一 GraphRAG 中按需抽取地点 Claim；
- 将个人语义地点与地图 POI 分层保存；
- 在足够证据支持时训练位置角色和迁移规律；
- 热图只是地点证据和概率分布的一个展示视图，不是 Runtime 的世界本体。

### Phase 6：评测与灰度

- 离线重放历史时间点，禁止看到未来证据；
- 比较不同 Routine 版本的校准误差；
- 灰度评测主动消息的准确率、打扰率和沉默质量；
- 所有决策可追踪到证据、模型版本和策略结果。

## 14. 验收标准

### 14.1 GraphRAG

- 任一进入 Prompt 的长期事实都能追踪到 Evidence；
- 能同时返回支持与反对材料；
- 不泄漏 `evidence_cutoff` 之后的信息；
- 同一查询在同一快照和版本下可重放；
- 地点/人物消歧错误可通过重跑派生层修正，不修改原始聊天。

### 14.2 RoutineModel

- 每个概率都有模型版本、样本量、覆盖率和适用阶段；
- 在稀疏数据下会回退或返回 unknown；
- 不使用 Agent 自产内容自我训练；
- 搬家、换工作等变化不会与旧阶段永久平均；
- 概率经过时间切分的离线校准评测。

### 14.3 SituationalState

- Observed 和 Inferred 在存储及 API 上不可混淆；
- 每个槽位会按 TTL 自动过期；
- 新鲜直接证据能够覆盖 Routine 推测；
- 只有推测时不会输出确定的当前事实；
- 推测不会被普通长期记忆检索召回为证据。

### 14.4 Agent Runtime

- 没有新动因时不会只因 Routine 高概率而主动发消息；
- 并发 Cycle 不会重复发消息或执行工具；
- 每个动作可说明使用了哪些证据和哪一版模型；
- 策略阻止的动作不会以“计划”形式改变世界状态；
- 系统能正式选择沉默并合理安排下一次检查。

## 15. 明确不做的事情

- 不复制 Generative Agents 的固定 Tile Maze；
- 不构造一份模型自认为真实的每日完整日程；
- 不每分钟推理并把预测位置写入记忆；
- 不把 GraphRAG 变成“所有模型产物都塞进去”的垃圾场；
- 不让四个模块都成为能独立行动的 ReAct Agent；
- 不把地图 POI、个人语义地点和当前所在位置混为一谈；
- 不用主动消息数量衡量 Agent 是否“活着”。

## 16. 最终闭环

MoonlightBox 对 Generative Agents 最合适的吸收方式，是保留“持续感知、回忆、判断、行动、再形成新经验”的闭环，同时重新定义每一步的真实性边界：

```text
真实世界输入
→ GraphRAG 保存并检索证据
→ RoutineModel 从长期证据学习概率规律
→ SituationalState 形成有时效、区分观测与推测的当前快照
→ Agent Runtime 根据新动因、关系、风险和权限决定行动或沉默
→ 只有真实执行结果回流为新证据
```

这套设计中的“世界”不是一张固定地图，也不是一份虚构日程，而是由**可追溯的过去证据、可校准的长期规律、有期限的当前信念以及受策略约束的真实行动**共同构成。

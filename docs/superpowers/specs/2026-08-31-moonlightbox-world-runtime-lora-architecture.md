# MoonlightBox 人物世界、持续生活 Runtime 与 LoRA 协同架构

> 核心定义：**LightRAG 重建真实人物及其世界，Life Runtime 让数字人在分支中持续生活，Persona LoRA 让数字人以目标人物的方式表达。** 三者缺一不可，但各自只承担自己擅长的职责。

## 1. 文档目的

MoonlightBox 的目标不是做一个会模仿聊天语气的问答机器人，而是从真实聊天中重建一个具体人物的背景，再让这个人物在一条新的时间分支中持续生活、形成当下状态，并以接近本人的方式与用户互动。

要实现这个目标，必须同时解决三个问题：

1. **她是谁，她生活在怎样的世界里？**
   - 身份、姓名和别名；
   - 工作、学校和组织；
   - 家、公司、常去地点；
   - 家人、朋友、同事和用户之间的关系；
   - 兴趣、偏好、习惯、作息和生活阶段；
   - 重要事件及其相互关系。

2. **用户不说话时，她如何仍然持续生活？**
   - 时间继续前进；
   - 活动、位置、精力、情绪和关注点持续变化；
   - 能睡觉、上班、吃饭、通勤、休息和处理自己的事情；
   - 能记住未完成话题和承诺；
   - 能主动联系、延迟回复、等待或保持沉默；
   - 服务重启后仍能从上次状态继续。

3. **LoRA 小模型在其中做什么？**
   - 它是否只是把聊天数据拿来微调；
   - 它负责语言风格，还是也负责行动决策；
   - 它怎样获得知识图谱、当前状态和生活规律；
   - 如何避免把不断变化的事实硬塞进模型权重。

本文给出三者的统一架构，并坚持一个实施原则：

> 核心能力第一版就存在；复杂度只在真实问题出现后增加。

### 1.1 Windows 宿主机 + WSL2 Linux 运行基线

当前工程的宿主机是 **Windows**，但开发工具和 MoonlightBox 服务运行在 **WSL2 Linux** 中。因此应用层按 Linux 技术栈设计，Windows 负责承载 WSL2、提供宿主机 GPU 驱动和在开机后拉起 WSL 服务。设计不再假设 macOS、Apple Silicon、Metal 或 MLX-LM 存在。

- MoonlightBox API、Life Runtime、Worker、LightRAG 和训练进程默认作为 WSL2 中的 Linux Python 进程运行；
- 开发与启动命令使用 Bash/常见 Linux CLI；常驻进程由 WSL2 内的 `systemd` 管理；
- 现有源码工作树可以继续位于 `/mnt/c`，第一版不强制迁移；但 Python 虚拟环境、`node_modules`、SQLite、向量/图索引和模型文件默认位于 WSL Linux 文件系统，避免把高频 I/O 数据库和索引放在 `/mnt/c`；
- Windows 文件可以通过 `/mnt/c/...` 导入，但正式处理前先复制到 WSL 数据目录；核心代码仍使用 `pathlib` 和配置项，不写死某个用户目录；
- Docker 可以运行在 WSL2/Docker Desktop 集成中，但第一版不要求为了 Sidecar 必须容器化；
- LoRA 使用 Linux 上的 PyTorch + Transformers + PEFT/TRL 训练栈；
- 启动时探测实际硬件与 PyTorch 后端。宿主机有兼容的 NVIDIA GPU 且 WSL CUDA 可用时启用 GPU LoRA/QLoRA；否则仍可运行数据构建、测试与小模型 LoRA，但不承诺大模型 QLoRA 的本机速度。

硬件能力是可探测的运行条件，不应被写死成产品架构。训练与推理通过统一 `PersonaModelEngine` 接口接入，以后更换本机 GPU 后端或远程训练节点不改变 LightRAG 和 Runtime 契约。

## 2. 明确的产品决策

### 2.1 知识图谱不是可选增强

聊天导入完成后，系统必须先使用 LightRAG 建立真实人物背景，再进行重要节点选择。

节点不能只根据情绪突变、回复间隔和关键词产生。不了解人物的工作、地点、社交关系和生活阶段，就无法判断一句普通消息背后是否发生了真正重要的变化。

例如：

- “我以后不去那边了”是否重要，取决于“那边”是不是她工作多年的公司；
- “今天还是她陪我”是否代表关系变化，取决于“她”是谁；
- 回复时间突然从凌晨变成早晨，可能不是关系变化，而是换工作或搬家；
- “回家”可能指父母家、自己的住处或两人共同住处。

因此第一条主链路是：

```text
聊天导入
→ 对话 Episode
→ LightRAG 人物世界图
→ 人物背景档案
→ 基于背景的重要节点分析
→ 自动生成并排序节点
```

### 2.2 24 小时运行是核心能力

数字人不能只在收到用户消息时临时存在。第一版就必须拥有持续时间、生活状态、日程推进、延迟回复和主动行为。

但“24 小时运行”不等于每分钟调用一次大模型。正确实现是事件驱动的持久 Runtime：没有事件时系统休眠，下一次状态转换、承诺到期或主动检查时间到达时再唤醒。

### 2.3 LoRA 是人物 Actor，不是整个世界

LoRA 不负责保存全部背景知识，也不负责自己维护时钟、数据库和任务队列。

第一版将 LoRA 定位为 `PersonaActor`：

- 学习目标人物如何说话；
- 学习连续气泡、标点、表情、贴纸和回复节奏；
- 学习目标人物面对常见对话情境时的反应方式；
- 在 Runtime 给出的意图和当前处境下生成真正像本人的表达。

世界知识由 LightRAG 提供，持续生活由 Runtime 提供，LoRA 负责把两者转化成目标人物的外显行为。

## 3. 总体架构

```text
                    ┌──────────────────────────┐
                    │       真实聊天导入        │
                    └─────────────┬────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │ Episode 切分与媒体语义化  │
                    └─────────────┬────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │ LightRAG Historical World│
                    │ 人物、地点、组织、关系、   │
                    │ 工作、活动、规律、事件     │
                    └───────┬──────────┬───────┘
                            │          │
                 人物背景档案│          │节点上下文
                            ▼          ▼
                 ┌──────────────┐  ┌──────────────┐
                 │World Profile │  │ Node Analyzer│
                 └──────┬───────┘  └──────┬───────┘
                        │                 │
                        └────────┬────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  从节点创建时间分支       │
                    └─────────────┬────────────┘
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │                   Life Runtime                  │
        │                                                 │
        │ VirtualClock  DayPlan  LifeState  EventQueue    │
        │      │          │         │          │          │
        │      └──────────┴─────────┴──────────┘          │
        │                         │                       │
        │              Director / Life Decision          │
        └─────────────────────────┬───────────────────────┘
                                  │表达意图与当前处境
                                  ▼
                    ┌──────────────────────────┐
                    │ PersonaActor：Base + LoRA│
                    │ 文字、气泡、表情与节奏    │
                    └─────────────┬────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │ 消息/等待/状态变化/Wakeup │
                    └──────────────────────────┘
```

整个系统不是四个自由行动的 Agent，而是一条由代码控制的流水线。

## 4. 三个大脑与一个身体

为了避免职责混乱，可以把系统理解为“三个大脑与一个身体”。

### 4.1 历史大脑：LightRAG

回答：

- 她是谁；
- 她认识谁；
- 她在哪里生活和工作；
- 她过去经历过什么；
- 哪些事件、人物和地点相互关联；
- 当前话题与过去哪些经历有关。

LightRAG 使用图结构与向量表示进行双层检索，适合同时处理具体实体关系与跨对话的全局背景。官方实现提供 `local`、`global`、`hybrid`、`naive` 和 `mix` 查询模式，其中 `mix` 会组合知识图谱与原始文本块检索。

### 4.2 生活大脑：Life Runtime

回答：

- 分支中的时间现在走到哪里；
- 她现在在做什么；
- 精力、情绪和可用性如何；
- 下一件生活事件什么时候发生；
- 用户消息是否打断当前活动；
- 现在应该回复、主动联系、等待还是晚点继续。

这些能力主要依赖持久状态、时钟和调度，不应交给 LoRA 权重自行记忆。

### 4.3 表达大脑：Persona LoRA

回答：

- 她会怎样组织这句话；
- 会发一条还是连续多条；
- 使用什么口头禅、标点、表情和贴纸；
- 面对冲突、亲密、尴尬、关心或拒绝时怎样表达；
- 主动开场和被动回复分别是什么风格。

### 4.4 身体：Executor

负责真实执行：

- 保存消息；
- 按气泡延迟投递；
- 更新 LifeState；
- 安排 Wakeup；
- 取消过期动作；
- 保证幂等与分支隔离。

模型可以提出行动，只有 Executor 能让行动真正发生。

## 5. 导入后先构建人物世界

### 5.1 进入 LightRAG 前的规则预提取

Bundle 不是直接从未经处理的聊天文本拼接而来。第一版先执行一个以规则为主、必要时需要用户处理歧义的不调用 LLM 的预处理阶段，解决“谁在说话”和“同一个人有多个称呼”这两个 LightRAG 难以稳定解决的问题。这个阶段先处理原始数据中的发送者字段，再处理正文中的缩写和昵称；不提前判断朋友、恋人、工作关系等人物语义。

输入是数据库中的原始 `Message` 和 `Participant`；其中 `Participant.role` 是说话人身份的唯一权威来源：

```text
self    → USER（用户）
target  → TARGET_PERSON（目标人物）
other   → OTHER（其他已登记参与者）
```

每条消息先生成候选提及，再生成内部的 `NormalizedMessage`；原文永远保留：

```yaml
NormalizedMessage:
  message_id: string
  participant_id: string
  participant_role: self | target | other
  original_content: string        # 原始消息，不修改
  normalized_content: string      # 供 Bundle/LightRAG 使用的规范化文本
  replacements:                   # 仅记录确定执行的替换
    - original: string
      normalized: string
      participant_id: string
      rule: exact_alias | explicit_alias | safe_variant
  unresolved_mentions:            # 不能安全归属的缩写、昵称或引用对象
    - text: string
      reason: string
  quoted_segments:                 # 转发、引用或复制的内容边界
    - speaker_text: string | null
      content: string
```

预处理首先从原始导入格式的发送者字段建立参与者和别名。微信导出的每条消息通常已经带有发送者昵称；例如 `Feather`、`AMOS` 如果出现在消息发送者列，就属于**来源级参与者昵称**，可以用固定规则直接写入 `Participant` 的显示名或别名表，不应作为正文内容再送入 Bundle，也不需要用户逐项确认。只有正文中的缩写、昵称、签名或引用署名无法通过发送者字段确定归属时，才产生待用户处理的 `MentionCandidate`：

```yaml
MentionCandidate:
  text: yy
  occurrences: [m101, m205, m306]
  surrounding_samples: [string]
  suggested_participants: [participant_id]
  decision: pending | mapped | third_party | unresolved
  mapped_participant_id: string | null
```

系统在这里暂停进入 LightRAG 的流程，但只把正文中的未决候选交给用户处理。来源级发送者昵称已经由规则自动入表，不出现在候选确认列表中。用户在一个项目级映射界面中处理正文候选：

```text
yy       → 余熠（已有参与者）
小羽     → 洪欣羽（已有参与者）
某个签名 → 第三方人物（新建项目人物）
未知缩写 → 未确定，暂不替换
```

用户可以把正文候选映射到已有参与者、登记为第三方人物、标记为未解析，或合并多个候选称呼。只有 `mapped` 的候选才会参与规范化替换；`third_party` 只获得明确的新人物标识；`unresolved` 保留原文，不进入规范姓名替换。该确认是一次项目级身份整理，不是逐条消息审批，也不要求用户判断关系、偏好或事件。来源级发送者昵称不经过这一步，直接使用其原始参与者记录。

映射结果必须绑定项目和参与者，而不是写成全局字符串字典：

```yaml
AliasBinding:
  project_id: string
  alias_text: string
  participant_id: string | null
  third_party_person_id: string | null
  decision: mapped | third_party | unresolved
  decided_by: user
  mapping_version: string
```

预处理规则按以下顺序执行：

1. **固定说话人。** Bundle 中的发言标签使用 `[USER]`、`[TARGET_PERSON]` 和 `[OTHER]`，不使用消息正文里的“我”“她”“对方”来决定当前说话人。
2. **建立项目内称呼表。** 直接使用原始消息发送者字段创建或更新 `Participant`，并把 `Feather`、`AMOS` 等来源级昵称写入该参与者的显示名／别名。正文中明确出现“X 就是 Y”时，再记录为别名映射。所有映射必须绑定 `participant_id`，不能只保存字符串到字符串的全局字典。
3. **执行安全替换。** 只替换规范名称的大小写、空格、标点变体，以及已经明确绑定的昵称。替换时保留 `original_content` 和 `replacements`，不得覆盖原文。
4. **处理正文缩写和首字母。** `yy`、`hx y` 等不是发送者字段的缩写才作为 `MentionCandidate` 呈现给用户。只有用户将其绑定到参与者，或用户登记为第三方人物后，才生成替换；未处理的候选保持 `unresolved_mentions`，不能仅因拼音首字母相似就自动合并。
5. **处理引用和转发。** 引用中的“我”、署名和被引用昵称不继承当前消息的说话人身份。引用内容保留边界和原始署名，只作为被提及文本；无法确认引用者时，不创建新的 `Person`。
6. **写入身份说明。** 每个 Bundle 在送入 LightRAG 前附加简短说明：角色标签对应的 `participant_id` 和规范名称，以及“未列入映射表的名字只是被提及对象”。说明用于消除歧义，不作为人物事实抽取。

规范化文本示例：

```text
[WORLD_IDENTITY]
[USER] = 余熠（participant_id: p-user）
[TARGET_PERSON] = 洪欣羽（participant_id: p-target）
发送者字段中的昵称已经在 Participant 表中登记；未列入正文映射表的姓名、昵称和缩写只是被提及内容，不自动视为说话人。

[USER]：余熠（原文称呼：yy）今天去公司吗？
[TARGET_PERSON]：我今天不上班。
```

这里的“余熠（原文称呼：yy）”只在用户完成 `yy → 余熠` 映射后出现；未确定时保持原文并记录为 `unresolved_mentions`。预处理阶段不得根据“好朋友”“关系很好”“像家人”等词直接建立关系，也不得根据一次出现把对象提升为长期偏好或规律。

该阶段的结果可以只作为 Bundle 构建的内存中间对象，不需要新增 LightRAG 实体表。若需要持久化，保存 `message_id → normalized_content/replacements/unresolved_mentions` 的处理记录即可；LightRAG 的节点 ID 仍不是 MoonlightBox 的业务主键。

用户完成候选映射后，只有 `normalized_content` 进入下一阶段；`original_content`、替换记录、映射版本和消息映射继续保存在 MoonlightBox，用于来源追溯和重新处理。用户未完成映射时，任务保持 `awaiting_identity_mapping`，不向 LightRAG 写入未确定的规范姓名。

### 5.2 原始消息使用轻量会话窗口组织

不把所有聊天拼成一个无边界的巨型文档，也不把每条消息作为独立文档。第一版只使用很朴素的边界：

- 一次连续聊天作为一个 `ConversationBundle`；
- Bundle 太长时再按长度切开；
- 新 Bundle 带上前一个 Bundle 最后两三轮对话作为 `carry-in context`；
- 不先做复杂的“事件置信度”“主题变化阈值”“证据充足度判断”。

这里的 Episode 更接近“会话窗口”，不是系统声称已经准确识别出的现实事件。一个 Episode 是 LightRAG 的来源文档单位，不等于一个不可再分的 chunk；LightRAG 仍可以在较长 Episode 内部继续切块。

Episode 正文中的 message 标签也不是必须的。例如：

```text
[message:m101][19:22][用户]
```

这种写法实现简单，但技术标识会进入实体与关系抽取上下文，可能带来噪声。默认使用更干净的侧表映射：

```text
LightRAG 文档正文：只包含自然可读的对话
MoonlightBox 映射表：document_id → episode_id → message_ids
```

```yaml
lightrag_doc_id: doc-123
episode_id: ep-001
message_ids: [m101, m102, m103, m104]
```

查询命中 LightRAG 文档或 chunk 后，再通过映射表回到原始消息。这样既保留消息级可追溯性，也不需要让抽取模型反复看到 `m101` 之类的内部编号。

### 5.3 第一版实体类型

这里的“实体类型”不是客观百科分类，而是数字人从“我是谁、我如何生活、我如何认识别人”的角度组织记忆。SQLAlchemy 中保存的人物世界对象是唯一的持久化语义模型；LightRAG 不另建一套业务类型，只读取同一字段作为抽取提示和检索索引。

第一版保存在 SQLAlchemy 人物世界对象表中的 `entity_type` 只有以下 7 类：

| `entity_type` 字段值 | 在数字人世界中的含义 | 不应单独承担的语义 |
| --- | --- | --- |
| `Person` | 我认识、在意或谈到的人 | 不自动说明与我的关系 |
| `Place` | 对我的生活有意义的地方 | 不自动说明是住处、单位或常去地点 |
| `Organization` | 与我的工作、学习、社交或活动有关的组织 | 不自动说明我在其中任职或就读 |
| `Activity` | 我做过、正在做或计划做的事情 | 不自动说明它是一次事件还是长期规律 |
| `Event` | 发生在我身上、改变我状态或关系的具体事情 | 不把仅被提到的话题当作已经发生 |
| `Topic` | 我反复关注、讨论或回避的主题 | 不是事实、人物或事件本身 |
| `Object` | 对我有意义但不适合上述类别的对象 | 不能用作无描述的万能实体 |

“我的工作”“我的偏好”“我的生活规律”“我的人生阶段”“我与某人的关系”不再作为 `entity_type` 的独立值，而是 SQLAlchemy 中以人物为中心的档案字段或关联记录。每个对象都要保留规范名称、原文称呼、人物视角下的意义、时间状态和来源；未知信息可以为空，不得为了填满字段而猜测。

LightRAG 的实体类型提示文件应直接复用上述 7 个 `entity_type` 值。LightRAG 返回的节点和关系在写入 SQLAlchemy 前，必须映射到 `project_id`、`subject_person_id`、对象类型、人物视角描述和来源字段；LightRAG 的节点 ID、开放类型或摘要不是数据库的业务主键，也不是最终档案内容。

### 5.4 第一版关系类型

关系类型不是 `Relationship` 实体的字段，也不是另一套客观模型；它们是 SQLAlchemy 中 `social_relationships`、`work_and_education`、`places` 等人物中心字段或关联记录中的可选规范值。LightRAG 可以暂时把它们作为边标签帮助检索，但边标签必须回写为数据库关系记录，不能取代数据库字段。

| 规范关系 | 方向与语义 | 建模边界与必须保留的信息 |
| --- | --- | --- |
| `knows` | `Person ↔ Person`：双方存在可确认的认识或交往关系。 | 这是最低限度的熟人关系，不等于朋友。保留认识场景、起始时间和当前状态；仅听说过某人不能证明双方互相认识。 |
| `family_of` | `Person ↔ Person`：双方具有家庭或亲属关系。 | 必须保留具体子类型和方向，例如母女、姐弟、表亲、继亲；“像家人一样”只描述亲密感，不能直接归一为亲属关系。 |
| `friend_of` | `Person ↔ Person`：双方具有可确认的友情。 | 保留如何认识、友情阶段、亲密程度及当前／过去状态。“关系很好”只描述关系质量，如果不知道关系性质，不能单凭这句话判定为朋友。 |
| `colleague_of` | `Person ↔ Person`：双方在同一明确工作语境中共事。 | 保留组织、团队、职责和共事时期。同一行业、同一办公楼，甚至同属一个大组织，都不一定意味着直接共事。 |
| `partner_of` | `Person ↔ Person`：双方具有恋爱、婚姻或其他明确伴侣关系。 | 保留伴侣子类型、关系阶段、起止时间和状态。暧昧、单方喜欢或一次亲密互动不能自动归一为伴侣关系。 |
| `works_at` | `Person → Organization`：人物在某组织任职或工作。 | 保留职位、部门、工作地点、起止时间及当前／过去／计划状态。拜访某公司、与其合作或谈到该公司不等于在那里工作。 |
| `studies_at` | `Person → Organization`：人物在某学校或教育机构学习。 | 保留专业、年级、学籍身份、起止时间和状态。参观学校、参加短期活动或谈到学校不等于在那里就读。 |
| `lives_at` | `Person → Place`：人物在某地点居住。 | 保留地点精度、居住性质、起止时间和状态。短暂停留、经常拜访或“准备搬去”不能写成当前居住；“搬到”通常还应建立搬家 `Event`，再表达居住状态的变化。 |
| `often_visits` | `Person → Place`：人物在某时期反复前往某地。 | 保留目的、频率、典型时间和有效时期。一次到访不足以建立“常去”，但第一版不使用固定次数阈值，可忠实保存“经常去”等完整陈述。 |
| `participates_in` | `Person → Activity/Event`：人物实际参与某项活动或事件。 | 保留参与角色、方式和时间。提到、关注、旁观或观看某项活动不一定等于参与。若参与的是组织，应进一步表达其成员身份或具体活动，不能模糊处理。 |
| `likes` | `Person → 任意实体`：人物对对象具有明确的正向偏好。 | 保留喜欢的具体对象、方面、强度、情境和时间。一次选择、接受建议或礼貌赞同不能直接推导出稳定喜欢。 |
| `dislikes` | `Person → 任意实体`：人物对对象具有明确的负向偏好。 | 保留不喜欢的具体对象、方面、强度、情境和时间。一次拒绝可能源于价格、时间或身体状况，不能自动归一为长期不喜欢。 |
| `usually_does` | `Person → Activity/Routine`：人物在一定条件下通常会做某事。 | 保留频率、时间窗口、条件、有效时期和例外。一次行为不能建立该关系；不设置统一次数阈值，以原文是否表达稳定规律为第一版依据。 |
| `before` | `Event → Event`（也可用于档案中的阶段记录）：前者在时间上早于后者。 | 保留参照时间、粒度和确定性。时间先后本身不表示因果；时间信息不足时不要强排顺序。 |
| `after` | `Event → Event`（也可用于档案中的阶段记录）：前者在时间上晚于后者。 | 是 `before` 的反向查询关系，保留相同的时间限定。实现可以只存一个方向并在查询时生成反向视图，避免两边不一致。 |
| `related_to` | `任意实体 ↔ 任意实体`：存在有意义联系，但当前上下文不足以归入更具体的规范关系。 | 只作兜底，必须保留原始关系描述和无法归一的原因，不能把它当成丢弃细节的“万能边”。后续获得更多上下文时可升级为具体关系。 |
| `changed_by` | 人物的工作、生活阶段、习惯或关系被某个 `Event` 改变。 | 保留改变前后状态、方向、时间和表述的确定性；仅仅先后发生不能证明因果。 |

关系归一化必须读取完整关系陈述，而不是把孤立词语映射成谓词。处理顺序为：先从 Bundle 上下文中识别主体、客体和开放关系；再保留原始关系、时间、状态及限定信息；只有语义足够明确时才填写 `canonical_relation`。无法确认的关系保持开放或未决，不能为了得到一张“整齐”的图而强制归类。

例如，“小羽是我大学室友，现在也是我最好的朋友”可以归一为 `friend_of`，但还要保留“大学室友”这一认识来源、“最好”这一亲密程度和“现在”这一状态；只出现“我和小羽关系很好”时，关系性质仍可能是亲属、朋友、同事或伴侣，因此只能保留原始关系描述，不能直接写成 `friend_of`。

LightRAG 可以产生开放关系描述。MoonlightBox 保留 `raw_relation`、限定信息、时间、状态和来源，再视完整上下文填写规范关系；无法归一化的内容仍留在人物档案的 `unresolved_relationships` 中。

### 5.5 人物背景编译

`BackgroundCompiler` 不是把图谱翻译成客观人物百科，而是根据 LightRAG 检索到的原始上下文，编译“这个数字人眼中的自己和生活”。查询问题必须围绕人物主体组织：

```text
我是谁，我怎样称呼自己，别人怎样称呼我？
我现在和过去分别处于什么工作、学习和生活状态？
我认识哪些人，我如何理解自己与他们的关系？
哪些地方对我的生活有意义，我在哪里居住、工作、学习或经常活动？
我明确表达过哪些喜欢、排斥和有条件的偏好？
我有哪些被自己或他人描述过的生活规律，它们适用于什么时间和情境？
哪些事件改变了我的生活、工作、地点或人际关系？
我和用户的关系在不同时间如何变化？
```

查询使用 `mix` 或按问题选择 `local/global`。LightRAG 只返回检索上下文，编译模型必须遵守以下提示：

```text
你正在维护一个数字人的主观世界档案，不是在编写客观人物百科。
所有字段都从“这个人如何理解自己、他人和生活”的角度填写。
先保留原始陈述，再做简洁归纳；不要把孤立关键词当成事实或关系。
关系必须读取完整上下文，保留 raw_relation、对象、时间、状态、限定信息和来源。
“关系很好”不能单独归一为 friend_of；“搬到”应记录为生活地点变化事件，并在语义明确时更新居住状态。
不确定、矛盾、转述和计划必须分别标记，不因信息不完整而整条删除，也不为了填满字段而猜测。
LightRAG 的实体类型和关系标签只是检索辅助，不是最终档案的栏目名称。
输出严格符合 PersonWorldProfile 结构；每个条目附 source_message_ids 和 source_status。
```

### 5.6 PersonWorldProfile

```yaml
PersonWorldProfile:
  project_id: string
  subject_person_id: string
  identity:
    names: [string]
    aliases: [string]
    self_descriptions: [string]
    roles: [string]
  work_and_education: [object]       # 我当前和过去的工作、学习及角色
  places: [object]                   # 对我有意义的居住、工作、学习和活动地点
  social_relationships: [object]     # 我与他人的关系；relation 是可选规范值
  preferences: [object]              # 我明确表达的喜欢、排斥和条件偏好
  recurring_activities: [object]     # 我反复做的活动及其条件
  routine_summary: object             # 由 recurring_activities 归纳的生活节奏
  life_phases: [object]              # 我经历或正在经历的持续阶段
  relationship_with_user: object      # 我如何理解与用户的关系
  important_events: [object]          # 改变我状态、地点、工作或关系的事件
  unresolved_relationships: [object] # 尚不能确定性质的关系，不强行归一
  unresolved_candidates: [object]    # 其他待解释的对象或陈述
  source_message_ids: [string]
  graph_version: string
  compiler_version: string
```

档案中的字段不采用“证据不够就整条删除”的模式。每项可以带简单来源状态：

```text
direct       原始消息直接表达
summarized   多段历史归纳
inferred     由历史上下文推导
superseded   已被更新的历史内容覆盖
```

`inferred` 仍然展示和参与理解，不通过绝对阈值将它筛掉，但与原始消息直接表达的内容保持来源区分。

### 5.7 背景自动编译并直接进入下一步

背景编译完成后立即进入节点分析，不再设置人物背景内容的用户审批、人工校验或必须补全字段的阻塞环节。唯一的用户参与点是进入 LightRAG 之前的身份映射；人物背景页只用于展示系统已理解的内容：

- 基本身份；
- 社交关系图；
- 常见地点；
- 工作与生活阶段；
- 长期偏好和规律；
- 来源为 `inferred` 的推导内容。

如果日后的自然对话中出现了更新信息，运行时先把它保存为该分支的
`LifeEvent`/`BranchMemoryItem`，供当前分支上下文使用；不会在每轮对话后自动重建
LightRAG 或覆盖 `PersonWorldProfile`。当用户明确要求更新人物背景，或产品安排离线
profile refresh 时，才把这些带来源的分支证据送入别名审核、图谱版本和人物背景编译
流程，生成新的 `WorldGraphVersion`/`PersonWorldProfile`。这样人物档案仍是可回溯的
历史基线，分支记忆则是运行时增量。

## 6. 人物背景如何帮助节点选择

节点发现仍然可以使用时间间隔、情绪、主题和回复节奏等局部信号，但这些信号只负责找到候选窗口。

真正判断重要性时，需要加入人物世界：

```text
局部对话变化
+ 涉及的人物与关系
+ 涉及的地点、工作或生活阶段
+ 之前相关事件
+ 之后是否产生持续影响
→ 重要节点候选
```

例如：

- 换工作不是普通话题变化，而是 `LifePhase` 转换；
- 第一次提到某个人，如果该人后来持续出现，可能是社交关系节点；
- 搬家会改变地点图、作息和回复时间，不能被误判为感情降温；
- 一次争吵是否重要，要看它是否改变关系状态和后续互动。

因此节点分析分成：

1. 局部候选发现；
2. LightRAG 查询相关人物、地点、事件和长期主题；
3. 背景增强判断；
4. 全局排序与去重；
5. 自动生成可用的节点列表。

这里不要求每个候选达到多个独立阈值。系统保留排序结果并默认发布可用节点，不要求用户审核、批准或决定保留数量。

## 7. LightRAG 的三个知识范围

时间分支不能直接查询包含全部未来聊天的完整图，因为 LightRAG 合并后的实体描述本身可能已经吸收未来信息。只在返回文本块后按时间过滤并不充分。

因此需要三个隔离范围。

### 7.1 Full Historical World

包含项目的完整真实聊天，用于：

- 建立人物背景；
- 选择和解释重要节点；
- 训练数据分析；
- 项目管理界面的全局探索。

它不能直接进入历史分支 Runtime。

### 7.2 Origin World Snapshot

用户选择一个节点创建分支时，系统直接为该节点建立截止安全的知识快照：

```text
只索引 occurred_at <= origin_time 的 Episode
```

同一节点创建的多个分支可以共享这个只读快照。

项目数据量有限时，第一版允许重新构建节点快照。它比让未来信息悄悄进入实体摘要更安全，也比一开始开发复杂时态图版本系统更简单。

### 7.3 Branch World Overlay

每条分支拥有独立的增量知识范围，保存：

- 分支中实际发送的消息；
- 已发生的生活事件；
- 分支内形成的新关系状态；
- 该分支中的承诺、计划和共同经历。

运行时查询：

```text
Origin World Snapshot
+ 当前 Branch World Overlay
```

其他分支的 Overlay 永远不可见。

第一版可以将 Branch Overlay 保存在 MoonlightBox 数据库和简单向量索引中；当分支内容积累到需要跨事件关系检索时，再增量写入独立 LightRAG 知识库。LightRAG 官方支持增量更新，因此不需要每次重建完整图。

## 8. LightRAG 的工程接入方式

### 8.1 作为独立 Sidecar 服务

MoonlightBox 第一版采用独立 Sidecar。官方 LightRAG Server 的 `workspace` 是进程启动级配置，查询接口不能为每个请求切换 workspace；直接让所有 MoonlightBox 项目共用一个 Server workspace 会造成跨项目检索污染。为保持 REST 边界并确保项目隔离，Sidecar 内部使用 LightRAG Core，为每个不可变 `WorldGraphVersion` 初始化独立 workspace：

```text
MoonlightBox API
    │ REST
    ▼
MoonlightBox LightRAG Sidecar
    └─ LightRAG Core（每个图版本独立 workspace）
       ├─ KV storage
       ├─ Vector storage
       ├─ Graph storage
       └─ Document status storage
```

Sidecar 只暴露批量文档写入和 `only_need_context` 查询，不负责人物档案的最终回答。WSL2 第一版默认由 `systemd` 启动和监控该独立 Linux 进程，MoonlightBox 通过 `localhost` REST 访问。如果后续选择 Docker Desktop，只替换 Sidecar 的启动方式，不改变 API 契约。

`WorldGraphVersion` 的配置指纹包含 `MOONLIGHTBOX_LIGHTRAG_INDEX_REVISION`。更换抽取模型、嵌入模型、嵌入维度或切块配置时只需递增这个显式版本号，系统就会创建新的不可变 workspace；这是一条索引兼容性边界，不是筛选人物信息的阈值。

这样可以：

- 独立升级 LightRAG；
- 不让 LightRAG 的依赖污染 Persona 训练和推理环境；
- 单独管理索引任务和失败恢复；
- 在 WSL2 中的 Runtime 与独立知识服务之间保持边界。

### 8.2 模型角色不能混用

LightRAG 的实体关系抽取与长上下文查询需要比普通向量 RAG 更强的通用模型。Persona LoRA 小模型不承担图谱抽取。

第一版模型分工：

| 任务 | 模型 |
| --- | --- |
| LightRAG EXTRACT | 独立的通用抽取模型 |
| LightRAG KEYWORD | 快速通用模型 |
| BackgroundCompiler / Node Analyzer | 较强通用分析模型 |
| Runtime Director | 通用指令模型，可与 PersonaActor 共用基础权重但不加载人格 LoRA |
| PersonaActor | 基础模型 + 目标人物 LoRA |

本地资源不足时，离线图谱构建可以配置云端或局域网模型；是否发送聊天内容必须在界面中明确说明。

### 8.3 嵌入模型需要尽早固定

LightRAG 官方说明索引和查询必须使用同一嵌入模型，切换嵌入模型通常需要重新嵌入数据。MoonlightBox 应在项目首次建图时记录：

- embedding model；
- embedding dimension；
- chunking strategy；
- extraction model；
- entity type prompt version；
- LightRAG version。

这些共同组成 `WorldGraphVersion`。

## 9. 24 小时持续生活的正确含义

“24 小时运行”不是让 LLM 永不停止地思考，而是满足四个条件：

1. 数字人的分支时间持续前进；
2. 生活状态在用户离开后仍可演化；
3. 下一次有意义的事件会可靠触发 Runtime；
4. 进程停止后可以恢复并快进到当前时间。

一个真实的人大多数时间也不会给用户发消息。持续生活与持续输出不是一回事。

WSL2 内不依赖某个永不退出的前台终端。API、Runtime、Worker 和 LightRAG 由 `systemd` 管理，但不假设 `systemd` 本身能永久保持 WSL 实例存活。Windows 任务计划程序在开机或登录后执行 `wsl.exe -d <distro> -- systemctl start moonlightbox.target` 拉起服务组。Wakeup、VirtualClock 锚点和最近成功 Cycle 都持久化；WSL 或 Windows 重启后按第 17 节快进，而不补跑每一分钟。

## 10. VirtualClock：持续生活的时间基础

### 10.1 分支时间与现实时间映射

每条分支记录：

```yaml
VirtualClock:
  branch_id: string
  virtual_anchor: datetime
  wall_anchor: datetime
  time_scale: number
  status: running | paused
  timezone: string
```

默认映射：

```text
virtual_now
= virtual_anchor
+ (wall_now - wall_anchor) × time_scale
```

例如用户从 2024 年的某个节点创建分支，创建时 `virtual_anchor` 等于节点时间，之后可以按 1:1 速度继续，也可以由产品允许加速或暂停。

### 10.2 时钟由代码计算

LLM 不负责猜现在几点。所有 Cycle 都接收同一个确定的 `virtual_now`。

暂停、恢复和调整时间倍率时创建新的锚点，不修改已经发生的 LifeEvent 时间。

## 11. LifeState：数字人当前怎样生活

```yaml
LifeState:
  branch_id: string
  virtual_now: datetime
  location_role: string | null
  activity: string
  social_context: string | null
  availability: available | busy | resting | asleep
  energy: low | medium | high
  mood: string
  attention: string | null
  current_goal: string | null
  open_conversation_threads: [object]
  commitments: [object]
  last_transition_at: datetime
  current_plan_block_id: string | null
  version: integer
```

这是分支模拟现实，不是对真实人物此刻状态的断言。因此不要求每个字段拥有真实世界 Evidence。

它的约束来自：

- Origin World Snapshot；
- 人物长期规律；
- 当前分支已经发生的事情；
- 当前时间；
- Director 对下一段生活的合理安排。

## 12. DayPlan：一天的生活骨架

### 12.1 为什么需要日程骨架

如果每次 Wakeup 都完全重新猜测生活，状态会频繁跳变。如果预先生成精确到每分钟的日程，又会僵硬且昂贵。

第一版每天生成少量粗粒度生活块：

```yaml
DayPlan:
  date: 2026-08-31
  blocks:
    - start: 00:00
      end: 07:30
      activity: sleep
      location_role: home
    - start: 08:30
      end: 12:00
      activity: work
      location_role: workplace
    - start: 12:00
      end: 13:30
      activity: meal_and_rest
    - start: 18:30
      end: 23:30
      activity: personal_time
```

生活块来自：

- PersonWorldProfile；
- RoutineProfile；
- 工作日、周末和节假日；
- 当前生活阶段；
- 分支中的承诺和计划；
- 少量随机性。

### 12.2 日程允许被打断

用户消息、新承诺、情绪事件和上一活动延迟都可以修改当天剩余计划。已经发生的 LifeEvent 不重写，未发生的计划可以重新安排。

DayPlan 是生活骨架，不是世界真相。

## 13. EventQueue 与 Wakeup

Runtime 只在有意义的时间醒来：

```text
用户发送消息
计划块开始或结束
延迟回复到期
承诺或开放话题到期
Director 主动安排的下一次检查
外部 Provider 返回新事件
```

```yaml
Wakeup:
  id: string
  branch_id: string
  wake_at: datetime
  reason: string
  trigger_type: plan_transition | delayed_reply | proactive_review | commitment
  idempotency_key: string
  status: scheduled | executing | completed | cancelled
```

系统不需要每分钟产生一条心理活动。没有事件时只保存下一次 Wakeup。

## 14. Runtime Cycle

一次完整 Cycle：

```text
1. 根据 VirtualClock 计算 virtual_now
2. 加载上一个 LifeState 和未完成 DayPlan
3. 如果进程中断过，快进处理已经跨过的计划边界
4. 合并本轮 Trigger
5. 查询 Origin World Snapshot 与 Branch Overlay
6. 读取 PersonWorldProfile、RoutineProfile 和当前关系状态
7. Director 决定下一段生活和是否需要表达
8. 如果需要表达，调用 PersonaActor LoRA
9. Executor 保存 LifeEvent、消息和新 LifeState
10. 安排下一次 Wakeup
```

输入契约：

```yaml
RuntimeContext:
  trigger: object
  virtual_now: datetime
  person_world_profile: object
  routine_profile: object
  life_state: object
  day_plan: object
  recent_branch_events: [object]
  historical_context: [object]
  branch_context: [object]
```

Director 输出：

```yaml
LifeDecision:
  state_patch: object
  plan_patch: object | null
  action: speak | wait | continue_life | schedule
  speech_mode: reply | proactive | delayed_reply | null
  communication_intent: string | null
  content_points: [string]
  next_wakeup_at: datetime | null
  private_reason: string
```

它不输出连续 readiness 分数。行动是离散选择。

## 15. 主动行为如何产生

第一版允许普通而非任务驱动的主动联系。一个真人可以因为以下原因发消息：

- 想起用户；
- 刚发生一件想分享的小事；
- 到了平时常聊天的时间；
- 想继续未完成话题；
- 某种情绪使她想靠近或回避；
- 没有重大理由，只是产生自然的社交冲动。

Routine、LifeState、开放话题和关系状态一起进入 Director，由它选择 `speak` 或 `wait`，不经过“证据分 ≥ X、动机分 ≥ Y、准备度 ≥ Z”的串联门槛。

只保留确定性边界：

- 用户是否允许主动消息；
- 安静时段；
- 分支是否运行；
- 是否为重复 Wakeup；
- 上一条尚未投递的相同动作是否存在。

这些是运行一致性和用户权限，不是人物是否有资格生活的阈值。

## 16. 用户消息如何打断生活

用户消息具有实时优先级：

1. 保存用户消息；
2. 取消尚未投递但已过时的主动草稿；
3. 将当前活动和可用性提供给 Director；
4. Director 可选择立即回复或安排延迟回复；
5. PersonaActor 只在真正需要生成外显消息时运行。

忙碌不意味着接口不返回结果。产品可以先显示已送达状态，随后按 LifeDecision 投递回复气泡。

## 17. 服务重启与时间快进

24 小时能力必须建立在持久化上，而不是常驻线程恰好没死。

每个分支持久化：

- VirtualClock 锚点；
- 当前 LifeState；
- DayPlan；
- 未完成 Wakeup；
- 未完成承诺；
- 最近成功 Runtime Cycle。

重启恢复：

```text
读取 last_virtual_now
→ 计算 current_virtual_now
→ 找出中间跨过的计划块
→ 合并为少量 LifeEvent
→ 更新当前状态
→ 处理仍有意义的到期承诺和延迟回复
→ 丢弃已经失去意义的旧检查
→ 安排下一个 Wakeup
```

不逐分钟补跑，不为离线的每一分钟调用模型。

## 18. LoRA 不是“把全部数据丢进去训练”

直接把所有聊天按 `user/assistant` 转成 JSONL，可以学到一部分措辞和语气，但不足以得到真正的 PersonaActor。

它存在几个问题：

- 连续短消息可能被错误拆成多轮；
- 无法区分主动开场和被动回复；
- 丢失发送间隔、多气泡、表情和贴纸行为；
- 不知道当时处于哪个关系阶段；
- 不知道哪些上下文是真实历史，哪些只是训练提示；
- 容易记住具体事实，却不会在新状态下正确使用人格；
- 聊天记录只包含实际发送的消息，无法直接教会模型什么时候等待。

因此 LoRA 训练首先是数据建模问题，然后才是调用 WSL2 Linux 中的 PEFT/TRL 训练引擎。

## 19. PersonaActor LoRA 应该学习什么

### 19.1 应该进入 LoRA 的稳定能力

- 词汇和句式；
- 长短句比例；
- 标点、语气词和表情；
- 单气泡或多气泡结构；
- 气泡之间的相对延迟；
- 主动开场方式；
- 被关心、被质疑、冲突、亲密和拒绝时的典型反应；
- 是否倾向追问、转移话题、直接表达或保持含蓄；
- 目标人物作为一个主体的边界感。

### 19.2 不应该主要依赖 LoRA 保存的内容

- 姓名、地点、公司和社会关系；
- 某个历史节点前后的事实；
- 当前日期和时间；
- 此刻活动、位置、情绪和精力；
- 当前分支承诺；
- 新产生的共同经历；
- 下一次 Wakeup。

这些内容会变化，必须由 LightRAG 和 Runtime 在推理时提供。

## 20. LoRA 训练数据管线

```text
原始聊天
→ 参与者映射
→ Episode 切分
→ 同一发送者连续气泡合并为 Turn
→ 识别 responsive / proactive
→ 保留文本、表情、贴纸和气泡间隔
→ 关联当时的关系阶段与粗粒度时间情境
→ 构建 PersonaActor 训练样本
→ 按时间与 Episode 切分 train/valid/test
→ PyTorch + PEFT/TRL LoRA（硬件支持时可用 QLoRA）
→ 真实回复盲测与行为评测
```

第一版训练样本：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是目标人物。当前是工作日晚间；关系阶段为熟悉亲密；表达意图是关心但不过度追问。使用多气泡协议。"
    },
    {"role": "user", "content": "今天真的好累"},
    {
      "role": "assistant",
      "content": "<bubble delay_ms=0>怎么啦</bubble><bubble delay_ms=900>今天事情很多吗</bubble>"
    }
  ]
}
```

训练目标必须是目标人物真实发送的内容。时间、关系阶段和表达意图可以由离线分析产生，但不能让合成模型替换真实回复。

WSL2 Linux 训练使用 Transformers/PEFT 的适配器机制，并由 TRL `SFTTrainer` 处理对话数据。训练时只对目标人物的 assistant completion 计算损失，将 system 和 user prompt token 屏蔽为 `-100`，或使用聊天模板支持时的 `assistant_only_loss`。这样避免让模型把 Runtime 说明文本也当成需要复述的目标。

## 21. LoRA 与 Runtime 的两阶段协调

第一版不让 Persona LoRA 同时承担世界推理、日程规划、状态更新、JSON 协议和自然表达全部职责。

采用两阶段：

### 21.1 Director 阶段

使用通用指令模型读取完整 RuntimeContext，输出结构化 LifeDecision：

- 状态怎样变化；
- 是否说话；
- 是回复还是主动；
- 想表达什么；
- 何时再次醒来。

Director 可以与 PersonaActor 使用同一个基础模型，但不加载人格 LoRA，或使用独立的通用小模型。

### 21.2 PersonaActor 阶段

仅在 `action = speak` 时调用目标人物 LoRA：

```yaml
PersonaActorRequest:
  identity_digest: object
  current_situation: object
  communication_intent: string
  content_points: [string]
  recent_dialogue: [object]
  relevant_memories: [object]
  speech_mode: reply | proactive | delayed_reply
```

输出：

```yaml
PersonaActorOutput:
  bubbles:
    - type: text | emoji | sticker
      content_or_asset: string
      delay_ms: integer
```

### 21.3 为什么先分两阶段

- Runtime 结构错误不会污染人物表达训练；
- LoRA 可以集中容量学习“像她”；
- Director 可以升级而无需重训人物 LoRA；
- 人物 LoRA 更新不会改变时钟和调度规则；
- 可以分别评测“决定是否合理”和“说得是否像本人”。

当系统积累足够多且通过离线行为评测的 Runtime Cycle 后，可以把 Director 决策蒸馏进 Persona LoRA，尝试单次融合推理。但这应是性能优化，不是第一版前提。

## 22. 三类模型的版本协调

每次 Runtime Cycle 固定：

```yaml
RuntimeVersions:
  world_graph_version: string
  origin_snapshot_version: string
  branch_overlay_version: string
  person_world_profile_version: string
  routine_profile_version: string
  director_model_version: string
  persona_base_model: string
  persona_adapter_version: string
  runtime_prompt_version: string
  actor_prompt_version: string
```

这样才能解释：

- 是图谱背景错了；
- 是生活状态不合理；
- 是 Director 决策不合理；
- 还是 LoRA 表达不像本人。

## 23. 知识、生活与表达的完整数据流

### 23.1 项目准备阶段

```text
聊天导入
→ LightRAG 索引（原始上下文、对象和开放关系）
→ PersonWorldProfile（数字人主观世界）
→ RoutineProfile
→ 节点自动分析与排序
→ LoRA PersonaActor 训练与验收
```

### 23.2 分支创建阶段

```text
选择节点
→ 确定 origin_time
→ 构建 Origin World Snapshot
→ 初始化 VirtualClock
→ 根据人物背景和节点状态生成首日 DayPlan
→ 初始化 LifeState
→ 创建空的 Branch World Overlay
```

### 23.3 运行阶段

```text
用户消息或 Wakeup
→ 查询起点世界 + 分支世界
→ Director 推进生活并决定行动
→ PersonaActor LoRA 生成外显表达
→ Executor 投递、更新状态和安排下次唤醒
→ 有意义的分支事件写入 Branch Overlay
```

## 24. 阈值与证据的处理原则

### 24.1 图谱内容不因低分消失

LightRAG 抽取出的候选背景可以带来源和候选状态，但第一版不连续经过多个分数阈值。BackgroundCompiler 输出候选列表，用户或后续上下文可以纠正。

### 24.2 检索按预算取内容

检索使用排序与 Token 预算，不采用“低于绝对分数全部清空”。如果没有高相关历史，Runtime 仍可依靠人物背景、LifeState、Routine 和当前分支继续生活。

### 24.3 模拟生活不要求现实证据

LifeState 属于分支现实。它必须符合人物背景和分支连续性，但不需要证明真实人物此刻真的在做这件事。

### 24.4 行动使用离散决定

Director 直接选择 `speak/wait/continue_life/schedule`，不串联 readiness、motivation、evidence 和 confidence 阈值。

### 24.5 硬边界只保护不可违反的范围

- 项目、人物和分支隔离；
- 历史节点未来信息隔离；
- 用户权限与安静时段；
- 幂等和重复投递；
- 已删除或不可见数据；
- 分支生成内容不能回写真实历史训练集。

## 25. 最小数据对象

在现有模型基础上增加或收敛：

```text
WorldGraphVersion
PersonWorldProfile
RoutineProfile
OriginWorldSnapshot
VirtualClock
DayPlan
LifeState
LifeEvent
RuntimeCycle
Wakeup
BranchWorldOverlayVersion
```

第一版允许 Profile、Plan 和 State 使用版本化 JSON。只有查询、并发或迁移问题真实出现后再拆成大量关系表。

## 26. 与当前工作树的对应关系

### 26.1 可以复用的部分

- 导入、消息和 Episode 基础；
- 节点分析流水线；
- `IdentityKernel` 与 `behavioral-rhythm-v1`；
- 分支记忆与连续性索引；
- `MentalStateVersion`；
- `PerceptionEvent`、`CognitiveCycle`、`AgentWakeup`；
- 常驻人格推理调度器；
- 多气泡、贴纸、表情和延迟协议；
- 当前 LoRA 数据集中的 responsive/proactive 标记；
- 时间切分、风格评测和人格偏好训练。

### 26.2 当前缺少的核心部分

- 项目依赖中尚未接入 LightRAG；
- 没有导入后先构建人物世界的完整流程；
- 没有 `PersonWorldProfile` 编译与展示页面；
- 节点选择尚未真正建立在完整人物背景之上；
- 没有明确的 VirtualClock 与 DayPlan；
- 当前 SituationalState 更像真实观测状态，不等同于分支模拟生活状态；
- 当前主动决策包含样本量、主动率和 readiness 等多级固定阈值；
- 当前融合认知协议让人格模型承担了较多结构化 Runtime 职责；
- 当前 `mlx_adapter.py` 和 `mlx_generation.py` 耦合 Apple MLX，不是 WSL2 Linux 下的可用训练/推理后端；
- 没有 Origin World Snapshot 与 Branch World Overlay 的 LightRAG 隔离方案。

### 26.3 需要调整而不是删除的部分

1. 保留当前 Agent 持久化与调度基础；
2. 将 `situational-state-v2` 保留给真实外部观测；
3. 新增独立 `LifeState` 表示分支模拟生活；
4. 将主动评分改为 Director 离散决策，固定分数先降为观测指标；
5. 将融合推理拆成 Director 与 PersonaActor 两阶段；
6. 接入 LightRAG Sidecar，先完成完整人物世界和背景页；
7. 让节点分析消费背景档案与图谱上下文；
8. 保留现有训练和生成业务接口，在后面新增 `PeftTrainingEngine` 与 `PeftGenerationEngine`，逐步替换 MLX 具体实现。

## 27. 第一版实施顺序

### Phase 1：LightRAG 人物世界

- 部署 LightRAG Sidecar；
- 定义聊天 Episode 文档格式；
- 定义人物世界实体提示；
- 完成项目级完整图索引；
- 实现 BackgroundCompiler；
- 增加人物背景展示页，不设用户审批步骤。

验收：导入完成后，在选择节点前能看到身份、人物、地点、关系、工作、活动、规律和生活阶段。

### Phase 2：背景增强节点分析

- 候选窗口查询相关图谱；
- 将生活阶段、人物关系和地点背景加入判断；
- 自动排序并发布最终节点；
- 为发布节点生成正确的真实事件时间。

验收：节点解释能够说明“为什么这件事对这个人重要”，而不只是报告情绪变化。

### Phase 3：PersonaActor LoRA

- 固定 Actor 输入输出契约；
- 重构训练数据为 Turn、多气泡和主动/响应样本；
- 加入粗粒度时间、关系阶段和表达意图；
- 使用 WSL2 Linux 中的 PyTorch + PEFT/TRL 训练 LoRA，硬件支持时使用 QLoRA；
- 分别评测风格、气泡、主动开场和典型反应。

验收：在给定相同表达意图时，LoRA 版本比基础模型更像目标人物。

### Phase 4：持续生活内核

- 实现 VirtualClock；
- 实现 LifeState、DayPlan 和 LifeEvent；
- 实现计划块 Wakeup；
- 实现服务重启快进；
- 实现用户消息打断。

验收：用户离开一段时间后回来，数字人的虚拟时间、活动和状态已经合理前进，并且重启后结果连续。

### Phase 5：Director 与主动闭环

- 实现 LifeDecision；
- 拆分 Director 与 PersonaActor；
- 支持回复、主动、等待、延迟回复和继续生活；
- 将现有主动分数改为观测，不作为默认硬门；
- 支持下一次 Wakeup。

验收：数字人能够在没有用户新消息时继续生活并偶尔自然主动，同时不会因为多级分数不足而永久沉默。

### Phase 6：时间安全的分支知识

- 为选定节点构建 Origin World Snapshot；
- 实现 Branch World Overlay；
- 运行时组合查询两者；
- 禁止 Full Historical World 进入历史分支。

验收：LightRAG 的实体摘要、关系和文本块均不能泄漏起点之后的信息。

## 28. 第一版验收场景

### 28.1 背景与节点

导入聊天后，系统识别：

- 目标人物的称呼和别名；
- 工作单位或工作角色；
- 家、公司和常去地点；
- 主要人物关系；
- 工作日与周末的粗粒度规律；
- 生活阶段变化。

节点分析能利用这些背景解释一次搬家、换工作、关系升温或疏远。

### 28.2 持续生活

用户在虚拟晚上离开，第二天回来：

- VirtualClock 已前进；
- 昨晚计划和睡眠状态已结束；
- 当前进入新一天的活动；
- 未完成话题仍然存在；
- 数字人可能已经主动发过一条符合本人节奏的消息，也可能选择没有打扰。

### 28.3 用户消息打断

数字人当前处于工作状态，用户发消息：

- Director 读取忙碌状态；
- 可以先不立即生成长回复，而安排延迟回复；
- 到期后 PersonaActor 用目标人物风格继续；
- 中途如果用户又发新消息，旧草稿被取消或重新规划。

### 28.4 LoRA 协调

Director 输出“关心对方，但不要像客服式追问；使用简短主动语气”。PersonaActor LoRA 将其表达为目标人物真实风格，而不是复述 Director 的结构化说明。

### 28.5 时间分支安全

从历史节点创建分支后：

- Runtime 只能查询 Origin World Snapshot；
- 真实未来聊天不进入实体摘要；
- 分支新生活只写入自己的 Overlay；
- 同一节点的另一条分支看不到本分支经历。

## 29. 什么时候再做加法

| 真实出现的问题 | 再增加的能力 |
| --- | --- |
| LightRAG 人物或地点合并错误 | 自动冲突覆盖层与更强实体消歧 |
| 图谱抽取在聊天口语上遗漏严重 | 针对聊天的二次抽取或领域提示优化 |
| Origin Snapshot 构建太慢或占空间 | 增加时态图版本与增量快照 |
| DayPlan 太僵硬 | 加入局部重规划与更多生活阶段特征 |
| LifeState 经常忘记过期 | 只为出问题的字段增加 TTL |
| Director 决策不像本人 | 用通过离线行为评测的 Cycle 训练或蒸馏行为策略 |
| 两阶段延迟太高 | 将 Director 与 Actor 蒸馏为一次融合推理 |
| 主动消息过多或重复 | 针对实际失败增加冷却或 Policy 规则 |
| LoRA 只学到口头禅、没有反应方式 | 增加情境平衡、偏好训练和困难负样本 |
| 分支生活积累后检索变差 | 将 Branch Overlay 增量接入 LightRAG |

不提前为这些问题一次性实现最终系统。

## 30. 明确不做的事情

- 不删除知识图谱、生活规律、当前状态或主动能力；
- 不在人物背景尚未建立时直接选择重要节点；
- 不把 LightRAG 当成普通向量库；
- 不让完整未来图进入历史节点分支；
- 不让 LoRA 同时承担知识库、时钟、调度器和人格表达；
- 不把全部原始聊天不经 Turn 建模直接丢给 LoRA；
- 不每分钟调用大模型证明数字人仍然活着；
- 不把“持续生活”等同于持续给用户发消息；
- 不让串联阈值决定数字人是否有资格生活和表达；
- 不把分支模拟生活回写成真实人物历史。

## 31. 最终闭环

MoonlightBox 的正确闭环是：

```text
真实聊天
→ LightRAG 重建人物及其历史世界
→ 人物背景帮助发现真正重要的节点
→ 用户从节点创建一条新的时间分支
→ VirtualClock、DayPlan 和 LifeState 让数字人持续生活
→ Director 决定下一步生活与表达意图
→ Persona LoRA 用目标人物的方式说出来
→ Executor 让消息、等待、状态变化和 Wakeup 真正发生
→ 分支经历进入自己的世界 Overlay
→ 下一轮从连续的生活继续
```

LightRAG 解决“她从哪里来、认识谁、经历过什么”；Runtime 解决“她现在怎样生活、下一刻做什么”；LoRA 解决“这件事由她说出来会是什么样子”。

第一版三者都必须存在。做减法的地方，是每个模块内部先采用最简单可靠的实现，而不是删掉任何构成真人感的核心能力。

## 32. 技术参考

- [LightRAG 官方仓库与架构说明](https://github.com/HKUDS/LightRAG)
- [LightRAG Core 编程与 QueryParam 文档](https://github.com/HKUDS/LightRAG/blob/main/docs/ProgramingWithCore.md)
- [LightRAG Server 与 REST API 文档](https://github.com/HKUDS/LightRAG/blob/main/docs/LightRAG-API-Server.md)
- [Microsoft WSL 中使用 systemd](https://learn.microsoft.com/windows/wsl/systemd)
- [Microsoft WSL 文件系统与性能建议](https://learn.microsoft.com/windows/wsl/filesystems)
- [Microsoft WSL2 NVIDIA CUDA 支持](https://learn.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl)
- [Transformers 与 PEFT 适配器文档](https://huggingface.co/docs/transformers/peft)
- [TRL SFTTrainer 与 assistant-only loss](https://huggingface.co/docs/trl/sft_trainer)

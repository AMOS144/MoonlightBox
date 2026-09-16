# PersonWorldProfile v3：由 Agent 综合理解的人物世界与心理画像

> **设计状态：待实现的统一规格，不代表代码已经完成。**
>
> **2026-09-11 更新：** 第 3.3 节定义情境模块的实例和详细字段；第 5 节定义跨 Agent 读取、
> 固定版本与环境/人格区分；第 6.3 节定义模块纠正后的关联更新。它们均属于原有七栏结构。
>
> 本次修订：心理画像由 Agent 综合理解聊天后直接生成。允许使用语气、文风、表情、玩笑、回应方式、
> 零散生活信息及整体互动印象；不要求明确自述、独立场景数、跨日期次数、逐项引文或反证才能输出。
> 删除 Signal 编码、证据权重、Reducer 算人格、模板拼接正文等旧方案。
>
> 本文是 Profile 结构、生成流程、Prompt、审核界面和 Runtime 人格上下文的设计依据。
> 原 `2026-09-09-person-world-agent-and-graph-governance-design.md` 中候选图版本、Revision 多步确认、
> Executor 和发布规则继续有效；其中与本文冲突的 Profile 生成和心理推断要求以本文为准。

## 1. 核心目标与判断责任

产物应是一份能帮助用户认识目标人物、能帮助 PersonaActor 扮演目标人物的中文画像。
Agent 需要结合完整语境主动理解、归纳和推断，不应将任务退化为摘抄聊天、填写证据表或等待自我介绍。

心理特征通常没有一句话可以直接证明。允许 Agent 根据交谈的整体感觉推断外向程度、组织性、
相处方式、在意的事情和可能的动机；用成熟心理模型整理这些理解，用具体中文说明它们如何表现。

这里的“推断”仍然来自实际读到的材料。模型熟悉心理学概念不等于了解这个具体的人。
没读到材料、工具失败、人物身份弄错时不能用通用人格模板补齐。心理画像标为“AI 推断”，用户可以修改。
这项来源说明集中展示，不要求每段都重复“证据不足、不能确定、仅供参考”。

事实和心理理解采用不同要求：

| 内容 | Agent 应怎样处理 |
| --- | --- |
| 姓名、雇主、学校、具体工作时间等现实信息 | 根据聊天、上下文和用户补充填写；不知道具体名称或时间就保留未知，不从人格推导 |
| 性格、互动风格、价值取向、处事方式 | 综合所有可用表达和互动主动推断，不以明确自述或证据数量作为前提 |
| 对话中的事件 | 根据对理解此人生活的作用归纳；单次重要事件、承诺或角色介绍可以收录，不机械排除 |
| 心理状态与稳定特征的关系 | Agent 判断是最近状态、特定关系中的表现还是较一般的倾向，并用自然语言表达 |
| 临床诊断 | 本项目不根据聊天生成疾病诊断；人格模型维度中的术语不等于临床结论 |

## 2. 阅读材料与输出的关系

生成流程统一为：

```text
读取栏目目的、模型维度、参与者信息与用户纠正
→ LightRAG 获取整体信息和相关对话
→ 按需要阅读原始消息、上下文及不同话题
→ Agent 综合理解并直接生成栏目正文和结构化维度
→ 后端校验结构、保存、装配
→ 用户审核画像并对不准确处进行对话式修改
```

LightRAG 的摘要和图谱关系可以作为综合理解的输入。出现具体身份、时间或人物指代歧义时，
Agent 回到原始对话核对；不要求每一项心理理解都反查成一条可证明的原话。

原始消息仍保留，Phoenix 记录实际读取内容和模型调用。Profile 可附代表性参考片段，
但这些片段用于解释“为什么有这种印象”，不作为人格测量证明。
不要求先创建 `SituatedScene` 或 `PsychologicalSignal` 才能填写 Profile。

## 3. PersonWorldProfile 完整结构与各栏目填写要求

Profile 只有下面七个顶层业务栏目。人格、动机、人际风格和叙事身份都是对应栏目的子项，
与生活信息一同填写、保存、展示和修订。本文所有字段定义集中在本章；不另建 world/psychology 两套树。

### 3.1 完整字段树

```yaml
PersonWorldProfileV3:
  schema_version: v3
  subject_participant_id: UUID             # 后端绑定
  overview: 人物整体介绍
  identity:                              # 自我、身份与性格；identity Agent
    summary: 中文综述
    names_and_self_reference: DimensionEntry
    claimed_roles: DimensionEntry
    self_evaluations: DimensionEntry
    identity_tensions: DimensionEntry
    big_five_bfi2: ModelProfile             # 五域、十五 facets
    mbti: MBTIModelProfile                  # 四轴、候选类型
  life_context:                          # 现实生活；life_context Agent
    summary: 中文综述
    primary_engagements: DimensionEntry
    institutional_attachment: DimensionEntry
    living_and_care: DimensionEntry
    places_and_mobility: DimensionEntry
    resources_and_constraints: DimensionEntry
    active_projects: DimensionEntry
    context_modules: [ContextModule]
  social_world:                          # 第三方关系；social_world Agent
    summary: 中文综述
    household_and_family: DimensionEntry
    peer_and_collaborative_world: DimensionEntry
    care_and_support: DimensionEntry
    responsibility_and_power: DimensionEntry
    group_and_institutional_ties: DimensionEntry
    interpersonal_styles: [RelationStyle]   # 按第三人或关系群体保存 IPC
  agency:                                # 关切、选择与动机；agency Agent
    summary: 中文综述
    objects_of_concern: DimensionEntry
    decision_criteria: DimensionEntry
    goals_and_commitments: DimensionEntry
    avoidance_and_boundaries: DimensionEntry
    motivation: ModelProfile               # SDT 自主、胜任、联结
  practices:                             # 日常与做事方式；practices Agent
    summary: 中文综述
    ongoing_practices: DimensionEntry
    project_organization: DimensionEntry
    time_rhythms: DimensionEntry
    place_based_practices: DimensionEntry
    interruptions_and_exceptions: DimensionEntry
  life_course:                           # 经历、变化与自我叙事；life_course Agent
    summary: 中文综述
    past_anchors: DimensionEntry
    current_phase: DimensionEntry
    active_transitions: DimensionEntry
    future_horizons: DimensionEntry
    unfinished_matters: DimensionEntry
    self_authored_meaning: DimensionEntry   # Narrative Identity 贯穿本栏，不再复制一份时间线
  relationship_with_user:                 # 与用户相处；relationship_with_user Agent
    summary: 中文综述
    shared_referents: DimensionEntry
    interaction_language: DimensionEntry
    coordination_and_care: DimensionEntry
    boundaries_and_commitments: DimensionEntry
    shared_projects: DimensionEntry
    relationship_change: DimensionEntry
    interpersonal_style: ModelProfile      # 仅这对参与者之间的 IPC

DimensionEntry:
  id: UUID                               # 后端生成
  dimension_id: 本栏目定义的子项
  status: described | unknown | not_applicable
  basis: stated | inferred | user_corrected
  value: 该子项允许的结构化值或 null
  description: Agent 撰写的中文解释
  reference_message_ids: []                # 可选；填写时须是真实且已读的 ID
  context_module_refs: [ContextModuleRef]   # 可选；详见 3.3.5
```

`summary` 与 `description` 由 Agent 写作，后端不从代码标签或模板拼接。
结构约束限定“要回答哪些问题”，不限定“必须通过哪些措辞和证据得出答案”。

每个固定子项都返回状态。已读材料可以支持合理印象时应输出 `described + inferred`；
只有完全缺乏相关语境、无法形成有意义的判断时才用 `unknown`。
`not_applicable` 表示已知不适用，不能代替未知。

`PersonWorldInvestigationReport` 在 Profile 外保存运行状态、资料范围、失败和未完成子项。
工具失败必须显示为运行失败，不得让模型编写“聊天中没有资料”来掩盖。

### 3.2 七栏编辑要求（包括心理学子项）

| 栏目 | 必填子项 | 正文需要交代什么 |
| --- | --- | --- |
| identity 自我、身份与性格 | names_and_self_reference、claimed_roles、self_evaluations、identity_tensions、big_five_bfi2、mbti | 常用称呼、自我定位、性格特点、五大人格与 MBTI 倾向；综述将这些内容组织成对“她是什么样的人”的介绍 |
| life_context 现实生活 | primary_engagements、institutional_attachment、living_and_care、places_and_mobility、resources_and_constraints、active_projects、context_modules | 目前主要在做什么、组织和生活环境、工作学习或照料负担、重要安排、方便与困难之处 |
| social_world 社会关系 | household_and_family、peer_and_collaborative_world、care_and_support、responsibility_and_power、group_and_institutional_ties、interpersonal_styles（IPC） | 家人、朋友、同事等在生活里扮演什么角色；相处、依赖、协作或摩擦是什么样，并用 IPC 描述主张程度与亲和程度 |
| agency 关切与选择 | objects_of_concern、decision_criteria、goals_and_commitments、avoidance_and_boundaries、motivation（SDT） | 喜欢或在意什么、如何取舍、有什么目标和承诺、哪些事情不愿接受；结合 SDT 理解什么使她投入、抗拒或感到被支持 |
| practices 日常与做事方式 | ongoing_practices、project_organization、time_rhythms、place_based_practices、interruptions_and_exceptions | 如何过一天、安排事情、推进任务、应对变动；模糊时间可写“大致傍晚”，不编造精确作息 |
| life_course 经历与变化 | past_anchors、current_phase、active_transitions、future_horizons、unfinished_matters、self_authored_meaning（本栏整体采用 Narrative Identity） | 重要经历、眼下阶段、正在改变什么、期待什么、仍惦记或未完成什么；没有完整人生史也能描述当前阶段 |
| relationship_with_user 与用户相处 | shared_referents、interaction_language、coordination_and_care、boundaries_and_commitments、shared_projects、relationship_change、interpersonal_style（IPC） | 两人常聊什么、怎样玩笑和表达关心、谁更主动、怎样协商或闹别扭、有什么共同安排，以及这段关系里的主张与亲和方式 |

每个子项只有一个主归属。例如工作安排写在 life_context，性格上的组织倾向写在 identity.big_five_bfi2，
工作任务如何推进写在 practices；三者可利用相同聊天，但各自回答不同问题。
自我叙事统一在 life_course 表达，identity 可在综述中引用，不再保存 self_narratives 的重复正文。

### 3.3 life_context.context_modules：可填写的生活情境子表

情境模块保存在 `life_context.context_modules`，由 life_context Agent 统一编写。
每个实例对应一份具体工作、一段学习、一项照料责任或一次迁移。一个人可以同时有工作、学习、照料，
也可以有两份工作；`kind` 是结构类型，不是人物身份的唯一分类。

模块不是一句“她在工作”的标签，而是有固定字段的详细子表。Agent 依据整体理解选择模块、填入内容，
允许推断，不要求 activation scene、自述或引文数量。未知字段保留未知，已有内容可以先生成。

#### 3.3.1 模块实例与字段结构

```yaml
ContextModule:
  id: UUID                            # 后端生成；修改时沿用，不以名称作身份
  revision: integer                   # 后端维护，每次接受变更后递增
  schema_version: employment_v1        # 后端按 kind 绑定
  kind: employment | education | job_search | entrepreneurship | care_work |
        retirement | rest | migration | household_transition | other
  title: 主职工作                       # 用户可读标题，不必知道雇主名称
  status: current | planned | past | paused | unknown
  basis: stated | inferred | user_corrected
  period:
    start_at: null                    # 只保存已知日期
    end_at: null
    description: 最近这段时间
  summary: 这项生活情境的中文概括
  related_module_ids: []               # 比如求职与主职并存，或学习服务于转行
  details: EmploymentDetails           # 按 kind 选择专属 Pydantic 类型

ModuleField:
  status: described | unknown | not_applicable
  basis: stated | inferred | user_corrected
  value: 类型规定的值或 null
  description: 用中文说明内容或不确定之处
  reference_message_ids: []            # 可选；没有引用不阻止保存
```

模块的 `basis` 概括整体来源，各字段还保留自己的 `basis`。
例如“公司名由用户补充，工作氛围由 Agent 推断”，不能用模块级 user_corrected 把全部字段标成用户确认。
生命周期 status 与来源 basis 分开：`current + inferred` 是推断的当前工作，
`planned + stated` 是本人提到的未来工作。

新实例在接受 Agent 输出时分配 ID；模型只能使用工具返回的已有 ID，不能自行编造 UUID。
一份工作的名称补全或职位变化通常保留模块 ID；另一份独立工作新建实例。
是否同一份工作由 Agent 阅读和理解后提出，后端不根据公司名做模糊合并。

#### 3.3.2 工作模块：完整示例

以下是结构示意，不是当前目标人物的数据。所有叶子均为 ModuleField：

```yaml
EmploymentDetails:
  organization:
    name: ModuleField                  # 公司/单位名称；可未知
    sector: ModuleField                # 国企、民企、外企、事业单位等；按材料填写
    industry: ModuleField
  role:
    title: ModuleField
    work_content: ModuleField
    responsibilities: ModuleField
  working_arrangement:
    employment_form: ModuleField       # 全职、兼职、项目制等
    workplace: ModuleField
    schedule: ModuleField              # 单位安排，保留粗略表述及条件
    flexibility: ModuleField
  experience:
    workload: ModuleField
    autonomy: ModuleField
    satisfaction_and_frustrations: ModuleField
  outlook:
    current_priorities: ModuleField
    intended_changes: ModuleField
```

填写规则：

- organization / role：描述在哪里工作、做什么、承担什么；不知道名称或职称，不影响描述工作内容。
- working_arrangement：描述工作制度和安排，区分“单位要求几点到”和“她实际什么时候到”。
- experience：Agent 综合理解她如何体验这份工作，可写压力、受限感、满意或不满，不要求她明确概括。
- outlook：描述眼下的重点与改变意向；正式目标、取舍动机由 agency 综合，工作模块保留工作背景。

例如 schedule 的值可以是：

```yaml
status: described
basis: inferred
value: 白天到岗，傍晚结束；具体时间还不清楚
description: 对话给人的印象是以白天到岗为主，尚不能填出固定上下班时刻。
reference_message_ids: []
```

日程字段使用可读文本和可选已知时间，不要求 Agent 为每个模块伪造精确日历。
国企性质描述组织环境，不能自动推导稳定收入、保守性格、强服从性或高尽责性。

#### 3.3.3 其他模块的固定详细字段

下表每个分组内的字段都是 ModuleField；每种类型定义独立 details schema。
必填意味着必须给出字段状态，并不意味着必须给出非空值。

| kind / 类型 | details 的固定分组与字段 | 编辑目的 |
| --- | --- | --- |
| education / EducationDetails | institution：name、education_form；study：stage、field、courses_and_tasks、exams_and_deadlines；arrangement：location、schedule、learning_style；experience：interests、pressure_and_difficulties、peer_environment；outlook：next_steps | 描述学习阶段、具体任务、组织方法和体验，可涵盖在校、培训、自学 |
| job_search / JobSearchDetails | current_state：employment_background、search_stage；direction：target_roles、sector_preferences、location_preferences；progress：applications、interviews、offers_and_decisions；conditions：constraints、concerns；outlook：next_actions、alternatives | 区分寻找方向、申请中、等待结果与已经入职 |
| entrepreneurship / EntrepreneurshipDetails | venture：name、business_or_product、stage；participation：own_role、team_and_partners；operation：daily_work、time_commitment；conditions：resources、difficulties；outlook：near_term_goals | 描述创业在实际做什么，避免仅用“创业者”标签 |
| care_work / CareWorkDetails | recipient：relationship、care_needs；responsibilities：tasks、time_commitment；coordination：division_of_work、support_network；experience：life_impact、feelings；outlook：expected_changes | 明确照料的是谁、如何分工、如何影响日常 |
| retirement / RetirementDetails | transition：previous_engagement、retirement_arrangement；daily_life：main_activities、time_structure、social_connections；conditions：resources_and_constraints；outlook：plans_and_interests | 描述退休后的生活组织，不默认失去目标或社会联系 |
| rest / RestDetails | background：reason_or_context、voluntary_or_constrained；daily_life：activities、time_structure、social_connections；experience：recovery_and_feelings；outlook：return_or_next_steps | 覆盖间隔期、休整或暂时停工，不从休整推断失业或疾病 |
| migration / MigrationDetails | move：from_place、to_place、reason_or_context；arrangement：housing、logistics、timing；progress：completed_and_pending；experience：adaptation、life_impact | 描述地域移动和适应，不把临时出行默认写成迁居 |
| household_transition / HouseholdTransitionDetails | change：before、after、participants；arrangement：housing、responsibility_changes；progress：completed_and_pending；experience：feelings、unresolved_matters | 描述同住、分居、家庭成员或家庭安排变化，不重复整份迁移记录 |
| other / OtherContextDetails | topic：name、scope；situation：current_activity、people_and_environment；conditions：resources_and_constraints；experience：feelings；outlook：next_steps | 作为未覆盖情境的通用子表，字段固定；不允许模型临时发明 Schema |

这些是调查和写作的范围，不要求每个叶子各调用一次工具。
需要更具体的项目、申请、合作方清单时，在该字段中用可读概括保存；不在本版额外建立招聘或项目管理系统。

#### 3.3.4 公共字段与其他栏目：正文只保存一次

| 信息 | 唯一详细归属 | 其他字段怎样使用 |
| --- | --- | --- |
| 公司、岗位、工作制度、工作体验 | employment 模块 | life_context.institutional_attachment / primary_engagements 概括并引用 |
| 学校、学习任务与安排 | education 模块 | 公共字段概括当前学习投入，practices 描述实际学习习惯 |
| 居住或照料的具体情境 | 相应模块；没有适用模块时在 living_and_care | 公共字段介绍全局生活情况，不复制所有模块字段 |
| 某份工作的近期项目 | employment.outlook.current_priorities | active_projects 只概括全局重点；跨多模块的独立事务由 active_projects 保存详情 |
| 单位规定的时间 | employment.working_arrangement.schedule | practices.time_rhythms 结合聊天判断真实作息及例外 |
| 对工作的满意、不满与自主空间 | employment.experience | agency.motivation 解释动机，identity 判断是否体现更一般的性格 |
| 入职、离职等变化 | 模块保留 period / status 与当前情境 | life_course 描述变化过程及其意义，不另复制工作档案 |

公共字段可以保存一句综合判断和模块引用。它们不是对模块正文做机械拼接，也不是第二份可独立修改的详情。
无对应模块时允许公共字段先描述已知情况；后续模块形成后，Agent 在完成 life_context 时统一整理归属。

#### 3.3.5 字段引用与模块读版本

为公共字段和跨栏派生判断增加可选引用：

```yaml
ContextModuleRef:
  module_id: UUID
  module_revision: integer
  field_paths: [details.working_arrangement.schedule]
  usage: summary | interpretation_context

# DimensionEntry、ModelDimensionEntry 可包含：
context_module_refs: [ContextModuleRef]
```

这些引用只追踪“用了哪份背景”，不是新证据门槛，不要求引用数量，也不决定心理判断是否有效。
Agent 使用模块时选择所依赖的字段，后端核对 ID、版本及字段是否确实已读。
未细化到字段时允许引用整个模块；如果调用过模块读取工具却未声明具体用途，
运行时至少记录栏目级读取依赖，以免后续模块变化被遗漏。

life_context Agent 编辑模块；其他 Agent 只读。它们发现背景可疑时可以提出修正建议或回查聊天，
不能直接改别栏内容，也不能把疑点静默覆盖成自己的版本。

### 3.4 栏目内模型子项的详细定义

七栏采用五类心理学模型帮助填写相关子项，模型类型不构成独立栏目。聊天生成的结果不按正式问卷计分，也不输出伪精确分数、
测量概率或通过证据计数算出的人格结论。

| 唯一字段位置与模型 | 子项结构 | Agent 应写出的内容 |
| --- | --- | --- |
| identity.big_five_bfi2 · Big Five / BFI-2 | 五域、十五 facets | 整体性格倾向：社交主动性、体谅与信任、组织与责任、情绪反应、好奇与想象等 |
| identity.mbti · MBTI | EI、SN、TF、JP 与候选类型 | 根据整段交流推断较接近的偏好组合，说明最明显的轴和难判断的轴 |
| social_world.interpersonal_styles / relationship_with_user.interpersonal_style · IPC | agency、communion | 在不同关系里更主动还是跟随、更亲近还是保持距离，面对分歧怎样互动 |
| agency.motivation · SDT | autonomy、competence、relatedness | 怎样的安排让她更有动力，什么使她抗拒或受挫，怎样的支持可能有效 |
| life_course 全栏 · Narrative Identity | 经历、当前阶段、未来、未竟事项、自我赋义 | 她如何看待过去、眼下和未来，什么故事或目标参与塑造自我理解 |

BFI-2 facets：

```yaml
extraversion: [sociability, assertiveness, energy_level]
agreeableness: [compassion, respectfulness, trust]
conscientiousness: [organization, productiveness, responsibility]
negative_emotionality: [anxiety, depression, emotional_volatility]
open_mindedness: [intellectual_curiosity, aesthetic_sensitivity, creative_imagination]
```

负性情绪性中的 anxiety/depression 是模型维度名称，不能显示成焦虑症、抑郁症诊断。
Narrative Identity 使用 life_course 中已有的 past_anchors、current_phase、active_transitions、future_horizons、
unfinished_matters、self_authored_meaning 组织输出；这些是项目编辑字段，不宣称是标准心理量表。
可以只描述当前篇章，不要求同时具备过去—现在—未来的完整证据链。

模型背景参考：[Big Five](https://pubmed.ncbi.nlm.nih.gov/1635039/)、
[BFI-2](https://www.sciencedirect.com/science/article/pii/S0092656616301325)、
[IPC](https://pubmed.ncbi.nlm.nih.gov/16430329/)、
[SDT](https://selfdeterminationtheory.org/about-the-theory/)、
[MBTI 综述](https://onlinelibrary.wiley.com/doi/10.1002/jcad.70006)。
本文对聊天推断和角色扮演的使用方式是产品设计，不等同于这些工具的正式测量程序。

#### 3.4.1 栏目内心理推断可使用的材料

语气、表情、文风、玩笑、用词、话题选择、回应速度、互动频率、推荐、抱怨、拒绝、主动关心、
临时改变主意等，都可以参与整体理解。一次有特点的互动也可以触发推断或帮助形成倾向判断。

Agent 需要结合对话对象、情境、聊天风格和前后文衡量意义。例如回复慢可能与忙碌有关，
玩笑可能表达亲近也可能带着不满。这些替代解释交给 Agent 综合判断，
不转化为“必须找到第二次、第三次才能输出”的规则。

明确自述是可用材料之一，不是必要条件。没有代表性引文也可以输出画像：
“整体上，她表达直接、带玩笑感，比较愿意主动拉近距离。”
合理推断可直接写成自然中文，不必为每项附“不能证明她真正是什么人”的长尾说明。

完全未读到目标人物材料时仍应停止心理推断。允许 Agent 推断不等于允许用常识虚构具体人生事件。

#### 3.4.2 模型子项的共用值类型

```yaml
ModelProfile:                           # 可复用值类型，不是 Profile 顶层栏目
  summary: 本模型子项的中文综述
  dimensions: [ModelDimensionEntry]

MBTIModelProfile:                       # 继承 ModelProfile，仅 identity.mbti 使用
  candidates: []

RelationStyle:
  relationship_scope: 某个第三人或关系群体
  profile: ModelProfile                 # IPC

ModelDimensionEntry:
  id: UUID                              # 后端生成
  model: big_five_bfi2 | mbti | interpersonal_circumplex | sdt
  dimension_id: 模型内维度
  status: described | unknown
  basis: inferred | self_reported | user_corrected
  value: 由模型定义的值域或 null
  context: 整体倾向或具体关系、任务的中文说明
  description: 用中文说明该特征如何表现
  reasoning_summary: 简短说明整体印象来自什么；不要求引文或逐条证明
  roleplay_guidance: 当前语境下角色可以如何表现
  uncertainties: []                     # 只有影响使用的重要疑点才写
  reference_message_ids: []             # 可选，不设置数量下限
  context_module_refs: [ContextModuleRef] # 使用的生活模块及版本，不是心理证据计数
```

上述共用类型嵌入第 3.1 节的固定字段；不是另外一套人物档案。BFI-2 的 domains/facets 和 MBTI 四轴
必须作为具名子项完整返回，不能省略不熟悉的维度；未知用 unknown 表示。Narrative Identity 直接使用
life_course 的 DimensionEntry，不额外存 ModelProfile。

各模型采用不同值域：Big Five、IPC 用 low/moderate/high/mixed/unknown；
MBTI 用四轴各自的字母、mixed 或 unknown；SDT 用 supported/frustrated/mixed/unknown；
叙事身份用具体中文内容，不压缩成 present/absent。

MBTI 允许直接输出 Agent 判断的候选类型，不必等四轴都“证据充分”或用户确认。
例如“较像 ENFP，J/P 不太明确”。模型无法区分时可用 ENxP 或列出两个候选；
用户确认是纠正来源，不是首次展示门槛。
候选类型与四轴必须一致；后端只检查这种结构一致性，不重新计算人格。

心理模型目录只保存维度名、简短定义、输出格式和负责人。
删除 allowed_signal_codes、SignalWeight、心理 Reducer、固定验证任务三件套。
Agent 可以在一次推断中综合多个维度，不按 15 个 facets 分别强制检索 15 次。

## 4. 人话、正文长度与编辑质量

正文需要让用户直接看懂“这个人是什么样、现在怎样生活、与我怎样相处”。
采用自然中文，先给判断，再说明具体表现；心理学术语用于栏目和维度名，
不在正文里反复写“场景链、信号、主体意义、取向范畴”等抽象词。

以下是目标字数，不是强制补齐的最低量；汉字、字母、数字、标点各计一个可见字符，
不计 UUID、引用列表、JSON 键及格式标记。资料少就少写，不能为凑字数编故事。

| 输出 | 目标字数 | 编辑要求 |
| --- | --- | --- |
| 整体人物介绍 overview | 200–350 字 | 串起生活状态、性格印象和与用户相处的特点 |
| 每个栏目的 summary | 120–220 字 | 一至两段，交代该栏主要情况；不复述所有消息 |
| 单个已描述子项 description | 40–100 字 | 解释清楚这个子项，不只是一个标签 |
| 栏目内 ModelProfile.summary | 150–250 字 | 描述可辨识的倾向及常见表现；避免通用好人模板 |
| 单个心理维度 description | 40–90 字 | 说明特征怎样表现；允许矛盾和情境差异 |
| 单个情境模块 summary | 100–180 字 | 讲清这份工作、学习或照料的主要情况，避免重复公共综述 |
| 模块 details 的字段 description | 20–80 字 | 说明该字段，未知简短说明即可；不按叶子数凑篇幅 |
| reasoning_summary | 20–60 字 | 说明主要观察角度，例如“表达直接、会主动追问，也会用玩笑缓和提醒” |
| roleplay_guidance | 30–80 字 | 写一个可执行的相处或回应建议 |
| unknown 的说明 | 10–30 字 | 简短指出缺什么，不写长篇无证据声明 |

推荐正文风格：

> 她说话直接，常用玩笑和小称呼把提醒说得轻松一些。与用户聊天时比较主动，
> 会追问近况，也愿意分享吃到的东西和临时想到的安排。发生分歧时，可能更适合先接住她的话，
> 再商量具体怎么办。

这是输出风格示例，不是对当前数据中目标人物的实际判断，不得原样作为生成默认值。

不合格正文包括：把问候逐条列出；只写“外向性高”；每句追加无法确定的免责声明；
把“爱分享”固定解释成 E 型；填满华丽但适用于任何人的性格赞美。

字数作为编辑反馈，不以超出目标区间中断整次运行，不截断已生成正文。
只在明显重复、极端冗长或必填正文缺失时请求 Agent 修订。
目标字数不等于模型输出 token 上限；多维度结构化输出的预算需容纳整份结果。

## 5. 七个 Agent 的职责与跨栏上下文

七个 Agent 与第 3 章七个栏目一一对应。每个 Agent 对本栏目全部字段负责，包括其中的心理学子项。
例如 identity 同时负责称呼、自我定位、Big Five 和 MBTI，agency 同时负责目标、决策理由与 SDT。
social_world 的 IPC 只描述第三方关系，relationship_with_user 的 IPC 只描述与用户的关系。

字段路径也是保存、展示、修订和 Runtime 读取的唯一定位方式。例如：
`identity.mbti`、`agency.motivation`、`relationship_with_user.interpersonal_style`。
同一模型/维度/关系范围只有一个最终负责人，跨栏只共享材料与摘要。

### 5.1 调度流程：先共享可用背景，再按需补全

不让所有 Agent 等待“完整人生档案”才开始，也不允许它们在读取过程中混用不同版本。

1. 初始化：冻结本次 Run 的聊天范围、基准图版本和已发布 Profile 引用。
   将已有 life_context 模块目录放入七个 Agent 的起始上下文；没有模块就标明尚未生成。
2. 七栏开始草稿：life_context Agent 识别并填写模块；其他 Agent 同时理解本栏材料。
   第一阶段允许直接推断性格，不必等待模块。
3. life_context 输出完整栏目草稿后，后端为模块分配/更新稳定 ID、revision，
   形成一个不可变的 Run 内模块快照。不把 token 流中的半成品字段提供给其他 Agent。
4. 每个仍运行的 Agent 保持本轮原有读取版本；结束本轮后再提供新快照。
   已读模块的 Agent、明确等待生活背景的 Agent，以及 practices / agency / identity，
   会收到“生活背景已就绪或发生变化”的补全输入。
5. 收到补全输入的 Agent 自己判断是否需要读取具体模块、补查聊天或更新本栏。
   一次补全可处理多个模块，不按字段循环调用模型。没有影响可直接返回“无需修改”。
6. 汇总各栏最终结果，由 identity 结合已完成栏目撰写 overview，结构校验后保存整份候选 Profile。
   若发布期间还有用户修订或模块新版本，当前草稿必须先完成一致性处理，不能静默混入新版本。

初次编译只安排一次模块就绪后的集中补全阶段。模块已读且版本未变、栏目已覆盖其影响时不重复调用。
若消费方发现 life_context 本身需要修改，记录建议供用户修订，或使候选显示“背景有分歧”；
不创建 life_context → identity → life_context 的自动往返循环。

life_context 失败时，其他栏目可以完成自己的理解。之前已发布的模块可以作为明确标注来源的旧背景读取，
不能把旧背景冒充本次新结果。模块工具返回 unavailable/failed，与没有适用模块的空数组明确区分。

### 5.2 模块工具：发现、取定义、按需读取

以下是本次计划新增的接口，不表示现在已实现。实现后由 LangChain StructuredTool 注册，
LangGraph 和统一 AgentLoopController 执行。内部 Run、项目、候选版本由服务端绑定，
Agent 不传任意项目 ID，也不能通过工具读取其他项目。

| 工具 | 模型可见输入 | 返回 | 使用者 |
| --- | --- | --- | --- |
| get_context_module_spec | kind | 类型版本、details 字段结构、字段简短编辑要求 | life_context；需要理解字段时其他 Agent 也可读取 |
| list_context_modules | kinds 可选、statuses 可选 | 模块 ID、标题、类型、生命周期、revision、简短 summary、快照状态 | 七个 Agent |
| read_context_module | module_id、field_paths 可选 | 选中分组/字段的原值、description、basis、模块与快照版本 | 七个 Agent |

list 默认提供当前、计划中和暂停实例，过去实例可显式查询；这只是默认列表范围，
life_course 仍可读取历史。未过滤状态时也须明确响应采用了什么默认范围。
列表只给摘要，避免把所有工作、课程和照料细节塞入每个 Agent 的初始上下文。

read 默认返回整份模块，也可选 experience 或 working_arrangement 等分组。
不存在的字段路径返回明确参数错误，不以空值冒充未知。返回版本由工具绑定当前快照，
同一 Agent 回合读不到一半突然变成另一份内容。

工具响应带 snapshot_id、availability（ready / pending / failed）和来源类型
（本次候选 / 基准已发布版本）。ready 且 items=[] 才表示这个快照没有对应模块；
pending/failed 不解释为目标人物没有工作或没有某种生活情境。所有写入只通过当前候选的版本校验执行。

spec 是结构说明，不是工具 JSON Schema 的重复 dump。工具 Schema 仍通过框架传递；
模块内容遵循统一 JSON 序列化，不引入单独的 Prompt 拼接协议。

life_context 在工具获取 details 定义后，仍通过自己的结构化输出提交模块列表：
采用以 kind 为 discriminator 的 Pydantic 联合类型。模型不用调用数据库写工具。
后端只验证类型、字段和身份版本，不根据组织性质判定性格，也不对描述作正则分类。

### 5.3 其他 Agent 怎样理解模块

| 消费方 | 优先读取 | 需要自己完成的理解 |
| --- | --- | --- |
| practices | 工作/学习安排、照料时间、各模块 period / status | 把制度安排与实际习惯区分开；结合聊天解释作息、弹性与例外 |
| agency | 工作体验、学习压力、照料分工、求职顾虑 | 判断哪些需要被满足/受挫、怎样取舍；不能从“国企”直接推出看重稳定 |
| identity | 各情境的 summary、工作体验、实践和选择相关信息 | 比较环境约束与本人表现，形成 Big Five/MBTI 的整体印象 |
| social_world | 团队合作、学校同辈、照料分工 | 理解第三人和组织在其生活中的角色，区分同场与实际关系 |
| life_course | past / current / planned 模块、period、转变打算 | 整理过去、当前与未来的连接，允许阶段描述粗略 |
| relationship_with_user | 时间限制、共同迁移、照料或其他相关生活安排 | 理解互动可用性与相处方式，不能把工作忙直接写成疏远用户 |

工作时间例子：模块说“单位要求 9 点到岗”，practices 可以写“日常受早间到岗安排影响”，
但不能因此认定她每天 9 点实际到达，更不能替她补出起床和通勤时间。

人格例子：模块说“工作审批严格、截止时间明确”。identity 应理解这是外部条件。
如果聊天整体显得她喜欢自行安排、常为繁琐流程烦躁，可以推断她更重自主或灵活，
即使她实际上需要按流程工作。也可能她喜欢秩序并主动提前准备；需要 Agent 综合判断，
不是由系统选择固定结论。以上均为方法示例，不是目标人物已知特征。

可供 Agent 写在 reasoning_summary 中的表达：

> 这份工作要求按流程做事，但她谈起自己安排时更强调灵活，所以工作上的守时未必代表她偏爱计划。

这类区别由 Agent 理解，不新增“必须有两条本人选择”的证明门槛。
模块中 basis=inferred 的工作体验仍是前一个 Agent 的理解，不能因被另一个 Agent 引用就升级成 stated，
也不能把两个 Agent 对同一段材料的重复判断当作独立佐证。
消费方可直接读取聊天形成不同理解；结构引用只说明背景来源，不替心理推断打分。

### 5.4 背景版本、依赖与失效

模块内容保存在候选 Profile 的 life_context 内。Run 模块快照和依赖记录是辅助运行元数据，
不另建一个与 Profile 各自可写、互相不同步的“人物世界库”。

依赖记录保存 consuming_section / entry_id、snapshot_id、module_id、module_revision、field_paths。
这些来自工具读取和输出引用，保存在调查元数据，不复制 Phoenix 的模型/工具 trace。

变更影响按结构路径确定：整模块引用受任何内容变化影响，字段引用受该路径或父路径变化影响。
标题、排序等展示变更不要求重算人格；无法获得字段级依赖时退回栏目级“需要复查”。
标记受影响只代表让负责 Agent 重新考虑，不代表后端已经判断原性格结论错误。

有正在生成的消费任务时保留它读取的快照，不强杀或丢弃结果；
完成时若依赖已经过期，保存为旧尝试，不覆盖最新草稿，按最新背景安排一次合并后的更新。
幂等键包含目标草稿版本和消费栏目，合并短时间内多个模块变更，避免每改一格就重新跑七栏。

发布前，受影响且纳入本次变更的栏目必须更新或显式决定移除旧结论。
未受影响的栏目及其 ID 保留。已发布 Profile 与旧 Runtime Snapshot 不被该过程原地修改。

### 5.5 通用工具、循环与 Prompt

使用 LangChain 注册工具、LangGraph 编排流程、统一 AgentLoopController 管理循环。
已有 LightRAG 搜索、原始消息定位、上下文读取和当前 Profile/纠正读取工具继续复用。
参与者绑定通过现有工具返回的元数据完成；需要新增接口时单独实现，不将未实现的工具名写进 Prompt。

Agent 先广泛理解，再针对重要空白检索。没有要求每个维度执行 discovery/verify/counter 三次调用，
也不强制寻找反例。遇到显著矛盾，Agent 主动比较或缩小判断；已有合理理解时可以结束。

各 Agent Prompt 单独维护于 Markdown，写清栏目目的、固定子项、心理模型、中文编辑要求、
工具用法以及允许整体推断。模型 schema 和工具参数由框架传入，不在 system prompt 重复整份 JSON。
必要的简短模型定义可随任务提供，帮助不同模型版本保持相同字段含义。
模块 consumer 的 Prompt 说明“可读哪些生活背景、怎样区分外部要求与本人倾向”；字段定义放在模块 spec，
检索与快照读取实现放在工具模块。不要把完整十类模块 schema 重复写进每份 Prompt。

已完成、无进展、工具异常、取消和预算停止由统一 Controller 处理。
没有独立证据链、没有多日期、没有明确自述都不是停止原因。
读取结果为空或工具失效是真实运行问题，应通过 Phoenix 排查，而非继续编写画像。

## 6. 前端审核与纠正

### 6.1 七栏展示与模块子卡片

页面展示 overview 和七个栏目，各栏内展开其全部子项。
identity 内显示 Big Five/MBTI，agency 内显示动机，两个关系栏内显示各自 IPC，life_course 内显示叙事。
不增加单独的“心理画像”页面、顶层卡片组或第八栏。推断内容标注“AI 根据聊天推断，可修改”，参考片段默认折叠。
不显示“证据通过率”，不要求用户逐条批准参考片段才能看画像。

用户可见状态：已生成、尚不了解、当前不适用、运行失败。
不直接显示内部 status 或模型字段名；同时区分“还没有生成”和“已经生成但暂无判断”。

Revision 可针对整个栏目、单个维度或一段描述发起，以后端稳定 ID 绑定选择。
没有 reference_message_ids 也可修订；用户可以直接说“这不像她，她实际上比较……”。

修改心理画像时，Agent 理解用户希望改变的性格描述与表现，按既有流程澄清、预览和确认。
不要求用户举证自己为什么这么认识对方。用户纠正保留来源；有冲突时说明差异并协商。
若还要修改图谱，则沿用独立 Graph Patch 预览、批准和发布，不能把普通回答当作图谱批准。

### 6.2 life_context 模块卡片

现实生活栏先显示公共综述，再显示“主职工作”“兼职教学”“备考”“照料家人”等实例子卡片。
按当前/计划中/暂停/过去组织，默认折叠过去实例。每张卡片显示标题、状态和 summary，
展开后按 details 的固定分组显示内容。未知字段可集中折叠，不展示几十行空白表格。

正文与模型推断都可被圈选；用户可选择单个字段、分组或整个模块发起右侧纠正。
用户可用自然语言要求新增另一份工作、把工作改成过去、纠正类型或移除误归属的模块，
无需填写完整表单和 UUID。

通过 section + module_id + field_path 定位修改，页面排序或标题变化不影响选择。
模块被移除后，历史修订仍可找到旧内容；Profile 中的活动列表与当前引用同步处理。

### 6.3 模块纠正和相关栏目更新

例子：用户说“这段是她在说我的工作，她自己不是这种工作制度”。

1. Revision Agent 读取选中的工作实例和相关上下文，澄清应移除哪些字段或整个实例。
   不默认把其余工作信息也删掉，不要求用户提供正式证明。
2. 形成理解确认后，生成模块修订草稿。后端根据模块/字段依赖列出可能受影响的 practices、
   agency、identity 等栏目，展示为“可能需要重新考虑”，不直接宣告它们全错了。
3. 在候选草稿里应用模块变更，相关负责 Agent 读取新快照重新判断自己的条目。
   对新模块或删除模块，曾读取模块目录的相关栏目也纳入检查，避免只追踪已有引用而漏掉新增背景。
4. 将模块修改与这些更新合并成同一份 Profile Diff：每项显示修改前后及原因。
   用户批准的是这份完整结果，而不是模块改完后还有后台任务悄悄改人格。
5. 用户继续更改理解或范围时，旧 Diff/批准失效；保留变更历史，重新生成本次实际要批准的版本。
6. 涉及图谱时再生成 Graph Patch，遵守既有独立确认和发布流程。

若受影响条目仍因其他材料成立，Agent 可以保留正文并更新背景引用。
若不足以保留，提出更改或移除，不把其他人格维度清空。
受影响 Agent 运行失败时，保留当前发布版本，候选标明待完成；
允许用户明确选择将受影响的旧条目一并移除后继续审核，不把过期结论静默发布。

纠正类型时采用“新建正确类型实例 + 移除错误类型实例 + 更新引用”的原子候选变更，
旧 ID 留在修订历史。不能只改 kind 而保留另一种类型的 details。
结束一份真实工作用 status=past 并保留 ID；误把用户的工作归给目标人物时应移除误归属内容，
不能写成目标人物“过去做过这份工作”。

## 7. 图谱与 Runtime

心理理解默认保存在 Profile，不自动写成 LightRAG 的确定人物属性。
现实事实的图谱修改继续使用来源核对、候选版本、ChangeSet 和用户批准。
模块生成和跨栏补全不调用图谱写接口；module_id 也不默认等于 LightRAG entity_id。
减少心理画像的证据门槛不改变既有图谱操作权限。

PersonaActor 从七个栏目各自的字段读取结构化值与中文解释，包括栏目内的模型子项；不读取独立 psychology 对象。
PersonaContextAssembler 选择当前相关部分，保留原意，不通过固定模板改写成另一套人设。

上下文优先级：

1. 当前对话、当前分支事实和用户纠正；
2. 已发布的现实处境、共同项目和关系边界；
3. 相关关系和任务的 IPC、SDT 理解；
4. Big Five、MBTI 与叙事身份的画像及角色建议。

标为 inferred 的画像也可进入 Runtime，不要求 supported 或 user_confirmed。
生效条件是整个 Profile 已按现有流程发布；无需对每个心理维度再次批准。
不能根据 MBTI 编出新的职业、经历或共同回忆。

DayPlan 可按需读取分支绑定 Profile 的当前工作、学习、照料模块及 practices；
模块里的工作制度是规划约束或背景，practices 是对实际作息的理解，两者都不能直接替代当天安排。
PersonaActor 与 Director 读取同一发布版本，不能一个读取新模块、另一个保留旧人格。
修订在发布前只供编译和审核使用；已有分支是否重新基线化继续沿用既有显式版本流程。

## 8. 代码组织与迁移计划

```text
backend/moonlightbox/world/person_world/
├── contracts/
│   ├── profile_v3.py                  # 七栏完整字段与中文正文
│   ├── world_dimensions.py            # 七栏固定子项与格式
│   ├── psychological_models.py        # 栏目内复用的模型值类型
│   ├── context_modules.py             # 实例外壳、kind 判别联合、字段值与引用
│   └── context_module_details/        # 各类型 details；定义字段，非行为 Prompt
├── tools/
│   ├── context_module_spec.py         # 获取模块结构与简短编辑要求
│   └── context_modules.py             # 列目录、按固定版本读取
├── context_module_snapshots.py        # Run 内快照、版本、读取依赖、变更影响
├── subagents/                        # 七份 Agent 行为与编辑要求
├── assembly/
│   └── profile.py                    # 结构校验和装配
├── investigation/                    # 运行状态、资料范围、Phoenix 关联
└── coordinator.py                    # 栏目草稿 → 跨栏补全 → 审核
```

实施顺序：

1. 建立统一七栏 schema、ContextModule 外壳、十类 details、ModuleField、ContextModuleRef；
   kind 判别联合与工具 spec 共用同一字段定义，避免前后端出现两份不同表单。
2. 实现 spec/list/read 三个只读工具、Run 内快照和读取依赖；生成前验证工具注册集合与 Prompt 一致。
3. 更新 life_context Prompt：选择实例、按需获取类型定义、批量补足重要字段、输出完整栏目。
   其他 Prompt 增加模块读取和环境/个人区别，不增加证据计数要求。
4. 实现模块稳定 ID/revision 分配、公共字段引用、七栏草稿和模块就绪后的集中补全。
5. 完成模块子卡片、按稳定 ID 的 Revision Scope、依赖失效、相关栏目重算和合并 Profile Diff。
6. 接入 Runtime/DayPlan 的版本绑定模块读取；保留 inferred 可用和整体 Profile 发布规则。
7. 运行一轮真实数据烟测，检查模块具体程度、跨栏理解、主体归属与中文画像。
   再走一次“纠正工作安排 → 相关栏目更新 → 合并 Diff”的候选流程，不自动批准或写图。

保留 v1/v2 数据与旧 Snapshot，不原地覆盖历史。v3 从可用聊天、图谱和用户纠正重新生成。
旧参考资料可重用，但旧“通过/未通过证据”状态不能决定 v3 是否允许理解。
已有分支升级仍按项目原有版本机制处理。

## 9. 验收

文档与实现共同遵守以下标准：

1. 只有零散、玩笑式、间接表达而没有自我介绍的聊天，也能产生 Agent 的性格与相处方式判断。
2. 没有 reference_message_ids、多日期或多个场景时，心理画像仍能保存、审核、发布后供 Runtime 使用。
3. 表情、文风、回应习惯、玩笑均允许参与推断；单一特征不通过代码直接换算成人格。
4. MBTI 候选类型可直接展示为 AI 推断；四轴不明可写 x 或候选，不等待“证据充分”状态。
5. 未提供雇主、岗位、学校、精确作息时，不从心理倾向补出这些具体事实。
6. 心理状态不因检索不到明确自述就全变 unknown；工具失败也不能伪装成正常推断。
7. overview、栏目综述、模型综述和子项有明确中文编辑要求，展示内容能读成画像而非消息清单。
8. 目标字数不形成凑字、截断或整次失败条件。
9. 后端只做结构、版本、引用和权限校验，不加心理证据权重或计数门槛。
10. Profile、API、前端、Revision 与 Runtime 使用同一套七栏字段路径；无独立 world/psychology 树。
    同一个心理维度有唯一栏目与负责人；跨栏材料可共享，主体归属仍正确。
11. 无逐条引用的画像可被用户修改；图谱修改仍经过既有独立批准。
12. Phoenix 能关联实际调用、输入材料、阶段、输出及错误；不再创建一套自建 trace。
13. 同时有两份工作和一段学习时保存三个独立模块；未知公司名不阻止填写工作内容和体验。
14. spec、Pydantic details 和前端展示字段一致；不同 kind 的 details 不能混写。
15. 模块更新或标题变更保持正确实例 ID；新增另一份工作不能覆盖原工作。
16. 同一回合读版本一致；新模块就绪后进行集中补全，不触发 Agent 互相无限重跑。
17. 模块推断被跨栏读取后仍标为推断；同一材料被多个 Agent 引用不增加所谓证据强度。
18. 用“公司要求严格、本人更希望灵活”的材料检查 identity 的理解，不能通过规则直接判为高尽责性。
    这是生成质量检查，不是用字符串断言某种人格才算正确。
19. 纠正 schedule 只标记结构相关的消费者；标题或排序变动不重跑人格；新增/删除模块不会遗漏目录消费者。
20. 合并 Diff 包括模块和受影响条目；旧版消费任务不能覆盖新版草稿；用户没有批准前不更新已发布版本。
21. DayPlan 与 PersonaActor 读取分支固定版本，候选模块不泄漏到正在体验的已发布 Runtime。

## 10. 当前边界

这份文件描述的是计划中的文档和架构改动；只有实现并运行后才能报告生成效果。
当前的完成标准是让 Agent 用主流心理模型组织自己的理解，提供具体、自然、有用的中文人物画像，
并让用户能够纠正它。

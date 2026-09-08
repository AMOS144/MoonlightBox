# 聊天地点解析与个人带权地点图设计

## 1. 文档目的

本文只解决 MoonlightBox 在导入历史聊天记录时，如何从消息中提取空间证据并构建人物的个人空间模型。本文不设计 Agent 在模拟未来中的移动、地图探索、行动执行和实时感知；这些能力以后建立在本文产出的地点实体、到访观测和带权地点图之上。

本文回答以下问题：

1. 如何从聊天记录中提取地点提及。
2. 如何判断“家”“公司”“万象城”“那里”等文字指向哪个地点实体。
3. 如何区分提到某地、计划去某地、正在路上、已经到达和确实停留。
4. 如何在证据不足时保留歧义，而不是强行选定地址。
5. 如何把离散空间证据聚合为人物的带权地点图。
6. 地点节点和地点间连线包含哪些字段，以及每个派生字段如何计算。
7. 如何保证分析可重跑、可审计、可人工修正且不会把模型推断伪装成历史事实。

协议暂定为 `personal-place-graph-v1`。

其中“家中、到家、回家”等个人语义地点的身份归一、人物视角、时间有效期、合并审计和旧数据
回填，由补充规格《[语义地点合并与个人别称归一设计](./2026-08-20-semantic-place-alias-canonicalization-design.md)》
定义；实现语义地点合并时以该补充规格为准。

V1 的唯一主线固定为：

```text
所有聊天
→ Episode 化
→ Episode 级空间检索
→ 局部上下文分析
→ 一个个人地点图
```

其中只有最后的个人地点图是业务图。Episode 是消息分组，向量索引是搜索设施，二者都不是需要维护的额外知识图谱。

## 2. 范围与非目标

### 2.1 本阶段必须完成

- 将全部历史消息确定性地切分为可重放的 ConversationEpisode。
- 为 Episode 建立向量索引，通过空间语义检索选出候选 Episode，而不是逐消息调用地点模型。
- 扩展候选 Episode 的相邻上下文，在局部对话片段上发现地点证据。
- 直接接纳结构化位置消息和可用媒体地理元数据，不要求它们经过语义检索。
- 精确保存原文 span、发送者、目标人物、消息时间和证据消息 ID。
- 解析否定、假设、计划、引用、转述、到达、离开和在途状态。
- 将地点提及链接到已有个人地点、地图 API 候选或暂未定位的语义地点。
- 形成可竞争、可合并、可拆分的地点实体候选。
- 从地点提及生成到访观测和到访片段，但不制造缺失轨迹。
- 按人物构建有向带权地点图，并逐字段保留计算证据。
- 支持增量导入、全量重算、模型版本变更和地图供应商变更。

### 2.2 本阶段明确不做

- 不模拟 Agent 当前在哪里。
- 不让 Agent 根据地点图自动出行。
- 不生成未来日程。
- 不调用地图 API 代替主体决定目的地。
- 不把稀疏聊天补全成连续 GPS 轨迹。
- 不根据“公司”“家”等别称猜测精确门牌号。
- 不把用户转述、引用消息或计划自动当作目标人物的实际到访。
- 不把地图 API 返回的候选 POI 自动当作真实历史地点。

## 3. 核心原则

### 3.1 提及、实体、到访和习惯必须分层

系统必须区分四种不同对象：

```text
PlaceMention       消息里出现了某个地点表达
PlaceEntity        多个表达可能共同指向的地点
VisitObservation   有证据表明某个人与某地点处于某种到访状态
PersonalPlaceGraph 多条到访证据聚合出来的活动范围和转移规律
```

“我今天想去公司”可以产生 `PlaceMention`，但只能产生 `planned` 观测，不能增加实际到访次数。“到公司了”可以产生 `arrived` 观测，但如果“公司”无法定位，仍然可以链接到坐标未知的语义地点实体。

### 3.2 地图 API 是候选提供者，不是历史真相来源

地图 API 可以提供：

- 地理编码候选。
- POI 名称、类别、行政区和坐标。
- 两地点间距离、路线和预计耗时。
- 某坐标附近的设施。

地图 API 不能证明聊天人物去过某地，也不能证明“公司”对应候选列表中的某栋楼。真实历史只能由消息、结构化位置分享、可信媒体元数据或人工确认支撑。

### 3.3 可以先有语义地点，后有地理坐标

`妈妈的公司`、`家`、`老地方`在没有坐标时仍是有效地点实体。地点实体支持以下地理解析等级：

```text
semantic_only  只有人物语义，例如“公司”
district       只确定城市或行政区
area           确定商圈、园区或模糊区域
poi            确定到具体 POI
address        确定到具体地址
coordinate     来自可信位置分享的坐标
```

系统不得为了让图看起来完整而强制提升解析等级。

### 3.4 原始证据不可变，派生结论可版本化重算

原始 `Message` 和原始媒体元数据保持不变。地点提及、候选、消歧结果、到访片段和图权重都必须带分析运行版本。模型、Prompt、规则或地图数据变化时创建新版本，不覆盖旧结果。

### 3.5 宁可未解析，也不能错误合并

错误地把两个“万象城”合并，或者把“我家”和“你家”合并，会比保留两个临时实体造成更严重的长期污染。自动解析必须同时满足最低分数和领先第二候选的 margin；不满足时保留候选集合。

### 3.6 人物归属优先于地点识别

以下三句话不能得到相同的到访结论：

```text
我到公司了                 发送者本人到达公司
你到公司了吗               对另一个人的询问，不证明任何人到达
小王说他到公司了           第三方转述，低权限报告
```

每个地点提及必须先确定“谁与地点发生什么关系”，再做实体消歧。

### 3.7 全量聊天只做 Episode 化和向量化

所有历史消息都会进入确定性的 Episode 切分，并为每个 Episode 生成一次 embedding；不会把每条消息逐条交给语言模型扫描。较重的地点理解模型只处理被空间语义检索召回、并扩展过相邻上下文的少量 Episode bundle。

## 4. 总体处理管线

```text
全部聊天消息
  ↓
确定性 ConversationEpisode 化
  ↓
Episode embedding 与空间语义检索
  ↓
候选 Episode 的相邻上下文扩展
  ↓
局部空间分析：PlaceMention + VisitObservation 候选
  ↓
地点实体消歧：PlaceCandidate → PlaceEntity
  ↓
合并 VisitObservation → VisitEpisode
  ↓
聚合一个 PersonalPlaceGraph
  ↓
生成版本化图快照
```

系统只维护一种业务图：最终的个人地点图。ConversationEpisode 是现有聊天消息的分组和检索单元，不是第二张图；向量索引只是候选搜索工具。

所有阶段均可独立失败和重跑。后序阶段只能引用前序阶段的持久化 ID，不能只保存一段无法追溯的模型摘要。

## 5. ConversationEpisode 化与空间检索

### 5.1 Episode 是什么

`ConversationEpisode` 是一段时间连续、适合被模型一次理解的原始对话。它不是关系事件，不要求内容重要，也不产生摘要事实。

它只是检索 chunk 的稳定清单，不复制一套聊天存储。每个 Episode 最少只持久化：

- `id/project_id/import_id`。
- 有序 `message_ids`。
- `started_at/ended_at`。
- `segmentation_version`。
- `content_hash`，由有序消息 ID 和原始消息内容确定。

参与者、规范化对话文本、结构化位置和媒体都在建索引或分析时通过 `message_ids` 从原表读取，不在 Episode 表重复保存。前后 Episode 通过同一 import 内的时间排序得到，不需要持久化双向指针。

Episode 只是一种关系表投影，不需要图数据库。保留上述最小字段是为了能够引用原始证据、按时间过滤、增量检测内容变化，以及在切分算法升级后并行重建新版本。

### 5.2 如何切分全部聊天

消息先按时间和稳定序号排序。每条普通消息只属于一个基础 Episode，不使用重叠窗口，避免同一证据被重复分析。切分只服务于检索质量和模型输入预算，不表示人生事件边界。

切分条件为：

- 相邻消息间隔超过 `episode_gap`。
- Episode token 数达到 embedding 或局部分析模型的安全输入预算。
- 消息数达到防止异常超长片段的安全上限。

`episode_gap` 不应拍脑袋固定。首次实现先统计真实聊天相邻消息间隔分布，选择能分开明显离线期、又不会拆散连续对话的拐点；缺少统计时才使用 90 分钟作为回退值。token 预算根据实际 embedding 和局部模型上下文计算，不使用固定中文字数代替。所有参数写入 `segmentation_version` 配置。

达到长度上限时，即使仍在同一段活跃聊天中也切成连续 Episode；分析时仍可通过时间排序扩展相邻 Episode。跨自然日不是独立切分条件，避免把 23:50“上车了”和 00:20“到了”错误拆成无关对话。

结构化位置卡片、引用和媒体不触发单独切分；它们与附近消息留在同一 Episode。引用原文保留原作者，不能混入引用者发言。

Episode ID 由以下内容生成稳定哈希：

```text
project_id + import_id + segmentation_version + ordered_message_ids
```

相同输入和配置必须得到完全相同的 Episode 清单。

### 5.3 Episode 文档如何构造

embedding 文档保留时间顺序和说话者，不使用 LLM 摘要：

```text
[2024-03-12 12:01][妈妈] 中午去长泰吧
[2024-03-12 12:02][用户] 好呀
[2024-03-12 12:47][妈妈] 我到了
[2024-03-12 13:30][妈妈] 回单位了
```

结构化位置消息追加受控标记，例如 `[位置分享: 名称=长泰广场]`。图片 OCR、语音转写和媒体语义只有在已有结果时才附加，并标注来源；没有结果时只保留媒体占位，不阻塞 Episode 建索引。

### 5.4 输入规范化

- 保留原文和字符偏移，另生成规范化文本。
- 全角半角、重复空白和常见表情符号可规范化，但不能改变原 span。
- 微信 `>` 引用区、`[引用消息：...]` 和转发内容单独标记，不与发送者直接陈述混合。
- 语音转写、OCR 和图片语义注释必须保留其识别置信度和来源类型。
- 系统位置分享优先读取结构化经纬度、名称和地址，不从展示字符串反解析已有结构化字段。

### 5.5 Episode embedding

所有 Episode 都生成一次本地 embedding，并写入按项目隔离的向量集合。向量文档、embedding 模型版本和内容哈希必须绑定；内容未变化时不重复计算。

这一阶段没有地点提取模型，也不调用地图 API。它只把完整历史变成可进行语义搜索的 Episode 索引。

### 5.6 空间语义 query bank

空间检索不用一个宽泛的“地点”查询。下面列出的是 query family，不是只有六条固定查询。每个 family 包含一条语义描述、若干贴近真实聊天的短句原型，以及以后从标注集漏召回样本中加入的查询。

V1 query family 至少包括：

| family | 语义范围 | 初始口语原型示例 |
|---|---|---|
| `presence_arrival_departure` | 当前所在、到达、离开 | 到了、我还在这、刚从那边出来 |
| `home_work_school` | 家、公司、单位、学校等日常锚点 | 到家了、还在公司、刚下班、去学校 |
| `commute_transport` | 通勤、在途和交通节点 | 还在路上、上地铁了、堵车、还有三站 |
| `outing_consumption` | 吃饭、购物、娱乐和日常外出 | 去吃饭、逛商场、在咖啡店、去公园 |
| `meeting_social` | 见面、接送、聚会和共同地点 | 在哪见、我去接你、到你家、聚会地点 |
| `travel_lodging` | 旅行、出差、酒店、机场和车站 | 到机场了、住这个酒店、出差、回上海 |
| `medical_service` | 医院、办事和预约地点 | 到医院了、去复诊、在银行、办完出来了 |
| `moving_long_term_change` | 搬家、换公司和长期活动范围变化 | 搬到浦东、换办公室了、新家、调到新校区 |
| `relative_deictic` | 这里、那里、附近、楼下等空间指代 | 我在楼下、还是老地方、就在附近、到这边了 |
| `plan_cancel_route` | 地点计划、取消和路线变化 | 明天去、改去别的地方、不去了、从这里过去 |

每个口语原型作为独立 embedding query，不把整张表拼成一段长查询。每个 family 初始可以有 5 到 15 条原型，检索时取 family 内最高相似度，并设置 family 配额后求并集。因此实际 query 数量会是几十条，而不是上面曾写出的六条。

query bank 使用版本化 YAML 或 JSON 管理，字段至少包含 `query_id/family/text/source/version`。初始原型来自真实聊天抽样；标注集中的漏召回样本经过人工确认后才能增加新原型。第一遍 query bank 只表达通用空间行为，不混入任何人物的专属地点别称。已经确认的别称如需回查历史，走 5.9 的独立可选流程。

### 5.7 如何避免只召回最近或最常见片段

对多年聊天只做一次全局 top-k 会让大量“回家/公司”片段淹没早期旅行和搬家证据。因此检索按时间分区执行，默认按自然月：

1. 每个 query 在每个月份内独立取 top-k。
2. 保留超过最低相似度的结果。
3. 每月即使分数整体较低，也保留少量最高分供召回评估。
4. 所有带结构化位置消息的 Episode 无条件进入候选。

初始 `k` 和相似度阈值必须由标注集选择。这里优先保证候选 Episode recall，误召回可以由局部模型拒绝。

V1 默认关闭 EventNode 提示。如果后续离线召回评测证明它确实有增益，当前项目已有的 `outing/travel/date/family_social` 等 EventNode 引用消息才可以作为可选 `recall_hint`，给对应 Episode 加召回分；但不能让 Episode 自动通过局部空间分析，更不能成为地点事实证据。空间管线不得依赖 EventNode 是否已经生成；关闭该 hint 后仍必须能独立完成全流程，避免两条分析管线产生执行顺序依赖。

候选 Episode 记录：

- 命中的 query ID。
- 每个 query 的 similarity。
- 时间分区。
- 结构化位置强制召回原因，或可选 EventNode `recall_hint`。
- 检索器、embedding 和 query bank 版本。

### 5.8 相邻上下文扩展

单个候选 Episode 可能只有“中午去长泰吧”，而“我到了”落在下一个 Episode。局部分析前按确定规则扩展：

- 默认加入前一个和后一个 Episode。
- 只加入与候选 Episode 时间间隔不超过 6 小时的邻居。
- 加入候选消息明确引用的原消息。
- 多个候选的扩展范围重叠时合并成一个 `SpatialAnalysisBundle`，避免重复调用模型。
- Bundle 超过模型预算时按 Episode 边界拆开，并保留连续 bundle 关系。

邻居只是上下文，不自动成为空间证据。模型输出仍必须精确引用支持结论的 message ID。

### 5.9 已知别称的历史回查

这不是第一遍主链的必要步骤，而是第一遍结束后的可选 recall backfill。V1 默认关闭；只有标注集证明第一遍会漏掉有价值的历史片段时才开启。

它解决的是一个很具体的问题：系统可能先从表达完整的对话中确认“长泰”指向某个地点，但更早的聊天里只有“还是长泰？”“长泰吧”这种语义很弱的短句，第一遍通用空间 query 未必能召回。确认别称以后，可以定向找到这些早期出现，再交给局部模型判断，而不是直接认定它们指向同一地点。

执行位置固定为：

```text
第一遍通用空间召回
→ 第一遍 Bundle 分析
→ 第一遍实体消歧，产生候选 PlaceAlias
→ 可选：已知别称历史回查
→ 分析新增 Bundle，并重算受影响实体
→ 最终合并 VisitEpisode 和地点图
```

#### 5.9.1 哪些别称可以触发回查

别称必须满足以下任一条件：

1. 人工明确确认了 alias 与 PlaceEntity 的链接；或
2. 至少两个第一遍独立 Bundle 提供支持，并且 alias 链接置信度超过标注集校准的阈值。

“独立”要求证据来自不同 Bundle，且不能由本次别称回查召回。这样可以防止“用长泰搜到长泰，再拿搜索结果证明长泰”的循环证据。

别称按检索性质分三类：

| `alias_kind` | 示例 | 自动回查策略 |
|---|---|---|
| `stable_name` | 长泰、前滩太古里、瑞金 | 允许按已证实写法精确回查 |
| `role_anchor` | 公司、新单位、妈妈家 | 必须绑定人物和有效时间区间；所有命中都只是待判断假设 |
| `deictic` | 老地方、这里、那边、楼下 | 默认禁止全历史回查，只在局部 Bundle 内做指代消解；人工确认了明确时段和对象时才能例外开启 |

系统只使用证据中真实出现过的规范写法和人工确认的变体，不由模型凭空生成同义词、简称或模糊拼写。

#### 5.9.2 如何检索

这里不再使用“别称 × 空间行为”生成大量 embedding query。已知词面后，采用项目内全文索引或规范化精确匹配更可控：

1. 用 `normalized_alias` 查询 Episode 的规范化文本。
2. 保留原始字符 span、message ID 和命中的 alias ID。
3. 排除第一遍已经分析过的 Episode。
4. 对命中 Episode 执行与 5.8 相同的相邻上下文扩展。
5. 重叠范围合并为新的 `SpatialAnalysisBundle`。
6. 按 `alias_id + alias_version + episode_content_hash` 去重，使重跑幂等。

精确命中只是召回条件，不是地点证据。为了限制常见词造成的候选爆炸，每个 alias 设置单月和单次 run 的候选上限；超过上限时停止自动回查并进入抽样评估，而不是按相似度偷偷丢弃历史。

#### 5.9.3 新 Bundle 如何分析

局部模型除原始消息外，只额外收到一条候选提示：

```json
{
  "matched_text": "长泰",
  "candidate_place_id": "place_123",
  "alias_id": "alias_456",
  "alias_confidence": 0.91,
  "valid_time_range": ["2024-01-01", null],
  "instruction": "该链接只是召回假设，必须根据本 Bundle 判断 same_place / other_place / unresolved / non_spatial"
}
```

模型仍必须引用当前 Bundle 的 message ID，并允许输出：

- `same_place`：当前用法支持链接到候选地点。
- `other_place`：同一词面在这里指向其他地点。
- `unresolved`：上下文不足，保持未解析。
- `non_spatial`：只是品牌、活动名或其他非地点用法。

`same_place` 生成链接到候选地点的 PlaceMention；`other_place` 生成未链接到该候选的独立 PlaceMention，重新走正常实体消歧；`unresolved` 可以保留未解析 PlaceMention，但不能仅凭 alias 提示生成实际到访；`non_spatial` 不生成地点结果。随后只重算受这些新证据影响的实体、到访片段和图边。

#### 5.9.4 防止自我强化

- `retrieved_by_alias_id` 必须写入新 mention 的来源。
- 回查结果不能提高触发本次回查的 alias 置信度，也不能作为该 alias 达到“两条独立证据”的依据。
- 回查中新发现的 alias 不触发第三遍检索。
- 地点链接、到访和图权重仍只使用消息本身及局部上下文；“因为 alias 搜到了它”不增加任何证据权重。

因此第二遍的作用仅是补找可能漏掉的原文，不是扩充地点知识，也不是用已有结论重写历史。

### 5.10 相对时间规范化

每条消息必须先转换到项目时区。相对时间使用消息时间作为基准，例如：

```text
“刚到公司”       区间默认 [消息时间 - 30 分钟, 消息时间]
“下午去公司”     使用当地日期的下午模糊区间
“明天去公司”     形成 planned 区间，不形成实际到访
“昨天在公司”     形成当地昨天的模糊区间
```

时间结果始终保存上下界和解析精度，不能只保存一个假精确时间点。

## 6. 局部空间分析

### 6.1 模型一次分析什么

地点理解模型只读取一个 `SpatialAnalysisBundle`，不会从头扫描完整聊天，也不会每条消息调用一次。输入包含：

- Bundle 内按时间排序的原始消息。
- message ID、发送者和时间。
- 引用原作者和结构化位置内容。
- 已确认的少量个人地点别称及其 place ID。
- 不提供地图候选；这一阶段不调用地图 API。

模型一次完成：

1. 找出真正承担空间作用的原文 span。
2. 判断相关人物。
3. 判断 `at/go_to/arrive/leave/pass_by/near/mention`。
4. 判断陈述、问句、否定、计划、取消、假设、引用和转述。
5. 解析同一 Bundle 内“这里、那里、刚到、回去了”等指代。
6. 给出候选 PlaceMention 和 VisitObservation，不做具体地图 POI 消歧。

输出必须是结构化 JSON。每个结论必须包含 `evidence_message_ids`；显式地点必须返回原文字符 span；省略地点使用 `implicit_anchor`。不能引用 Bundle 外消息，不能创造输入中没有出现的具体地名。

### 6.2 可提取的地点表达

下面的类型是 Bundle 局部模型经过上下文分析后的输出标签，不是预先扫描聊天的字符串词典。局部模型允许类型未知，也允许同时保留多个类型及其分数。

地点提及的受控输出类型为：

| 类型 | 示例 | 说明 |
|---|---|---|
| `named_poi` | 万象城、瑞金医院 | 可查询地图候选的命名地点 |
| `address` | 南京西路 123 号 | 地址表达 |
| `administrative` | 上海、浦东、张江 | 行政区或片区 |
| `personal_anchor` | 家、公司、学校、妈妈家 | 与人物绑定的语义地点 |
| `relative_place` | 楼下、附近、对面、地铁口 | 依赖锚点解析 |
| `deictic_place` | 这里、那里、老地方 | 依赖对话指代 |
| `route_or_station` | 二号线、还有三站、虹桥站 | 交通路径或站点 |
| `remote_context` | 群里、视频里、网上 | 非物理地点，标记后不进入地理图 |
| `unknown_physical_place` | 搬到那片以后、还在园区里 | 可以判断为空间表达，但暂时无法稳定分类 |

`mention_type` 不能单凭 span 文字决定。例如“瑞金医院通知我复诊”中的“瑞金医院”首先是消息来源机构，不足以证明人物与医院存在空间关系；“我已经到瑞金医院了”才是带 `arrive` 关系的物理地点提及。“公司说要裁员”里的“公司”是组织主体，“我还在公司”里的“公司”才是个人空间锚点。

类型只是模型分析后的受控标签，不是用于预扫描聊天的字符串词表。模型先判断候选表达是否在当前上下文中承担 `physical / remote / organization_only / non_spatial / uncertain` 作用，再输出地点类型、人物和空间关系。

对于可能属于 `named_poi/address/administrative/route_or_station` 的结果，后续实体消歧才调用地图 API。对于 `家/公司/这里/楼下`，先创建或链接人物语义地点，不强制查询地图。

不同例子的实际判定：

| 原文 | Bundle 局部分析 | 上下文判定 | 结果 |
|---|---|---|---|
| 我到万象城了 | 找到“万象城”及到达关系 | 物理地点、arrive、发送者本人 | `named_poi`，再查地图候选 |
| 万象城今天发了公告 | 找到“万象城”但无人物移动 | 机构主体或 uncertain，没有人物空间关系 | 不形成到访；可拒绝空间 mention |
| 公司说要裁员 | 找到“公司”但它是施事主体 | `organization_only` | 不形成地点提及 |
| 我还在公司 | 找到“公司”及在场关系 | 物理地点、present、发送者本人 | `personal_anchor` |
| 我在公司附近 | 找到“公司/附近”及相对关系 | 公司为锚点，附近为 relative relation | 两个关联 mention，不猜具体坐标 |
| 瑞金医院通知我复诊 | 找到“瑞金医院”但它是消息来源 | 机构来源；“复诊”可能形成 planned 事件 | 不形成实际到访 |
| 我已经到瑞金医院了 | 找到“瑞金医院”及到达关系 | 物理地点、arrive | `named_poi` |
| 你到公司了吗 | 找到“公司”但整句是询问 | question、主体为对方 | 可保留询问 mention，不形成到访 |
| 刚到 | Bundle 模型发现省略地点 | arrive，但地点未知 | `implicit_anchor`，在局部上下文中解析 |
| 还有三站 | Bundle 模型结合前后消息发现 | en_route，线路/目的地未知 | `route_or_station`，不创造站名 |

系统不维护世界地点词典。第一遍局部分析不依赖个人地点词表；只有通过第一遍分析形成且满足 5.9 资格的项目个人别称，才可以参与可选的历史回查。

### 6.3 PlaceMention 字段

每一条地点提及包含：

| 字段 | 来源或计算方式 |
|---|---|
| `id` | UUID |
| `project_id/import_id/message_id` | 原始消息外键 |
| `analysis_run_id` | 本次空间分析运行 |
| `span_start/span_end/raw_text` | 原消息中的精确字符位置 |
| `normalized_text` | 去除无意义空白后的规范表达，不改变语义 |
| `mention_type` | 上述受控枚举 |
| `type_distribution` | 各候选类型的校准分布；最高分不够时保留 unknown |
| `source_episode_ids/source_bundle_id` | 召回和局部分析来源 |
| `candidate_sources` | `structured_location/episode_model/confirmed_alias` 中的一个或多个 |
| `speaker_id` | 消息发送者 |
| `subject_id` | 与地点发生关系的人物；未知时为空 |
| `subject_resolution` | `explicit/self/second_person/coreference/reported/unknown` |
| `relation` | `at/go_to/arrive/leave/pass_by/from/to/near/mention` |
| `movement_phase` | `none/planned/en_route/arrived/present/departed/cancelled` |
| `assertion_mode` | `asserted/question/negated/hypothetical/conditional/reported/quoted` |
| `time_start_lower/time_start_upper` | 开始时间上下界 |
| `time_end_lower/time_end_upper` | 结束时间上下界 |
| `time_precision` | `exact/minute/hour/part_of_day/day/unknown` |
| `context_message_ids` | 本次解析实际读取的上下文 |
| `evidence_message_ids` | 直接支持该提及结论的消息 |
| `extractor_method` | `structured_location/episode_model/structured_plus_model` |
| `raw_score` | 未校准提取分数 |
| `confidence` | 经验证集校准后的置信度 |
| `status` | `active/rejected/superseded/needs_review` |

### 6.4 提取分数

结构化位置和 Episode 局部分析结果统一形成未校准分数：

```text
raw_mention_score = clamp(
    0.25 × source_structure
  + 0.20 × span_specificity
  + 0.20 × relation_clarity
  + 0.15 × subject_clarity
  + 0.10 × time_clarity
  + 0.10 × context_support
  - contradiction_penalty,
  0, 1
)
```

各特征定义：

- `source_structure`：结构化位置分享为 1；普通文本为 0.5；OCR/语音按识别置信度缩放。
- `span_specificity`：完整地址或唯一 POI 接近 1；“那里”“附近”接近 0.2。
- `relation_clarity`：“到 X 了”“正在 X”高；仅出现地点名低。
- `subject_clarity`：显式人物或发送者自述高；省略主语或第三方转述低。
- `time_clarity`：明确时间高；只有消息时间的弱推断中等；无可用时间低。
- `context_support`：相邻消息重复确认、位置卡片或一致路线信息提高。
- `contradiction_penalty`：否定、问句误判、引用归属不明、同时地点冲突等惩罚之和，最高 0.6。

`raw_mention_score` 不是概率。上线前使用人工标注验证集做 isotonic regression 或 Platt scaling，得到 `confidence`，并保存校准器版本。

### 6.5 必须拒绝的常见假阳性

- “公司说要裁员”中的公司是组织，不是人物所在地点。
- “回家看看这部电影”若“回家”属于影片名或引用标题，不是移动。
- “给家里打电话”表示通讯对象，不证明当前或目标地点。
- “你到公司了吗？”是询问，不证明对方到达。
- “本来想去万象城但没去”只保留取消计划，不形成到访。
- 引用消息中的“我在家”属于原作者，不属于引用者。
- “外卖送到公司”可以支撑收货目的地，不自动证明人物当时在公司。

## 7. 地点实体消歧

### 7.1 消歧不是一次地图搜索

从逻辑上看，地点实体消歧使用五类操作：

1. 查询人物已有地点实体和别称。
2. 查询同一对话中最近出现、尚可承接指代的地点。
3. 根据城市、行政区、路线和共现地点生成地理约束。
4. 必要时调用地图 Provider 获取 POI/地址候选。
5. 对候选打分并决定链接、暂定、创建语义地点或保持未解析。

优先匹配个人已有地点，能够保持“公司”“单位”“办公室”长期指向同一个人物地点；但个人先验最多贡献有限权重，不能压过新的明确地址证据。

上面的五条是证据处理逻辑，不应直接理解成五个 ReAct Agent。实际实现采用一个可回放的有界 workflow，其中大部分节点是数据库查询、纯函数或受控 Provider 调用。Bundle 局部模型已经完成自然语言理解，实体消歧阶段不再让多个 Agent 自由决定查什么、何时停止。

#### 7.1.1 全局调度

一次 `spatial_analysis_run` 不能简单按照数据库返回顺序逐条解析，否则先处理哪条消息会改变“已有地点”。调度器使用固定顺序：

1. 冻结上一个已发布地点图快照，作为本次 run 的历史先验。
2. 先处理人工确认、结构化位置、完整地址和高唯一性 POI 等强证据，形成当前 run 的种子地点。
3. 再处理命名地点和人物角色锚点，例如“万象城”“公司”。
4. 最后处理“那里”“楼下”“老地方”等依赖先行词的弱指代。
5. 同一级别按 `message_time + message_id` 稳定排序；当前 run 新增的种子只在下一级开始时一次性发布，避免并发完成顺序影响结果。

“冻结快照”不是锁住地点表，也不是禁止并行，而是给本次 run 记录一个只读 `base_snapshot_id`。同一层的所有任务读取相同版本；整层完成并汇总后，调度器才产生下一层可见的内部 seed snapshot：

```text
S0：上次已发布地点图，只读
→ 强证据任务并行计算
→ 汇总和实体物化
S1：S0 + 本轮强证据种子，只读
→ 普通地点任务并行计算
→ 汇总和实体物化
S2：S1 + 本轮普通地点结果，只读
→ 弱指代任务并行计算
```

这样并行任务完成的先后不会改变它们读到的先验。这里的“发布”仅指在本次分析 run 内形成下一阶段可读取的版本，不等于立即把中间结果发布给最终用户。

#### 7.1.2 单条 PlaceMention 的 workflow

每条 mention 使用同一个 `ResolutionState`：

```text
mention 与 Bundle 上下文
→ 读取内部候选
→ 构造约束
→ 判断是否需要地图候选
→ 候选特征补全
→ 统一打分与校准
→ 阈值决策
```

`ResolutionState` 至少包含：

- `mention_id/subject_id/speaker_id/time_range`。
- 原文 span、mention 类型、relation 和 assertion mode。
- `candidate_place_ids` 及每个候选的来源。
- 城市、行政区、人物、时间、先行地点等约束及其证据 message ID。
- Provider 请求、响应哈希和重试次数。
- 候选特征、冲突、分数和最终 decision。
- workflow、特征、地图数据和校准器版本。

具体节点如下：

| workflow 节点 | 做什么 | 实现方式 | 是否 ReAct |
|---|---|---|---|
| `load_internal_candidates` | 查询历史快照、当前 run 种子地点、人物别称和 Bundle 内先行地点 | SQL/索引查询 | 否 |
| `build_constraints` | 从已提取字段构造人物、时间、城市、行政区、类型和先行地点约束 | 纯规则；输入来自 Bundle 模型结果 | 否 |
| `should_query_map` | 判断内部候选是否已足够，以及 mention 是否具有可查询词面 | 条件分支 | 否 |
| `query_map_candidates` | 对命名 POI 或地址执行有上限的 `search_poi/geocode` | 直接调用 MapProvider | 否 |
| `enrich_candidate_features` | 计算名称、类型、距离和冲突；只对初筛后的少量候选按需查询路线 | 纯函数加受控 Route API | 否 |
| `score_candidates` | 计算特征分、校准置信度并稳定排序 | 可版本化打分器 | 否 |
| `decide_resolution` | 根据 top score、margin 和硬冲突决定链接、暂定或未解析 | 确定性阈值 | 否 |

地图调用也不是自由循环。例如命名 POI 最多执行预先定义的查询序列：

```text
search_poi(name, city_hint, near)
→ 无结果时去掉 near 扩大到 city_hint
→ address 类型仍无结果时调用一次 geocode
→ 停止并保持未解析
```

查询次数、退化条件和停止条件都写在配置中。Provider 超时只产生可重试状态，不允许模型临时发明新查询策略。

#### 7.1.3 LLM 在消歧中的边界

V1 不在上述节点中使用 ReAct。确实存在候选非常接近、需要重新阅读措辞的情况时，可以以后增加一个可选 `semantic_candidate_judge`：它一次性接收当前 Bundle、最多若干个候选及已经计算好的地图事实，只输出结构化的 `candidate_text_fit` 和对应 message ID。

这个可选节点：

- 不能调用地图、数据库或路线工具。
- 不能创建输入中不存在的候选。
- 不能直接作出最终链接决定。
- 只贡献一个有上限的特征，最终仍由校准分数和 margin 决策。

因此它是 bounded LLM judge，不是 ReAct Agent。只有离线评测证明它比现有特征有稳定增益时才加入。

#### 7.1.4 workflow 的终态

一次消歧只能进入以下终态之一：

| 终态 | 条件与含义 |
|---|---|
| `linked` | 候选超过自动阈值和领先 margin，链接已有或地图地点 |
| `provisional` | 有较强首选但证据还不足，只能低权重使用 |
| `new_semantic_place` | “公司”“家”等个人锚点证据明确，但没有也不需要精确地图实体 |
| `ambiguous` | 保留多个候选，等待更多消息或人工处理 |
| `unresolved` | 无合法候选或上下文不足，不猜地点 |
| `rejected` | 实际为机构、品牌、引用误归属或其他非空间用法 |

workflow 节点必须幂等保存输入哈希和版本。重跑相同输入应得到相同候选、分数和终态；人工确认通过独立 resolution 记录覆盖自动决策，不改写原始 PlaceMention。

#### 7.1.5 候选并行与写竞争

一个 mention 有多个候选时，可以并行执行候选级的特征补全，但不能让每个候选独立作出决策或修改 PlaceEntity。执行结构是标准 fan-out/fan-in：

```text
候选生成与廉价初筛
→ Candidate A ─┐
→ Candidate B ─┼─ 并行补全名称、类型、地理和路线特征
→ Candidate C ─┘
→ 等待本批候选完成或到达统一 deadline
→ reducer 统一排序、校准并计算 top-second margin
→ 生成唯一 ResolutionProposal
→ EntityMaterializer 事务写入
```

候选 worker 只允许：

- 读取该阶段冻结的 snapshot。
- 读取缓存的 Provider 数据或执行受并发上限控制的 Provider 请求。
- 写入自己唯一的 `PlaceCandidate` 结果。

候选 worker 不允许：

- 修改 PlaceEntity 或 PlaceAlias。
- 宣布自己获胜或提前提交链接。
- 根据其他候选的完成顺序改变分数。
- 把请求失败当成候选不匹配；缺失特征必须记为 `unknown`。

并行度不是“地图返回多少候选就启动多少个 LLM”。先用名称、城市、类型和直线距离做廉价初筛，再只对配置上限内的候选执行路线等昂贵操作。若以后启用 `semantic_candidate_judge`，也应把初筛后的候选放进同一次比较请求，而不是每个候选各调用一个模型，否则独立生成的置信度不可直接比较。

fan-in reducer 使用稳定的 `candidate_key` 作为同分 tie-breaker，并且只在统一 deadline 后计算 margin。数据库至少设置以下幂等约束：

```text
UNIQUE(run_id, mention_id, candidate_key)  -- 每个候选只有一份结果
UNIQUE(run_id, mention_id)                 -- 每条 mention 只有一个自动 resolution
UNIQUE(provider, provider_place_id)        -- 同一地图实体不会重复物化
```

不同 mention 仍可能同时提出创建同一个新地点，例如两段对话都提到此前不存在的“新单位”。因此单条 mention workflow 最后只生成不可变 `ResolutionProposal`；真正创建、合并或更新 PlaceEntity 由 `EntityMaterializer` 完成。它先按 provider ID、人物、角色、有效时间和地点相似度对 proposal 分组，再按分组键加事务锁或使用 compare-and-set。它可以按人物或候选实体分片并行，但同一分组在逻辑上只有一个 writer。

因此这里有两种不同的“竞争”：

- 候选的分数竞争是预期行为，由 reducer 用 top score 和 margin 解决。
- 并发写竞争是工程问题，由只读 snapshot、阶段 barrier、唯一约束和单 writer materialization 消除。

### 7.2 PlaceCandidate 字段

| 字段 | 说明 |
|---|---|
| `mention_id` | 来源提及 |
| `candidate_key` | 已有 place ID 或 provider + provider_place_id |
| `candidate_source` | `existing_place/dialogue_anchor/map_provider/new_semantic` |
| `canonical_name` | 候选标准名称 |
| `place_type` | 住宅、办公、餐饮、商场、医院、交通等 |
| `latitude/longitude` | 可为空 |
| `uncertainty_radius_m` | 坐标不确定半径 |
| `address_components` | 国家、省市区、道路等结构 |
| `provider_payload_hash` | 地图响应审计哈希，不默认保存不必要的完整敏感响应 |
| `feature_scores` | 各项消歧特征 |
| `contradictions` | 冲突原因 |
| `raw_score/confidence` | 未校准分数和校准后置信度 |
| `rank` | 当前提及下候选排名 |

### 7.3 候选打分

```text
raw_entity_score = clamp(
    0.24 × name_similarity
  + 0.18 × type_compatibility
  + 0.16 × geographic_consistency
  + 0.14 × route_continuity
  + 0.12 × dialogue_coreference
  + 0.10 × personal_prior
  + 0.06 × source_quality
  - contradiction_penalty,
  0, 1
)
```

字段计算方法：

- `name_similarity`：标准化别称精确匹配为 1；拼音、简称、编辑距离和地图别名综合计算。
- `type_compatibility`：“公司”与办公楼兼容，“吃饭”与餐饮 POI 兼容；类型冲突可降为 0。
- `geographic_consistency`：与明确城市、行政区、已确认锚点和位置分享的一致程度。
- `route_continuity`：前后已知地点与候选之间的地图耗时是否符合消息时间间隔。路线可行只用于排除和降权，不能证明发生过移动。
- `dialogue_coreference`：候选是否是当前对话中“这里”“那边”“楼下”的最近合法先行词。
- `personal_prior`：人物过去将同一别称链接到该地点的可靠比例，最高封顶为 0.8，避免早期误判永久自我强化。
- `source_quality`：结构化坐标、人工确认、地图唯一结果和纯文本候选依次降低。
- `contradiction_penalty`：行政区冲突、物理上不可达、地点类型冲突和同时位置冲突，每项记录原因。

实体分数同样需要使用标注集校准。自动决策默认要求：

```text
自动链接：top_confidence >= 0.78 且 top - second >= 0.15
暂定链接：top_confidence >= 0.55 且 top - second >= 0.08
保持歧义：其他情况
```

暂定链接可以参与候选展示和低权重聚合，但不能被当作精确地址事实。阈值必须通过验证集确定，不把本文默认值视为最终经验真理。

### 7.4 “家”和“公司”的特殊处理

`家`和`公司`默认创建人物专属语义实体：

```text
person:{person_id}:home:1
person:{person_id}:workplace:1
```

以下表达可以成为同一实体别称：

```text
公司、单位、办公室
家、我家、家里
```

但合并前必须保证人物视角一致：

```text
妈妈说“我家”      → 妈妈的家
用户说“我家”      → 用户的家
用户对妈妈说“你家” → 妈妈的家
```

一个人可以拥有多个 `home` 或 `workplace` 实体。搬家、换工作、两地居住不能通过覆盖旧坐标处理，而应使用有效时间区间和竞争角色分数。

### 7.5 模糊坐标和隐私

- 未确认家庭住址默认只保留语义实体或较粗 geohash，不向地图 Provider 发送完整聊天上下文。
- 地图请求只包含完成候选查询所需的最少地点文字、城市和附近锚点。
- 多个互斥候选不得通过平均经纬度生成一个虚假的中心点。
- 同一真实地点的多次带误差坐标可以使用加权 medoid 计算代表点，`uncertainty_radius_m` 使用加权 90% 分位距离。
- 精确家庭坐标、工作地址和医疗地点属于敏感字段，必须支持加密、导出隐藏和删除后重算。

## 8. PlaceEntity 数据模型

地点实体包含权威字段和派生字段。权威字段由确认或消歧结果给出；活动权重不得写回权威字段。

### 8.1 身份与地理字段

| 字段 | 产生方式 |
|---|---|
| `id/project_id` | UUID 和项目外键 |
| `owner_person_id` | 个人锚点所属人物；公共 POI 可为空 |
| `canonical_name` | 人工确认优先，否则选最高置信别称或地图标准名 |
| `place_type` | 候选类型证据加权投票；冲突时保留分布 |
| `role_distribution` | `home/workplace/social/food/shopping/transit/medical/travel/other` 概率分布 |
| `geo_resolution` | `semantic_only/district/area/poi/address/coordinate` |
| `latitude/longitude` | 确认坐标或同一地点坐标观测的加权 medoid |
| `uncertainty_radius_m` | 坐标观测到代表点的加权 90% 分位距离 |
| `address_components` | 地图 Provider 或人工确认的结构化地址 |
| `provider_refs` | Provider 名、place ID、数据版本，不把 Provider ID 当内部主键 |
| `valid_from/valid_to` | 搬家、换工作等地点角色的有效区间 |
| `status` | `provisional/active/ambiguous/superseded/rejected` |

个人地点图是 MoonlightBox 自己的业务数据，长期保存在项目数据库中。每个名称、地址和坐标观测仍需记录 `source_type/source_ref`，至少区分 `chat_evidence/structured_location/manual/map_provider`；地图 Provider 返回的标准名称、POI ID、地址和坐标属于可拆卸 enrichment，不能成为 PlaceEntity 身份成立的唯一依据。

### 8.2 别称字段

别称单独保存 `PlaceAlias`，包含：

- `raw_alias` 和 `normalized_alias`。
- `alias_kind`：`stable_name/role_anchor/deictic`。
- 使用该别称的 `speaker_id`。
- 指代视角，例如 `self_home`、`second_person_home`。
- 适用时间区间。
- 支持和反对的 mention ID。
- 有效支持质量之和。
- 该别称链接到当前地点的置信度。
- `retrieval_eligible`、确认来源和 alias 版本；用于控制 5.9 的可选历史回查。

同一文本别称可以因人物和时间指向不同地点，不能在项目范围内设置全局唯一。

## 9. 从地点提及形成到访观测

### 9.1 VisitObservation 类型

| `visit_state` | 示例 | 是否计入实际到访 |
|---|---|---|
| `mentioned` | “万象城挺大的” | 否 |
| `planned` | “下午去公司” | 否 |
| `cancelled` | “今天不去公司了” | 否，用于抵消计划 |
| `en_route` | “在去公司的路上” | 只计移动证据 |
| `arrived` | “到公司了” | 是 |
| `present` | “还在公司” | 是 |
| `departed` | “刚从公司出来” | 是，结束停留 |
| `passed_by` | “路过万象城” | 单独统计，不计常规停留 |
| `reported_presence` | “小王说妈妈在公司” | 低权限，不默认计入可靠到访 |

### 9.2 VisitObservation 字段

| 字段 | 说明 |
|---|---|
| `subject_id/place_id` | 人物和地点；地点可暂为空 |
| `mention_ids` | 直接来源提及 |
| `visit_state` | 上述状态 |
| `assertion_mode` | 原陈述权限 |
| `started_at_lower/upper` | 到访或移动开始时间范围 |
| `ended_at_lower/upper` | 结束时间范围，可为空 |
| `purpose_distribution` | 工作、回家、吃饭、社交等受控概率分布 |
| `companion_person_ids` | 明确共同出现的人物，不从关系猜测 |
| `transport_mode` | 明确出现的交通方式；未知为空 |
| `source_authority` | 结构化位置、自述、第三方报告、模型推断等 |
| `verification_status` | `confirmed/observed/self_reported/third_party/inferred/disputed` |
| `confidence` | 到访结论置信度 |
| `contradiction_group_id` | 与同一时间互斥地点的竞争组 |
| `evidence_message_ids` | 原始证据 |
| `analysis_run_id` | 分析版本 |

### 9.3 到访置信度

```text
raw_visit_score = clamp(
    0.30 × mention_confidence
  + 0.25 × entity_confidence
  + 0.15 × subject_confidence
  + 0.15 × time_confidence
  + 0.15 × presence_strength
  - contradiction_penalty,
  0, 1
)
```

`presence_strength` 默认参考：

```text
结构化实时位置分享  1.00
present              0.95
arrived/departed     0.90
en_route             0.65
passed_by            0.55
planned              0.25
mentioned            0.05
cancelled            0.00
```

`source_authority` 单独保存，防止一个高语言置信度的第三方转述伪装成目标人物的直接观测。建议初始权限系数：

```text
可信结构化位置          1.00
目标人物明确自述        0.85
对话双方共同确认        0.80
用户关于目标人物的转述  0.45
其他人物转述            0.35
纯模型推断              0.20
```

最终用于图聚合的证据质量为：

```text
evidence_quality = calibrated_visit_confidence × source_authority
```

### 9.4 合并为 VisitEpisode

多个观测满足以下条件时可以合并为一次到访片段：

- `subject_id` 和 `place_id` 相同。
- 时间区间不矛盾。
- 中间没有可靠证据表明人物去了另一个互斥地点。
- 相邻时间间隔小于地点类型阈值，例如短时 POI 4 小时、工作地点 12 小时、家庭 18 小时。
- 路线时间没有证明两次观测不可能属于同一停留。

一次 VisitEpisode 保存：

- 最早到达范围。
- 最晚在场范围。
- 明确离开范围。
- 停留时长下界和上界。
- 是否为左截断或右截断，即聊天只覆盖停留中间的一部分。
- 合并的 observation ID 和独立证据根。

没有离开证据时不得把“到公司了”延长到下一条任意地点消息。未知停留时长保持未知，不使用人物平均停留时长回填历史事实。

## 10. 带权地点图

### 10.1 图的定义

每个人物拥有自己的 `PersonalPlaceGraph`：

```text
G_person = (V, E)

V：人物知道或历史上涉及的地点实体
E：有证据支持的人物地点转移关系
```

公共 PlaceEntity 可以被多个人物图引用，但节点权重和边权重始终按人物独立计算。

### 10.2 进入聚合的证据权重

对每个 VisitEpisode `i`：

```text
q_i = confidence_i
    × source_authority_i
    × state_factor_i
    × conflict_factor_i
```

其中：

- `state_factor`：arrived/present/departed 为 1；passed_by 为 0.25；en_route 为 0，不作为地点停留；planned/mentioned/cancelled 为 0。
- `conflict_factor`：无竞争为 1；竞争组中按各候选归一化置信度分配；已被可靠反证则为 0。
- 同一独立证据根重复抽取时只保留最高 `q_i`，避免模型重跑或摘要重复增加权重。

### 10.3 PlaceNode 字段及计算

#### 基础统计

| 字段 | 计算 |
|---|---|
| `first_seen_at` | 最早有效地点提及时间，不要求实际到访 |
| `last_seen_at` | 最新有效地点提及时间 |
| `first_visit_at` | 最早 `q_i > 0` 的到访片段时间 |
| `last_visit_at` | 最新 `q_i > 0` 的到访片段时间 |
| `mention_count` | 独立原始消息中的地点提及数，不含模型重跑重复项 |
| `effective_visit_count` | `Σ q_i`，不是简单整数计数 |
| `raw_visit_episode_count` | 实际 VisitEpisode 行数，供审计 |
| `effective_dwell_hours` | `Σ(q_i × known_dwell_hours_i)`；时长未知的片段不参与 |
| `dwell_coverage` | 已知停留时长的有效到访质量 / 总有效到访质量 |
| `evidence_strength` | `1 - Π(1 - min(q_i, 0.95))`，同根证据只算一次 |

#### 访问占比

为了避免少量数据导致 100% 极端权重，使用带 Dirichlet 先验的收缩估计：

```text
visit_share_p = (effective_visit_count_p + alpha × prior_p)
              / (Σ effective_visit_count + alpha)
```

初始 `alpha = 3`。`prior_p` 不根据地点名称猜测；已有确认的 home/workplace 可以使用人物角色先验，否则各 active 地点均分。保存 `alpha` 和 prior 版本。

#### 停留占比

仅在 `dwell_coverage >= 0.35` 时计算：

```text
dwell_share_p = (effective_dwell_hours_p + beta × dwell_prior_p)
              / (Σ effective_dwell_hours + beta)
```

初始 `beta = 6 小时`。覆盖不足时字段为空，最终活动权重重新归一化其他分量，不得用零表示“没有停留”。

#### 新近度

```text
recency_p = Σ(q_i × exp(-ln(2) × age_days_i / half_life_p)) / Σ q_i
```

默认半衰期：

```text
home/workplace  180 天
日常餐饮购物     60 天
交通地点         30 天
旅行临时地点     14 天
未知类型         60 天
```

地点角色尚不稳定时使用未知类型半衰期，避免用待预测角色反过来强化自身。

#### 时间分布

时间分布使用项目时区，形成 `7 × 24` 小时槽位。一个到访区间按与每个小时槽的重叠时长分配权重；只有时间点的观测只投到对应槽，并降低到 0.5 权重。

```text
slot_mass[p,d,h] = Σ(q_i × overlap_fraction_i,d,h)

P(place=p | weekday=d, hour=h)
  = (slot_mass[p,d,h] + gamma × global_prior[p])
  / (Σ_place slot_mass[place,d,h] + gamma)
```

初始 `gamma = 2`。同时保存：

- `hourly_mass[168]`：原始加权槽质量。
- `place_given_time[168]`：该时间槽人物位于本地点的收缩概率。
- `temporal_reliability[168] = min(1, supporting_quality / 8)`。

低可靠度时间槽以后只能作为弱先验。

#### 活动场权重

`activity_weight` 表示地点在人物正常活动范围中的总体权重，不等同于“人物现在位于这里的概率”。

先获得可用分量并重新归一化：

```text
base_activity_p = weighted_mean(
  visit_share_p,       weight 0.40,
  dwell_share_p,       weight 0.30 when available,
  recency_p,           weight 0.15,
  evidence_strength_p, weight 0.15
)

activity_weight_p = base_activity_p / Σ base_activity
```

如果停留覆盖不足，0.30 权重按比例分配给其余三个分量。`activity_weight` 不使用情绪重要度，避免一次重大但偶发的医院到访变成日常热点。

#### 地点显著性

`salience_score` 与活动权重分开，表示地点对人物的主观或人生意义：

```text
salience_score = clamp(
    0.35 × max_event_importance
  + 0.25 × emotional_association
  + 0.20 × relationship_association
  + 0.20 × evidence_strength,
  0, 1
)
```

这些输入只能来自已经通过证据复核的历史事件和关系经历。一个地点可以低频但高显著，例如婚礼场地或医院。

#### 角色分布

对每个角色 `r`：

```text
role_raw[p,r] =
    0.50 × explicit_role_evidence
  + 0.20 × schedule_likelihood
  + 0.15 × dwell_pattern_likelihood
  + 0.15 × recurrence_likelihood
```

- `explicit_role_evidence`：目标人物明确说“我家”“公司”或人工确认。
- `schedule_likelihood`：例如工作日白天对 workplace 的似然；这里只是弱特征。
- `dwell_pattern_likelihood`：夜间长停留对 home 的弱支持。
- `recurrence_likelihood`：跨周重复出现程度。

对 `role_raw` 做 softmax 后得到 `role_distribution`。自动激活 `primary_role` 默认要求 top >= 0.75 且领先第二名 >= 0.20；否则为空。任何时间模式都不能单独把一个地点判定为家或公司。

### 10.4 PlaceEdge 字段及计算

边是有方向的，`家 → 公司` 和 `公司 → 家` 分开保存。

#### 如何形成转移证据

对按时间排序的 VisitEpisode `(A, B)`，只有满足以下条件才形成转移候选：

- A 和 B 地点不同。
- A 有离开/最后在场上界，B 有到达/首次在场下界。
- 中间没有第三个可靠地点。
- 时间间隔小于配置上限，默认 12 小时。
- 地图最短可行耗时不超过观测间隔上界；否则进入冲突组。

边证据质量：

```text
transition_quality = min(q_A, q_B)
                   × temporal_link_confidence
                   × route_feasibility
```

地图路线只提供 `route_feasibility`，不能单独创建边。

#### 边字段

| 字段 | 计算或来源 |
|---|---|
| `from_place_id/to_place_id` | 有向地点对 |
| `raw_transition_count` | 独立转移候选数 |
| `effective_transition_count` | `Σ transition_quality` |
| `first_transition_at/last_transition_at` | 首末有效转移时间 |
| `time_slot_mass[168]` | 按 A 的离开时间槽累计质量 |
| `P(to|from,slot)` | 对同一 from 和时间槽做 Dirichlet 收缩归一化 |
| `observed_duration_min/max` | 有明确离开和到达证据时的观测范围 |
| `observed_duration_median` | 对可靠时长使用质量加权中位数 |
| `provider_duration_seconds` | 地图 API 基准时长，按交通方式和查询版本保存 |
| `distance_meters` | 地图路线距离；语义地点无坐标时为空 |
| `transport_distribution` | 明确交通方式的质量加权分布，未知不猜测 |
| `recency` | 与节点相同的指数衰减方法，默认半衰期 60 天 |
| `evidence_ids` | 转移两端 VisitEpisode 及原消息 ID |
| `confidence` | `1 - Π(1 - transition_quality)`，同根证据去重 |

转移概率计算：

```text
P(to=j | from=i, slot=s)
  = (edge_mass[i,j,s] + eta × destination_prior[j])
  / (Σ_k edge_mass[i,k,s] + eta)
```

初始 `eta = 2`。样本不足时以目标地点总体 `activity_weight` 作为弱先验；同时保存 `transition_reliability = min(1, outgoing_mass / 8)`，防止把先验概率展示为已学习规律。

### 10.5 图快照字段

`PersonalPlaceGraphSnapshot` 包含：

- `project_id/person_id`。
- `analysis_run_id`。
- `graph_version`。
- `cutoff_message_id/cutoff_at`。
- `timezone`。
- 节点 ID、节点派生指标和使用的参数版本。
- 边 ID、边派生指标和使用的地图数据版本。
- 未解析 mention 数、歧义实体数和冲突数。
- 数据覆盖日期范围。
- 图级可靠度与警告。
- 内容哈希，保证相同输入可重复产生相同快照。

### 10.6 个人活动热图投影

热图不是新的事实表，也不是连续 GPS 轨迹。它是 `PersonalPlaceGraphSnapshot` 在人物、时间范围和隐私级别下的一种可重建视图；所有热点都必须能回到 PlaceEntity、VisitEpisode 和原消息证据。

#### 10.6.1 默认热度

全历史默认使用 10.3 已计算的 `activity_weight`，不使用简单 mention 次数，也不使用 `salience_score`。这样一次重大但偶发的医院事件不会变成日常活动中心。

用户选择时间范围、星期或小时后，在筛选后的 VisitEpisode 上重新计算同一套活动权重：

```text
heat_raw[p, filter] = activity_weight_p(filter)
```

颜色渲染可以使用平方根拉伸，避免最高频的家和公司完全淹没其他地点：

```text
heat_display_p = sqrt(heat_raw_p / max_place(heat_raw))
```

`heat_display` 只控制颜色，tooltip 和详情必须展示原始 `activity_weight`、`effective_visit_count`、已知停留时长覆盖率和证据可靠度。不同人物或不同筛选条件的颜色默认各自归一化，因此不能仅凭颜色跨图比较；需要比较时提供统一 scale 模式。

#### 10.6.2 地理精度与隐私

地点按 `geo_resolution` 分层展示：

| 地理精度 | 展示方式 |
|---|---|
| `coordinate/poi/address` | 进入热力点层；高缩放时叠加可点击 Place marker |
| `area` | 使用较大半径的半透明区域层，不伪装成精确点 |
| `district` | 展示行政区面或粗粒度区域，不把区中心当作实际到访点 |
| `semantic_only` | 不放到地图上，进入“未定位地点”侧栏 |

敏感地点的模糊必须由后端完成，前端不能先收到精确家庭或医疗坐标再做视觉遮盖。后端根据项目权限返回精确点、量化网格中心、粗粒度区域或完全不返回坐标。每条热力数据包含 `geo_resolution/uncertainty_radius_m/privacy_level/coordinate_crs`。

热力核半径至少覆盖地点不确定范围。V1 可以把精确 POI、粗粒度 area 和 district 拆成不同图层，分别使用固定米制半径；不能把所有点使用同一个像素半径，否则缩放地图时会改变热点代表的实际地理范围。

#### 10.6.3 后端接口

建议新增：

```text
GET /api/projects/{project_id}/people/{person_id}/place-heatmap
```

查询参数：

```text
snapshot_id
from/to
weekdays
hour_start/hour_end
metric=activity|visits|dwell
min_confidence
privacy_level
normalization=relative|fixed
```

`activity` 是默认产品视图；`visits` 和 `dwell` 是审计视图，并在停留覆盖不足时显示警告。

响应主体使用 GeoJSON `FeatureCollection`，方便直接交给 Loca 2.0：

```json
{
  "type": "FeatureCollection",
  "properties": {
    "snapshot_id": "graph_snapshot_1",
    "metric": "activity",
    "located_mass": 0.82,
    "unlocated_mass": 0.18,
    "normalization": "relative",
    "data_warning": null
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [121.45, 31.20]},
      "properties": {
        "place_id": "person:mother:workplace:1",
        "display_name": "公司",
        "primary_role": "workplace",
        "heat_raw": 0.34,
        "heat_display": 1.0,
        "effective_visit_count": 12.7,
        "evidence_strength": 0.91,
        "geo_resolution": "area",
        "uncertainty_radius_m": 800,
        "privacy_level": "blurred",
        "coordinate_crs": "GCJ-02"
      }
    }
  ],
  "unlocated_places": []
}
```

`located_mass/unlocated_mass` 必须展示给用户，防止地图看起来完整但实际大量地点没有坐标。API 缓存键包含 graph snapshot、全部过滤条件、隐私级别和聚合器版本。

#### 10.6.4 前端展示

MoonlightBox 当前前端是 React + Vite + Mantine。V1 使用高德 JS API 2.0 作为底图、Loca 2.0 `HeatMapLayer` 作为活动热力层；Loca 2.0 接受 GeoJSON，并支持以地理单位聚合热力。页面至少包含：

- 人物选择器和图快照版本。
- 全部时间、自定义范围、工作日/周末、小时范围过滤。
- `活动范围/到访次数/已知停留`指标切换；默认活动范围。
- 热力层、地点标记、转移边三个独立开关；转移边默认关闭，避免把热点图误读成连续轨迹。
- 数据覆盖、已定位质量占比、未解析和冲突警告。
- 未定位地点侧栏。
- 点击地点后的详情抽屉：角色、有效时间、原始/显示权重、最近到访、置信度和证据入口。

视觉层级建议：

```text
低缩放：城市/区域级活动场
中缩放：米制半径热力层
高缩放：热力淡出，展示 Place marker 和不确定范围
选中地点：按需展示与该地点相关的有向转移边
```

前端只负责渲染后端已脱敏的数据，不计算地点事实或活动权重。生产环境使用高德 JS API 的 Web Key，并按官方安全方案通过代理保护安全密钥。

## 11. 冲突、竞争和人工确认

### 11.1 同时地点冲突

同一人物在重叠时间被解析到物理上互斥的两个地点时：

- 不删除任一证据。
- 创建 `contradiction_group_id`。
- 根据来源权限、时间精度、实体置信度和路线可行性计算竞争质量。
- 未形成明确赢家时，双方按归一化质量进入低权重聚合。
- 人工确认后创建 resolution 记录，新快照应用 resolution，旧快照不变。

### 11.2 地点合并

仅当以下证据充分时自动合并 PlaceEntity：

- 共享稳定 provider place ID；或
- 高置信坐标重叠、类型兼容且别称长期共指；或
- 人工确认。

合并使用 `superseded_by_id`，不改写原 mention。两个个人语义地点即使距离很近也不能只凭坐标自动合并，例如“家”和“公司”可能位于同一园区。

### 11.3 地点拆分

当“公司”在不同时间对应明显不同城市、地址或路线模式时，创建新 PlaceEntity，并按有效时间重新分配 mention。拆分后重算所有 VisitEpisode 和图快照。

### 11.4 人工确认界面最小需求

- 展示原消息及必要上下文。
- 展示人物归属、时间、移动阶段。
- 在地图上展示候选但默认模糊敏感坐标。
- 支持选择已有地点、创建新地点、保持未知、合并和拆分。
- 每次操作填写简短原因并形成不可变 resolution 记录。

## 12. 与现有 MoonlightBox 管线的集成

空间分析不应塞入现有 `EventNode`。建议新增独立模块：

```text
backend/moonlightbox/spatial/
  models.py
  episodes.py
  retrieval.py
  bundle_analysis.py
  time_resolution.py
  candidate_generation.py
  entity_resolution.py
  visits.py
  graph_builder.py
  map_provider.py
  jobs.py
  schemas.py
  router.py
```

建议新增表：

```text
spatial_analysis_runs
spatial_conversation_episodes
spatial_episode_candidates
spatial_analysis_bundles
place_mentions
place_candidates
place_entities
place_aliases
place_provider_enrichments
place_resolutions
visit_observations
visit_episodes
place_graph_nodes
place_graph_edges
place_graph_snapshots
```

处理时机：

1. `ImportSource` 确认、消息和媒体语义导入完成。
2. 创建独立 `spatial_analysis_run`，按配置生成稳定 ConversationEpisode 清单。
3. 为所有 Episode 生成或复用 embedding。
4. 执行分时间区间的空间 query bank 检索，强制加入结构化位置 Episode；已有事件结果只作为可关闭的召回加分提示。
5. 扩展相邻 Episode，合并成不重复的 SpatialAnalysisBundle。
6. 对每个 Bundle 运行一次局部空间分析，生成 PlaceMention 和 VisitObservation 候选。
7. 在整个 import 范围统一做实体消歧，避免 Bundle 内局部决定长期地点身份。
8. 可选：评测确认有必要时，使用满足 5.9 资格的别称做一次有界历史回查，只分析新增 Bundle；V1 默认跳过。
9. 合并最终 VisitObservation 和 VisitEpisode。
10. 在统一地点图表中按 `person_id` 生成个人地点图快照。
11. 事件分析可以引用已解析地点 ID，但不能修改空间分析结果。

增量导入时只重新切分受新消息影响的尾部 Episode，为新增或内容变化的 Episode 建 embedding，再执行检索和局部分析。随后重新处理受影响的别称、指代链、VisitEpisode 和图快照。旧 Episode manifest、分析结果和图快照继续保留。

## 13. 地图 Provider 接口

### 13.1 V1 选择

中国大陆 V1 主 Provider 使用高德地图 Web 服务 API，但业务模型仍只依赖内部 `MapProvider` 协议。高德提供地点关键字/周边/ID 搜索、地理编码、行政区查询、坐标转换、距离测量和多种路线规划，覆盖当前地点消歧所需能力：

- [高德地点搜索](https://developer.amap.com/api/webservice/guide/api-advanced/newpoisearch)
- [高德地理与逆地理编码](https://developer.amap.com/api/webservice/guide/api/georegeo)
- [高德距离测量与路径规划](https://developer.amap.com/api/webservice/guide/api/direction)
- [高德路径规划 2.0](https://developer.amap.com/api/webservice/guide/api/newroute)

腾讯位置服务提供城市/区域、周边、矩形、多边形、POI 详情、路线和距离矩阵等相近能力，可以用同一标注集做离线 Provider 对照评测；V1 不在一次消歧中混合高德和腾讯候选，以免把不同 POI ID、分类体系和排序分数误当成可直接比较。若腾讯在目标人物所在城市的 `candidate recall@5` 明显更高，再整体切换主 Provider，而不是对单条 mention 动态降级。

百度 V1 暂不接入：除第三套 POI ID 和类型体系外，还会引入 BD-09 坐标处理；其公开使用条款也需要单独确认服务器持久化权限。海外地点在 V1 保持 `semantic_only` 或城市级；确有需求后再选择明确允许永久 geocoding 存储的全球 Provider，或者评估自托管 OSM 数据服务。OSMF 的公共 Nominatim 服务限制重度和批量使用，不可作为聊天历史批处理后端。

最终选型仍要用真实数据验证。建议从目标聊天人工标注至少 100 至 200 条命名 POI/地址，分别测量：

- `candidate recall@5` 和 top-1 accuracy。
- 同名 POI 在 city/adcode/near 约束下的区分能力。
- 商场子 POI、医院院区、地铁站出入口等细粒度覆盖。
- 延迟、失败率、配额和实际账单。

### 13.2 内部接口

业务代码不直接出现高德请求参数，只通过协议和标准化 DTO 访问：

```python
class MapProvider(Protocol):
    def search_poi(self, query, *, city_hint, adcode_hint, near, category, limit): ...
    def get_poi(self, provider_place_id): ...
    def geocode(self, address, *, city_hint, adcode_hint, limit): ...
    def reverse_geocode(self, coordinate, *, radius_m): ...
    def measure(self, origins, destinations, *, mode): ...
    def route(self, origin, destination, *, mode): ...
    def convert_coordinate(self, coordinate, *, target_crs): ...
```

V1 消歧优先调用 `measure` 获取少量候选的距离和大致耗时，不保存或处理完整导航 polyline；只有需要公交可达性等信息时才调用 `route`。历史消息不能使用“当前拥堵 ETA”证明当时路线，只能把当前道路距离作为弱可行性约束。

标准化 DTO 必须保留：

- `provider/provider_place_id`，不能把 Provider ID 当内部 PlaceEntity 主键。
- Provider 原始坐标和 `coordinate_crs`；不得混用高德 GCJ-02、百度 BD-09 和 WGS84。
- 名称、地址、行政区、分类、父子 POI、入口点及各字段是否缺失。
- `queried_at/provider_api_version/request_hash/response_hash`。

### 13.3 数据与许可边界

MoonlightBox 始终长期保存自己的个人地点图，包括：内部 PlaceEntity ID、人物归属、聊天中出现的原始别称、语义角色、有效时间、到访观测、图边、权重、置信度和 evidence message ID。这些事实来自用户导入的聊天和系统分析，不是地图 Provider 数据。

地图厂商只负责可选 enrichment。建议独立保存到 `place_provider_enrichments`，包含 Provider POI ID、Provider 标准名称、Provider 地址、Provider 坐标、分类、查询版本和许可状态。删除这张表后，“妈妈有一个公司地点、她何时提到或到访它、它与家的活动关系”等个人地点图仍必须完整，只是可能退化为没有精确地图坐标的 `semantic_only` 地点。

边界示例：

| 数据 | 归属与保存策略 |
|---|---|
| 聊天原文“我到长泰了” | MoonlightBox 长期保存 |
| “长泰”是人物使用的地点别称 | MoonlightBox 长期保存 |
| 妈妈在该地点的到访时间和图权重 | MoonlightBox 长期保存 |
| 用户位置卡片自身携带的坐标 | 按导入源条款和用户授权保存，来源标为 `structured_location` |
| 高德返回的 POI ID、标准地址、分类和坐标 | Provider enrichment，按高德许可决定保存字段和期限 |
| 系统作出的候选选择及置信度 | MoonlightBox 长期保存，但保留所依据 Provider 请求的审计引用 |

Provider 调用要求：

- 先按规范化参数做当前 run 内去重和短期失败重试；原始响应能否跨 run 缓存、缓存多久、哪些字段能持久化，必须服从所购买服务的协议或书面授权。
- 在确认许可前，不把完整 Provider POI 响应复制为 MoonlightBox 的长期 enrichment 库。长期个人地点图照常保存，并且必须能与 Provider enrichment 分离删除和重建。
- 仅保存审计所需的 provider 名、请求时间、版本和哈希；`provider_place_id`、标准地址、坐标等字段是否持久化由许可配置控制。
- 完整聊天、人物真实姓名和无关关系信息不得发送给 Provider。
- Provider 不可用时仍能产生 semantic-only 地点图。
- Provider 变更只重跑候选与地理字段，不重新生成原始 PlaceMention。
- 上线或商业化前必须复核当时有效的[高德开放平台服务协议](https://developer.amap.com/pages/terms/)和账户套餐。普通 API Key 不能被默认理解为允许缓存、索引或生成长期衍生 POI 数据库。

## 14. 示例

假设目标人物“妈妈”的历史消息为：

```text
08:12 妈妈：到公司了
12:05 妈妈：去长泰吃个饭
13:10 妈妈：回单位了
18:42 妈妈：到家了
```

Episode 化后形成的检索文档为：

```text
[08:12][妈妈] 到公司了
[12:05][妈妈] 去长泰吃个饭
[13:10][妈妈] 回单位了
[18:42][妈妈] 到家了
```

空间 query“人物到达、离开或返回某个地方”将该 Episode 召回。系统加入时间相邻 Episode 后形成 Bundle；本例不需要额外邻居。模型只分析这个 Bundle，一次产生四组提及：

```text
公司   subject=妈妈 relation=arrive movement_phase=arrived
长泰   subject=妈妈 relation=go_to movement_phase=planned/en_route 待上下文判断
单位   subject=妈妈 relation=go_to movement_phase=arrived/present 待上下文判断
家     subject=妈妈 relation=arrive movement_phase=arrived
```

第二步实体消歧：

- `公司` 创建 `妈妈的公司#1`，初始 semantic-only。
- `单位` 因人物相同、当天路线连续、别称类型兼容，候选链接到 `妈妈的公司#1`；满足 margin 后确认别称。
- `家` 创建 `妈妈的家#1`，初始 semantic-only。
- `长泰` 查询项目城市中的地图候选。如果存在多个长泰广场，结合公司区域、午饭语义和 65 分钟内往返可行性排序；仍不满足 margin 时保持多个候选。

第三步到访观测：

```text
08:12 arrived 妈妈的公司#1
12:05 planned/en_route 长泰候选
13:10 arrived/present 妈妈的公司#1
18:42 arrived 妈妈的家#1
```

即使长泰没有消歧完成，也可以确认：

- 妈妈上午到过语义上的公司。
- 午间存在一次去某个“长泰”地点的计划或移动表达。
- 下午回到公司。
- 晚上到达语义上的家。

系统不得自动推断：

- 妈妈具体在哪栋办公楼。
- 她一定在长泰吃完饭。
- 13:10 到 18:42 始终留在公司。
- 她从公司直接回家且使用了某种交通工具。

随着更多日期出现相同模式，`妈妈的公司#1` 和 `妈妈的家#1` 的 `effective_visit_count`、时间槽质量和 `activity_weight` 才会逐渐提高。

## 15. 验收标准

### 15.1 标注集

从真实聊天中分层抽取至少以下样本：

- 明确 POI、地址、行政区。
- 家、公司、学校等个人锚点。
- 这里、那里、楼下、附近等指代。
- 问句、否定、计划、取消、假设和转述。
- 同名 POI 和跨城市地点。
- 搬家、换公司和多住所。
- 引用消息、OCR、语音转写和位置分享。
- 物理不可达与同时地点冲突。

### 15.2 指标

- 空间 Episode 候选召回率：人工标注的空间证据所在 Episode 是否进入候选集。
- Bundle 证据覆盖率：支持同一空间结论的消息是否处于同一或相邻可连接 Bundle。
- 地点 span precision/recall/F1。
- subject attribution accuracy。
- assertion mode 和 movement phase macro-F1。
- 时间区间覆盖率和过度精确率。
- 实体消歧 top-1 accuracy、top-k recall。
- 自动链接 precision 和 abstention quality。
- VisitObservation precision，尤其 planned/question 不得误计到访。
- PlaceEntity 错误合并率和错误拆分率。
- home/workplace 角色 precision。
- 相同输入、版本和 Provider 缓存下图快照哈希一致。

初始上线硬门槛建议：

```text
空间 Episode 候选召回率 >= 0.98
Bundle 证据覆盖率 >= 0.98
自动实体链接 precision >= 0.95
实际到访判定 precision >= 0.95
问句/否定/取消误计到访率 <= 1%
跨人物地点归属错误率 <= 0.5%
错误地点合并率 <= 0.5%
```

召回不足可以通过人工复核和后续模型版本改善；错误写入个人活动范围会持续影响未来世界，因此自动写入优先保证 precision。

## 16. 实施顺序

### 阶段一：Episode 检索层

- 空间分析运行、稳定 Episode manifest、Episode embedding 和 query bank。
- 相邻 Episode 扩展、Bundle 去重和人工召回标注工具。
- 只验收 Episode 候选召回率与 Bundle 证据覆盖率，暂不提取地点。

### 阶段二：局部空间分析

- 每个 Bundle 一次结构化模型调用。
- 生成 PlaceMention 和 VisitObservation 候选。
- 验证人物归属、时间、问句、否定、计划、取消和省略地点。

### 阶段三：地点实体与到访片段

- PlaceEntity、PlaceAlias 和 MapProvider 候选；按召回评测决定是否开启一次有界的别称历史回查。
- VisitObservation、冲突组和 VisitEpisode 合并。
- 验证不把计划、问句和转述变成真实到访。

### 阶段四：带权地点图

- 节点和边的证据质量聚合。
- 时间条件分布、活动权重、显著性和可靠度。
- 图快照、重算、合并、拆分和人工修正。

完成以上四阶段并通过历史标注集验收后，才进入 Agent 世界运行时、地点选择和地图探索设计。

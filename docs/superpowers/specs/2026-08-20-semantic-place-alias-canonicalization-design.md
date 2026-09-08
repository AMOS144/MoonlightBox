# 语义地点合并与个人别称归一设计

## 1. 文档目的

本文细化《聊天地点解析与个人带权地点图设计》中的语义地点实体化部分，专门解决以下问题：

```text
家中、到家、回家、在家、家里
```

这些表达在人物、时间和上下文一致时，应当共同链接到一个个人地点实体，而不是分别创建五个
`PlaceEntity`。本文同时规定哪些相似表达绝对不能自动合并，例如“我家”“你家”“妈妈家”以及
“家”“老家”“新家”。

本文不靠维护一个无限扩张的地点字符串词典解决问题，也不允许简单删除“到、回、在、中”等字后
直接合并。主方法是：模型做局部语义拆分，确定人物和地点头；确定性代码验证证据、生成候选、评分
和提交合并；原始消息与 `PlaceMention` 始终保持可追溯。

协议版本暂定为：

```text
semantic-place-canonicalization-v1
personal-place-graph-v2
```

## 2. 当前问题与根因

当前 `_semantic_candidate` 使用以下身份键：

```text
semantic:{owner}:{normalize(raw_text)}
```

因此：

```text
家中   → semantic:person-1:家中
到家   → semantic:person-1:到家
回家中 → semantic:person-1:回家中
```

这实际上把“消息里的表面文字”误当成了“地点身份”。运动动词、方位助词和状态表达进入了实体名，
同一地点被拆成多个节点。后续权重归一化会进一步放大问题：每个错误节点都获得平滑先验，未定位
质量被重复计算，转移边也会出现“家中 → 到家”这种并不存在的地点移动。

当前设计还混淆了三个不同对象：

```text
evidence span     消息中不可改写的证据，例如“回家中”
place head        证据中的地点中心词，例如“家”
place entity      人物世界中的稳定地点，例如“洪欣羽的家#1”
```

必须把三者拆开。

## 3. 目标与系统不变量

实现后必须满足以下不变量：

1. 一个 `PlaceMention` 只记录一次原始证据，不因实体合并而改写。
2. 地点实体身份不由单个原始字符串决定。
3. 私人语义地点至少受 `project_id + owner_person_id + semantic_role + temporal_slot`
   约束。
4. 同一别称可以因说话者、指代对象和时间不同而指向不同地点。
5. 错误合并的代价高于暂时重复；有冲突时保留歧义。
6. 合并地点后，从原始 mention 和 visit 重新计算图权重，不能直接相加旧节点权重。
7. `semantic_only` 地点可以参与个人地点图，但无坐标时不能进入地图几何层。
8. `personal_anchor` 不因字符串看起来像地点就直接发送给地图 Provider。

## 4. 三层对象边界

### 4.1 PlaceMention：不可变证据

`PlaceMention` 表示消息中出现了一个空间表达。建议新增：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `raw_text` | text | 原消息中的完整连续证据，例如“正在回家中” |
| `span_start/span_end` | int | `raw_text` 在原消息中的严格位置 |
| `place_head_text` | text/null | 地点中心词，例如“家” |
| `head_span_start/head_span_end` | int/null | 地点头在原消息中的位置 |
| `head_normalized` | text/null | 确定性 Unicode、空白和标点归一结果 |
| `semantic_role_hint` | enum/null | `home/workplace/school/hometown/family_home/other` |
| `possessor_person_id` | UUID/null | “谁的家/公司”，解析不了则为空 |
| `subplace_text` | text/null | “正大广场的摩天轮”中的“摩天轮” |
| `modifier_tags` | list | `new/old/former/temporary/nearby/inside` 等 |
| `normalization_version` | string | 产生上述结构的规则与 Prompt 版本 |
| `normalization_confidence` | float | 结构拆分置信度，不等于实体链接置信度 |

`raw_text` 和 `place_head_text` 承担不同职责：

```text
消息：我还没到家
raw_text：到家
place_head_text：家
relation：to
movement_phase：arriving
assertion_mode：negated
```

系统仍保存“到家”这个完整证据，但地点候选生成只使用“家”及其人物语义角色。

### 4.2 PlaceAlias：人物使用的称呼

`PlaceAlias` 表示某个人、从某个视角、在某段时间内如何称呼一个地点。它不是地点实体，也不是
简单的全局同义词表。

建议在现有字段上明确以下语义：

| 字段 | 规则 |
| --- | --- |
| `raw_alias` | 保存消息里实际出现的称呼，如“家中”“单位” |
| `normalized_alias` | 只做字符级归一，不负责语义合并 |
| `alias_kind` | `surface/role_anchor/deictic/subplace/provider_name/manual` |
| `speaker_id` | 谁使用了这个称呼 |
| `perspective` | `self/addressee/third_person/shared/unknown` |
| `valid_from/valid_to` | 搬家、换公司后限制别称的有效期 |
| `supporting_mention_ids` | 支持此别称链接的原始证据 |
| `opposing_mention_ids` | 表明此别称可能指向别处的证据 |
| `retrieval_eligible` | 是否允许用于第二遍历史回查 |

`normalized_alias` 相同不代表实体相同。例如两个人都说“家”，应产生两个不同人物作用域的别称。

### 4.3 PlaceEntity：稳定的个人地点身份

语义地点实体应具备稳定的内部身份。建议新增或明确：

| 字段 | 含义 |
| --- | --- |
| `semantic_key` | 内部身份键，例如 `person-1:home:slot-1`，不可由 raw text 直接生成 |
| `owner_person_id` | 地点属于谁的个人语义空间 |
| `semantic_role` | 当前最可信角色，如 `home/workplace` |
| `role_distribution` | 多证据累积后的角色概率 |
| `canonical_name` | UI 展示名；语义身份不依赖它 |
| `valid_from/valid_to` | 此地点承担该角色的有效时间 |
| `geo_resolution` | `semantic_only/district/area/poi/address/coordinate` |
| `status` | `provisional/active/ambiguous/superseded/rejected` |
| `superseded_by_id` | 合并后指向保留实体，不删除旧实体 |

第一处被发现的家可以创建：

```text
semantic_key = person-1:home:slot-1
canonical_name = 家
geo_resolution = semantic_only
```

以后“家中、到家、回家”都链接到这个 ID。获得结构化位置或人工确认后，在同一实体上补充坐标，
不能另建一个“有坐标的家”。

## 5. 局部模型如何拆分地点表达

### 5.1 模型负责什么

局部云模型读取 Bundle，负责：

1. 找到完整空间证据 span。
2. 在 span 内标出地点头。
3. 判断人物主体与地点占有者。
4. 判断语义角色提示、运动阶段、否定、计划和时间。
5. 标出“新、旧、老家、妈妈家”等可能改变实体身份的修饰信息。

模型不负责：

- 直接输出内部 PlaceEntity ID。
- 决定两个历史地点永久合并。
- 编造原文没有的地址或城市。
- 把“家”查询为地图 POI。

### 5.2 Prompt 必须增加的约束

地点提取 Prompt 应加入：

```text
raw_text 必须是原消息中的连续原文证据。
place_head_text 必须是 raw_text 内部的连续子串，表示地点中心词，不包含到达、离开、
正在、回、去、来、在、中、里等动作或方位成分，除非它们本身是专名的一部分。

示例：
“我到家了” → raw_text="到家", place_head_text="家", semantic_role_hint="home"
“正在家中做饭” → raw_text="家中", place_head_text="家", semantic_role_hint="home"
“回老家过年” → place_head_text="老家", semantic_role_hint="hometown"
“去妈妈家” → place_head_text="妈妈家", possessor=妈妈，不得归到说话者自己的家
“正大广场的摩天轮” → place_head_text="正大广场", subplace_text="摩天轮"
```

### 5.3 确定性验证

模型输出后必须执行：

1. `raw_text == message.content[span_start:span_end]`。
2. `place_head_text` 必须能在 `raw_text` 中找到且位置唯一，或模型同时提供合法 head span。
3. `possessor_person_id` 必须属于当前项目参与者。
4. `personal_anchor` 必须有合法 `semantic_role_hint`，否则降为 `other/unknown`。
5. 模型不得将“回老家”的地点头改成“家”，也不得将“妈妈家”改成“家”。
6. 验证失败只拒绝当前 mention，不接受模型改写后的字符串。

规则的作用是验证结构，不是独立完成中文语义解析。

## 6. 人物、视角与地点占有者

合并语义地点前必须先解析两个人物字段：

```text
subject_id     谁正在到达、离开、停留或讨论这个地点
possessor_id   这个私人语义地点属于谁
```

它们可能不同：

| 消息 | subject | possessor | 结果 |
| --- | --- | --- | --- |
| 妈妈：“我到家了” | 妈妈 | 妈妈 | 妈妈的家 |
| 我：“我去妈妈家” | 我 | 妈妈 | 妈妈的家 |
| 妈妈：“你到家了吗” | 我 | 我 | 询问我的家，不证明到访 |
| 妈妈：“去你家吧” | 我或对话对象 | 对话对象 | 不能链接到妈妈的家 |
| “小王说他回家了” | 小王 | 小王 | 第三方低权限报告 |

若 `possessor_id` 不确定，不能仅因地点头都是“家”自动合并。候选可以保留，等待更多上下文。

## 7. 候选生成

对每条语义 mention，按冻结的地点图快照生成候选：

1. 同项目、同 `possessor_id`、角色兼容且时间区间可重叠的 PlaceEntity。
2. 同说话者和视角下，已确认或满足检索资格的 PlaceAlias。
3. 同一 Bundle 中已经由强证据建立的地点锚点。
4. 最近可承接的指代地点，仅用于“那里、楼下、附近”。
5. 没有合适候选时的 `new_semantic_place`。

`personal_anchor` 候选生成不调用地图 Provider。只有以下证据才能提升其地理解析等级：

- 聊天平台结构化位置。
- 明确地址或命名 POI 与语义锚点的共指证据。
- 已确认路线和附近关系形成的稳定约束。
- 人工确认。

## 8. 合并评分与决策

### 8.1 评分特征

语义地点候选评分建议使用：

```text
score =
  0.25 * owner_compatibility
+ 0.20 * semantic_role_compatibility
+ 0.15 * confirmed_alias_match
+ 0.10 * temporal_compatibility
+ 0.10 * dialogue_context_fit
+ 0.08 * subject_route_continuity
+ 0.07 * lexical_head_similarity
+ 0.05 * personal_prior
- contradiction_penalty
```

说明：

- `owner_compatibility`：明确人物相同为 1；明确不同直接否决，不只是扣少量分。
- `semantic_role_compatibility`：“家中”和既有 home 为 1；“老家”和普通 home 不能直接为 1。
- `confirmed_alias_match`：只使用已确认或达到独立证据门槛的别称。
- `temporal_compatibility`：同一时期支持合并；搬家前后的冲突降低分数。
- `dialogue_context_fit`：同一 Bundle 中的指代和路线是否支持。
- `lexical_head_similarity` 权重较低，避免字符串支配身份判断。
- `personal_prior` 必须封顶，防止早期错误持续自我强化。

### 8.2 硬冲突

出现以下任一情况，不允许自动合并：

- `possessor_id` 明确不同。
- 一个是 `home`，另一个明确是 `hometown/family_home`。
- 出现“新家、旧家、以前的公司、搬走”等身份切换证据，时间关系尚未解析。
- 两个已确认 POI 坐标明显冲突，且没有搬迁或多地点角色证据。
- 人工标记为不同地点。
- 候选仅由本次别称回查产生，构成循环证明。

### 8.3 决策门槛

建议：

```text
自动链接：top_score >= 0.86 且 margin >= 0.15，且无硬冲突
暂定链接：top_score >= 0.68 且 margin >= 0.10，且无硬冲突
歧义：有多个接近候选，保存候选集合，不选实体
新建：个人锚点证据明确，但没有兼容候选
未解析：主体、占有者或地点头不足以建立安全身份
```

“同一人物已有唯一 home”可以提供强候选，但不能变成永久唯一约束。人物搬家后允许存在
`home:slot-1` 和 `home:slot-2`。

## 9. “家”类表达的具体规则

### 9.1 默认可合并

人物和时间一致、没有冲突时，以下地点头归入同一个 home 候选：

```text
家、家中、家里、在家、到家、回家、从家出发
```

这里不是把所有表面字符串预先写入地点库。模型先将完整表达拆成 `place_head_text=家`，代码再按
人物、角色和时间链接实体。

### 9.2 默认不可直接合并

| 表达 | 原因 |
| --- | --- |
| 我家 / 你家 | possessor 不同 |
| 妈妈家 / 爸妈家 | family_home，属于特定人物或家庭锚点 |
| 老家 | hometown，不等于当前住所 |
| 娘家 / 婆家 | 家庭关系角色不同 |
| 新家 / 旧家 | 可能代表时间上不同的 home slot |
| 临时住处 / 酒店 | lodging，不应提升为 home |
| 公司宿舍 | workplace/lodging 复合地点，需要独立实体或父子关系 |
| 楼下 / 小区门口 | 相对地点，需要 anchor，不等于家本身 |

### 9.3 “新家”和搬迁

发现“搬家、新家、以前住的地方”时：

1. 不立即把新家与旧 home 合并。
2. 创建 `home_change_candidate`，收集时间、地址、路线和后续称呼证据。
3. 确认后关闭旧实体的 `valid_to`，建立 `home:slot-2.valid_from`。
4. “家”这个别称按时间分别链接到 slot-1 和 slot-2。
5. 无法确定搬迁时间时保留时间区间，不伪造精确日期。

## 10. 别称确认与历史回查

地点链接与别称可用于检索是两个门槛。

单条“到家了”可以基于人物和唯一 home 候选暂定链接，但“家”成为可触发历史回查的别称，必须满足：

- 至少两条来自不同 Bundle 的独立消息；或
- 一条结构化位置/明确地址强证据；或
- 人工确认。

别称回查只能召回历史消息，不能直接给召回结果增加实体链接分数。新消息仍要经过局部语义分析，
防止“用家搜到家，再用搜索结果证明所有家相同”的循环。

## 11. 合并提交与审计

### 11.1 不物理删除旧实体

建议新增 `PlaceEntityMergeDecision`：

| 字段 | 含义 |
| --- | --- |
| `source_place_id` | 被合并实体 |
| `target_place_id` | 保留实体 |
| `decision` | `proposed/accepted/rejected/reverted` |
| `reason_codes` | owner、role、alias、time、manual 等 |
| `supporting_mention_ids` | 支持证据 |
| `opposing_mention_ids` | 冲突证据 |
| `score/margin` | 决策时分数 |
| `run_id` | 哪次分析产生 |
| `reviewed_by/reviewed_at` | 人工审核信息 |

接受合并后：

1. `source.status = superseded`。
2. `source.superseded_by_id = target.id`。
3. 历史 `PlaceMention` 和旧 `PlaceResolution` 不改写。
4. 新查询沿 `superseded_by_id` 找到规范实体。
5. 图投影时把旧、新实体映射到同一 canonical ID。
6. 必须检测并禁止 superseded 链形成环。

这样可以完整撤销合并，也能解释旧版本为什么曾显示多个地点。

### 11.2 并发与固定处理顺序

候选生成可以并行，实体归并必须基于冻结快照并按固定阶段提交：

```text
冻结上次地点图
→ 解析结构化位置和人工确认
→ 解析明确命名 POI/地址
→ 解析人物角色锚点：家、公司、学校
→ 解析相对地点和指代
→ 生成合并提案
→ 单线程/事务化提交规范实体
→ 重建 VisitEpisode 与图快照
```

同一 run 中不能让任务完成先后改变另一个 mention 看到的候选集合。

## 12. 合并后的到访与权重重算

不能这样做：

```text
merged_weight = 家中.activity_weight + 到家.activity_weight
```

两个旧节点的权重已经分别做过平滑和归一化，直接相加会重复先验。正确流程是：

1. 将每条有效 `PlaceResolution` 映射到 canonical PlaceEntity。
2. 按 message/evidence ID 去重 `VisitObservation`。
3. 在 canonical ID 上重新合并连续 `VisitEpisode`。
4. 重新计算有效到访、停留、近期性和证据强度。
5. 对全体 canonical 节点重新归一化活动权重。
6. 生成新的不可变 `PlaceGraphSnapshot`。

合并后应满足：

```text
家中 + 到家 + 回家
→ 一个“家”节点
→ 多个别称
→ 多条原始 mention
→ 按证据聚合后的到访序列
→ 一个活动权重
```

语义节点继续参与全局地点图权重，但没有坐标时只进入 `unlocated_places`。地图几何层在有坐标节点
内部重新归一化显示强度，同时单独展示地理覆盖率。

## 13. 现有数据回填

### 13.1 回填原则

- 不重新导入聊天。
- 不覆盖原始 mention。
- 先 dry-run 输出合并提案，再建立新图快照。
- 回填前备份数据库。
- 默认只自动接受高精度角色锚点合并，复杂搬迁交给人工审核。

### 13.2 回填步骤

```text
读取现有 PlaceMention
→ 用新 Prompt/结构化模型补 place_head、role、possessor、modifier
→ 确定性验证
→ 按 owner + role + temporal compatibility 形成候选簇
→ 生成 merge proposal
→ 自动接受无冲突高分提案
→ 标记 superseded，不删除旧实体
→ 重建 resolution canonical view
→ 重建 visit 和 personal-place-graph-v2 snapshot
→ 对比旧新节点数、权重和未定位质量
```

对于当前示例，预期是：

```text
洪欣羽：家中、到家 → 洪欣羽的家#1
我：回家中、小屋子里 → 不能与洪欣羽的家合并；是否彼此相同仍需证据
庭中 → 不能仅因语义接近居住环境就并入家
老家/妈妈家/新家 → 默认保持独立候选
```

### 13.3 回填报告

每次回填至少输出：

- 原始实体数、规范实体数、合并提案数、自动接受数、待审核数。
- 每个提案的支持与反对 message ID。
- owner/role/time 冲突统计。
- 合并前后活动权重总和与节点排序变化。
- 地理覆盖率变化；语义合并本身不应伪造新坐标。
- 可用于完整回滚的旧 snapshot ID。

## 14. API 与审核界面

地点详情页应展示：

```text
规范地点：洪欣羽的家#1
角色：home
地理状态：semantic_only
别称：家、家中、到家、回家
有效时间：2026-04 起，结束未知
支持证据：N 条
冲突证据：M 条
合并来源：自动 / 人工
```

审核操作：

- “这些称呼是同一个地点”。
- “不是同一个地点”。
- “从某日期起换了新地点”。
- “修改地点占有者”。
- “撤销上次合并”。
- “将语义地点链接到命名 POI/结构化位置”。

人工操作不能修改原消息，只产生版本化 resolution、alias 或 merge decision。

## 15. 测试与验收

### 15.1 必测样例

| 输入 | 预期 |
| --- | --- |
| 我到家了 / 我在家中 / 我正在回家 | 同一人物、同一 home 实体 |
| 我还没到家 | 链接 home，但不产生实际到访 |
| 你到家了吗 | 正确解析 subject；不证明到访 |
| 去妈妈家 | family_home/妈妈，不是说话者的 home |
| 回老家过年 | hometown，不合并当前 home |
| 新家还没收拾好 | 新 home slot 候选，不直接合并旧家 |
| 公司楼下等我 | 楼下为相对地点，以公司为 anchor |
| “《回家》这部电影” | 非空间语境，不产生 PlaceMention |
| 我家和你家很近 | 产生两个不同 possessor 的地点实体 |
| 家中 → 到家 | 合并后不得生成地点转移边 |

### 15.2 指标

第一阶段验收指标：

- 高风险 owner 错误合并：0。
- `home/hometown/family_home` 跨角色错误合并：0。
- 已标注语义地点 pair 的自动合并 precision ≥ 99%。
- “家中/到家/回家”同人物同时间样本召回率 ≥ 95%。
- 合并前后所有原始 evidence message ID 保留率 100%。
- 权重重建后节点权重总和误差小于 `1e-9`。
- 同一输入、同一版本重复运行得到相同 canonical mapping 和图 hash。

## 16. 实施顺序

建议按以下顺序落地：

1. 扩展 `BundlePlaceMention` 与 `PlaceMention` 的地点头、角色、占有者和修饰字段。
2. 更新云端 Prompt，并增加 span/head 的确定性验证测试。
3. 将 semantic candidate key 从 `normalize(raw_text)` 改为内部 identity proposal。
4. 实现 owner、role、temporal 候选生成和硬冲突规则。
5. 增加 `PlaceEntityMergeDecision` 与 canonical ID 解析。
6. 修改 VisitEpisode 和图投影，使其按 canonical ID 重算。
7. 增加 dry-run 回填命令和审核 API。
8. 对现有聊天执行回填，人工检查所有自动合并提案。
9. 发布 `personal-place-graph-v2`，旧 snapshot 保留只读。

## 17. 最终数据流

```text
原始消息：“我还没到家”
  ↓
PlaceMention
  raw_text=到家
  place_head=家
  subject=洪欣羽
  possessor=洪欣羽
  role=home
  assertion=negated
  ↓
候选生成
  洪欣羽的家#1  score=0.93
  洪欣羽的老家   硬角色冲突
  我的家         硬 owner 冲突
  ↓
PlaceResolution → 洪欣羽的家#1
  ↓
PlaceAlias 增加“到家”的证据
  ↓
不生成实际 VisitObservation，因为 assertion=negated
  ↓
图中仍只有一个“家”节点，不产生“家 → 到家”的假边
```

这个结构保证语义地点既能在没有地图坐标时成为人物世界的一部分，又不会因聊天表达变化被拆成
大量虚假节点。

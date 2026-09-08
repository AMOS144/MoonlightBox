# Generative Agents 设计思路与各模块作用

## 1. 文档目的

本文解释 `vendor/generative_agents` 中 Generative Agents 的原始设计思路、运行流程和各个模块的职责。

本文首先回答以下问题：

1. Generative Agents 到底解决什么问题。
2. Agent、世界、记忆、检索、反思、计划和行动之间是什么关系。
3. `Scratch`、`SpatialMemory` 和 `AssociativeMemory` 为什么是三种不同的状态。
4. 一条世界事件如何经过感知、记忆和规划，最终改变 Agent 行动。
5. 这套系统为什么能表现出一定的行为连续性。
6. 它与 GraphRAG、知识图谱和真实世界地图分别有什么区别。

本文只解释原项目，不在主体部分混入 MoonlightBox 的改造方案。原项目中的类名 `Persona` 就是论文所说的
Generative Agent；`AssociativeMemory` 对应论文中的 Memory Stream；`Reverie` 是整个模拟框架。

## 2. 一句话理解 Generative Agents

Generative Agents 不是让大模型在每一步自由决定所有事情，而是把大模型放进一个有明确状态、时间、地图、
记忆和执行规则的模拟器中：

> Agent 在一个持续变化的世界中选择性感知事件，把事件写入个人记忆，从记忆中取回与当前情况相关的内容，
> 维持并修订自己的日程，必要时形成高层反思，然后把文字计划落实为地图中的具体行动。

它的核心不是某一个 Prompt，而是以下闭环：

```text
权威世界状态 Maze
       ↓
Perceive：看见附近空间与事件，并写入记忆
       ↓
Retrieve：找回和当前事件有关的过去事件与想法
       ↓
Plan：维持日程、选择动作、决定是否响应别人
       ↓
Reflect：累计到足够重要的信息后形成高层想法
       ↓
Execute：寻路并执行一个确定的地图动作
       ↓
新的位置、对象状态、人物事件重新进入世界
```

代码中的真实调用顺序是：

```python
perceived = perceive(maze)
retrieved = retrieve(perceived)
plan = plan(maze, personas, new_day, retrieved)
reflect()
execute(maze, personas, plan)
```

因此反思通常不会回头重做本时间步已经生成的计划，而是作为新记忆影响后续时间步。入口位于
[`Persona.move`](../../../vendor/generative_agents/reverie/backend_server/persona/persona.py)。

## 3. 总体架构

系统可以分为四层。

```text
┌──────────────────────────────────────────────────────────────┐
│ 1. 模拟调度层：ReverieServer                                 │
│ 管理统一时间、模拟步数、存档，并依次驱动所有 Agent             │
└───────────────────────────┬──────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. 世界层：Maze                                              │
│ Tile、碰撞、地点层级、对象、人物位置和当前世界事件             │
└───────────────────────────┬──────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. Agent 状态层：Persona                                     │
│ Scratch + SpatialMemory + AssociativeMemory                  │
└───────────────────────────┬──────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. 认知与行动层                                               │
│ Perceive → Retrieve → Plan → Reflect → Execute               │
└──────────────────────────────────────────────────────────────┘
```

这四层的边界很重要：

- `Maze` 保存当前世界实际上是什么样。
- `Persona` 保存某个 Agent 知道什么、记得什么、正在做什么。
- 认知模块根据有限输入更新 Persona，不直接随意改写整个世界。
- `Execute` 把计划转换为路径；模拟调度器再把行动结果写回世界。

## 4. ReverieServer：模拟调度器

`ReverieServer` 是模拟的总控制器，代码位于
[`reverie.py`](../../../vendor/generative_agents/reverie/backend_server/reverie.py)。

它不负责扮演某个人物，而是负责维护所有 Agent 共享的模拟过程。

### 4.1 保存的全局状态

- 模拟开始时间和当前时间。
- 每个 simulation step 对应多少游戏秒。
- 当前 step 编号。
- `Maze` 实例。
- 全部 `Persona` 实例。
- 每个 Persona 当前所在的 Tile。
- 每一步的环境文件与移动结果。

原项目默认一个模拟步可以代表 10 秒。前端完成上一轮移动后写出环境状态，后端读取它，再驱动所有 Agent
产生下一步移动。

### 4.2 每一步做什么

每个时间步大致执行：

1. 读取前端返回的人物最新坐标。
2. 清理上一时间步的临时对象事件。
3. 在 Maze 中更新人物所在 Tile。
4. 把每个人当前的动作事件写入对应 Tile。
5. 依次调用每个 `Persona.move()`。
6. 收集下一格坐标、Emoji、动作描述和对话。
7. 写出 movement JSON 供前端执行。
8. 增加 step 和世界时间。

因此 Generative Agents 仍然是一个离散时间模拟器。Agent 并不是持续运行的独立进程，而是由调度器按时间步
依次调用。

### 4.3 Fork 与存档

一次新模拟从已有模拟目录 fork。保存时会持久化：

- 世界和全局时间。
- 每个 Agent 的三类记忆。
- 人物当前位置。
- movement 与 environment 历史。

这让同一个模拟可以暂停、继续或从某一状态创建分支。

## 5. Maze：权威世界状态

`Maze` 位于 [`maze.py`](../../../vendor/generative_agents/reverie/backend_server/maze.py)，使用固定的二维 Tile
矩阵表示 Smallville。

### 5.1 地点层级

每个 Tile 同时属于一个分层地址：

```text
world → sector → arena → game object
```

例如：

```text
the Ville
  → Isabella Rodriguez's apartment
    → main room
      → bed
```

代码使用冒号拼接成地址：

```text
the Ville:Isabella Rodriguez's apartment:main room:bed
```

这个地址既是语言模型选择地点时使用的符号，也是执行模块查找目标 Tile 时使用的键。

### 5.2 每个 Tile 保存什么

每个 Tile 包含：

- `world`
- `sector`
- `arena`
- `game_object`
- `spawning_location`
- 是否为碰撞区域
- 当前发生在该 Tile 上的 `events`

事件通常是四元组：

```text
(subject, predicate, object, description)
```

例如：

```text
("Isabella Rodriguez", "is", "sleeping", "sleeping")
("...:bed", "is", "used", "being used")
```

### 5.3 正向和反向寻址

Maze 同时支持：

- 从坐标读取地点路径和事件。
- 从地点地址找到属于该地点的全部 Tile。

后者由 `address_tiles` 完成。规划模块输出地点地址，执行模块再用 `address_tiles` 找到候选坐标并寻路。

### 5.4 Maze 的真正作用

Maze 不只是前端背景图，它提供了四种约束：

1. **感知边界**：Agent 只能看到附近 Tile 和同一 arena 内的事件。
2. **行动空间**：Agent 只能选择地图中存在的地点和对象。
3. **物理约束**：碰撞矩阵与路径规划决定人物如何移动。
4. **共享现实**：所有 Agent 都通过同一个 Maze 彼此看见并改变环境。

## 6. Persona：一个 Agent 的容器

`Persona` 位于
[`persona.py`](../../../vendor/generative_agents/reverie/backend_server/persona/persona.py)。它自身没有复杂推理逻辑，
主要作用是组合三类记忆并按顺序调用认知模块。

```text
Persona
├── scratch：身份、当前状态、计划和工作记忆
├── s_mem：已经认识的空间结构
└── a_mem：事件、对话和反思形成的长期记忆流
```

这三类状态不能合并理解成同一种“记忆”。

## 7. Scratch：身份、工作记忆与当前控制状态

`Scratch` 位于
[`scratch.py`](../../../vendor/generative_agents/reverie/backend_server/persona/memory_structures/scratch.py)。

它虽然叫“短期记忆”，实际承担的职责更接近 Agent 的运行状态总表。

### 7.1 身份稳定集

Scratch 保存：

- 姓名、年龄。
- `innate`：先天或核心人格词，例如 friendly、outgoing。
- `learned`：人物背景和较稳定特质。
- `currently`：当前人生阶段或正在关注的事情。
- `lifestyle`：作息和习惯。
- `living_area`：居住地点。

`get_str_iss()` 会把它们组合成 Identity Stable Set，供大量 Prompt 使用。

它的作用是让不同时间、不同认知模块看到相对稳定的同一个人，而不是每次只根据最新消息临时扮演。

### 7.2 感知控制参数

默认参数包括：

- `vision_r = 4`：可见 Tile 半径。
- `att_bandwidth = 3`：一次最多认真感知几个附近事件。
- `retention = 5`：最近多少条事件用于避免重复感知。

这是一种人为施加的注意力瓶颈。Agent 不会把整个世界都塞进大模型上下文。

### 7.3 日程与当前动作

Scratch 同时保存：

- 一天的粗粒度目标 `daily_req`。
- 小时级原始日程 `f_daily_schedule_hourly_org`。
- 已经逐步分解的细粒度日程 `f_daily_schedule`。
- 当前动作地址、描述、开始时间和持续时间。
- 当前动作对应的事件三元组。
- 当前使用的对象及对象事件。
- 已规划路径。

### 7.4 社交临时状态

Scratch 还保存：

- 正在与谁聊天。
- 当前完整对话内容。
- 对话结束时间。
- 与同一人物再次聊天前的缓冲计数。

这些字段防止 Agent 每个模拟步都重新发起相同对话。

### 7.5 反思触发状态

Scratch 保存累计重要性阈值：

- `importance_trigger_max = 150`
- `importance_trigger_curr`
- 自上次反思以来的新元素数量
- 新近度、相关度和重要性权重

新事件会减少剩余阈值，累计到零时触发反思。

## 8. SpatialMemory：Agent 主观认识的地点目录

`SpatialMemory` 在代码中叫 `MemoryTree`，位于
[`spatial_memory.py`](../../../vendor/generative_agents/reverie/backend_server/persona/memory_structures/spatial_memory.py)。

其结构与 Maze 的地点层级一致：

```json
{
  "the Ville": {
    "Hobbs Cafe": {
      "cafe": [
        "refrigerator",
        "cafe customer seating",
        "piano"
      ]
    }
  }
}
```

### 8.1 它解决什么问题

语言模型要为“睡觉”“吃饭”“买东西”等动作选择位置，但不能把整个 Tile 地图放入 Prompt。

SpatialMemory 提供一个压缩后的可行动空间：

- 当前世界有哪些可进入 sector。
- 某个 sector 有哪些 arena。
- 某个 arena 有哪些可交互对象。

计划模块逐层选择 sector、arena 和 game object，最终形成可执行地址。

### 8.2 它如何增长

感知模块遍历视野范围内的 Tile，把新看到的 world、sector、arena 和 game object 增量加入树中。

因此 Maze 是客观世界，SpatialMemory 是某个 Agent 已经见过或预先知道的世界子集。

### 8.3 它不是什么

SpatialMemory 不是：

- 到访历史。
- 热力图。
- 地点权重图。
- 带时间的轨迹。
- 对真实 POI 的实体消歧系统。

它只回答“这个 Agent 知道哪些可用地点和对象”，不回答“Agent 多常去这里”或“聊天中的家在哪里”。

## 9. AssociativeMemory：长期记忆流

`AssociativeMemory` 位于
[`associative_memory.py`](../../../vendor/generative_agents/reverie/backend_server/persona/memory_structures/associative_memory.py)，
对应论文中的 Memory Stream。

它将经历保存为一个按时间增长的 ConceptNode 集合。

### 9.1 三类 ConceptNode

| 类型 | 含义 | 示例 |
|---|---|---|
| `event` | 感知到的世界事件 | Isabella 正在睡觉；Klaus 走进咖啡馆 |
| `chat` | 一段人物对话 | Isabella 与 Maria 讨论派对 |
| `thought` | 从事件或对话归纳出的想法 | Isabella 认为 Maria 可能愿意帮助组织活动 |

### 9.2 ConceptNode 字段

| 字段 | 作用 |
|---|---|
| `node_id` | 节点唯一 ID |
| `type` | `event/thought/chat` |
| `depth` | 思想的抽象层级；事件和聊天通常为 0 |
| `created` | 创建时间 |
| `expiration` | 可选过期时间 |
| `last_accessed` | 最近被检索的时间，用于新近度 |
| `subject/predicate/object` | 简化的事件三元组 |
| `description` | 自然语言描述，也是主要语义内容 |
| `embedding_key` | 用来查嵌入向量的文本键 |
| `poignancy` | 对当前 Agent 的重要性，通常为 1 到 10 |
| `keywords` | 快速按人物或对象召回的关键词 |
| `filling` | 支撑该节点的其他节点 ID 或完整对话 |

### 9.3 `filling` 为什么重要

反思形成的 `thought` 不只是一个孤立摘要。其 `filling` 会保存支持这条想法的事件或思想节点 ID。

```text
事件 node_12 ─┐
事件 node_27 ─┼→ thought node_41
想法 node_33 ─┘
```

这使记忆不只是平铺列表，而是形成一张浅层的证据依赖图。`thought.depth` 会根据其证据节点深度增加。

但它不是完整知识图谱：三元组没有严格本体，实体没有全局消歧，`filling` 也主要服务于反思可追溯性。

### 9.4 存储和索引

原实现维护：

- `id_to_node`
- 按时间倒序的 event、thought、chat 列表
- keyword 到 event/thought/chat 的倒排索引
- keyword 强度计数
- 文本到 embedding 的字典

因此它同时支持按关键词快速找相关记忆，以及按向量计算语义相关度。

## 10. Perceive：选择性地把世界变成个人经历

感知模块位于
[`perceive.py`](../../../vendor/generative_agents/reverie/backend_server/persona/cognitive_modules/perceive.py)。

### 10.1 感知空间

系统先读取 `vision_r` 范围内的 Tile，将看到的地点层级和对象写入 SpatialMemory。

### 10.2 感知事件

事件感知受到三层限制：

1. 只看视野范围内的 Tile。
2. 只处理与自己位于同一 arena 的事件。
3. 按距离排序，只选择最近的 `att_bandwidth` 个事件。

### 10.3 去重

系统读取最近 `retention` 条 event 的三元组。如果一个事件刚刚已经被记录，就不会在每个 10 秒时间步重复写入。

### 10.4 写入长期记忆

对于新事件，系统会：

1. 生成自然语言描述。
2. 提取 subject 和 object 关键词。
3. 生成 embedding。
4. 用大模型给事件评估 1 到 10 的 `poignancy`。
5. 创建 event ConceptNode。
6. 从反思阈值中扣除本事件的重要性。

感知模块因此是“客观世界事件”转化为“某个 Agent 的主观经历”的入口。

## 11. Retrieve：按当前需要找回记忆

代码中实际存在两种检索方式。

### 11.1 当前感知事件的快速检索

`retrieve(persona, perceived)` 针对每个刚感知到的事件，使用事件的 subject、predicate 和 object 关键词召回：

- 相关历史 event。
- 相关 thought。

返回结构是：

```python
{
  "当前事件描述": {
    "curr_event": 当前事件,
    "events": 相关历史事件,
    "thoughts": 相关想法
  }
}
```

这个结果主要用于判断是否应对当前人物或事件作出反应。

### 11.2 通用记忆检索 `new_retrieve`

反思、对话和部分规划使用更完整的检索。对于一个 `focal_point`，它给所有 event 和 thought 计算三个分数：

#### 新近度 Recency

设计意图是越近访问的记忆分数越高，使用指数衰减：

```text
recency_decay ^ 排名
```

默认 `recency_decay = 0.99`。

需要注意，当前仓库实现先按 `last_accessed` 升序排列，再依次赋予 `0.99 ^ i`，直接按代码执行可能反而让
更早访问的节点得到略高分。这应视为研究原型的实现瑕疵；“最近记忆应更容易取回”才是架构本身的设计原则。

#### 相关度 Relevance

计算 focal point embedding 与记忆 embedding 的余弦相似度。

#### 重要性 Importance

直接使用记忆节点的 `poignancy`。

代码先分别归一化三项，再组合：

```text
score = 0.5 × recency + 3 × relevance + 2 × importance
```

最后取前 N 条，并把它们的 `last_accessed` 更新为当前时间。

### 11.3 检索的设计意义

Agent 不是把完整生平放进每个 Prompt，而是根据当前焦点临时拼出一个有限上下文。

这正是 Generative Agents 能维持较长模拟的关键：长期记忆可以不断增长，但每次推理只读取少量高价值内容。

## 12. Reflect：把经历压缩成高层认识

反思模块位于
[`reflect.py`](../../../vendor/generative_agents/reverie/backend_server/persona/cognitive_modules/reflect.py)。

如果只保存事件，Agent 只能记得大量细节，却难以形成“某个人值得信任”“我最近一直很孤独”“大家可能会参加
派对”这样的高层认识。Reflection 负责从经历中产生这类抽象想法。

### 12.1 何时触发

每个新事件都有 1 到 10 的重要性。系统持续从 `importance_trigger_curr` 中扣除重要性，累计达到阈值时触发反思。

这意味着：

- 大量日常小事最终也能触发反思。
- 少量高重要性事件会更快触发反思。
- 反思不是每一步都运行，可以控制成本。

### 12.2 反思过程

1. 收集上次反思以来的新事件与想法。
2. 让模型生成若干重要的高层问题，即 focal points。
3. 对每个 focal point 使用 `new_retrieve` 找相关记忆。
4. 让模型从这些记忆中产生多个 insight。
5. 要求每个 insight 指明支撑它的记忆序号。
6. 将 insight 作为新的 thought 节点写回 AssociativeMemory。

例如：

```text
事件：Maria 主动帮助 Isabella 布置咖啡馆
事件：Maria 询问派对还缺什么
事件：Maria 邀请 Klaus 参加
                      ↓ Reflect
想法：Maria 很重视 Isabella 的派对，并愿意主动帮助
```

### 12.3 反思形成递归记忆

新的 thought 本身可以在下一次反思中成为证据，于是形成：

```text
事件 → 一阶想法 → 更高层想法
```

这让 Agent 能从零散经历形成逐步抽象的自我认识和社会认识，但也带来错误反思被继续放大的风险。

## 13. Plan：从人格和记忆生成持续日程

规划模块位于
[`plan.py`](../../../vendor/generative_agents/reverie/backend_server/persona/cognitive_modules/plan.py)。

它同时解决三个不同时间尺度的问题：

```text
一天的粗计划
    ↓
小时级日程
    ↓
当前几分钟的具体动作与地点
```

### 13.1 新一天的长期规划

新的一天开始时，系统：

1. 根据 lifestyle 生成起床时间。
2. 根据身份稳定集和当前状态生成一天的粗计划。
3. 为 24 小时分别生成活动。
4. 合并连续相同活动，形成小时级日程。
5. 把当天计划作为 thought 写入记忆流。

这保证 Agent 在没有突发事件时也有连续、可预测的生活，而不是每一步临时随机决定。

### 13.2 Just-in-time 任务分解

系统不会一开始就把全天每一分钟都规划完。某个较长任务即将发生时，才将其拆成若干分钟级子任务。

例如：

```text
09:00–12:00 准备课程
```

临近执行时拆成：

```text
复习课程标准 15 分钟
构思活动 30 分钟
制作材料 30 分钟
检查课程计划 30 分钟
……
```

这样既保留长期方向，也允许未来计划根据新事件调整。

### 13.3 从动作描述选择地点

当需要执行一个新动作时，计划模块逐层选择：

1. world。
2. sector。
3. arena。
4. game object。

模型只能从 SpatialMemory 提供的候选中选择。例如“睡觉”最终可能落到：

```text
the Ville:Isabella Rodriguez's apartment:main room:bed
```

### 13.4 对新事件作出反应

Agent 感知到另一个人物或事件后，会结合检索出的历史判断：

- 是否发起聊天。
- 是否等待对方完成某事。
- 是否继续原计划。

如果决定聊天或等待，系统会把新活动插入当前日程，并重新生成受影响时间段的细分计划。

这体现了 Generative Agents 的一个核心思想：

> 行为既不是完全照表执行，也不是完全即时生成，而是稳定计划与环境反应的结合。

## 14. Conversation：社会互动如何进入循环

对话不是一个独立于世界的聊天接口，而是地图中的一种行动。

### 14.1 发起条件

两个 Agent 必须能够相互感知。发起者再结合：

- 当前双方正在做什么。
- 上一次对话时间与主题。
- 与对方相关的事件和想法。
- 聊天缓冲状态。

决定是否交谈。

### 14.2 对话结果

系统生成双方完整对话，估算对话时长，并同时修改双方状态：

- 动作地址变成 `<persona> 对方姓名`。
- 当前事件变成 `chat with`。
- 双方日程插入这次对话。
- 对话内容写入 chat ConceptNode。

### 14.3 对话结束后的记忆

对话结束时，反思模块还会生成：

- 对未来计划有何影响的 thought。
- 从这段对话记住了什么的 thought。

这些 thought 通过 `filling` 指回 chat 节点，使对话能够影响以后的计划和人物关系判断。

## 15. Execute：把符号计划变成物理移动

执行模块位于
[`execute.py`](../../../vendor/generative_agents/reverie/backend_server/persona/cognitive_modules/execute.py)。

这一层主要是确定性代码，不依赖大模型自由生成坐标。

### 15.1 目标解析

根据计划地址确定目标：

- 普通地址：从 `Maze.address_tiles` 获取地点的候选 Tile。
- `<persona>`：追踪目标人物当前位置。
- `<waiting>`：停留在指定 Tile。
- `<random>`：在指定区域随机选 Tile。

### 15.2 路径规划

执行模块：

1. 从候选 Tile 中抽样。
2. 尽量避开已被其他人物占据的位置。
3. 对候选目标运行 path finder。
4. 选择最短路径。
5. 每个模拟步只前进一格。

### 15.3 输出

每一步返回：

```text
(next_tile, pronunciatio, description)
```

例如：

```text
((58, 9), "💤", "sleeping @ the Ville:apartment:main room:bed")
```

执行层的意义是把自然语言规划限制在真实地图可执行范围内。模型负责选择“做什么、去哪里”，确定性系统负责
“能否到达、下一格怎么走”。

## 16. 一条事件的完整生命周期

假设 Isabella 在咖啡馆看到 Klaus 走进来。

### 第一步：世界产生事件

Klaus 当前所在 Tile 上出现：

```text
(Klaus, is, entering Hobbs Cafe, ...)
```

### 第二步：Isabella 感知

如果 Klaus 在她的视野内、处于同一 arena，并进入注意带宽，她会创建一个 event memory。

### 第三步：计算重要性

模型根据 Isabella 的身份和 Klaus 的行为给事件打 poignancy 分数。

### 第四步：检索相关记忆

通过 Klaus 关键词找回：

- 以前与 Klaus 的对话。
- Klaus 是否知道派对。
- 关于 Klaus 性格的 thought。

### 第五步：保持或打断计划

Isabella 原计划可能是布置咖啡馆。计划模块判断是否与 Klaus 交谈；若需要，则把对话插入日程。

### 第六步：执行

如果要交谈，执行模块规划到 Klaus 附近的路径并逐格移动。

### 第七步：新事件写回世界

双方的 `chat with` 事件出现在 Maze 中，其他附近 Agent 也可能感知到它。

### 第八步：形成反思

当重要性累计达到阈值后，Isabella 可能形成：

```text
Klaus 对情人节派对很感兴趣。
```

这个 thought 会在未来决定是否邀请 Klaus 或向别人提到他时被检索。

## 17. 为什么这套系统能显得“像一个持续存在的人”

可信行为主要来自多个机制的组合，而不是单次大模型生成质量。

### 17.1 稳定身份

身份稳定集持续进入 Prompt，使人物长期保持相近的背景、作息和性格。

### 17.2 选择性记忆

Agent 不需要记住一切，但重要、相关或最近的内容更容易被取回。

### 17.3 反思抽象

零散事件被压缩成高层认识，高层认识又影响后续行为。

### 17.4 分层计划

Agent 有一整天的基本方向，同时能在分钟级对环境作出反应。

### 17.5 世界中的信息传播

一个 Agent 告诉另一个 Agent 的信息会进入对方记忆，之后又可能通过新对话传播给第三个人。

### 17.6 行为进入共享环境

计划不是只生成一段文字，而是变成人物位置、对象状态和可被别人感知的事件，因此能够形成多 Agent 因果循环。

## 18. 各模块职责与非职责

| 模块 | 主要回答的问题 | 不负责什么 |
|---|---|---|
| `ReverieServer` | 现在是几点、该驱动谁、如何保存模拟 | 不扮演具体人物 |
| `Maze` | 世界中有什么、人物和事件在哪里、哪里可达 | 不保存个人主观回忆 |
| `Scratch` | 我是谁、今天计划什么、现在在做什么 | 不保存完整长期经历 |
| `SpatialMemory` | 我知道哪些地点和对象可用 | 不统计到访、热度或轨迹 |
| `AssociativeMemory` | 我经历、谈论和反思过什么 | 不代表权威世界事实 |
| `Perceive` | 当前附近有什么值得进入记忆 | 不检索完整生平 |
| `Retrieve` | 当前问题最相关的过去是什么 | 不决定最终行动 |
| `Reflect` | 从多条经历能得到什么高层认识 | 不直接移动人物 |
| `Plan` | 接下来做什么、在哪里做、是否响应别人 | 不负责寻路 |
| `Execute` | 怎样把目标地址落实为下一格移动 | 不决定人格和长期目标 |
| `Conversation` | 两个人是否聊天以及聊什么 | 不维护权威人物关系图 |

## 19. Generative Agents 与 GraphRAG 的关系

两者解决的问题不同。

### 19.1 Generative Agents 解决什么

它提供的是 Agent runtime architecture：

```text
感知什么
→ 回忆什么
→ 如何形成认识
→ 如何维持计划
→ 如何行动并影响世界
```

### 19.2 GraphRAG 解决什么

GraphRAG 更偏向数据组织与检索：

```text
大量证据如何形成节点和关系
→ 如何沿图和向量找到局部上下文
→ 如何把上下文提供给模型
```

### 19.3 原项目是不是 GraphRAG

严格说不是现代意义上的 GraphRAG，但它包含一些相似元素：

- ConceptNode 是记忆节点。
- keyword 索引和 embedding 用于检索。
- thought 的 `filling` 指向证据节点。
- Agent 根据检索结果生成新的 thought 或行动。

它缺少典型 GraphRAG 中的全局实体图、社区发现、跨文档关系图和统一图查询。每个 Agent 主要维护自己的个人
Memory Stream。

因此可以这样理解：

> GraphRAG 可以成为更强的记忆与证据检索底座；Generative Agents 提供使用这些记忆驱动长期行为的认知循环。

二者可以组合，但不能互相替代。

## 20. 原项目空间机制与真实地图的区别

原项目的空间问题相对简单，因为：

- 地图是人工制作的固定 Tile 世界。
- 所有地点名字都由开发者预先定义。
- 每个地点有唯一的层级地址。
- 地点地址可以直接反查 Tile。
- 不存在“多个城市都有正大广场”的开放世界问题。
- Agent 只需要从候选列表中选择，不需要调用地图 Provider 做 POI 消歧。

因此原项目的 SpatialMemory 不能直接承担真实聊天中的地点解析。它值得借鉴的是：

- 世界状态与个人空间认知分离。
- 规划先产生语义动作，再选择可执行地点。
- 地点最终必须转换为确定性执行系统能理解的地址。

但真实世界还需要额外处理别称、指代、地点所有者、历史搬迁、同名 POI、坐标来源和不确定性。

## 21. 原项目的主要限制

### 21.1 固定世界

世界、地点层级和对象均由 Tiled 地图预先制作，扩展新城市需要重新制作地图资产。

### 21.2 世界状态较简单

世界事件是弱类型自然语言三元组，没有复杂物理状态、资源约束或稳定领域模型。

### 21.3 没有严格实体消歧

人物和地点名称主要依赖预定义字符串。它不处理真实开放世界中的同名实体和含糊指代。

### 21.4 记忆缺少修订机制

Memory Stream 基本追加写入，没有完善的冲突、撤销、版本、置信度演化和人工纠正体系。

### 21.5 反思可能放大错误

模型生成的错误 insight 会作为 thought 再次参与检索，并可能成为更高层反思的证据。

### 21.6 检索扩展性有限

`new_retrieve` 会遍历 event 和 thought 节点计算分数，适合实验规模，不适合超大长期个人数据。

### 21.7 Prompt 和调度逻辑较脆弱

原项目包含大量专用 Prompt、硬编码阈值、字符串协议和异常兜底，更接近研究原型而不是生产框架。

### 21.8 模型调用较多

感知重要性、日程、任务分解、地点选择、事件三元组、反应判断、对话和反思都可能调用模型，多 Agent 长时间
运行成本较高。

## 22. 阅读代码的推荐顺序

如果要继续阅读源码，建议按以下顺序：

1. [`persona.py`](../../../vendor/generative_agents/reverie/backend_server/persona/persona.py)：先看完整认知循环。
2. [`scratch.py`](../../../vendor/generative_agents/reverie/backend_server/persona/memory_structures/scratch.py)：理解人物运行状态。
3. [`associative_memory.py`](../../../vendor/generative_agents/reverie/backend_server/persona/memory_structures/associative_memory.py)：理解 Memory Stream。
4. [`perceive.py`](../../../vendor/generative_agents/reverie/backend_server/persona/cognitive_modules/perceive.py)：理解世界如何进入记忆。
5. [`retrieve.py`](../../../vendor/generative_agents/reverie/backend_server/persona/cognitive_modules/retrieve.py)：理解记忆评分。
6. [`reflect.py`](../../../vendor/generative_agents/reverie/backend_server/persona/cognitive_modules/reflect.py)：理解事件如何变成想法。
7. [`plan.py`](../../../vendor/generative_agents/reverie/backend_server/persona/cognitive_modules/plan.py)：理解分层规划和反应。
8. [`spatial_memory.py`](../../../vendor/generative_agents/reverie/backend_server/persona/memory_structures/spatial_memory.py)：理解已知地点树。
9. [`execute.py`](../../../vendor/generative_agents/reverie/backend_server/persona/cognitive_modules/execute.py)：理解计划如何落到路径。
10. [`maze.py`](../../../vendor/generative_agents/reverie/backend_server/maze.py) 与
    [`reverie.py`](../../../vendor/generative_agents/reverie/backend_server/reverie.py)：最后看完整模拟器如何驱动所有 Agent。

## 23. 最终心智模型

理解 Generative Agents 时，可以一直保留以下五个对象：

```text
World
  当前客观发生了什么

Agent State
  这个人是谁、正在做什么、原本打算做什么

Memory Stream
  这个人过去经历和想到过什么

Cognition
  当前应该注意、回忆、反思和决定什么

Execution
  如何把决定变成共享世界中的真实变化
```

系统的设计重点不是让模型一次给出完美行为，而是让这五部分持续循环，使每一次行动都来自人物身份、历史记忆、
当前环境和既有计划的共同约束。

这也是 Generative Agents 最值得借鉴的部分：

> 记忆不是一个独立的聊天检索功能，世界也不是一张展示地图；记忆、世界、计划和行动必须组成一个能持续产生
> 新证据的闭环。

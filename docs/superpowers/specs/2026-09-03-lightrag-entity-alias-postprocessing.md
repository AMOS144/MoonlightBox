# LightRAG 建图后实体别名归并与聊天昵称预处理设计

## 1. 目标

本设计解决两类不同问题：

1. 聊天导入时，识别聊天格式中明确存在的发送者昵称，并将其归档到发送者资料中；
2. LightRAG 完成建图后，从已经存在的人物节点中生成别名归并建议，由人工审核后调用 LightRAG 原生合并能力。

不再扫描全文，把任意短英文词、普通句子或职业/地点当作人物别名。

## 2. 当前聊天数据的事实

当前支持的 wxecho 数据已经包含结构化发送者字段：

```json
{
  "time": "2024-01-01 12:00:00",
  "sender": "洪欣羽",
  "type_name": "文本",
  "content": "……"
}
```

TXT 格式也有明确的行结构：

```text
[2024-01-01 12:00:00] 洪欣羽: ……
```

因此，`sender` 或 TXT 行首发送者是可靠的“原始发送者”，不需要从正文猜测。当前 `test1` 项目中，`AMOS` 和 `Feather` 出现在正文的微信引用格式中。这里的名称不是普通正文，而是微信发送引用消息时自动显示的“被引用消息发送者昵称”，例如：

```text
真的吗
> AMOS：我终于知道……
```

它们不是当前这条消息的顶层 `sender`，而是被引用消息里的显示名称。这个结构必须进入昵称处理和归档；普通正文里的 `ai`、`bug`、`docx` 没有发送者字段或引用结构，不进入昵称处理。

## 3. 预处理：只处理有格式依据的昵称

第一版固定流水线为：

```text
识别数据格式
→ 固定聊天双方角色
→ 解析并归档微信引用昵称
→ 清除技术载荷
→ 保留原文并生成干净 Bundle
```

每一步只能使用导入器类型、参与者角色或微信消息结构等确定信息，不能扫描短词猜测人物。

### 3.1 原始发送者

导入器继续使用现有字段：

```text
sender → Participant.name
```

同时保留：

```text
Message.raw.original_sender
Message.raw.source_format
```

原始发送者字段不经过 LLM，不经过候选猜测，也不需要人工确认。

### 3.2 引用消息中的微信显示昵称

微信引用署名不是“疑似缩写”，而是一个有明确产品语义的昵称字段。只识别导入格式明确支持的引用结构，例如：

```text
> 名称：引用内容
＞名称：引用内容
```

处理步骤：

1. 解析引用署名和引用正文；
2. 用引用正文精确匹配同一项目中的原始消息；
3. 如果能唯一反查到原始 `sender`，将引用署名归档为该 Participant 的 `alias`，状态为 `resolved`；
4. 如果无法反查，仍然归档为 `quoted_sender`，状态为 `unresolved`，不能把它降级成普通正文词；
5. 如果同一署名反查到不同发送者，记录为 `conflicting`，不自动合并；
6. Bundle 送入 LightRAG 时，可将已解析的引用署名替换为统一参与者显示名，同时保留原始消息和昵称映射，确保可追溯。

例如：

```text
AMOS → 我
Feather → 洪欣羽
```

这类归档属于微信格式解析结果，不属于 LightRAG 的实体判断。已经反查成功的 `AMOS`、`Feather` 不再作为 LightRAG 的待归并候选；只有无法从导入消息确定归属的引用昵称，才可在后处理审核列表中单独显示。

### 3.3 正文中的普通词

不对全文执行短词扫描，也不维护 `ai`、`bug` 等黑名单。

只有出现明确的别名表达时才记录，例如：

```text
我的昵称是 xx
英文名是 xx
大家叫我 xx
xx 是我的简称
```

普通表达中的：

```text
用 ai
改 bug
打开 docx
```

保持原文，不进入昵称归档，不进入人物别名归并。

## 4. 别名归档数据

别名是发送者资料的附属信息，不新建一套“实体类型表”。建议在现有 `Participant` 体系下增加别名归档记录：

```text
ParticipantAlias
  id
  project_id
  participant_id              # 已知时填写
  alias                       # Feather、AMOS 等
  origin                      # source_sender / quoted_sender / explicit_statement
  source_message_ids
  resolution_status           # resolved / unresolved / conflicting
  created_at
```

这张表描述的是“聊天中出现过的称呼”，不是 LightRAG 的图节点，也不代替 LightRAG 图谱。

预处理只负责写入来源和归档信息。Bundle 正文默认保留自然聊天文本；如需给模型稳定角色，可在 Bundle 渲染层使用已有的 `self`、`target` 角色，而不是把内部 ID 写入文本。

## 5. LightRAG 建图后处理

### 5.1 读取全部图节点

建图完成后，Sidecar 调用 LightRAG：

```python
await rag.get_graph_labels()
```

这是读取当前工作区全部节点名称，不是普通 Top-K 检索。

随后通过 `get_entity_info` 读取节点类型、描述、关系和来源信息。

### 5.2 筛选人物节点

把完整节点列表交给模型，要求模型只能从列表中选择人物节点：

```text
只返回代表现实人物、人物昵称或人物称呼的节点。
不要返回地点、组织、职业、活动、技术词或抽象概念。
不能创造列表中不存在的名称。
```

程序校验模型返回的每个名称必须原样存在于 `get_graph_labels()`。因此 `ai`、`bug`、地点和职业不会因为字符串形式进入别名比较；它们即使存在于图中，也会停留在普通节点层。

### 5.3 生成归并候选

模型只比较人物节点对，不直接修改图谱。每条建议包含：

```json
{
  "source_entity": "Feather",
  "target_entity": "洪欣羽",
  "decision": "likely_same_person",
  "reason": "……",
  "evidence": [
    {
      "entity": "Feather",
      "source_document_id": "doc-123",
      "text": "……"
    }
  ]
}
```

判断依据只能来自节点描述、关系、来源消息和已归档的发送者信息：

- 明确的昵称/英文名/简称表达；
- 引用署名精确反查到同一个原始发送者；
- 同一个聊天角色、同一组个人经历和关系网络。

仅凭名称相似、同时出现或都属于 `Person`，不能判定为同一人。证据不足时返回 `unknown`，不生成可执行合并。

## 6. 人工审核与执行归并

每条模型建议进入审核列表：

```text
Feather ↔ 洪欣羽
模型判断：可能是同一人
证据：……

[合并] [不是同一个人] [暂不处理]
```

审核结果：

- `合并`：调用 LightRAG 原生 `amerge_entities`；
- `不是同一个人`：保存否定结果，后续不重复推荐；
- `暂不处理`：保留建议，不修改图谱。

当前安装的 `lightrag-hku 1.5.6` 已提供：

```python
await rag.amerge_entities(
    source_entities=["Feather"],
    target_entity="洪欣羽",
)
```

LightRAG 负责合并节点描述、关系、来源 chunk 和向量数据。MoonlightBox 不直接操作 LightRAG 底层图数据库。

## 7. Sidecar 接口

需要在现有 `documents:batch` 和 `query` 之外增加：

```text
GET  /v1/workspaces/{workspace}/entities
POST /v1/workspaces/{workspace}/entity-alias-proposals
POST /v1/workspaces/{workspace}/entity-merges
GET  /v1/workspaces/{workspace}/entity-merge-history
```

其中：

- `entities` 读取节点；
- `entity-alias-proposals` 只生成候选；
- `entity-merges` 只接受人工审核通过的归并；
- `entity-merge-history` 用于追溯和回滚审计。

## 8. 最小实现顺序

1. 增加结构化引用署名解析和 `ParticipantAlias` 归档；
2. 暂不改写普通正文，不做短词扫描；
3. Sidecar 暴露 `get_graph_labels` 和 `get_entity_info`；
4. 生成“人物节点归并候选”，但不自动执行；
5. 前端增加人工审核列表；
6. 审核通过后调用 `amerge_entities`；
7. 保存归并历史和 LightRAG 返回结果。

整个流程中，预处理解决“聊天格式本身已经告诉我们的昵称”，后处理解决“图谱中多个节点是否是同一现实人物”。两者不再混用。

## 9. 别名归一化 Agent workflow

候选生成不是把整份聊天直接塞给模型的单一 prompt，而是一个有界 LangGraph
workflow。节点按固定顺序执行，不能自行写图谱或无限循环调用工具：

```text
read_person_nodes
  → locate_original_sources
  → query_lightrag_context
  → propose_aliases
```

节点注册三个 LangChain 只读工具：

| 工具 | 输入 | 返回 |
| --- | --- | --- |
| `list_person_nodes` | 无 | LightRAG 全部人物节点、类型和描述 |
| `locate_original_mentions` | 节点名称列表 | 原始消息 ID、Bundle 文档、时间、发送者和原文摘录 |
| `query_lightrag` | 一个节点名称 | 该节点的局部关系网络、描述和检索上下文 |

`propose_aliases` 只接受上述工具结果，并通过结构化输出生成审核提案。程序
要求 `source_entities` 是别称，`target_entity` 必须来自节点描述明确表达姓名/实名/原名
或“昵称为”的稳定人物节点（项目主体和用户也可作为配置的实名节点）；关系角色和
泛称不能充当目标。因此“胖笨笨”“笨胖胖”“入”“此入”等名称只能作为待审核的
source，是否归入“余熠”“洪欣羽”等真实姓名必须由工具证据和人工确认共同决定，
不会再生成昵称到昵称的映射。每条 evidence 同时保留 `message_id`、Bundle 文档
和摘录，审核界面可以继续追溯原文。人工批准仍是唯一调用 `amerge_entities` 的
节点。

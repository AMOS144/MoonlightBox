# 人物表达资料：画像编译、审核与 Skill 渐进读取

## 数据归属

不生成每个人各自的共享 Skill 文件，也不增加第八个 Agent。现有“与用户关系”Agent 在编译
PersonWorldProfile 时填写 `relationship_with_user.expression_profile`，与该人物的相处理解放在一起。
资料版本是 `expression-v1`；保持 v3 画像历史可读，不覆盖已有发布。

六类固定资料是称呼、语气词与口癖、分句节奏、标点、玩笑方式、不同场景下的变化。
每类保留状态、概括、推断/用户纠正来源和可选参考消息；patterns 内每种表达同时描述
`form`（怎么说）、`use_when`（什么时候怎样用）、`avoid_when`（什么时候不用）。
同一个称呼的场景不能被拆成无条件的口癖词表。不以固定次数、词频或正则判断人的习惯。

Agent 根据整体互动推断方式；具体词形查真实原文，避免将检索总结或用户的话当成目标人物口癖。
不引入视觉模型，不在静态资料里塞可直接发送的素材 ID。表情包仍在当次互动中查询并验权。

## 编译和审核

栏目提交工具要求新输出包含六类资料，缺项可在工具回执中修正。历史存储模型允许缺失，默认未知，
不假装已有结果。资料使用现有条目 ID 和字段路径，在审核页按六类查看、选择、对话纠正。
沿用已有草稿、预览、多步确认、最终发布流程，不另开绕过审核的自动发布通道。

## 运行时

1. 分支绑定已发布画像时，资料随原有快照投影进入内存；不直接读最新草稿，也不跨人物查找。
2. Director 普通上下文不重复携带这份详细资料；`read_skill(speaking)` 返回共享指导、业务工具
   用法及本次人物的资料副本。共享文件不会被修改。
   资料通过正文开场之后的 `{{ expression_profile }}` 原位注入；frontmatter 的
   `context_placeholder: expression_profile` 声明位置名称，不再由加载器追加到末尾。
3. 资料按内容 hash 标识版本，纳入 Skill 的检查点契约；同次运行冻结副本。切换人物或资料版本
   不会改变模型侧工具 Schema，但会使依赖旧资料的检查点不再误用。
4. Skill 先使用匹配当前场景的常用方式，必要时通过固定 `execute_tool` 检索真实表达和表情包。
   静态资料给常态，检索给当前相似互动；二者都不要求机械照抄。
5. 当前 PersonaActor 仍负责最终措辞，因此其隔离表达上下文也接收同一份资料，不继续只用旧版
   IdentityKernel 的泛化风格。无新资料的历史分支保留旧输入兼容。

普通服务重启不会更新已发布画像或分支快照。已有项目需重新调查“与用户关系”栏目、审核发布，
再使用现有显式背景刷新流程绑定新版本；本次代码修改未触发付费编译、发布或业务数据修改。

## 实现入口

- `world/person_world/contracts/profile_v3.py`：固定结构与历史读取默认值。
- `world/person_world/subagents/relationship_with_user.md`：调查和整理方式。
- `frontend/src/features/world/V3ProfileGrid.tsx`：资料展示与选择纠正。
- `runtime_v1/expression_profile.py`：只读资料与版本摘要。
- `agent_runtime/skills.py`：调用方动态资料的冻结与读取，不修改工具目录。
- `runtime_v1/director.py`、`context_views.py`：Director 渐进读取与 Actor 表达输入。

离线测试覆盖编译必填、历史兼容、审核定位、人物隔离、资料冻结、Schema 稳定、版本变化以及
Director/Actor 输入差异；不将这些测试称为已验证真人表达质量，效果需编译真实资料后体验。

本轮验证：画像结构/技能/表达链路一组 33 项通过，异步编译/同级协作/资料接线另一组
19 项通过（部分用例重复执行）；Ruff 和前端 TypeScript + Vite 构建通过。

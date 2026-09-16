# PersonWorld Prompt 维护入口

本目录直接存放当前 v3 使用的七份栏目 Prompt：
identity、life_context、social_world、agency、practices、life_course、
relationship_with_user。修改对应 Markdown 即可维护该栏目的行为。

- `_understanding.md`：当前七栏共用的理解原则。
- `revision.md`：审核纠正对话使用的 Prompt，仍在使用。
- 旧版七栏 Prompt 和生成、重试实现已退役，不再保留另一套作者文件。

默认加载器读取主目录。历史 v1/v2 画像仍可读取，但不能调用旧版栏目重试；
需要更新时，点击“重新生成 v3 画像”，基于独立图谱副本生成新候选，再由用户审核。
输出字段与心理模型定义分别位于 `../contracts/profile_v3.py` 和
`../contracts/psychological_models.py`，工具 Schema 仍由 LangChain 注册。

目录整理没有修改任何 Prompt 正文，也不会重新生成、批准或发布画像。

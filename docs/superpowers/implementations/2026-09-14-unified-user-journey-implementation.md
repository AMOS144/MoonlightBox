# 前端用户主线统一：实施与验收记录

日期：2026-09-14。工作目录：`runtime-prune`。

## 1. 完成范围

本次不是完整历史分支链路的验收。已完成前端主线、统一组件基础与流程投影；后端历史资产编译仍未接通，必须明确保留为剩余工作。

| 项目 | 本次结果 |
| --- | --- |
| 设计冲突 | 修订导入确认顺序、导航层次、状态含义、重复批准、历史背景范围、追加资料、侧栏会话和恢复位置 |
| 主导航 | 概览、人物资料、选择起点、我的分支；高级能力下沉；窄屏使用同一导航的 Drawer |
| 概览 | 读取后端 Journey 投影；恢复上次工作，保留分支快捷入口和业务待办，不查询模型/事件数量猜下一步 |
| 导入 | 上传解析与正式导入分开；预览与人物输入可跨页恢复；导入成功只表示记录保存，不再串联所有后台分析 |
| 人物资料 | 统一查看参与者、头像、记录范围、追加导入和别名审核；正式导入是否完成以服务端 ImportSource 为准 |
| 人物背景 | 已发布版与新草稿可切换，发布后有起点交接；诊断默认折叠；保留现有纠正和图谱批准机制 |
| 节点调查 | 按调查 ID 恢复，稳定状态不高频轮询；统一 Agent 外壳；确认后明确保存的只是起点批准 |
| 分支准备 | 删除旧事件/LoRA 表单，显示真实历史能力缺口；已有分支准备页显示绑定背景与具体阶段，不显示虚假百分比，不自动跳页 |
| 聊天 | 保留原聊天结构与数据；准备态在稳定分支入口展示，不来回重定向；查看旧消息时出现新消息入口；修正窄屏留白 |
| 组件库 | Mantine 与 Tabler 保留；统一 theme、CSS 变量、状态、异步反馈、交接卡和 Agent 对话外壳；开发示例 `/dev/ui` |
| 路由清理 | `/data` → `/setup/import`，`/events`、`/timeline` → `/nodes`；旧事件只读；已有分支 URL 不变 |

核心新增模块：

- 后端：`backend/moonlightbox/projects/journey.py`，只读聚合已有业务状态，不新增 Journey 表、不调用模型。
- 前端：`features/journey/`、`components/feedback/`、`components/agent/`。
- `ParticipantsPage` 承载导入后的资料确认；`LegacyEventsPage` 只读展示旧资料。
- `scripts/frontend_journey_smoke.mjs` 通过独立 Chrome CDP 检查真实页面，不发送消息或启动调查。

## 2. 删除与保留

已删除无应用引用的 `NodeReviewPage.tsx`、`TimelinePage.tsx` 及旧节点审核页测试；旧事件资料由只读页接替。删除 ImportWizard 的跨 Agent 轮询、旧模型训练门槛、旧首页推断和重复阶段菜单。

原始消息、媒体、模型文件、历史事件、画像、图谱、已有分支及聊天均保留。源码删除可以从 Git 恢复；本次没有提交 Git，也没有覆盖其他任务的工作区改动。

旧 CSS 仅移除重复主题变量并避免标题覆盖 Mantine，聊天样式不整体重写。尚有历史页面专用样式可继续按引用清理，不宣称所有旧选择器已经移除。

## 3. 实际环境

- 当前体验数据库原为 `0059_runtime_event_received_at`，未包含节点调查表。
- 使用 SQLite 在线备份后升级到 `0060_node_investigations`，此次迁移只新增表。
- 备份：`/home/yuyi/project/moonlightbox_clean/MoonlightBox/.worktrees/.runtime-data/data/moonlightbox.db.before-journey-20260914-032049.bak`。
- API 使用原运行进程的配置重启在 `127.0.0.1:8001`，前端继续使用 `127.0.0.1:5175`。
- 交付后发现 API 收到终止信号并退出，前端代理返回 502。随后沿用原进程配置，改由用户级临时服务 `moonlightbox-experience-api.service` 托管，异常退出自动重启；日志使用 `journalctl --user -u moonlightbox-experience-api` 查看。该服务不代表开机自启，手动停止仍会保持停止。
- 原 cognition/realtime Worker 保留；补启动 background Worker 以承接节点调查、人物纠正等用户提交的后台任务。没有代用户开始节点调查或重新编译人物背景。
- 真实 Journey 读取成功：一个导入、一个已发布背景引用、两条已有分支均可读。

未用开发目录的默认空数据库替换体验数据库；迁移与重启均采用原 API 的环境配置。

## 4. 验证

- `npm run build`：通过。
- `npm run lint`：通过。
- `npm run test -- --run --maxWorkers=1`：12 个测试文件、23 个用例通过。
- `pytest backend/tests/projects`：当次收集的 28 个用例通过。
- `pytest backend/tests/projects/test_journey.py backend/tests/nodes -q`：13 个用例通过，与上一组存在重叠，不相加计数。
- 改动的项目投影模块与测试 Ruff 检查通过，`git diff --check` 通过。
- 独立 Windows Chrome 检查 1440px 与 390px：概览、资料、背景、起点、分支、旧链接、已有聊天、准备预览，共 16 次读取检查通过；无浏览器异常、无横向溢出，窄屏导航可打开。
- 最终浏览器截图留在 `/tmp/moonlight-journey-ui-QoayT6`，已包含窄屏聊天留白修复，并再次人工检查聊天布局。

一次并行前端测试曾遇到辅助地图测试超时及 teardown 调度错误，单 Worker 完整重跑通过；不通过修改生产业务来掩盖测试超时。Playwright 浏览器下载失败，改用本机已安装 Chrome 的独立临时实例验收，没有使用用户日常浏览器会话。

这些检查不代表已经完成新的历史分支编译，也没有对真实聊天执行一次云端模型效果测试。

## 5. 尚未完成的必要后端工作

1. 用批准的精确消息边界（不仅是时间字符串）建立新的分支创建契约与幂等回执。
2. 为所选起点编译独立的人物背景，并确保图谱检索、原文、记忆和初始化都不能读到起点后的资料。
3. 从普通新建接口移除旧 EventNode/LoRA 必填依赖，绑定服务端当前认知配置。
4. 将历史背景、DayPlan、Director 初始化纳入可恢复的准备流程，支持必要的用户澄清。
5. 契约接通后，才将“确认起点”与“准备分支”合并为一个前端主操作，并验收失败只重试创建、不重新批准。
6. 完成真实新项目从导入到新历史分支聊天的端到端验收。

当前 `historical_branch_creation.available=false` 是对实际实现能力的声明，不是把失败任务伪装为产品限制。前端不会使用最新背景或旧创建接口兜底。此列表完成前，整个设计仍属部分交付。

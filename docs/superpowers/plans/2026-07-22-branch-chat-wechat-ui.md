# 分支聊天微信式界面实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 将分支聊天页改为保留深色主题的全高贴边桌面微信布局，移除浮窗、任务条和旧分支提示，同时保持现有聊天行为。

**架构：** 在 `ProjectLayout` 中根据当前路由切换聊天专用外壳，避免影响其他项目页面；在 `BranchChatPage` 中精简顶栏和只读状态；在全局样式中集中重写聊天布局及消息视觉。现有查询、轮询、分页、乐观发送和人格记忆数据流不做改动。

**技术栈：** React 19、TypeScript、React Router 7、TanStack Query 5、Vitest、Testing Library、纯 CSS

---

## 文件结构

- 修改 `frontend/src/features/projects/ProjectLayout.tsx`：识别分支聊天路由，切换内容区样式并隐藏任务条。
- 新建 `frontend/src/features/projects/ProjectLayout.test.tsx`：验证聊天路由和普通项目路由的外壳差异。
- 修改 `frontend/src/features/branches/BranchChatPage.tsx`：精简顶栏、移除升级提示、提供只读输入状态。
- 修改 `frontend/src/features/branches/BranchChatPage.test.tsx`：验证更多入口、只读状态和原有发送行为。
- 修改 `frontend/src/index.css`：实现贴边全高布局、微信式消息区、输入区、人格记忆侧栏和响应式规则。

### 任务 1：建立聊天路由专用项目外壳

**文件：**
- 新建：`frontend/src/features/projects/ProjectLayout.test.tsx`
- 修改：`frontend/src/features/projects/ProjectLayout.tsx`

- [ ] **步骤 1：编写聊天路由外壳的失败测试**

创建测试，使用内存路由分别进入聊天页和普通项目页，并模拟 `ProjectTaskBar`：

```tsx
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'

import { ProjectLayout } from './ProjectLayout'

vi.mock('./ProjectTaskBar', () => ({
  ProjectTaskBar: () => <div data-testid="project-task-bar">任务状态</div>,
}))

afterEach(cleanup)

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/projects/:projectId" element={<ProjectLayout />}>
          <Route path="branches/:branchId" element={<div>聊天内容</div>} />
          <Route path="data" element={<div>数据内容</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  )
}

test('分支聊天路由使用无内边距外壳并隐藏任务条', () => {
  renderAt('/projects/p1/branches/b1')
  expect(screen.getByRole('main')).toHaveClass('project-content--branch-chat')
  expect(screen.getByRole('main').parentElement).toHaveClass(
    'project-layout--branch-chat',
  )
  expect(screen.queryByTestId('project-task-bar')).not.toBeInTheDocument()
})

test('普通项目路由保持原有外壳和任务条', () => {
  renderAt('/projects/p1/data')
  expect(screen.getByRole('main')).not.toHaveClass('project-content--branch-chat')
  expect(screen.getByTestId('project-task-bar')).toBeInTheDocument()
})
```

- [ ] **步骤 2：运行测试并确认失败**

运行：

```bash
cd frontend && npm test -- --run src/features/projects/ProjectLayout.test.tsx
```

预期：聊天路由仍渲染任务条，且 `main` 缺少 `project-content--branch-chat`。

- [ ] **步骤 3：实现路由专用外壳**

在 `ProjectLayout.tsx` 中使用 `useMatch` 精确识别聊天路由：

```tsx
import { NavLink, Outlet, useMatch, useParams } from 'react-router-dom'

const { projectId } = useParams()
const isBranchChat = Boolean(
  useMatch('/projects/:projectId/branches/:branchId'),
)
```

将根节点开标签替换为：

```tsx
<div
  className={`project-layout${isBranchChat ? ' project-layout--branch-chat' : ''}`}
>
```

将项目内容节点替换为：

```tsx
<main
  className={`project-content${isBranchChat ? ' project-content--branch-chat' : ''}`}
>
  {projectId && !isBranchChat ? (
    <ProjectTaskBar projectId={projectId} />
  ) : null}
  <Outlet />
</main>
```

- [ ] **步骤 4：运行测试并确认通过**

运行：

```bash
cd frontend && npm test -- --run src/features/projects/ProjectLayout.test.tsx
```

预期：2 个测试全部通过。

### 任务 2：精简聊天顶栏和只读状态

**文件：**
- 修改：`frontend/src/features/branches/BranchChatPage.test.tsx`
- 修改：`frontend/src/features/branches/BranchChatPage.tsx`

- [ ] **步骤 1：更新顶栏入口测试**

将原来查找“人格记忆”的断言改为查找可访问名称“更多”：

```tsx
expect(
  screen.getByRole('button', { name: '更多' }).querySelector('svg'),
).toBeNull()
```

新增点击后打开人格记忆面板的断言：

```tsx
fireEvent.click(screen.getByRole('button', { name: '更多' }))
expect(await screen.findByRole('heading', { name: '人格记忆' })).toBeInTheDocument()
```

- [ ] **步骤 2：新增只读分支失败测试**

预先写入只读分支查询数据，确认不显示升级横幅且输入区只读：

```tsx
queryClient.setQueryData(['branches', 'p1'], [
  {
    id: 'b1',
    lifecycle_status: 'archived',
    baseline_status: 'ready',
    baseline_manifest_id: null,
  },
])

expect(screen.queryByText('这是旧版只读分支')).not.toBeInTheDocument()
expect(screen.getByPlaceholderText('此分支仅供查看')).toBeDisabled()
expect(screen.getByRole('button', { name: '发送' })).toBeDisabled()
```

- [ ] **步骤 3：运行相关测试并确认失败**

运行：

```bash
cd frontend && npm test -- --run src/features/branches/BranchChatPage.test.tsx
```

预期：顶栏仍使用“人格记忆”，只读输入框仍使用“发消息…”占位文案。

- [ ] **步骤 4：实现精简顶栏**

删除未再使用的 `upgrade` mutation 和 `.branch-upgrade-banner` JSX。将顶栏按钮改为：

```tsx
<button
  aria-label="更多"
  className="branch-memory-button"
  onClick={() => setMemoryOpen(true)}
  type="button"
>
  <span aria-hidden="true">···</span>
</button>
```

联系人头像、名称和“对方正在输入…”逻辑保持不变。

- [ ] **步骤 5：实现只读输入状态和空内容禁用**

将输入框和发送按钮调整为：

```tsx
<textarea
  aria-label="发消息"
  id="branch-message"
  onKeyDown={(event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }}
  onChange={(event) => setContent(event.target.value)}
  disabled={isReadOnly}
  placeholder={isReadOnly ? '此分支仅供查看' : '发消息…'}
  value={content}
/>
<button
  className="primary-button"
  disabled={isReadOnly || content.trim().length === 0}
  type="submit"
>
  发送
</button>
```

发送失败提示继续保留在输入区内部。

- [ ] **步骤 6：运行测试并确认通过**

运行：

```bash
cd frontend && npm test -- --run src/features/branches/BranchChatPage.test.tsx
```

预期：顶栏、只读状态以及既有发送测试全部通过。

### 任务 3：重写微信式聊天视觉

**文件：**
- 修改：`frontend/src/index.css`

- [ ] **步骤 1：实现项目内容区贴边布局**

在项目外壳样式附近加入：

```css
.project-content {
  min-width: 0;
  padding: 48px;
}

.project-content--branch-chat {
  height: 100vh;
  padding: 0;
  overflow: hidden;
}
```

保留普通页面现有 `48px` 内边距。

- [ ] **步骤 2：移除聊天浮窗并建立三段结构**

替换 `.branch-chat`：

```css
.branch-chat {
  position: relative;
  display: grid;
  width: 100%;
  min-width: 0;
  height: 100%;
  overflow: hidden;
  grid-template-rows: 64px minmax(0, 1fr) 138px;
  background: #1b1820;
}
```

删除原有 `max-width`、外边距、边框、22px 圆角和阴影。

- [ ] **步骤 3：调整顶栏和更多按钮**

```css
.branch-chat__header {
  min-height: 64px;
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 0 22px;
  border-bottom: 1px solid #2e2933;
  background: #211d25;
}

.branch-memory-button {
  min-width: 44px;
  margin-left: auto;
  padding: 8px 10px;
  border: 0;
  color: #d7d2dc;
  background: transparent;
  font-size: 1.3rem;
  letter-spacing: 0.12em;
}
```

- [ ] **步骤 4：调整消息区、头像和微信式气泡**

```css
.message-list {
  min-width: 0;
  min-height: 0;
  padding: 24px 28px;
  overflow-x: hidden;
  overflow-y: auto;
  background: #1b1820;
}

.chat-row {
  gap: 10px;
  max-width: 78%;
  margin-bottom: 14px;
}

.chat-avatar {
  width: 36px;
  height: 36px;
  border-radius: 4px;
}

.chat-bubble {
  position: relative;
  padding: 9px 12px;
  border-radius: 5px;
  background: #2b2730;
}

.chat-bubble--user {
  background: #7650a7;
}
```

为普通文本气泡使用 `::before` 绘制左右尖角；引用、媒体占位和通话气泡沿用同一颜色体系。图片、视频和贴纸本体不添加尖角。

- [ ] **步骤 5：实现微信式固定输入区**

```css
.message-form {
  position: relative;
  display: grid;
  min-height: 138px;
  padding: 16px 20px;
  border-top: 1px solid #2e2933;
  background: #211d25;
  grid-template-rows: 1fr auto;
}

.message-form textarea {
  min-height: 72px;
  padding: 0;
  border: 0;
  outline: 0;
  color: #f2eef5;
  background: transparent;
  resize: none;
}

.message-form .primary-button {
  width: 78px;
  justify-self: end;
}
```

禁用按钮保持低对比度；有内容时沿用项目主按钮强调色。

- [ ] **步骤 6：让人格记忆面板贴边而非悬浮**

保留绝对定位和现有宽度，移除独立浮窗感：

```css
.branch-memory-panel {
  position: absolute;
  z-index: 20;
  top: 0;
  right: 0;
  width: min(460px, 100%);
  height: 100%;
  overflow-y: auto;
  border-left: 1px solid #3d3742;
  background: #17141b;
  box-shadow: none;
}
```

- [ ] **步骤 7：补充移动端全屏规则**

```css
@media (max-width: 720px) {
  .project-layout {
    display: block;
  }

  .project-content--branch-chat {
    height: 100dvh;
  }

  .project-layout--branch-chat .sidebar {
    display: none;
  }

  .message-list {
    padding: 20px 14px;
  }

  .chat-row {
    max-width: 92%;
  }
}
```

### 任务 4：完整回归与质量检查

**文件：**
- 验证：`frontend/src/features/projects/ProjectLayout.test.tsx`
- 验证：`frontend/src/features/branches/BranchChatPage.test.tsx`
- 验证：`frontend/src/features/branches/MessageBubble.test.tsx`
- 验证：`frontend/src/features/branches/BranchMemoryPanel.test.tsx`
- 验证：`frontend/src/features/branches/useAdaptiveBranchHistory.test.ts`

- [ ] **步骤 1：运行前端测试**

运行：

```bash
cd frontend && npm test -- --run
```

预期：全部 Vitest 测试通过。

- [ ] **步骤 2：运行静态检查**

运行：

```bash
cd frontend && npm run lint
```

预期：oxlint 退出码为 0。

- [ ] **步骤 3：运行生产构建**

运行：

```bash
cd frontend && npm run build
```

预期：TypeScript 编译和 Vite 构建成功。

- [ ] **步骤 4：手动视觉验收**

启动现有本地应用并进入任意已就绪分支，逐项确认：

1. 聊天区占满项目导航右侧，没有浮窗圆角、阴影和外围留白。
2. 顶栏、消息区和输入区始终构成完整三段布局。
3. 训练任务条和旧分支横幅不显示。
4. 更多按钮打开贴边人格记忆面板。
5. 文本、图片、视频、语音、引用、系统消息和分支起点显示正常。
6. 历史向上加载、滚动位置恢复、发送和回复延迟展示正常。
7. 窄屏时项目导航隐藏，聊天区占满视口。

本计划不创建提交；完成后由用户决定是否提交当前工作区。

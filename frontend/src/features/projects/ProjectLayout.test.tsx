import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { Link, MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProjectLayout } from './ProjectLayout'
import { TestThemeProvider } from '../../test/TestThemeProvider'

vi.mock('./ProjectTaskBar', () => ({
  ProjectTaskBar: () => <div data-testid="project-task-bar" />,
}))

afterEach(cleanup)

function renderProjectLayout(path: string) {
  render(
    <TestThemeProvider>
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/projects/:projectId/*" element={<ProjectLayout />}>
          <Route path="*" element={<div>项目内容</div>} />
        </Route>
      </Routes>
    </MemoryRouter>
    </TestThemeProvider>,
  )
}

describe('ProjectLayout', () => {
  it('设置页隐藏所有项目并提供返回入口', () => {
    render(<TestThemeProvider><MemoryRouter initialEntries={['/settings']}><Routes>
      <Route element={<ProjectLayout />}><Route path="settings" element={<div>配置表单</div>} /></Route>
    </Routes></MemoryRouter></TestThemeProvider>)
    expect(screen.queryByRole('link', { name: '所有项目' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '返回' })).toBeInTheDocument()
  })
  it('设置使用独立的底部导航，不再混入项目工具', () => {
    renderProjectLayout('/projects/p1')
    const link = screen.getByRole('link', { name: '设置' })
    expect(link).toHaveAttribute('href', '/settings')
    expect(screen.getByRole('navigation', { name: '应用设置' })).toContainElement(link)
    expect(screen.queryByText('高级设置 / 诊断')).not.toBeInTheDocument()
  })
  it('项目列表与项目内页切换时保留同一个外壳，且新建页不冒充项目', () => {
    render(<TestThemeProvider><MemoryRouter initialEntries={['/']}><Routes>
      <Route element={<ProjectLayout />}>
        <Route index element={<Link to="/projects/p1">打开项目</Link>} />
        <Route path="projects/:projectId" element={<Link to="/projects/new">新建</Link>} />
        <Route path="projects/new" element={<div>新建表单</div>} />
      </Route>
    </Routes></MemoryRouter></TestThemeProvider>)
    const shell = screen.getByRole('main').parentElement
    expect(screen.queryByRole('link', { name: '人物资料' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('link', { name: '打开项目' }))
    expect(screen.getByRole('main').parentElement).toBe(shell)
    expect(screen.getByRole('link', { name: '人物资料' })).toHaveAttribute('href', '/projects/p1/setup/participants')
    fireEvent.click(screen.getByRole('link', { name: '所有项目' }))
    expect(screen.getByRole('main').parentElement).toBe(shell)
    fireEvent.click(screen.getByRole('link', { name: '打开项目' }))
    fireEvent.click(screen.getByRole('link', { name: '新建' }))
    expect(screen.getByText('新建表单')).toBeInTheDocument()
    expect(screen.queryByTestId('project-task-bar')).not.toBeInTheDocument()
  })
  it('分支聊天路由启用专用布局并隐藏任务栏', () => {
    renderProjectLayout('/projects/p1/branches/b1')

    const main = screen.getByRole('main')
    expect(main).toHaveClass('project-content--branch-chat')
    expect(main.parentElement).toHaveClass('project-layout--branch-chat')
    expect(screen.queryByTestId('project-task-bar')).not.toBeInTheDocument()
  })

  it('普通项目路由保留默认布局并显示任务栏', () => {
    renderProjectLayout('/projects/p1/data')

    const main = screen.getByRole('main')
    expect(main).not.toHaveClass('project-content--branch-chat')
    expect(main.parentElement).not.toHaveClass('project-layout--branch-chat')
    expect(screen.getByTestId('project-task-bar')).toBeInTheDocument()
  })

  it('新建分支路由保留默认布局并显示任务栏', () => {
    renderProjectLayout('/projects/p1/branches/new')

    const main = screen.getByRole('main')
    expect(main).not.toHaveClass('project-content--branch-chat')
    expect(main.parentElement).not.toHaveClass('project-layout--branch-chat')
    expect(screen.getByTestId('project-task-bar')).toBeInTheDocument()
  })

  it('分支准备路由保留默认布局并显示任务栏', () => {
    renderProjectLayout('/projects/p1/branches/b1/preparing')

    const main = screen.getByRole('main')
    expect(main).not.toHaveClass('project-content--branch-chat')
    expect(main.parentElement).not.toHaveClass('project-layout--branch-chat')
    expect(screen.getByTestId('project-task-bar')).toBeInTheDocument()
  })
})

import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
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

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'
import { ProjectOverviewPage } from './ProjectOverviewPage'
import { TestThemeProvider } from '../../test/TestThemeProvider'

afterEach(() => { cleanup(); localStorage.clear(); vi.unstubAllGlobals() })
function mount(extra: Record<string, unknown> = {}) {
  vi.stubGlobal('fetch', vi.fn(async (input: string) => new Response(JSON.stringify(input.endsWith('/journey') ? {
    project_id: 'p1', processing: false, stages: [], tasks: [], branches: [],
    next_action: { label: '导入记录', action: 'import', detail: '先解析再确认人物', object_id: null },
    ...extra,
  } : { id: 'p1', name: '我们的故事' }))))
  render(<TestThemeProvider><QueryClientProvider client={new QueryClient()}><MemoryRouter initialEntries={['/projects/p1']}>
    <Routes><Route path="/projects/:projectId" element={<ProjectOverviewPage />} /></Routes>
  </MemoryRouter></QueryClientProvider></TestThemeProvider>)
}
test('空项目按后端投影引导导入，不查询训练模型与旧事件', async () => {
  mount()
  expect(await screen.findByRole('link', { name: '导入记录' })).toHaveAttribute('href', '/projects/p1/setup/import')
  expect(screen.queryByRole('link', { name: '打开下一步' })).not.toBeInTheDocument()
  expect(vi.mocked(fetch).mock.calls.every(([url]) => !/models|events/.test(String(url)))).toBe(true)
})

test('已有对话显示真实预览', async () => {
  mount({ participants: [{ id: 't', name: '小洪', role: 'target', avatar_asset_id: null }],
    branches: [{ id: 'b', title: '五月', origin_time: '2026-05-08T10:00:00Z', lifecycle_status: 'active', latest_message: { role: 'self', text: '周末一起吃饭吧', virtual_time: null } }] })
  expect(await screen.findByRole('link', { name: '继续聊天' })).toHaveAttribute('href', '/projects/p1/branches/b')
  expect(screen.getByText('你：周末一起吃饭吧')).toBeInTheDocument()
})

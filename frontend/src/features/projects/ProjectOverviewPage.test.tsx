import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'

import { ProjectOverviewPage } from './ProjectOverviewPage'
import { TestThemeProvider } from '../../test/TestThemeProvider'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function renderOverview(models: unknown[] = [], events: unknown[] = []) {
  vi.stubGlobal('fetch', vi.fn().mockImplementation((input: string) => {
    let body: unknown = []
    if (input === '/api/projects/p1') {
      body = { id: 'p1', name: '我们的故事', status: 'created', created_at: '2026-01-01', updated_at: '2026-01-01' }
    } else if (input.endsWith('/events')) body = events
    else if (input.endsWith('/models')) body = models
    return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } }))
  }))
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <TestThemeProvider>
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/projects/p1']}>
        <Routes><Route path="/projects/:projectId" element={<ProjectOverviewPage />} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>
    </TestThemeProvider>,
  )
}

test('空项目明确引导用户导入聊天', async () => {
  renderOverview()
  expect(await screen.findByRole('heading', { name: '我们的故事' })).toBeInTheDocument()
  expect(screen.getByRole('link', { name: /开始导入聊天/ })).toHaveAttribute('href', '/projects/p1/data')
})

test('数字人可用后将唯一下一步指向关系时间轴', async () => {
  renderOverview([{ id: 'm1', active: true }])
  expect(await screen.findByText('选择一个想回去的时刻')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: /进入关系时间轴/ })).toHaveAttribute('href', '/projects/p1/timeline')
})

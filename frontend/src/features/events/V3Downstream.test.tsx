import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'

import { BranchCreatePage } from '../branches/BranchCreatePage'
import { TimelinePage } from '../timeline/TimelinePage'
import { TestThemeProvider } from '../../test/TestThemeProvider'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function response(body: object, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function v3Event() {
  return {
    id: 'travel-1',
    project_id: 'project-1',
    lane: 'shared_experience',
    source_lanes: ['shared_experience'],
    event_status: 'occurred',
    type: 'travel',
    title: '杭州共同旅行',
    summary: '双方完成了一次难忘的共同旅行',
    start_message_id: 'm1',
    end_message_id: 'm3',
    before_state: null,
    after_state: null,
    emotion_labels: ['开心'],
    topic: '旅行',
    conflict_level: 0,
    importance: 0.9,
    reason: '到达与回程消息证明旅行发生',
    evidence_ids: ['m1', 'm2', 'm3'],
    started_at: '2026-05-01T02:00:00Z',
    ended_at: '2026-05-03T10:00:00Z',
    status: 'active',
    created_at: '2026-07-01T00:00:00Z',
    score_components: null,
    evidence_summaries: [],
    analysis_version: 'hybrid-v3',
    prompt_version: 'v3',
    model: 'test-model',
  }
}

function renderRoute(initialEntry: string, element: ReactNode, path: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <TestThemeProvider>
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route path={path} element={element} />
          <Route path="/projects/:projectId/branches/:branchId" element={<p>分支已创建</p>} />
          <Route
            path="/projects/:projectId/branches/:branchId/preparing"
            element={<p>分支正在准备</p>}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
    </TestThemeProvider>,
  )
}

test('时间线使用V3标题、摘要和事件开始时间', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response([v3Event()])))
  renderRoute(
    '/projects/project-1/timeline',
    <TimelinePage />,
    '/projects/:projectId/timeline',
  )

  expect(await screen.findByText('杭州共同旅行')).toBeInTheDocument()
  expect(screen.getByText('双方完成了一次难忘的共同旅行')).toBeInTheDocument()
  expect(screen.getByText(/2026/)).toBeInTheDocument()
  const link = screen.getByRole('link', { name: '从这里创建分支' })
  const target = new URL(link.getAttribute('href') ?? '', 'https://example.test')
  expect(target.searchParams.get('eventId')).toBe('travel-1')
  expect(target.searchParams.get('originTime')).toBe('2026-05-01T02:00:00Z')
})

test('共同经历创建分支时只提交节点和已验收模型', async () => {
  let submitted: Record<string, unknown> | undefined
  const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input)
    if (path.endsWith('/events')) {
      return Promise.resolve(response([v3Event()]))
    }
    if (path.endsWith('/models')) {
      return Promise.resolve(
        response([
          { id: 'model-1', base_model: 'test', recommended: false, active: true },
        ]),
      )
    }
    submitted = JSON.parse(String(init?.body))
    return Promise.resolve(response({ id: 'branch-1' }, 201))
  })
  vi.stubGlobal('fetch', fetch)
  renderRoute(
    '/projects/project-1/branches/new?eventId=travel-1',
    <BranchCreatePage />,
    '/projects/:projectId/branches/new',
  )

  expect(
    await screen.findByRole('option', {
      name: '杭州共同旅行 · 双方完成了一次难忘的共同旅行',
    }),
  ).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('分支名称'), {
    target: { value: '如果那天换一种说法' },
  })
  fireEvent.click(screen.getByRole('button', { name: '进入这条时间线' }))

  await waitFor(() => expect(submitted).toBeDefined())
  expect(await screen.findByText('分支正在准备')).toBeInTheDocument()
  expect(submitted).toMatchObject({
    origin_event_id: 'travel-1',
    model_version_id: 'model-1',
    origin_time: '2026-05-01T02:00:00Z',
  })
  expect(submitted).not.toHaveProperty('state_snapshot')
})

test('创建分支失败时页面显示错误而不是静默无响应', async () => {
  const fetch = vi.fn((input: RequestInfo | URL) => {
    const path = String(input)
    if (path.endsWith('/events')) {
      return Promise.resolve(response([v3Event()]))
    }
    if (path.endsWith('/models')) {
      return Promise.resolve(
        response([
          { id: 'model-1', base_model: 'test', recommended: true, active: true },
        ]),
      )
    }
    return Promise.resolve(response({ detail: '创建失败' }, 500))
  })
  vi.stubGlobal('fetch', fetch)
  renderRoute(
    '/projects/project-1/branches/new?eventId=travel-1',
    <BranchCreatePage />,
    '/projects/:projectId/branches/new',
  )
  await screen.findByRole('option', {
    name: '杭州共同旅行 · 双方完成了一次难忘的共同旅行',
  })
  fireEvent.change(screen.getByLabelText('分支名称'), {
    target: { value: '错误可见性测试' },
  })
  fireEvent.click(screen.getByRole('button', { name: '进入这条时间线' }))

  expect(await screen.findByText('创建失败，请检查节点和模型后重试')).toBeInTheDocument()
})

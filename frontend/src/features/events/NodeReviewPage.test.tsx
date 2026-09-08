import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'

import { NodeReviewPage } from './NodeReviewPage'
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

function event(overrides: Record<string, unknown> = {}) {
  return {
    id: 'event-1',
    project_id: 'project-1',
    type: 'cold_war',
    start_message_id: 'm1',
    end_message_id: 'm2',
    before_state: '仍在沟通',
    after_state: '停止回复',
    emotion_labels: ['失望'],
    topic: '冲突',
    conflict_level: 4,
    importance: 0.9,
    reason: '回复突然中断',
    evidence_ids: ['m1', 'm2'],
    lane: 'relationship',
    event_status: 'occurred',
    title: '冷战',
    summary: '回复突然中断',
    display_summary: '那次对话停在了彼此都需要冷静的时刻，关系也因此短暂陷入沉默。',
    summary_status: 'ready',
    summary_model: 'models/Qwen3-8B-4bit',
    started_at: '2026-01-01T20:00:00Z',
    ended_at: '2026-01-01T20:01:00Z',
    source_lanes: ['relationship'],
    status: 'active',
    created_at: '2026-01-01T00:00:00Z',
    score_components: {
      state_change_strength: 0.88,
      persistence: 0.77,
      evidence_quality: 0.66,
      model_confidence: 0.55,
    },
    evidence_summaries: [
      {
        message_id: 'm1',
        sender: '甲',
        timestamp: '2026-01-01T20:00:00Z',
        content: '我想先冷静一下',
      },
      {
        message_id: 'm2',
        sender: '乙',
        timestamp: '2026-01-01T20:01:00Z',
        content: '那我们明天再说',
      },
    ],
    analysis_version: 'hybrid-v2',
    prompt_version: 'event-review-v2',
    model: 'gpt-test',
    revision_number: 1,
    analysis_run_id: 'run-1',
    ...overrides,
  }
}

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <TestThemeProvider>
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/projects/project-1/events']}>
        <Routes>
          <Route path="/projects/:projectId/events" element={<NodeReviewPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
    </TestThemeProvider>,
  )
}

test('节点卡片只显示标题和本地生成的回忆摘要', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response([event()])))
  renderPage()

  expect(await screen.findByText('冷战')).toBeInTheDocument()
  expect(
    screen.getByText('那次对话停在了彼此都需要冷静的时刻，关系也因此短暂陷入沉默。'),
  ).toBeInTheDocument()
  expect(screen.queryByText(/综合分数/)).not.toBeInTheDocument()
  expect(screen.queryByText('状态变化')).not.toBeInTheDocument()
  expect(screen.queryByText(/我想先冷静一下/)).not.toBeInTheDocument()
  expect(screen.queryByText(/hybrid-v2/)).not.toBeInTheDocument()
})

test('共同经历显示回忆摘要并支持通道筛选', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      response([
        event({ id: 'relationship-event', type: 'conflict' }),
        event({
          id: 'travel-event',
          lane: 'shared_experience',
          source_lanes: ['shared_experience'],
          type: 'travel',
          title: '杭州共同旅行',
          event_status: 'confirmed',
          summary: '双方确认五一共同旅行',
          display_summary: '两个人认真约好了五一去杭州，也开始一起期待这次出发。',
          before_state: null,
          after_state: null,
          score_components: {
            event_significance: 0.91,
            relationship_impact: 0.82,
            evidence_quality: 0.93,
            persistence: 0.44,
            type_support: 0.88,
            model_confidence: 0.86,
          },
        }),
      ]),
    ),
  )
  renderPage()

  expect(await screen.findByText('杭州共同旅行')).toBeInTheDocument()
  expect(
    screen.getByText('两个人认真约好了五一去杭州，也开始一起期待这次出发。'),
  ).toBeInTheDocument()
  expect(screen.queryByText('事件重要性')).not.toBeInTheDocument()

  fireEvent.click(screen.getByRole('radio', { name: '共同经历' }))
  expect(screen.getByText('杭州共同旅行')).toBeInTheDocument()
  expect(screen.queryByText('冷战')).not.toBeInTheDocument()

  fireEvent.click(screen.getByRole('radio', { name: '关系变化' }))
  expect(screen.getByText('冷战')).toBeInTheDocument()
  expect(screen.queryByText('杭州共同旅行')).not.toBeInTheDocument()
})

test('误报操作使用紧凑危险按钮和独立底部操作区', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response([event()])))
  renderPage()

  const button = await screen.findByRole('button', {
    name: '标记冷战节点 event-1 为误报',
  })
  expect(button).toHaveClass('event-card__reject-button')
  expect(button.parentElement).toHaveClass('event-card__actions')
})

test.each([
  '<wrapper><secret>不能泄漏</secret></wrapper>',
  '<msg><title>不能泄漏</title></msg>',
  '<div><br>不能泄漏</div>',
])('完整协议证据 %s 不会出现在精简节点卡片中', async (content) => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      response([
        event({
          evidence_summaries: [
            {
              message_id: 'm1',
              sender: '甲',
              timestamp: '2026-01-01T20:00:00Z',
              content,
            },
          ],
        }),
      ]),
    ),
  )
  renderPage()

  expect(await screen.findByText('冷战')).toBeInTheDocument()
  expect(screen.queryByText(/\[不可显示的协议消息\]/)).not.toBeInTheDocument()
  expect(screen.queryByText(/不能泄漏/)).not.toBeInTheDocument()
})

test('隐藏 rejected 与 superseded 节点', async () => {
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockResolvedValue(
        response([
          event(),
          event({ id: 'event-rejected', status: 'rejected', after_state: '误报节点' }),
          event({ id: 'event-old', status: 'superseded', after_state: '旧节点' }),
        ]),
      ),
  )
  renderPage()
  expect(
    await screen.findByText('那次对话停在了彼此都需要冷静的时刻，关系也因此短暂陷入沉默。'),
  ).toBeInTheDocument()
  expect(screen.queryByText('误报节点')).not.toBeInTheDocument()
  expect(screen.queryByText('旧节点')).not.toBeInTheDocument()
})

test('标记误报时显示 pending 并在失败后显示错误', async () => {
  let rejectRequest: ((response: Response) => void) | undefined
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(response([event()]))
    .mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          rejectRequest = resolve
        }),
    )
  vi.stubGlobal('fetch', fetch)
  renderPage()

  const button = await screen.findByRole('button', {
    name: '标记冷战节点 event-1 为误报',
  })
  fireEvent.click(button)
  expect(
    await screen.findByRole('button', {
      name: '正在标记冷战节点 event-1 为误报',
    }),
  ).toBeDisabled()
  rejectRequest?.(response({ detail: '失败' }, 500))
  expect(await screen.findByRole('alert')).toHaveTextContent('标记失败，请重试')
  expect(
    screen.getByRole('button', { name: '标记冷战节点 event-1 为误报' }),
  ).toBeInTheDocument()
})

test('按时间展示节点并确认剩余时间轴后创建训练任务', async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(
      response([
        event({
          id: 'event-late',
          title: '较晚节点',
          started_at: '2026-02-02T00:00:00Z',
        }),
        event({
          id: 'event-early',
          title: '较早节点',
          started_at: '2026-01-01T00:00:00Z',
        }),
      ]),
    )
    .mockResolvedValueOnce(
      response({
        confirmation_id: 'confirmation-1',
        training_job_id: 'job-1',
        status: 'queued',
      }),
    )
  vi.stubGlobal('fetch', fetch)
  renderPage()

  expect(await screen.findByText('较早节点')).toBeInTheDocument()
  const cards = screen.getAllByRole('article')
  expect(within(cards[0]).getByText('较早节点')).toBeInTheDocument()
  expect(screen.getByText('有效节点 2 个')).toBeInTheDocument()

  fireEvent.click(
    screen.getByRole('button', { name: '确认时间轴并开始训练' }),
  )

  await waitFor(() => {
    expect(fetch).toHaveBeenLastCalledWith(
      '/api/projects/project-1/events/confirm-and-train',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          analysis_run_id: 'run-1',
          event_revisions: [
            { event_id: 'event-early', revision_number: 1 },
            { event_id: 'event-late', revision_number: 1 },
          ],
        }),
      }),
    )
  })
})

test('每个节点独立显示误报 pending 与 error，并防止对应节点重复提交', async () => {
  const pending = new Map<string, (response: Response) => void>()
  const fetch = vi.fn((input: RequestInfo | URL) => {
    const path = String(input)
    if (path.endsWith('/events')) {
      return Promise.resolve(
        response([
          event({ id: 'event-1', type: 'conflict' }),
          event({ id: 'event-2', type: 'reconciliation' }),
        ]),
      )
    }
    return new Promise<Response>((resolve) => {
      pending.set(path, resolve)
    })
  })
  vi.stubGlobal('fetch', fetch)
  renderPage()

  const conflict = await screen.findByRole('article', { name: /冲突.*event-1/ })
  const reconciliation = screen.getByRole('article', { name: /和解.*event-2/ })
  const conflictButton = within(conflict).getByRole('button', {
    name: /标记冲突节点 event-1 为误报/,
  })
  const reconciliationButton = within(reconciliation).getByRole('button', {
    name: /标记和解节点 event-2 为误报/,
  })
  fireEvent.click(conflictButton)
  const pendingConflictButton = await within(conflict).findByRole('button', {
    name: /正在标记冲突节点 event-1 为误报/,
  })
  fireEvent.click(pendingConflictButton)
  fireEvent.click(reconciliationButton)

  expect(pendingConflictButton).toBeDisabled()
  const pendingReconciliationButton = await within(reconciliation).findByRole(
    'button',
    {
      name: /正在标记和解节点 event-2 为误报/,
    },
  )
  expect(pendingReconciliationButton).toBeDisabled()
  expect(fetch).toHaveBeenCalledTimes(3)
  pending.get('/api/projects/project-1/events/event-1')?.(
    response({ detail: '失败' }, 500),
  )
  expect(await within(conflict).findByRole('alert')).toHaveTextContent('标记失败，请重试')
  expect(within(reconciliation).queryByRole('alert')).not.toBeInTheDocument()
  expect(pendingReconciliationButton).toBeDisabled()
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render as testingRender, screen, waitFor } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'

import { BranchChatPage } from './BranchChatPage'
import { assistantRevealSchedule, mergeBranchMessages } from './chatUtils'
import type { Branch, BranchMessage } from './types'
import { TestThemeProvider } from '../../test/TestThemeProvider'

function render(ui: ReactElement) {
  return testingRender(<TestThemeProvider>{ui}</TestThemeProvider>)
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

test('合并已保存消息与临时消息时优先按客户端消息 ID 去重', () => {
  const persisted = [
    {
      id: 'u1',
      role: 'user',
      turn_id: 'ut1',
      client_message_id: 'client-1',
      bubble_index: 0,
      content: '对不起',
    },
    {
      id: 'a1',
      role: 'assistant',
      turn_id: 'at1',
      bubble_index: 0,
      content: '我听到了',
    },
  ] as BranchMessage[]
  const staged = [
    {
      id: 'local-u1',
      role: 'user',
      turn_id: 'local-turn',
      client_message_id: 'client-1',
      bubble_index: 0,
      content: '对不起',
    },
    {
      id: 'a1',
      role: 'assistant',
      turn_id: 'at1',
      bubble_index: 0,
      content: '我听到了',
    },
  ] as BranchMessage[]

  expect(mergeBranchMessages(persisted, staged)).toHaveLength(2)
})

test('同一轮数字人气泡按累计延迟依次显示', () => {
  const schedule = assistantRevealSchedule([
    { delay_ms: 0 },
    { delay_ms: 800 },
    { delay_ms: 500 },
  ] as BranchMessage[])

  expect(schedule).toEqual([0, 800, 1300])
})

test('发送消息立即确认用户消息，数字人回复异步进入列表', async () => {
  let sent = false
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((_path: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        sent = true
        const body = JSON.parse(String(init.body)) as {
          client_message_id: string
        }
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'u1',
              branch_id: 'b1',
              sequence: 0,
              role: 'user',
              content: '对不起',
              type: 'text',
              media_asset_id: null,
              turn_id: 'ut1',
              bubble_index: 0,
              delay_ms: 0,
              generation_status: 'completed',
              generation_metadata: {},
              client_message_id: body.client_message_id,
              observed_at: null,
              expression_plan_id: null,
              actor_intent: null,
              is_proactive: false,
              created_at: new Date().toISOString(),
            }),
            { status: 202, headers: { 'Content-Type': 'application/json' } },
          ),
        )
      }
      const messages = sent
        ? [
            { id: 'u1', role: 'user', content: '对不起' },
            { id: 'a1', role: 'assistant', content: '我听到了' },
            { id: 'a2', role: 'assistant', content: '你继续说呀' },
          ]
        : []
      return Promise.resolve(
        new Response(JSON.stringify(messages), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
    }),
  )
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/projects/p1/branches/b1']}>
        <Routes>
          <Route
            path="/projects/:projectId/branches/:branchId"
            element={<BranchChatPage />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )

  expect(screen.queryByText('输入新的选择')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '发送' }).querySelector('svg')).toBeNull()
  expect(
    screen.getByRole('button', { name: '更多' }).querySelector('svg'),
  ).toBeNull()
  fireEvent.change(screen.getByPlaceholderText('发消息…'), {
    target: { value: '对不起' },
  })
  fireEvent.click(screen.getByRole('button', { name: '发送' }))

  queryClient.setQueryData(['branch-messages', 'b1'], [
    {
      id: 'u1',
      role: 'user',
      content: '对不起',
      client_message_id: 'client-1',
    },
    { id: 'a1', role: 'assistant', content: '我听到了' },
    { id: 'a2', role: 'assistant', content: '你继续说呀' },
  ])

  await waitFor(() => {
    expect(screen.getAllByText('对不起')).toHaveLength(1)
    expect(screen.getAllByText('我听到了')).toHaveLength(1)
    expect(screen.getAllByText('你继续说呀')).toHaveLength(1)
  })
})

test('数字人处理上一条消息时仍可继续发送', async () => {
  const pendingPosts: Array<{
    clientMessageId: string
    messageContent: string
    resolve: (response: Response) => void
  }> = []
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((_path: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        const body = JSON.parse(String(init.body)) as {
          client_message_id: string
          content: string
        }
        return new Promise<Response>((resolve) => {
          pendingPosts.push({
            clientMessageId: body.client_message_id,
            messageContent: body.content,
            resolve,
          })
        })
      }
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
    }),
  )
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/projects/p1/branches/b1']}>
        <Routes>
          <Route
            path="/projects/:projectId/branches/:branchId"
            element={<BranchChatPage />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )

  fireEvent.change(screen.getByPlaceholderText('发消息…'), {
    target: { value: '第一条' },
  })
  fireEvent.click(screen.getByRole('button', { name: '发送' }))

  await waitFor(() => {
    expect(screen.getByText('第一条')).toBeInTheDocument()
  })
  expect(screen.getByRole('button', { name: '发送' })).toBeDisabled()
  expect(screen.getByPlaceholderText('发消息…')).not.toBeDisabled()
  fireEvent.change(screen.getByPlaceholderText('发消息…'), {
    target: { value: '第二条' },
  })
  expect(screen.getByRole('button', { name: '发送' })).not.toBeDisabled()
  fireEvent.click(screen.getByRole('button', { name: '发送' }))

  await waitFor(() => {
    expect(pendingPosts).toHaveLength(2)
    expect(screen.getByText('第二条')).toBeInTheDocument()
  })

  pendingPosts.forEach((pending, index) => {
    pending.resolve(
      new Response(
        JSON.stringify({
          id: `u${index + 1}`,
          branch_id: 'b1',
          sequence: index,
          role: 'user',
          content: pending.messageContent,
          type: 'text',
          media_asset_id: null,
          turn_id: `ut${index + 1}`,
          bubble_index: 0,
          delay_ms: 0,
          generation_status: 'completed',
          generation_metadata: {},
          client_message_id: pending.clientMessageId,
          observed_at: null,
          expression_plan_id: null,
          actor_intent: null,
          is_proactive: false,
          created_at: new Date().toISOString(),
        }),
        { status: 202, headers: { 'Content-Type': 'application/json' } },
      ),
    )
  })
  await waitFor(() => {
    expect(queryClient.isMutating()).toBe(0)
  })
})

test('消息仅包含空白字符时禁用发送按钮', () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify([]), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    ),
  )
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/projects/p1/branches/b1']}>
        <Routes>
          <Route
            path="/projects/:projectId/branches/:branchId"
            element={<BranchChatPage />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )

  fireEvent.change(screen.getByPlaceholderText('发消息…'), {
    target: { value: '   ' },
  })

  expect(screen.getByRole('button', { name: '发送' })).toBeDisabled()
})

test('点击更多按钮会打开人格记忆面板', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((path: string) => {
      const data = path.endsWith('/memory')
        ? {
            identity_kernel: { content: {} },
            current_state: {
              relationship_state: {},
              emotional_tendency: {},
              user_model: {},
            },
            active_beliefs: [],
            competing_beliefs: [],
            recent_reflections: [],
            pending_jobs: 0,
            failed_jobs: 0,
            evolution_frozen: false,
          }
        : []
      return Promise.resolve(
        new Response(JSON.stringify(data), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
    }),
  )
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/projects/p1/branches/b1']}>
        <Routes>
          <Route
            path="/projects/:projectId/branches/:branchId"
            element={<BranchChatPage />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )

  const moreButton = screen.getByRole('button', { name: '更多' })
  expect(moreButton.querySelector('svg')).toBeNull()
  fireEvent.click(moreButton)

  expect(
    await screen.findByRole('heading', { name: '人格记忆' }),
  ).toBeInTheDocument()
})

test('只读分支禁用消息输入和发送且不显示旧版提示', () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify([]), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    ),
  )
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  queryClient.setQueryData<Branch[]>(['branches', 'p1'], [
    {
      id: 'b1',
      project_id: 'p1',
      origin_event_id: 'e1',
      model_version_id: 'm1',
      title: '只读分支',
      origin_time: new Date().toISOString(),
      state_snapshot: {},
      lifecycle_status: 'archived',
      replacement_branch_id: null,
      generation_policy_version: 'v1',
      origin_import_id: null,
      origin_boundary_message_id: null,
      baseline_manifest_id: null,
      baseline_job_id: null,
      baseline_status: 'ready',
      baseline_error_code: null,
      baseline_error_message: null,
      baseline_ready_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
    },
  ])

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/projects/p1/branches/b1']}>
        <Routes>
          <Route
            path="/projects/:projectId/branches/:branchId"
            element={<BranchChatPage />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )

  expect(
    screen.queryByText('这是旧版只读分支，不会继续使用受污染的历史上下文。'),
  ).not.toBeInTheDocument()
  expect(screen.getByPlaceholderText('此分支仅供查看')).toBeDisabled()
  expect(screen.getByRole('button', { name: '发送' })).toBeDisabled()
})

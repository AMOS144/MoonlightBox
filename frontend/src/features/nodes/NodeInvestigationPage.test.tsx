import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'
import { TestThemeProvider } from '../../test/TestThemeProvider'
import { NodeInvestigationPage } from './NodeInvestigationPage'

afterEach(() => { cleanup(); sessionStorage.clear(); vi.unstubAllGlobals() })

function candidate(id: string) {
  return {
    id,
    revision: 1,
    title: `经历${id}`,
    summary: `详情${id}`,
    occurrence: '五月初',
    why_branch: '当时还有不同选择',
    uncertainties: [],
    status: 'ready',
    boundaries: [{ label: '决定之前', kind: 'message', message_ref: `m-${id}`, side: 'before' }],
  }
}

function workspace() {
  return {
    id: 'inv',
    revision: 1,
    status: 'completed',
    cursor: 20,
    total_messages: 100,
    timezone: 'Asia/Shanghai',
    graph_id: 'g',
    candidates: [candidate('a'), candidate('b')],
    confirmed: null as { candidate_id: string; cutoff_at: string; preview_hash: string } | null,
  }
}

function mount(fetcher: typeof fetch) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  vi.stubGlobal('fetch', vi.fn(fetcher))
  const view = render(<TestThemeProvider><QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={['/projects/project/nodes']}><Routes>
      <Route path="/projects/:projectId/nodes" element={<NodeInvestigationPage />} />
    </Routes></MemoryRouter>
  </QueryClientProvider></TestThemeProvider>)
  return { queryClient, ...view }
}

test('页面只让用户选择 Agent 已整理好的候选', async () => {
  const value = workspace()
  mount(async () => new Response(JSON.stringify(value), { status: 200 }))
  await screen.findByText('详情a')
  expect(screen.queryByRole('textbox')).not.toBeInTheDocument()
  expect(screen.queryByText('审阅背景')).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: /经历b/ }))
  expect(await screen.findByText('详情b')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '选择这个起点' })).toBeEnabled()
})

test('选择候选后确认唯一边界并停留在本页', async () => {
  let value = workspace()
  const calls: string[] = []
  mount(async (input, options) => {
    const url = String(input)
    if (options?.method === 'POST') calls.push(url)
    if (url.endsWith('/preview')) {
      return new Response(JSON.stringify({ preview_hash: 'boundary', candidate_revision: 1, cutoff_at: '2026-05-01T10:00:00+08:00' }), { status: 200 })
    }
    if (url.endsWith('/confirm')) {
      value = { ...value, status: 'paused', confirmed: { candidate_id: 'a', preview_hash: 'boundary', cutoff_at: '2026-05-01T10:00:00+08:00' } }
      return new Response(JSON.stringify(value), { status: 200 })
    }
    return new Response(JSON.stringify(value), { status: 200 })
  })
  fireEvent.click(await screen.findByRole('button', { name: '选择这个起点' }))
  await screen.findByText('已选择为起点')
  expect(screen.getByText('已选择')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /经历b/ })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '选择这个起点' })).not.toBeInTheDocument()
  await waitFor(() => expect(calls.some((url) => url.endsWith('/preview'))).toBe(true))
  expect(calls.some((url) => url.endsWith('/confirm'))).toBe(true)
})

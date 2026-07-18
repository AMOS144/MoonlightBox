import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'

import { NodeReviewPage } from './NodeReviewPage'

afterEach(() => {
  vi.unstubAllGlobals()
})

test('节点审核页显示变化前后状态与证据', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify([
          {
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
            status: 'active',
            created_at: '2026-01-01T00:00:00Z',
          },
        ]),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    ),
  )
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/projects/project-1/events']}>
        <Routes>
          <Route path="/projects/:projectId/events" element={<NodeReviewPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )

  expect(await screen.findByText('停止回复')).toBeInTheDocument()
  expect(screen.getByText(/m1 · m2/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '标记为误报' })).toBeInTheDocument()
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'

import { BranchChatPage } from './BranchChatPage'

afterEach(() => {
  vi.unstubAllGlobals()
})

test('发送消息后显示数字人的分支回复', async () => {
  let sent = false
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((_path: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        sent = true
        return Promise.resolve(
          new Response(
            JSON.stringify({ id: 'a1', role: 'assistant', content: '我听到了' }),
            { status: 201, headers: { 'Content-Type': 'application/json' } },
          ),
        )
      }
      const messages = sent
        ? [
            { id: 'u1', role: 'user', content: '对不起' },
            { id: 'a1', role: 'assistant', content: '我听到了' },
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

  fireEvent.change(screen.getByLabelText('输入新的选择'), {
    target: { value: '对不起' },
  })
  fireEvent.click(screen.getByRole('button', { name: '发送' }))

  expect(await screen.findByText('我听到了')).toBeInTheDocument()
})

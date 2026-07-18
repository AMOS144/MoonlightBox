import { render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import App from '../../App'

afterEach(() => {
  vi.unstubAllGlobals()
})

test('显示已有项目和创建项目入口', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            id: '2d0d2bc8-d2e5-4d8e-a0d2-e3f395313c76',
            name: '月光测试',
            status: 'created',
            created_at: '2026-07-18T08:00:00Z',
            updated_at: '2026-07-18T08:00:00Z',
          },
        ]),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    ),
  )

  render(<App />)

  expect(await screen.findByText('月光测试')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '创建项目' })).toBeInTheDocument()
})

import { cleanup, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'

import { ProjectTaskBar } from './ProjectTaskBar'
import { TestThemeProvider } from '../../test/TestThemeProvider'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

test('刷新或切换页面后从后端恢复正在训练的任务', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            id: 'job-1',
            kind: 'digital_human_training_v1',
            payload: { project_id: 'project-1' },
            status: 'running',
            progress: 0.255,
            checkpoint: {
              stage: 'training',
              iteration: 90,
              total_iterations: 600,
            },
            error_code: null,
            error_message: null,
            created_at: '2026-07-20T09:00:00Z',
            updated_at: '2026-07-20T09:30:00Z',
          },
        ]),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    ),
  )

  render(
    <TestThemeProvider>
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter>
          <ProjectTaskBar projectId="project-1" />
        </MemoryRouter>
      </QueryClientProvider>
    </TestThemeProvider>,
  )

  expect(await screen.findByText('正在训练数字人')).toBeInTheDocument()
  expect(screen.getByText(/26%/)).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '查看训练进度' })).toHaveAttribute(
    'href',
    '/projects/project-1/training/job-1',
  )
})

test('历史失败任务不会冒充当前未完成任务', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            id: 'old-failed-job',
            kind: 'digital_human_training_v1',
            payload: { project_id: 'project-1' },
            status: 'failed',
            progress: 0.9,
            checkpoint: { stage: 'model_acceptance' },
            error_code: 'training_quality_gate_failed',
            error_message: '旧候选没有通过',
            created_at: '2026-08-04T05:00:00Z',
            updated_at: '2026-08-04T05:45:00Z',
          },
        ]),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    ),
  )

  render(
    <TestThemeProvider>
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter>
          <ProjectTaskBar projectId="project-1" />
        </MemoryRouter>
      </QueryClientProvider>
    </TestThemeProvider>,
  )

  await vi.waitFor(() => expect(fetch).toHaveBeenCalled())
  expect(screen.queryByText(/未完成/)).not.toBeInTheDocument()
  expect(screen.queryByRole('status')).not.toBeInTheDocument()
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render as testingRender, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'

import { TrainingProgressPage } from './TrainingProgressPage'
import { TestThemeProvider } from '../../test/TestThemeProvider'

const render = (ui: React.ReactNode) =>
  testingRender(<TestThemeProvider>{ui}</TestThemeProvider>)

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function response(body: object) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

test('显示训练迭代和损失并允许取消', async () => {
  const running = {
    id: 'job-1',
    kind: 'digital_human_training_v1',
    status: 'running',
    progress: 0.5,
    checkpoint: {
      stage: 'training',
      iteration: 300,
      total_iterations: 600,
      train_loss: 1.23,
      validation_loss: 1.11,
    },
    error_code: null,
    error_message: null,
  }
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(response(running))
    .mockResolvedValueOnce(response({ ...running, status: 'cancelled' }))
  vi.stubGlobal('fetch', fetch)
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/projects/project-1/training/job-1']}>
        <Routes>
          <Route
            path="/projects/:projectId/training/:jobId"
            element={<TrainingProgressPage />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )

  expect(await screen.findByText('300 / 600')).toBeInTheDocument()
  expect(screen.getByText('1.23')).toBeInTheDocument()
  expect(screen.getByText('1.11')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '取消训练' }))
  expect(await screen.findByText('训练已取消')).toBeInTheDocument()
})

test('展示高保真候选、数据覆盖和双门槛失败原因', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      response({
        id: 'job-hf',
        kind: 'digital_human_training_v1',
        status: 'failed',
        progress: 0.88,
        checkpoint: {
          stage: 'model_acceptance',
          dataset_coverage: {
            target_message_count: 120,
            covered_target_message_count: 114,
            split_counts: { train: 92, valid: 11, test: 11 },
          },
          candidate_runs: {
            conservative: {
              status: 'succeeded',
              metrics: { validation_loss: 1.02, style_score: 0.76 },
            },
            expressive: {
              status: 'failed',
              error: '验证损失无效',
            },
          },
          best_candidate_id: 'conservative',
          acceptance: {
            passed: false,
            semantic_passed: true,
            style_passed: false,
            sticker_passed: true,
            memorization_passed: true,
            comparisons: {
              candidate: 0.76,
              base: 0.61,
              active: 0.79,
            },
            failure_reasons: ['风格分数未优于当前活动模型'],
          },
        },
        error_code: 'training_quality_gate_failed',
        error_message: '风格分数未优于当前活动模型',
      }),
    ),
  )
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/projects/project-1/training/job-hf']}>
        <Routes>
          <Route
            path="/projects/:projectId/training/:jobId"
            element={<TrainingProgressPage />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )

  expect(await screen.findByText('114 / 120')).toBeInTheDocument()
  expect(screen.getByText('train 92 · valid 11 · test 11')).toBeInTheDocument()
  expect(screen.getByText('conservative（优胜）')).toBeInTheDocument()
  expect(screen.getByText('base 0.61')).toBeInTheDocument()
  expect(screen.getByText('active 0.79')).toBeInTheDocument()
  expect(screen.getByText('candidate 0.76')).toBeInTheDocument()
  expect(screen.getByText('风格分数未优于当前活动模型')).toBeInTheDocument()
})

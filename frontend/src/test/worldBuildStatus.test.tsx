import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { TestThemeProvider } from './TestThemeProvider'
import { WorldBuildStatus } from '../features/projects/WorldBuildStatus'
import type { Journey } from '../features/journey/journey'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })
const build: NonNullable<Journey['world_build']> = { id: 'job1', status: 'failed', progress: .4, indexed_bundles: 20, bundle_count: 50,
  error_message: '连接中断', recovery: { status: 'waiting', retries: 1, retry_at: '2026-09-15T12:00:00Z' } }
function mount(value = build) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } })
  render(<TestThemeProvider><QueryClientProvider client={client}><WorldBuildStatus projectId="p" build={value} /></QueryClientProvider></TestThemeProvider>)
}
test('退避时展示进度和恢复时间，操作失败后可以重试', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 503 })))
  mount()
  expect(screen.getByText('图谱构建等待自动恢复')).toBeInTheDocument()
  expect(screen.getByText('已确认 20 / 50 个会话片段')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '重试构建' }))
  await waitFor(() => expect(screen.getByText('暂时无法完成操作，请稍后重试。')).toBeInTheDocument())
  expect(screen.getByRole('button', { name: '重试构建' })).toBeEnabled()
  expect(fetch).toHaveBeenCalledWith('/api/jobs/job1/resume', expect.objectContaining({ method: 'POST' }))
})
test('等待当前请求退出时不允许重复取消', () => {
  mount({ ...build, status: 'cancelling' })
  expect(screen.getByText('正在取消，等待当前请求退出')).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '取消构建' })).not.toBeInTheDocument()
})
test('内容被模型拒绝时不自动或手动重复提交', () => {
  mount({ ...build, error_message: '模型服务拒绝处理部分聊天内容，资料已保留。请检查受阻片段后再恢复。',
    failure: { code: 'lightrag_content_rejected', document_id: 'bundle-1', chunk_id: 'bundle-1-chunk-004' },
    recovery: { status: 'terminal', retries: 0, reason: 'lightrag_content_rejected' } })
  expect(screen.getByText('资料整理已暂停，等待处理')).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '立即恢复' })).not.toBeInTheDocument()
  expect(screen.getByText(/不会自动重复请求/)).toBeInTheDocument()
})

import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'
import { ProjectTaskBar } from './ProjectTaskBar'
import { TestThemeProvider } from '../../test/TestThemeProvider'
import { journeyKey } from '../journey/journey'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })
test('概览不重复呈现全局流程报错', () => {
  const client = new QueryClient({ defaultOptions: { queries: { enabled: false } } })
  client.getQueryCache().build(client, { queryKey: journeyKey('p1') }).setState({ status: 'error', error: new Error('请求失败') })
  render(<TestThemeProvider><QueryClientProvider client={client}><MemoryRouter initialEntries={['/projects/p1']}>
    <ProjectTaskBar projectId="p1" />
  </MemoryRouter></QueryClientProvider></TestThemeProvider>)
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
})
test('没有运行 Job 时仍显示需要用户回答的阶段', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({
    processing: false, tasks: [], next_action: { action: 'import' }, stages: [{ key: 'import', label: '导入记录', state: 'needs_user', detail: '那几天去旅行了吗？' }],
  }))))
  render(<TestThemeProvider><QueryClientProvider client={new QueryClient()}><MemoryRouter initialEntries={['/projects/p1/setup/import']}>
    <ProjectTaskBar projectId="p1" />
  </MemoryRouter></QueryClientProvider></TestThemeProvider>)
  expect(await screen.findByText('需要你操作')).toBeInTheDocument()
  expect(screen.getByText('那几天去旅行了吗？')).toBeInTheDocument()
  expect(screen.queryByText(/训练/)).not.toBeInTheDocument()
})
test('起点选择页内已展示调查状态，顶部不再重复提示', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({
    processing: false, tasks: [], next_action: { action: 'nodes' }, stages: [{ key: 'nodes', label: '调查并选择起点', state: 'needs_user', detail: '那几天去旅行了吗？' }],
  }))))
  render(<TestThemeProvider><QueryClientProvider client={new QueryClient()}><MemoryRouter initialEntries={['/projects/p1/nodes']}>
    <ProjectTaskBar projectId="p1" />
  </MemoryRouter></QueryClientProvider></TestThemeProvider>)
  await waitFor(() => expect(screen.queryByText('调查并选择起点')).not.toBeInTheDocument())
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
})

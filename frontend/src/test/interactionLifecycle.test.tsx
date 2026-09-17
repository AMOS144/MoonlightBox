import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { Link, MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'
import { TestThemeProvider } from './TestThemeProvider'
import { PersonWorldRevisionPanel } from '../features/world/PersonWorldRevisionPanel'
import { NodeProfilePage } from '../features/nodes/NodeProfilePage'
import { ProjectCreatePage } from '../features/projects/ProjectCreatePage'
import { ImportWizard } from '../features/data/ImportWizard'
import { BranchPreparationPage } from '../features/branches/BranchPreparationPage'

vi.mock('../features/world/V3ProfileGrid', () => ({ V3ProfileGrid: () => null }))
Object.defineProperty(document, 'fonts', { configurable: true, value: { addEventListener() {}, removeEventListener() {} } })
const clients: QueryClient[] = []
afterEach(() => {
  cleanup(); clients.forEach(c => c.clear()); clients.length = 0
  sessionStorage.clear(); localStorage.clear(); vi.unstubAllGlobals()
})
function mount(ui: ReactNode, path = '/') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  clients.push(client)
  return { client, ...render(<TestThemeProvider><QueryClientProvider client={client}><MemoryRouter initialEntries={[path]}>{ui}</MemoryRouter></QueryClientProvider></TestThemeProvider>) }
}
const json = (data: unknown) => new Response(JSON.stringify(data), { status: 200 })
const revision = { id: 'r1', status: 'waiting_for_user', selected_statements: [], messages: [], session_revision: 1, pending_turn_id: 't1', scope: {} }
const panel = <PersonWorldRevisionPanel projectId="p" opened selections={[]} onClose={() => {}} onLockChange={() => {}} onRestoreSelections={() => {}} onStartNew={() => {}} />

test('纠正发送期间锁定输入；失败保留草稿并允许重试', async () => {
  let finish!: (r: Response) => void
  vi.stubGlobal('fetch', vi.fn(async (url, init) => init?.method === 'POST'
    ? new Promise<Response>(resolve => { finish = resolve }) : json(String(url).endsWith('/r1') ? revision : [])))
  mount(panel)
  const input = screen.getByRole('textbox')
  fireEvent.change(input, { target: { value: '不能丢失的纠正' } })
  fireEvent.click(screen.getByRole('button', { name: '开始核对' }))
  await waitFor(() => expect(input).toBeDisabled())
  await act(async () => finish(new Response('{}', { status: 503 })))
  await waitFor(() => expect(input).toBeEnabled())
  expect(input).toHaveValue('不能丢失的纠正')
  fireEvent.click(screen.getByRole('button', { name: '开始核对' }))
  await waitFor(() => expect(input).toBeDisabled())
  await act(async () => finish(json(revision)))
  await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue(''))
})

test('取消与发送互斥，成功状态读取结束前不解锁旧操作', async () => {
  const posts: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (url, init) => {
    if (init?.method === 'POST') { posts.push(String(url)); return new Promise<Response>(() => {}) }
    return json(revision)
  }))
  mount(panel, '/?revision=r1')
  fireEvent.click(await screen.findByRole('button', { name: '取消本次纠正' }))
  await waitFor(() => expect(posts).toHaveLength(1))
  expect(screen.getByRole('textbox')).toBeDisabled()
  expect(screen.getByRole('button', { name: '发送' })).toBeDisabled()
  fireEvent.click(screen.getByRole('button', { name: '发送' }))
  expect(posts).toHaveLength(1)
})

test('迟到 GET 与旧 SSE 均不能回滚较新会话版本', async () => {
  let finish!: (r: Response) => void
  let emit!: (event: { data: string }) => void
  vi.stubGlobal('EventSource', class { addEventListener(_name: string, cb: typeof emit) { emit = cb } close() {} })
  vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(resolve => { finish = resolve })))
  mount(panel, '/?revision=r1')
  await waitFor(() => expect(emit).toBeDefined())
  await act(async () => emit({ data: JSON.stringify({ ...revision, status: 'agent_running', session_revision: 2 }) }))
  await act(async () => finish(json(revision)))
  await act(async () => emit({ data: JSON.stringify(revision) }))
  expect(screen.queryByRole('button', { name: '发送' })).not.toBeInTheDocument()
  expect(screen.getByText('正在核对人物资料')).toBeInTheDocument()
})

const review = { profile_id: 'p1', profile_v3: {}, failed_sections: ['identity'], approval_hash: 'h', publication_id: null, agent_run_id: 'a1', source_draft_id: 'd1', node_scope: { cutoff_at: '2026-05-01', timezone: 'Asia/Shanghai', preview_hash: 'h', investigation_id: 'i1' } }
const nodePage = <Routes><Route path="/projects/:projectId/node-profiles/:profileId" element={<NodeProfilePage />} /></Routes>
test('节点栏目重试跟踪 Job，刷新可恢复，完成后自动读取新草稿', async () => {
  let status = 'running', refreshed = 0
  vi.stubGlobal('fetch', vi.fn(async (url, init) => {
    if (String(url).endsWith('/jobs/job1')) return json({ status })
    if (String(url).endsWith('/review')) { refreshed++; return json({ ...review, profile_id: 'p2', failed_sections: [] }) }
    if (init?.method === 'POST') return json({ job_id: 'job1', section: 'identity' })
    return json(String(url).endsWith('/p2') ? { ...review, profile_id: 'p2', failed_sections: [] } : review)
  }))
  const first = mount(nodePage, '/projects/p/node-profiles/p1')
  fireEvent.click(await screen.findByRole('button', { name: '重试 identity' }))
  await waitFor(() => expect(screen.getByRole('button', { name: '重试 identity' })).toBeDisabled())
  await screen.findByText(/正在恢复所选栏目/)
  first.unmount()
  const second = mount(nodePage, '/projects/p/node-profiles/p1')
  expect(await screen.findByRole('button', { name: '重试 identity' })).toBeDisabled()
  status = 'succeeded'
  await act(async () => { await second.client.invalidateQueries({ queryKey: ['node-section-retry'] }) })
  await waitFor(() => expect(refreshed).toBe(1))
  await waitFor(() => expect(screen.queryByRole('button', { name: '重试 identity' })).not.toBeInTheDocument())
  expect(await screen.findByRole('button', { name: '开始对话' })).toBeEnabled()
})

test('创建请求期间离开页面，迟到回执不再导航', async () => {
  let finish!: (r: Response) => void
  vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(r => { finish = r })))
  mount(<><Link to="/other">离开</Link><Routes><Route path="/" element={<ProjectCreatePage />} /><Route path="/other" element={<div>其他页面</div>} /><Route path="/projects/p/setup/import" element={<div>被拉回导入</div>} /></Routes></>)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '测试' } })
  fireEvent.click(screen.getByRole('button', { name: '继续导入聊天' }))
  await waitFor(() => expect(finish).toBeDefined())
  fireEvent.click(screen.getByRole('link', { name: '离开' }))
  await act(async () => finish(json({ id: 'p' })))
  expect(screen.getByText('其他页面')).toBeInTheDocument()
  expect(screen.queryByText('被拉回导入')).not.toBeInTheDocument()
})

function importPreview() {
  sessionStorage.setItem('import-preview:p', JSON.stringify({ id: 'imp', message_count: 2, participants: ['我', '她'], time_range: null, media_file_count: 0, errors: [] }))
  sessionStorage.setItem('import-self:p', '"我"'); sessionStorage.setItem('import-target:p', '"她"')
}
test('导入完成后返回页面保留已保存状态，更换目录不沿用旧确认', async () => {
  importPreview()
  vi.stubGlobal('fetch', vi.fn(async () => json({ import_id: 'imp', message_count: 2, created: true })))
  const view = mount(<ImportWizard projectId="p" stage="participants" />)
  fireEvent.click(screen.getByRole('button', { name: '确认人物并导入' }))
  await screen.findByText(/导入完成，已保存/)
  view.unmount()
  mount(<ImportWizard projectId="p" stage="import" />)
  expect(screen.queryByText(/已解析，尚未保存/)).not.toBeInTheDocument()
  expect(screen.getByText(/导入完成，已保存/)).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('WxEcho 导出目录'), { target: { files: [new File(['next'], 'chat.csv')] } })
  expect(screen.queryByText(/导入完成，已保存/)).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '扫描并预览' })).toBeEnabled()
})
test('旧版缺少确认缓存时，也能从后台导入记录恢复状态', () => {
  importPreview()
  mount(<ImportWizard projectId="p" savedImports={[{ id: 'i', preview_id: 'imp', message_count: 2 }]} />)
  expect(screen.getByText(/导入完成，已保存/)).toBeInTheDocument()
  expect(screen.queryByText(/尚未保存/)).not.toBeInTheDocument()
})

test('分支恢复请求失败后重读成功，不再显示旧操作错误', async () => {
  let reads = 0
  vi.stubGlobal('fetch', vi.fn(async (_url, init) => init?.method === 'POST'
    ? new Response(JSON.stringify({ message: '恢复请求失败' }), { status: 503 })
    : json({ status: ++reads > 1 ? 'ready' : 'failed', stage: 'completed' })))
  mount(<Routes><Route path="/projects/:projectId/branches/:branchId/preparing" element={<BranchPreparationPage />} /></Routes>, '/projects/p/branches/b/preparing')
  fireEvent.click(await screen.findByRole('button', { name: '恢复准备' }))
  fireEvent.click(await screen.findByRole('button', { name: '重新读取' }))
  await screen.findByRole('link', { name: '进入聊天' })
  expect(screen.queryByText('恢复请求失败')).not.toBeInTheDocument()
})

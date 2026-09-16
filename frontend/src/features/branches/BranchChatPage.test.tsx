import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { Link, MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'
import { TestThemeProvider } from '../../test/TestThemeProvider'
import { BranchChatPage } from './BranchChatPage'

vi.mock('./BranchTimeline', () => ({
  BranchTimeline: ({ messages }: { messages: Array<{content: string}> }) => <div data-testid="bubbles">{messages.map(m => m.content).join('|')}</div>,
}))
vi.mock('./ChatContextPanel', () => ({ ChatContextPanel: () => null }))
vi.mock('./useAdaptiveBranchHistory', () => ({
  useAdaptiveBranchHistory: () => ({ items: [], hasNextPage: false, fetchNextPage: async () => {} }),
}))
afterEach(() => { cleanup(); sessionStorage.clear(); vi.unstubAllGlobals() })
type Sent = { content: string; client_message_id: string }
function mount(post: (body: Sent) => Promise<Response>) {
  const calls: Sent[] = []
  vi.stubGlobal('fetch', vi.fn(async (input, init) => {
    const url = String(input)
    if (init?.method === 'POST') { const body = JSON.parse(init.body); calls.push(body); return post(body) }
    const value = url.endsWith('/branches') ? ['b', 'b2'].map(id => ({ id, baseline_status: 'ready', lifecycle_status: 'active' }))
      : url.endsWith('/clock') ? { virtual_now: '2026-05-09T10:00:00+08:00', timezone: 'Asia/Shanghai' } : []
    return new Response(JSON.stringify(value), { status: 200 })
  }))
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const view = render(<TestThemeProvider><QueryClientProvider client={client}><MemoryRouter initialEntries={['/projects/p/branches/b']}>
    <Link to="/projects/p/branches/b2">另一分支</Link><Link to="/projects/p/branches/b">原分支</Link><Routes>
    <Route path="/projects/:projectId/branches/:branchId" element={<BranchChatPage />} />
  </Routes></MemoryRouter></QueryClientProvider></TestThemeProvider>)
  return { calls, client, ...view }
}
function receipt(body: Sent) {
  return new Response(JSON.stringify({message:{...body, id:body.client_message_id, branch_id:'b', role:'user'}}), {status:200})
}
test('失败保留原文；用同一幂等 ID 重试且不覆盖后续草稿', async () => {
  let count = 0
  const { calls, client } = mount(async body => ++count === 1 ? new Response('offline', {status:503}) : receipt(body))
  const input = await screen.findByRole('textbox', {name:'发消息'})
  fireEvent.change(input, {target:{value:'不能丢'}})
  fireEvent.click(screen.getByRole('button', {name:'发送'}))
  fireEvent.change(input, {target:{value:'下一句话'}})
  fireEvent.click(await screen.findByRole('button', {name:'重试发送'}))
  await waitFor(() => expect(calls).toHaveLength(2))
  expect(calls[0]).toEqual(calls[1])
  expect(input).toHaveValue('下一句话')
  expect(screen.getByTestId('bubbles')).toHaveTextContent('不能丢')
  await waitFor(() => expect(screen.queryByRole('button', {name:'重试发送'})).not.toBeInTheDocument())
  client.clear()
})
test('输入法选字 Enter 不发送，普通 Enter 发送', async () => {
  const { calls, client } = mount(async body => receipt(body))
  const input = await screen.findByRole('textbox', {name:'发消息'})
  fireEvent.change(input, {target:{value:'正在选字'}})
  fireEvent.keyDown(input, {key:'Enter', isComposing:true, keyCode:229})
  expect(calls).toHaveLength(0)
  fireEvent.keyDown(input, {key:'Enter', isComposing:false, keyCode:13})
  await waitFor(() => expect(calls).toHaveLength(1))
  client.clear()
})
test('分支切换隔离在途气泡，迟到回执只更新原分支', async () => {
  let resolve!: (response: Response) => void
  const { calls, client } = mount(() => new Promise(done => { resolve = done }))
  const input = await screen.findByRole('textbox', {name:'发消息'})
  fireEvent.change(input, {target:{value:'只属于原分支'}})
  fireEvent.click(screen.getByRole('button', {name:'发送'}))
  await waitFor(() => expect(calls).toHaveLength(1))
  fireEvent.click(screen.getByRole('link', {name:'另一分支'}))
  expect(screen.getByTestId('bubbles')).not.toHaveTextContent('只属于原分支')
  resolve(receipt(calls[0]))
  await waitFor(() => expect(client.getQueryData<Array<{status:string}>>(['branch-outbox','p','b'])?.[0].status).toBe('sent'))
  expect(screen.getByTestId('bubbles')).not.toHaveTextContent('只属于原分支')
  fireEvent.click(screen.getByRole('link', {name:'原分支'}))
  expect(screen.getByTestId('bubbles')).toHaveTextContent('只属于原分支')
  client.clear()
})
test('刷新后仍保留失败消息与原 client_message_id', async () => {
  const first = mount(async () => new Response('offline', {status:503}))
  fireEvent.change(await screen.findByRole('textbox', {name:'发消息'}), {target:{value:'重载也不能丢'}})
  fireEvent.click(screen.getByRole('button', {name:'发送'}))
  await screen.findByRole('button', {name:'重试发送'})
  first.unmount(); first.client.clear()
  const second = mount(async body => receipt(body))
  fireEvent.click(await screen.findByRole('button', {name:'重试发送'}))
  await waitFor(() => expect(second.calls).toHaveLength(1))
  expect(second.calls[0]).toEqual(first.calls[0])
  second.client.clear()
})

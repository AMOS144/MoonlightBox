import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'
import { TestThemeProvider } from '../../test/TestThemeProvider'
import { NodeCompilationPanel } from './NodeCompilationPanel'
import { NodeProfilePage } from './NodeProfilePage'

vi.mock('../world/V3ProfileGrid', () => ({ V3ProfileGrid: () => <div>节点画像内容</div> }))
vi.mock('../world/PersonWorldRevisionPanel', () => ({ PersonWorldRevisionPanel: () => null }))
afterEach(() => { cleanup(); sessionStorage.clear(); vi.unstubAllGlobals() })

test('编译、审核后一键开始对话：自动发布并建立分支进入对话', async () => {
  let published = false
  const calls: { url: string; body: Record<string, unknown> }[] = []
  vi.stubGlobal('fetch', vi.fn(async (input, options) => {
    const url = String(input), body = JSON.parse(options?.body ?? '{}')
    if (options?.method === 'POST') calls.push({ url, body })
    let value: unknown
    if (url.includes('node-compilations?')) value = null
    else if (url.endsWith('node-compilations')) value = { job_id: 'job', status: 'succeeded' }
    else if (url.endsWith('node-compilations/job')) value = { job_id: 'job', status: 'succeeded', draft: { id: 'draft' } }
    else if (url.endsWith('/approve')) { published = true; value = { publication_id: 'pub' } }
    else if (url.endsWith('/branches')) value = { id: 'branch' }
    else value = { profile_id: 'profile', profile_v3: {}, failed_sections: [], approval_hash: 'hash',
      publication_id: published ? 'pub' : null, node_scope: { investigation_id: 'inv', preview_hash: 'boundary', cutoff_at: '2026-05-01', timezone: 'Asia/Shanghai' } }
    return new Response(JSON.stringify(value), { status: 200 })
  }))
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<TestThemeProvider><QueryClientProvider client={client}><MemoryRouter initialEntries={['/start']}><Routes>
    <Route path="/start" element={<NodeCompilationPanel projectId="p" investigationId="inv" graphId="g" previewHash="boundary" />} />
    <Route path="/projects/:projectId/node-profiles/:profileId" element={<NodeProfilePage />} />
    <Route path="/projects/p/branches/branch" element={<div>分支对话页</div>} />
  </Routes></MemoryRouter></QueryClientProvider></TestThemeProvider>)
  await waitFor(() => expect(screen.getByRole('button', { name: '整理此刻的人物背景' })).not.toBeDisabled())
  fireEvent.click(screen.getByRole('button', { name: '整理此刻的人物背景' }))
  fireEvent.click(await screen.findByRole('button', { name: '审核节点背景' }))
  await screen.findByText('节点画像内容')
  expect(screen.queryByRole('button', { name: '确认并发布这个版本' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '准备分支' })).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '开始对话' }))
  await screen.findByText('分支对话页')
  expect(calls.find(c => c.url.endsWith('/branches'))?.body).toMatchObject({
    publication_id: 'pub', investigation_id: 'inv', preview_hash: 'boundary',
  })
  expect(calls.filter(c => c.url.endsWith('/approve'))).toHaveLength(1)
  client.clear()
})

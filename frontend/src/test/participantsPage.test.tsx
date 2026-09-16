import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'
import { ParticipantsPage } from '../features/data/ParticipantsPage'
import { ProjectTaskBar } from '../features/projects/ProjectTaskBar'
import { TestThemeProvider } from './TestThemeProvider'

afterEach(() => { cleanup(); vi.unstubAllGlobals(); sessionStorage.clear() })
function mount(status = 'building') {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({
    processing: false, tasks: [], next_action: { action: 'participants' }, stages: [],
    participants: [{ id: '1', name: '自己', role: 'self' }, { id: '2', name: '目标名字', role: 'target' }, { id: '3', name: '系统', role: 'other' }],
    imports: [{ id: 'i', message_count: 7020 }], time_range: [null, null], publication: null,
    graph: { id: 'g', status },
    world_build: status === 'building' ? { id: 'j', status: 'running', progress: .08, indexed_bundles: 0, bundle_count: 30, recovery: {} } : null,
  }))))
  render(<TestThemeProvider><QueryClientProvider client={new QueryClient()}><MemoryRouter initialEntries={['/projects/p/setup/participants']}><Routes>
    <Route path="/projects/:projectId/setup/participants" element={<><ProjectTaskBar projectId="p" /><ParticipantsPage /></>} />
    <Route path="/projects/:projectId/setup/graph" element={<p>建立图谱页</p>} />
  </Routes></MemoryRouter></QueryClientProvider></TestThemeProvider>)
}
test('图谱开始构建后自动前进到建立图谱步骤', async () => {
  mount('building')
  expect(await screen.findByText('建立图谱页')).toBeInTheDocument()
})
test('图谱已就绪时直接进入建立图谱步骤', async () => {
  mount('ready')
  expect(await screen.findByText('建立图谱页')).toBeInTheDocument()
})
test('待审阅背景也前进到建立图谱步骤', async () => {
  mount('awaiting_profile_review')
  expect(await screen.findByText('建立图谱页')).toBeInTheDocument()
})
test('别名审核时留在本页，仅展示目标人物', async () => {
  mount('awaiting_alias_review')
  expect(await screen.findByText('目标名字')).toBeInTheDocument()
  expect(screen.queryByText('自己')).not.toBeInTheDocument()
  expect(screen.queryByText('系统')).not.toBeInTheDocument()
  expect(screen.queryByText('建立图谱页')).not.toBeInTheDocument()
})

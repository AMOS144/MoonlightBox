import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'
import { TestThemeProvider } from '../../test/TestThemeProvider'
import { ProjectListPage } from './ProjectListPage'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })
function mount(projects: object[]) {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(projects))))
  render(<TestThemeProvider><QueryClientProvider client={new QueryClient()}><MemoryRouter><ProjectListPage /></MemoryRouter></QueryClientProvider></TestThemeProvider>)
}
test('空项目列表只保留一个创建入口', async () => {
  mount([])
  await screen.findByText('还没有项目')
  expect(screen.getAllByRole('link').filter(a => a.getAttribute('href') === '/projects/new')).toHaveLength(1)
})
test('同一人物的项目以项目名称区分，不重复展示同名标题', async () => {
  mount(['大学时期', '工作之后'].map((name, i) => ({ id: String(i), name, target_name: '小洪', updated_at: '2026-09-14T00:00:00Z' })))
  expect(await screen.findByRole('link', { name: /大学时期/ })).toHaveAttribute('href', '/projects/0')
  expect(screen.getByRole('link', { name: /工作之后/ })).toHaveAttribute('href', '/projects/1')
})

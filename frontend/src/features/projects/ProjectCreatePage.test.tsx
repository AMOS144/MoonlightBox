import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'
import { TestThemeProvider } from '../../test/TestThemeProvider'
import { ProjectCreatePage } from './ProjectCreatePage'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })
test('创建失败保留输入，重试成功后交接导入页', async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(new Response('{}', { status: 503 }))
    .mockResolvedValueOnce(new Response(JSON.stringify({ id: 'p1' }), { status: 201 }))
  vi.stubGlobal('fetch', fetchMock)
  render(<TestThemeProvider><MemoryRouter initialEntries={['/projects/new']}><Routes>
    <Route path="/projects/new" element={<ProjectCreatePage />} />
    <Route path="/projects/p1/setup/import" element={<div>导入页面</div>} />
  </Routes></MemoryRouter></TestThemeProvider>)
  expect(screen.getByRole('button', { name: '继续导入聊天' })).toBeDisabled()
  fireEvent.change(screen.getByLabelText(/项目名称/), { target: { value: '大学时期' } })
  fireEvent.click(screen.getByRole('button', { name: '继续导入聊天' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('项目创建失败')
  expect(screen.getByLabelText(/项目名称/)).toHaveValue('大学时期')
  fireEvent.click(screen.getByRole('button', { name: '继续导入聊天' }))
  expect(await screen.findByText('导入页面')).toBeInTheDocument()
  expect(fetchMock).toHaveBeenCalledTimes(2)
})

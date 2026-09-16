import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, test } from 'vitest'
import { PageBack } from '../components/PageBack'
import { TestThemeProvider } from './TestThemeProvider'

afterEach(() => { cleanup(); window.history.replaceState(null, '') })
test('直接打开页面时回到合理的上级', () => {
  render(<TestThemeProvider><MemoryRouter initialEntries={['/settings']}><Routes>
    <Route path="/settings" element={<PageBack />} /><Route path="/" element={<p>首页</p>} />
  </Routes></MemoryRouter></TestThemeProvider>)
  fireEvent.click(screen.getByRole('button', { name: '返回' }))
  expect(screen.getByText('首页')).toBeInTheDocument()
})
test('存在应用内历史时返回实际来源而不是固定首页', () => {
  window.history.replaceState({ idx: 1 }, '')
  render(<TestThemeProvider><MemoryRouter initialEntries={['/projects/p/nodes', '/settings']} initialIndex={1}><Routes>
    <Route path="/settings" element={<PageBack />} /><Route path="/projects/p/nodes" element={<p>起点调查</p>} />
  </Routes></MemoryRouter></TestThemeProvider>)
  fireEvent.click(screen.getByRole('button', { name: '返回' }))
  expect(screen.getByText('起点调查')).toBeInTheDocument()
})

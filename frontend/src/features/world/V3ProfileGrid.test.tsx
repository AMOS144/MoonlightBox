import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { TestThemeProvider } from '../../test/TestThemeProvider'
import { V3ProfileGrid } from './V3ProfileGrid'

afterEach(cleanup)

function mount(locked = false, editing = true) {
  const client = new QueryClient()
  client.setQueryData(['profile-v3-catalog'], { sections: [{ key: 'identity', label: '身份', fields: { name: '姓名' } }], labels: {}, model_groups: {}, modules: {} })
  const onToggle = vi.fn()
  render(<TestThemeProvider><QueryClientProvider client={client}><V3ProfileGrid editing={editing} projectId="p1" locked={locked} selectedKeys={new Set()} onToggle={onToggle}
    profile={{ identity: { name: { id: 'f1', description: '小洪', status: 'described', basis: 'inferred', reference_message_ids: ['m1'] } } }}
  /></QueryClientProvider></TestThemeProvider>)
  return onToggle
}

test('紧凑字段仍保留来源和完整选择回执', () => {
  const onToggle = mount()
  fireEvent.click(screen.getByRole('button', { name: '选择姓名' }))
  expect(onToggle).toHaveBeenCalledWith(expect.objectContaining({ key: 'v3:identity:f1:name', section: 'identity', field_path: 'name', entry_id: 'f1', statement: expect.objectContaining({ text: '小洪', source_message_ids: ['m1'] }) }))
})

test('锁定审核范围时不能重新选择', () => {
  const onToggle = mount(true)
  const button = screen.getByRole('button', { name: '选择姓名' })
  expect(button).toBeDisabled()
  fireEvent.click(button)
  expect(onToggle).not.toHaveBeenCalled()
})

test('阅读模式显示内容和来源，不显示编辑按钮', () => {
  mount(false, false)
  expect(screen.getByText('小洪')).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '选择姓名' })).not.toBeInTheDocument()
  expect(screen.getByRole('navigation', { name: '人物背景栏目' })).toBeInTheDocument()
})

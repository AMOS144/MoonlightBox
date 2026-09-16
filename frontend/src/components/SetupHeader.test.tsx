import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { SetupHeader } from './SetupHeader'
import { TestThemeProvider } from '../test/TestThemeProvider'

afterEach(cleanup)

test('渲染三个设置步骤并标记当前步', () => {
  render(<TestThemeProvider>
    <SetupHeader step={1} title="导入记录" description="选择聊天记录文件" />
  </TestThemeProvider>)

  const stepper = screen.getByLabelText('资料准备步骤')
  const labels = within(stepper).getAllByRole('button').map(item => item.textContent ?? '')
  expect(labels).toEqual(['1创建项目', '2导入记录', '3确认人物', '4建立图谱'])
  const current = within(stepper).getAllByRole('button').find(item => item.getAttribute('aria-current') === 'step')
  expect(current?.textContent).toContain('导入记录')
  expect(screen.getByRole('heading', { level: 1, name: '导入记录' })).toBeInTheDocument()
})

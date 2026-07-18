import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { ImportWizard } from './ImportWizard'

afterEach(() => {
  vi.unstubAllGlobals()
})

test('预览完成并映射两个角色后允许确认导入', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: 'preview-1',
          message_count: 2,
          participants: ['甲', '乙'],
          time_range: ['2026-01-01T20:00:00', '2026-01-01T20:00:12'],
          kind_counts: { text: 2 },
          sample_messages: [],
          errors: [],
        }),
        { status: 201, headers: { 'Content-Type': 'application/json' } },
      ),
    ),
  )

  render(<ImportWizard projectId="project-1" />)
  const file = new File(['时间,发送者,类型,内容'], 'chat.csv', { type: 'text/csv' })

  fireEvent.change(screen.getByLabelText('聊天记录文件'), {
    target: { files: [file] },
  })
  fireEvent.click(screen.getByRole('button', { name: '扫描并预览' }))

  expect(await screen.findByText('共 2 条消息')).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('我'), { target: { value: '乙' } })
  fireEvent.change(screen.getByLabelText('复刻对象'), { target: { value: '甲' } })

  expect(screen.getByRole('button', { name: '确认导入' })).toBeEnabled()
})

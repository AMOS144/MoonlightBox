import { cleanup, render, screen } from '@testing-library/react'
import { Alert } from '@mantine/core'
import { afterEach, expect, test } from 'vitest'
import { TestThemeProvider } from './TestThemeProvider'
import { userMessage, mediaNotice } from '../components/feedback/messages'

afterEach(cleanup)
test('用户错误文案不输出技术异常，具体业务提示仍保留', () => {
  expect(userMessage('tool_result_context_limit')).toBe('暂时无法完成操作，请稍后重试。')
  expect(userMessage(new Error('Failed to fetch'))).toContain('检查连接')
  expect(userMessage('姓名不能为空')).toBe('姓名不能为空')
  expect(userMessage([{ loc: ['body'], msg: 'bad' }], 422)).toContain('格式不正确')
  expect(userMessage('', undefined, 'quota_exhausted')).toContain('套餐用量已用完')
  expect(userMessage('', undefined, 'billing_unavailable')).toContain('余额不足')
  expect(userMessage('', undefined, 'rate_limited')).toContain('频率超限')
  expect(userMessage('', undefined, 'authentication')).toContain('API Key')
  expect(mediaNotice('missing_strong_media_reference', 16)).toContain('16 条表情消息暂未找到对应文件')
  expect(mediaNotice('unknown_internal_error', 2)).not.toContain('unknown_internal_error')
})
test('错误和警告共用中性表面，不使用大块高饱和底色', () => {
  render(<TestThemeProvider><Alert color="red" title="暂时无法读取">请稍后重试</Alert></TestThemeProvider>)
  expect(screen.getByRole('alert')).toHaveStyle({ background: 'var(--surface)' })
  expect(screen.getByText('暂时无法读取')).toBeInTheDocument()
})

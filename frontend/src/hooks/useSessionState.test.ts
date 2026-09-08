import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'

import { useSessionState } from './useSessionState'

afterEach(() => {
  cleanup()
  window.sessionStorage.clear()
})

test('按项目或分支键恢复页面临时状态', () => {
  const first = renderHook(() => useSessionState('branch:one:draft', ''))
  act(() => first.result.current[1]('还没发出的消息'))
  first.unmount()

  const restored = renderHook(() => useSessionState('branch:one:draft', ''))
  const isolated = renderHook(() => useSessionState('branch:two:draft', ''))

  expect(restored.result.current[0]).toBe('还没发出的消息')
  expect(isolated.result.current[0]).toBe('')
})

import { expect, test } from 'vitest'

import { adaptiveHistoryLimit } from './useAdaptiveBranchHistory'

test('历史首屏按视口自适应并限制在 20 到 80 条', () => {
  expect(adaptiveHistoryLimit(320, 72)).toBe(20)
  expect(adaptiveHistoryLimit(1440, 72)).toBe(30)
  expect(adaptiveHistoryLimit(10000, 72)).toBe(80)
})

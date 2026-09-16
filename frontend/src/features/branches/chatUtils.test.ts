import { describe, expect, it } from 'vitest'

import { formatChatTime, shouldShowChatTime } from './chatUtils'
import type { BranchMessage } from './types'

function message(timestamp: string): BranchMessage {
  return {
    id: timestamp,
    branch_id: 'branch',
    sequence: 0,
    role: 'user',
    content: '测试',
    type: 'text',
    media_asset_id: null,
    turn_id: timestamp,
    bubble_index: 0,
    delay_ms: 0,
    generation_status: 'completed',
    generation_metadata: {},
    created_at: timestamp,
  }
}

describe('chat time separators', () => {
  it('shows time for the first message, day changes, and five-minute gaps', () => {
    const first = message('2026-05-08T10:00:00Z')
    const close = message('2026-05-08T10:04:59Z')
    const distant = message('2026-05-08T10:05:00Z')
    const nextDay = message('2026-05-09T00:00:00Z')

    expect(shouldShowChatTime(undefined, first)).toBe(true)
    expect(shouldShowChatTime(first, close)).toBe(false)
    expect(shouldShowChatTime(first, distant)).toBe(true)
    expect(shouldShowChatTime(distant, nextDay)).toBe(true)
  })

  it('formats recent days like WeChat and older history with its date', () => {
    const reference = new Date(2026, 4, 9, 12, 0)

    expect(formatChatTime('2026-05-09T10:38:00Z', reference)).toBe('今天 10:38')
    expect(formatChatTime('2026-05-08T17:06:00Z', reference)).toBe('昨天 17:06')
    expect(formatChatTime('2026-04-01T08:00:00Z', reference)).toBe('2026年4月1日 08:00')
  })
})

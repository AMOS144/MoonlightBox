import { useCallback, useEffect, useLayoutEffect, useRef } from 'react'

import { MessageBubble } from './MessageBubble'
import type { BranchHistoryMessage, BranchMessage } from './types'

type Props = {
  history: BranchHistoryMessage[]
  messages: BranchMessage[]
  hasMore: boolean
  isFetchingMore: boolean
  loadMore: () => Promise<unknown>
  projectId: string
  timelineKey: string
  selfAvatarAssetId?: string | null
  selfAvatarName?: string
  targetAvatarAssetId?: string | null
  targetAvatarName?: string
}

const scrollPositions = new Map<string, number>()

export function BranchTimeline({
  history,
  messages,
  hasMore,
  isFetchingMore,
  loadMore,
  projectId,
  timelineKey,
  selfAvatarAssetId,
  selfAvatarName,
  targetAvatarAssetId,
  targetAvatarName,
}: Props) {
  const listRef = useRef<HTMLDivElement>(null)
  const previousHeight = useRef<number | null>(null)
  const initialized = useRef(false)
  const loading = useRef(false)

  const loadOlder = useCallback(() => {
    const element = listRef.current
    if (!element || loading.current || !hasMore) return
    loading.current = true
    previousHeight.current = element.scrollHeight
    void loadMore().finally(() => {
      loading.current = false
    })
  }, [hasMore, loadMore])

  useLayoutEffect(() => {
    const element = listRef.current
    if (!element) return
    if (previousHeight.current !== null) {
      element.scrollTop += element.scrollHeight - previousHeight.current
      previousHeight.current = null
      return
    }
    if (!initialized.current && history.length + messages.length > 0) {
      element.scrollTop = readScrollPosition(timelineKey) ?? element.scrollHeight
      initialized.current = true
    }
  }, [history.length, messages.length, timelineKey])

  useEffect(() => {
    const element = listRef.current
    if (
      !element ||
      !hasMore ||
      isFetchingMore ||
      element.scrollHeight >= element.clientHeight * 1.5
    ) {
      return
    }
    loadOlder()
  }, [hasMore, history.length, isFetchingMore, loadOlder])

  const historyMessages = history.map((message, index) => ({
    id: `history:${message.id}`,
    branch_id: '',
    sequence: index,
    role: message.role === 'self' ? ('user' as const) : ('assistant' as const),
    content: message.content,
    type: message.type,
    media_asset_id: message.media_asset_id,
    turn_id: `history:${message.id}`,
    bubble_index: 0,
    delay_ms: 0,
    generation_status: 'completed' as const,
    generation_metadata: { baseline_history: true },
    created_at: message.timestamp,
  }))

  return (
    <div
      className="message-list"
      onScroll={(event) => {
        const element = event.currentTarget
        scrollPositions.set(timelineKey, element.scrollTop)
        window.sessionStorage.setItem(
          `branch:${timelineKey}:scroll`,
          String(element.scrollTop),
        )
        if (element.scrollTop < 120 && hasMore && !isFetchingMore) {
          loadOlder()
        }
      }}
      ref={listRef}
    >
      {isFetchingMore ? <p className="history-loading">正在加载更早消息…</p> : null}
      {historyMessages.map((message, index) => (
        <MessageBubble
          key={message.id}
          message={message}
          projectId={projectId}
          avatarAssetId={
            message.role === 'user'
              ? selfAvatarAssetId
              : targetAvatarAssetId
          }
          avatarName={
            message.role === 'user' ? selfAvatarName : targetAvatarName
          }
          showAvatar={
            index === 0 ||
            historyMessages[index - 1]?.role !== message.role
          }
        />
      ))}
      {history.length > 0 ? (
        <div className="branch-timeline-divider">
          <span>从这里开始，进入新的时间分支</span>
        </div>
      ) : null}
      {messages.map((message, index) => (
        <MessageBubble
          key={`${message.id}-${index}`}
          message={message}
          projectId={projectId}
          avatarAssetId={
            message.role === 'user'
              ? selfAvatarAssetId
              : targetAvatarAssetId
          }
          avatarName={
            message.role === 'user' ? selfAvatarName : targetAvatarName
          }
          showAvatar={
            index === 0 || messages[index - 1]?.turn_id !== message.turn_id
          }
        />
      ))}
    </div>
  )
}

function readScrollPosition(timelineKey: string): number | undefined {
  const inMemory = scrollPositions.get(timelineKey)
  if (inMemory !== undefined) return inMemory
  const raw = window.sessionStorage.getItem(`branch:${timelineKey}:scroll`)
  if (raw === null) return undefined
  const stored = Number(raw)
  return Number.isFinite(stored) ? stored : undefined
}

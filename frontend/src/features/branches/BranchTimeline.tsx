import { Fragment, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Button } from '@mantine/core'

import { MessageBubble } from './MessageBubble'
import { formatChatTime, messageChatTime, shouldShowChatTime } from './chatUtils'
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

type ScrollPosition = {
  top: number
  max: number
}

const scrollPositions = new Map<string, ScrollPosition>()

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
  const nearBottom = useRef(true)
  const previousMessageCount = useRef(messages.length)
  const [hasUnread, setHasUnread] = useState(false)

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
    const addedMessages = messages.length > previousMessageCount.current
    previousMessageCount.current = messages.length
    if (previousHeight.current !== null) {
      element.scrollTop += element.scrollHeight - previousHeight.current
      previousHeight.current = null
      return
    }
    if (!initialized.current && history.length + messages.length > 0) {
      // 只有曾在可滚动列表中记录的位置才恢复。旧版本只保存一个数字；当首屏
      // 内容尚未撑满列表时它必然是 0，之后历史消息加载完成再重挂载时便会错误地
      // 把用户送回最早消息。这样的旧记录直接忽略，首次进入始终定位到最新消息。
      const saved = readScrollPosition(timelineKey)
      const max = Math.max(0, element.scrollHeight - element.clientHeight)
      element.scrollTop = saved ? Math.min(saved.top, max) : element.scrollHeight
      initialized.current = true
      nearBottom.current = element.scrollHeight - element.clientHeight - element.scrollTop < 80
    } else if (addedMessages) {
      // 正在读旧记录时只提示，不把阅读位置拉回最新消息。
      if (nearBottom.current) element.scrollTop = element.scrollHeight
      else setHasUnread(true)
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
    observed_at: message.timestamp,
    created_at: message.timestamp,
  }))
  const visibleHistoryMessages = historyMessages.filter(isVisibleMessage)
  const visibleMessages = messages.filter(isVisibleMessage)

  return (
    <div
      className="message-list"
      onScroll={(event) => {
        const element = event.currentTarget
        nearBottom.current = element.scrollHeight - element.clientHeight - element.scrollTop < 80
        if (nearBottom.current) setHasUnread(false)
        const position = {
          top: element.scrollTop,
          max: Math.max(0, element.scrollHeight - element.clientHeight),
        }
        scrollPositions.set(timelineKey, position)
        window.sessionStorage.setItem(
          `branch:${timelineKey}:scroll`,
          JSON.stringify(position),
        )
        if (element.scrollTop < 120 && hasMore && !isFetchingMore) {
          loadOlder()
        }
      }}
      ref={listRef}
    >
      {isFetchingMore ? <p className="history-loading">正在加载更早消息…</p> : null}
      {visibleHistoryMessages.map((message, index) => (
        <Fragment key={message.id}>
          <ChatTimeSeparator
            message={message}
            show={shouldShowChatTime(visibleHistoryMessages[index - 1], message)}
          />
          <MessageBubble
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
              visibleHistoryMessages[index - 1]?.role !== message.role
            }
          />
        </Fragment>
      ))}
      {history.length > 0 ? (
        <div className="branch-timeline-divider">
          <span>从这里开始，进入新的时间分支</span>
        </div>
      ) : null}
      {visibleMessages.map((message, index) => (
        // Runtime 消息的 id 在本地乐观气泡和服务端气泡中均唯一。不要把数组下标
        // 放进 key：轮询后若服务端补入消息，React 会把后续气泡当成新节点重建，
        // 列表高度变化时就容易出现视觉跳动。
        <Fragment key={message.id}>
          <ChatTimeSeparator
            message={message}
            show={shouldShowChatTime(visibleMessages[index - 1], message)}
          />
          <MessageBubble
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
              visibleMessages[index - 1]?.turn_id !== message.turn_id
            }
          />
        </Fragment>
      ))}
      {hasUnread && <Button size="xs" variant="light" style={{ position: 'sticky', bottom: 8, alignSelf: 'center', zIndex: 2 }} onClick={() => {
        if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight
        nearBottom.current = true
        setHasUnread(false)
      }}>有新消息，查看最新</Button>}
    </div>
  )
}

function ChatTimeSeparator({
  message,
  show,
}: {
  message: BranchMessage
  show: boolean
}) {
  const timestamp = messageChatTime(message)
  if (!show || !timestamp) return null
  return (
    <time className="chat-time-separator" dateTime={timestamp}>
      {formatChatTime(timestamp)}
    </time>
  )
}

function isVisibleMessage(message: BranchMessage): boolean {
  return message.generation_metadata?.retracted !== true
}

function readScrollPosition(timelineKey: string): ScrollPosition | undefined {
  const inMemory = scrollPositions.get(timelineKey)
  if (inMemory !== undefined) return inMemory
  const raw = window.sessionStorage.getItem(`branch:${timelineKey}:scroll`)
  if (raw === null) return undefined
  try {
    const parsed = JSON.parse(raw) as Partial<ScrollPosition>
    if (
      typeof parsed.top === 'number'
      && Number.isFinite(parsed.top)
      && typeof parsed.max === 'number'
      && Number.isFinite(parsed.max)
      && parsed.max > 0
    ) {
      return { top: Math.max(0, parsed.top), max: parsed.max }
    }
  } catch {
    // 旧版保存的是裸数字；无法判断它是否只是内容未加载时产生的 0，安全地忽略。
  }
  return undefined
}

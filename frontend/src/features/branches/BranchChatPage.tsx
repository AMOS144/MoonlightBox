import { useQuery } from '@tanstack/react-query'
import { Alert, Button, Textarea } from '@mantine/core'
import { useState } from 'react'
import { useParams } from 'react-router-dom'

import { request } from '../../api/client'
import { useSessionState } from '../../hooks/useSessionState'
import { BranchTimeline } from './BranchTimeline'
import { BranchPreparationPage } from './BranchPreparationPage'
import { ChatContextPanel } from './ChatContextPanel'
import { mergeBranchMessages } from './chatUtils'
import type {
  Branch,
  BranchMessage,
  BranchVirtualClock,
} from './types'
import { useAdaptiveBranchHistory } from './useAdaptiveBranchHistory'
import { useBranchOutbox } from './useBranchOutbox'

type Avatar = {
  role: 'self' | 'target'
  name: string
  asset_id: string | null
}

export function BranchChatPage() {
  const { projectId, branchId } = useParams()
  return <BranchChat key={`${projectId}:${branchId}`} projectId={projectId ?? ''} branchId={branchId ?? ''} />
}

function BranchChat({ projectId, branchId }: { projectId: string; branchId: string }) {
  const [content, setContent] = useSessionState(
    `branch:${projectId ?? ''}:${branchId ?? ''}:draft`,
    '',
  )
  const [contextOpen, setContextOpen] = useState(false)
  const branches = useQuery({
    queryKey: ['branches', projectId],
    queryFn: () => request<Branch[]>(`/api/projects/${projectId}/branches`),
    enabled: Boolean(projectId),
  })
  const branch = branches.data?.find((item) => item.id === branchId)
  const baselineReady = Boolean(
    branch &&
      (branch.baseline_status === undefined ||
        branch.baseline_status === 'ready'),
  )
  const isReadOnly = branch ? branch.lifecycle_status !== 'active' : false
  const avatars = useQuery({
    queryKey: ['project-avatars', projectId],
    queryFn: () =>
      request<Avatar[]>(`/api/projects/${projectId}/media/avatars/list`),
    enabled: Boolean(projectId),
  })
  const selfAvatar = avatars.data?.find((item) => item.role === 'self')
  const targetAvatar = avatars.data?.find((item) => item.role === 'target')
  const messages = useQuery({
    queryKey: ['branch-messages', branchId],
    queryFn: () =>
      request<BranchMessage[]>(
        `/api/projects/${projectId}/branches/${branchId}/messages`,
      ),
    enabled: Boolean(projectId && branchId && baselineReady),
    refetchInterval: 750,
  })
  const clock = useQuery({
    queryKey: ['branch-virtual-clock', branchId],
    queryFn: () =>
      request<BranchVirtualClock>(
        `/api/projects/${projectId}/branches/${branchId}/clock`,
      ),
    enabled: Boolean(projectId && branchId && baselineReady),
    // 只显示到分钟，定时读取足以让展示和服务器的 VirtualClock 保持一致。
    refetchInterval: 15_000,
  })
  const baselineHistory = useAdaptiveBranchHistory(
    projectId,
    branchId,
  )
  const outbox = useBranchOutbox(projectId, branchId, messages.data ?? [])
  const allMessages = mergeBranchMessages(messages.data ?? [], outbox.pending.map(item => item.message))

  if (branches.isError) {
    return <section className="chat-route-state"><h1>无法打开这条时间线</h1><Alert color="red">分支信息读取失败，请检查服务后重试。</Alert><Button onClick={() => branches.refetch()} type="button">重新读取</Button></section>
  }
  if (branches.isLoading) return <section className="chat-route-state">正在读取分支……</section>
  if (!branch) return <Alert color="red">这条分支不存在，请返回我的分支重新选择。</Alert>
  if (!baselineReady) return <BranchPreparationPage />
  return (
    <section className="branch-chat">
      <div>
        <header className="branch-chat__header">
        {targetAvatar?.asset_id ? (
          <img
            alt={`${targetAvatar.name}头像`}
            className="chat-avatar"
            src={`/api/projects/${projectId}/media/${targetAvatar.asset_id}`}
          />
        ) : (
          <div className="chat-avatar">
            {(targetAvatar?.name ?? '对方').slice(0, 1)}
          </div>
        )}
        <div>
          <h1>{targetAvatar?.name ?? '对方'}</h1>
        </div>
        {clock.data ? (
          <div
            className="branch-virtual-clock"
            title={`分支时区：${clock.data.timezone}`}
          >
            <span>虚拟时间</span>
            <time dateTime={clock.data.virtual_now}>
              {formatVirtualTime(clock.data.virtual_now)}
            </time>
          </div>
        ) : null}
        <Button variant="subtle" onClick={() => setContextOpen(true)}>资料</Button>
        </header>
        <ChatContextPanel projectId={projectId ?? ''} branchId={branchId ?? ''} opened={contextOpen} onClose={() => setContextOpen(false)} />
        {clock.data?.day_plan_preparation && clock.data.day_plan_preparation.status !== 'ready' ? (
          <div role="status" className="muted">
            {clock.data.has_day_plan
              ? clock.data.day_plan_preparation.status === 'blocked'
                ? '日程更新受阻，目前保留原日程，需要处理后重试。'
                : clock.data.day_plan_preparation.status === 'retryable_failed'
                  ? '日程更新暂未成功，目前保留原日程，稍后自动重试。'
                  : '正在更新日程，目前仍使用已提交的安排。'
              : clock.data.day_plan_preparation.status === 'blocked'
              ? '当天日程准备受阻，需要处理后重试；仍可聊天，但暂时无法确认具体安排。'
              : clock.data.day_plan_preparation.status === 'retryable_failed'
                ? '当天日程暂未准备成功，稍后自动重试；仍可聊天。'
                : '正在准备当天日程；现在发送的消息可能需要等待准备完成，具体安排暂待确认。'}
          </div>
        ) : null}
      </div>
      <BranchTimeline
        hasMore={Boolean(baselineHistory.hasNextPage)}
        history={baselineHistory.items}
        isFetchingMore={baselineHistory.isFetchingNextPage}
        loadMore={() => baselineHistory.fetchNextPage()}
        messages={allMessages}
        projectId={projectId ?? ''}
        timelineKey={branchId ?? ''}
        selfAvatarAssetId={selfAvatar?.asset_id}
        selfAvatarName={selfAvatar?.name}
        targetAvatarAssetId={targetAvatar?.asset_id}
        targetAvatarName={targetAvatar?.name}
      />
      {messages.isError && <Alert color="red">消息读取失败，已有内容保留。<Button variant="subtle" onClick={() => void messages.refetch()}>重新读取消息</Button></Alert>}
      {outbox.pending.filter(item => item.status === 'failed').map(item =>
        <div className="message-error" role="alert" key={item.message.client_message_id}>
          未确认发送：{item.message.content}
          <Button type="button" variant="subtle" disabled={isReadOnly} onClick={() => outbox.enqueue(item.message)}>重试发送</Button>
        </div>,
      )}
      <form
        className="message-form"
        onSubmit={(event) => {
          event.preventDefault()
          const messageContent = content.trim()
          if (messageContent && !isReadOnly) {
            const clientMessageId = crypto.randomUUID()
            setContent('')
            outbox.enqueue({
                id: `local-${crypto.randomUUID()}`,
                branch_id: branchId ?? '',
                sequence: allMessages.length,
                role: 'user',
                content: messageContent,
                type: 'text',
                media_asset_id: null,
                turn_id: `local-${crypto.randomUUID()}`,
                bubble_index: 0,
                delay_ms: 0,
                generation_status: 'completed',
                generation_metadata: {},
                client_message_id: clientMessageId,
                // 乐观气泡属于分支时间线，先使用上次服务器返回的虚拟钟面；
                // 服务端落盘后会用精确 observed_at 替换，不混入浏览器真实时间。
                observed_at: clock.data?.virtual_now ?? null,
                expression_plan_id: null,
                actor_intent: null,
                is_proactive: false,
                created_at: new Date().toISOString(),
            })
          }
        }}
      >
        <Textarea
          className="message-form__input"
          aria-label="发消息"
          id="branch-message"
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) {
              event.preventDefault()
              event.currentTarget.form?.requestSubmit()
            }
          }}
          onChange={(event) => setContent(event.target.value)}
          placeholder={isReadOnly ? '此分支仅供查看' : '发消息…'}
          value={content}
          disabled={isReadOnly}
        />

        <Button
          className="message-form__send"
          disabled={isReadOnly || content.trim().length === 0}
          type="submit"
        >
          发送
        </Button>
      </form>
    </section>
  )
}

function formatVirtualTime(virtualNow: string) {
  // 后端已经按分支的 VirtualClock 给出钟面时间；这里不调用浏览器时区转换，避免
  // 访问者所在时区覆盖分支时区。FastAPI ISO 字符串的前 16 位恰为 YYYY-MM-DDTHH:mm。
  const value = virtualNow.replace('T', ' ').slice(0, 16)
  return value
}

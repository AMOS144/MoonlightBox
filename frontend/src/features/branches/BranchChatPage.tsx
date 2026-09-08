import { useMutation, useQuery } from '@tanstack/react-query'
import { ActionIcon, Alert, Button, Textarea } from '@mantine/core'
import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import { request } from '../../api/client'
import { useSessionState } from '../../hooks/useSessionState'
import { BranchMemoryPanel } from './BranchMemoryPanel'
import { BranchTimeline } from './BranchTimeline'
import { mergeBranchMessages } from './chatUtils'
import type {
  Branch,
  BranchMessage,
  ConversationActorState,
} from './types'
import { useAdaptiveBranchHistory } from './useAdaptiveBranchHistory'

type Avatar = {
  role: 'self' | 'target'
  name: string
  asset_id: string | null
}

type RuntimeMessageQueued = {
  event_id: string
  message_id: string
  message: BranchMessage
}

export function BranchChatPage() {
  const { projectId, branchId } = useParams()
  const navigate = useNavigate()
  const [content, setContent] = useSessionState(
    `branch:${projectId ?? ''}:${branchId ?? ''}:draft`,
    '',
  )
  const [stagedBubbles, setStagedBubbles] = useState<BranchMessage[]>([])
  const [sendError, setSendError] = useState(false)
  const [memoryOpen, setMemoryOpen] = useState(false)
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
  useEffect(() => {
    if (branch && !baselineReady) {
      navigate(`/projects/${projectId}/branches/${branch.id}/preparing`, {
        replace: true,
      })
    }
  }, [baselineReady, branch, navigate, projectId])
  useEffect(() => {
    if (!projectId || !branchId || !baselineReady || isReadOnly) return
    const timeout = window.setTimeout(() => {
      void request(
        `/api/projects/${projectId}/branches/${branchId}/typing`,
        {
          method: 'PUT',
          body: JSON.stringify({ typing: content.length > 0 }),
        },
      )
    }, 250)
    return () => window.clearTimeout(timeout)
  }, [baselineReady, branchId, content, isReadOnly, projectId])
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
  const actorState = useQuery({
    queryKey: ['branch-conversation-state', branchId],
    queryFn: () =>
      request<ConversationActorState>(
        `/api/projects/${projectId}/branches/${branchId}/conversation-state`,
      ),
    enabled: Boolean(projectId && branchId && baselineReady),
    refetchInterval: 500,
  })
  const baselineHistory = useAdaptiveBranchHistory(
    projectId,
    branchId,
    branch?.baseline_manifest_id,
  )
  const send = useMutation({
    mutationFn: ({
      messageContent,
      clientMessageId,
    }: {
      messageContent: string
      clientMessageId: string
    }) =>
      request<RuntimeMessageQueued | BranchMessage>(
        `/api/projects/${projectId}/branches/${branchId}/runtime/messages`,
        {
          method: 'POST',
          body: JSON.stringify({
            content: messageContent,
            // Runtime EventQueue 用幂等键防止网络重试让同一消息进入两次 Director。
            idempotency_key: clientMessageId,
            client_message_id: clientMessageId,
          }),
        },
      ),
    onSuccess: (queued) => {
      // 兼容本地开发时仍在运行的旧 API 响应，正式 Runtime 返回 event_id + message。
      const message = 'message' in queued ? queued.message : queued
      setSendError(false)
      setStagedBubbles((current) => [
        ...current.filter(
          (item) => item.client_message_id !== message.client_message_id,
        ),
        message,
      ])
      void messages.refetch().then(() => {
        setStagedBubbles((current) =>
          current.filter(
            (item) => item.client_message_id !== message.client_message_id,
          ),
        )
      })
    },
    onError: (_error, variables) => {
      setSendError(true)
      setStagedBubbles((current) => {
        return current.filter(
          (item) => item.client_message_id !== variables.clientMessageId,
        )
      })
    },
  })
  const allMessages = mergeBranchMessages(messages.data ?? [], stagedBubbles)

  if (branches.isError) {
    return <section className="chat-route-state"><h1>无法打开这条时间线</h1><Alert color="red">分支信息读取失败，请检查服务后重试。</Alert><Button onClick={() => branches.refetch()} type="button">重新读取</Button></section>
  }
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
            {(targetAvatar?.name ?? '小肥入').slice(0, 1)}
          </div>
        )}
        <div>
          <h1>{targetAvatar?.name ?? '小肥入'}</h1>
          {actorState.data?.typing ? (
            <p>对方正在输入…</p>
          ) : null}
        </div>
        <ActionIcon
          aria-label="更多"
          className="branch-memory-button"
          onClick={() => setMemoryOpen(true)}
          size="lg"
          type="button"
          variant="subtle"
        >
          <span aria-hidden="true">···</span>
        </ActionIcon>
        </header>
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
      <form
        className="message-form"
        onSubmit={(event) => {
          event.preventDefault()
          const messageContent = content.trim()
          if (messageContent) {
            const clientMessageId = crypto.randomUUID()
            setContent('')
            setSendError(false)
            setStagedBubbles((current) => [
              ...current,
              {
                id: `local-${crypto.randomUUID()}`,
                branch_id: branchId ?? '',
                sequence: current.length,
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
                observed_at: null,
                expression_plan_id: null,
                actor_intent: null,
                is_proactive: false,
                created_at: new Date().toISOString(),
              },
            ])
            send.mutate({ messageContent, clientMessageId })
          }
        }}
      >
        <Textarea
          aria-label="发消息"
          id="branch-message"
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              event.currentTarget.form?.requestSubmit()
            }
          }}
          onChange={(event) => setContent(event.target.value)}
          placeholder={isReadOnly ? '此分支仅供查看' : '发消息…'}
          value={content}
          disabled={isReadOnly}
        />
        {sendError ? <p className="message-error">发送失败，请重试</p> : null}
        <Button
          disabled={isReadOnly || content.trim().length === 0}
          type="submit"
        >
          发送
        </Button>
      </form>
      {import.meta.env.DEV && (
        <div
          style={{
            position: 'fixed',
            top: 70,
            right: 12,
            zIndex: 9999,
            padding: 8,
            background: '#700',
            color: '#fff',
            fontSize: 12,
            maxWidth: 320,
          }}
        >
          msgLen={messages.data?.length ?? -1} all={allMessages.length} hist={baselineHistory.items.length}
          <br />
          isLoading={String(messages.isLoading)} isError={String(messages.isError)} err={messages.error?.message ?? 'none'}
          <br />
          baselineReady={String(baselineReady)} branchStatus={branch?.baseline_status ?? 'no-branch'}
          <br />
          first={messages.data?.[0]?.content ?? 'none'}
        </div>
      )}
      {memoryOpen && projectId && branchId ? (
        <BranchMemoryPanel
          branchId={branchId}
          onClose={() => setMemoryOpen(false)}
          projectId={projectId}
        />
      ) : null}
    </section>
  )
}

import { useMutation, useQuery } from '@tanstack/react-query'
import { Alert, Button, Paper, Stack, Text, Textarea, Title } from '@mantine/core'
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'

import { request } from '../../api/client'
import { useSessionState } from '../../hooks/useSessionState'
import type { RuntimeBranch, RuntimeBranchMessage } from './types'

type QueuedMessage = { event_id: string; message_id: string; message: RuntimeBranchMessage }

export function BranchChatPage() {
  const { projectId, branchId } = useParams()
  const [content, setContent] = useSessionState(`runtime-branch:${branchId ?? ''}:draft`, '')
  const [bootstrapped, setBootstrapped] = useState(false)
  const branches = useQuery({
    queryKey: ['runtime-branches', projectId],
    queryFn: () => request<RuntimeBranch[]>(`/api/projects/${projectId}/runtime/branches`),
    enabled: Boolean(projectId),
  })
  const messages = useQuery({
    queryKey: ['runtime-branch-messages', projectId, branchId],
    queryFn: () => request<RuntimeBranchMessage[]>(`/api/projects/${projectId}/runtime/branches/${branchId}/messages`),
    enabled: Boolean(projectId && branchId && bootstrapped),
    refetchInterval: 1000,
  })
  useEffect(() => {
    if (!projectId || !branchId) return
    void request(`/api/projects/${projectId}/branches/${branchId}/runtime/bootstrap`, { method: 'POST' })
      .then(() => setBootstrapped(true))
  }, [projectId, branchId])
  const send = useMutation({
    mutationFn: (message: string) => request<QueuedMessage>(
      `/api/projects/${projectId}/branches/${branchId}/runtime/messages`,
      {
        method: 'POST',
        body: JSON.stringify({
          content: message,
          idempotency_key: crypto.randomUUID(),
          client_message_id: crypto.randomUUID(),
        }),
      },
    ),
    onSuccess: () => { setContent(''); void messages.refetch() },
  })
  const branch = branches.data?.find((item) => item.id === branchId)

  return <Stack maw={780} mx="auto">
    <div>
      <Text c="moon.4" fw={700} size="xs">RUNTIME V1</Text>
      <Title order={2}>{branch?.title ?? '虚拟时间线'}</Title>
      <Text c="dimmed" size="sm">消息进入 EventQueue，由 Director、PersonaActor 与 Executor 顺序处理。</Text>
    </div>
    {branches.isError || messages.isError ? <Alert color="red">无法读取 Runtime 分支。</Alert> : null}
    {!bootstrapped ? <Text role="status">正在初始化世界快照与虚拟时钟…</Text> : null}
    <Paper p="md" withBorder>
      <Stack gap="sm">
        {messages.data?.map((message) => <Paper key={message.id} p="sm" bg={message.role === 'user' ? 'moon.0' : 'gray.0'}>
          <Text c="dimmed" size="xs">{message.role === 'user' ? '你' : '对方'} · {message.observed_at ? new Date(message.observed_at).toLocaleTimeString('zh-CN') : ''}</Text>
          <Text>{message.content}</Text>
        </Paper>)}
        {messages.isSuccess && messages.data.length === 0 ? <Text c="dimmed">尚未有消息。Runtime 已就绪后即可开始对话。</Text> : null}
      </Stack>
    </Paper>
    <form onSubmit={(event) => { event.preventDefault(); const text = content.trim(); if (text) send.mutate(text) }}>
      <Stack>
        <Textarea aria-label="发消息" disabled={!bootstrapped || branch?.lifecycle_status !== 'active'} onChange={(event) => setContent(event.target.value)} placeholder="发消息…" value={content} />
        {send.isError ? <Alert color="red">消息入队失败，请重试。</Alert> : null}
        <Button disabled={!content.trim() || send.isPending || !bootstrapped} loading={send.isPending} type="submit">发送</Button>
      </Stack>
    </form>
  </Stack>
}

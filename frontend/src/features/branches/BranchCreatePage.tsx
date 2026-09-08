import { useMutation, useQuery } from '@tanstack/react-query'
import { Alert, Button, NativeSelect, Paper, Stack, Text, TextInput, Title } from '@mantine/core'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'

import { request } from '../../api/client'
import { useSessionState } from '../../hooks/useSessionState'
import type { EventNode } from '../events/types'
import { modelDisplayName, type ModelVersion } from '../models/types'
import type { RuntimeBranch } from './types'

export function BranchCreatePage() {
  const { projectId } = useParams()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const routeEventId = searchParams.get('eventId') ?? ''
  const key = `runtime-branch-create:${projectId ?? ''}:${routeEventId}`
  const [title, setTitle] = useSessionState(`${key}:title`, '')
  const [eventId, setEventId] = useSessionState(`${key}:event`, routeEventId)
  const [modelId, setModelId] = useSessionState(`${key}:model`, '')
  const events = useQuery({
    queryKey: ['events', projectId],
    queryFn: () => request<EventNode[]>(`/api/projects/${projectId}/events`),
  })
  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => request<ModelVersion[]>(`/api/projects/${projectId}/models`),
  })
  const create = useMutation({
    mutationFn: () => {
      const event = events.data?.find((item) => item.id === eventId)
      const ready = models.data?.filter((item) => item.status === 'ready') ?? []
      const selected = modelId || ready.find((item) => item.active)?.id || ready[0]?.id
      if (!event || !selected) throw new Error('请选择节点和已就绪的 LoRA 模型')
      return request<RuntimeBranch>(`/api/projects/${projectId}/runtime/branches`, {
        method: 'POST',
        body: JSON.stringify({
          origin_event_id: event.id,
          model_version_id: selected,
          title: title.trim(),
          origin_time: searchParams.get('originTime') ?? event.started_at ?? event.created_at,
        }),
      })
    },
    onSuccess: (branch) => navigate(`/projects/${projectId}/branches/${branch.id}`),
  })
  const selectedEvent = events.data?.find((item) => item.id === eventId)
  const readyModels = models.data?.filter((item) => item.status === 'ready') ?? []

  return <Paper component="form" maw={680} onSubmit={(event) => { event.preventDefault(); create.mutate() }} p="xl" withBorder>
    <Stack>
      <Text c="moon.4" fw={700} size="xs">RUNTIME 时间线</Text>
      <Title order={2}>开始一条虚拟生活线</Title>
      <Text c="dimmed" size="sm">将使用当前完成的世界图谱；LightRAG 暂不按节点时间回溯。</Text>
      {events.isError || models.isError ? <Alert color="red">创建所需资料读取失败。</Alert> : null}
      {create.isError ? <Alert color="red">{create.error.message}</Alert> : null}
      <TextInput aria-label="分支名称" label="分支名称" onChange={(event) => setTitle(event.target.value)} required value={title} />
      <NativeSelect aria-label="起点节点" label="起点节点" onChange={(event) => setEventId(event.target.value)} required value={eventId}>
        <option value="">请选择</option>
        {events.data?.map((event) => <option key={event.id} value={event.id}>{event.title || event.topic}</option>)}
      </NativeSelect>
      <NativeSelect aria-label="模型版本" label="LoRA 模型" onChange={(event) => setModelId(event.target.value)} value={modelId}>
        <option value="">使用当前活动模型</option>
        {readyModels.map((model) => <option key={model.id} value={model.id}>{modelDisplayName(model)}</option>)}
      </NativeSelect>
      {selectedEvent ? <Text c="dimmed" size="sm">起点：{new Date(selectedEvent.started_at ?? selectedEvent.created_at).toLocaleString('zh-CN')}</Text> : null}
      <Button disabled={create.isPending || !selectedEvent || !title.trim() || readyModels.length === 0} loading={create.isPending} type="submit">创建并进入 Runtime</Button>
    </Stack>
  </Paper>
}

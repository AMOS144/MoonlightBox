import { useMutation, useQuery } from '@tanstack/react-query'
import { Alert, Button, NativeSelect, Paper, Stack, Text, TextInput, Title } from '@mantine/core'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'

import { request } from '../../api/client'
import { Icon } from '../../components/Icon'
import { useSessionState } from '../../hooks/useSessionState'
import type { EventNode } from '../events/types'
import { modelDisplayName, type ModelVersion } from '../models/types'
import type { Branch } from './types'

export function BranchCreatePage() {
  const { projectId } = useParams()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const routeEventId = searchParams.get('eventId') ?? ''
  const draftKey = `branch-create:${projectId ?? ''}:${routeEventId}`
  const [title, setTitle] = useSessionState(`${draftKey}:title`, '')
  const [eventId, setEventId] = useSessionState(
    `${draftKey}:event`,
    routeEventId,
  )
  const [modelId, setModelId] = useSessionState(`${draftKey}:model`, '')
  const events = useQuery({
    queryKey: ['events', projectId],
    queryFn: () => request<EventNode[]>(`/api/projects/${projectId}/events`),
  })
  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => request<ModelVersion[]>(`/api/projects/${projectId}/models`),
    refetchInterval: (query) =>
      query.state.data?.some((model) => model.active) ? false : 2000,
  })
  const create = useMutation({
    mutationFn: () => {
      const event = events.data?.find((item) => item.id === eventId)
      const availableModels = models.data?.filter((model) => model.active) ?? []
      const recommended = availableModels.find((model) => model.recommended)
      const selectedModelId =
        modelId || recommended?.id || availableModels[0]?.id || ''
      if (!event) throw new Error('请选择有效的时间节点')
      if (!selectedModelId) throw new Error('当前没有可用模型')
      return request<Branch>(`/api/projects/${projectId}/branches`, {
        method: 'POST',
        body: JSON.stringify({
          origin_event_id: eventId,
          model_version_id: selectedModelId,
          title,
          origin_time:
            searchParams.get('originTime') ?? event?.started_at ?? event?.created_at,
        }),
      })
    },
    onSuccess: (branch) =>
      navigate(`/projects/${projectId}/branches/${branch.id}/preparing`),
  })
  const selectedEvent = events.data?.find((item) => item.id === eventId)
  const activeModels = models.data?.filter((model) => model.active) ?? []
  const selectedModel =
    activeModels.find((model) => model.id === modelId) ??
    activeModels.find((model) => model.recommended) ??
    activeModels[0]
  const hasActiveModel = models.data?.some((model) => model.active) ?? false

  return (
    <Paper
      component="form"
      maw={680}
      onSubmit={(event) => {
        event.preventDefault()
        create.mutate()
      }}
      p="xl"
      withBorder
    >
      <Stack>
      <Text c="moon.4" fw={700} size="xs">创建平行时间线</Text>
      <Title order={2}>回到这个时刻</Title>
      {events.isLoading || models.isLoading ? <Text role="status">正在确认时间节点和数字人……</Text> : null}
      {events.isError || models.isError ? <Alert color="red" role="alert">创建所需信息读取失败，请返回时间轴后重试。</Alert> : null}
      {routeEventId && events.isSuccess && !selectedEvent ? <Alert color="red" role="alert">原时间节点已不存在，请重新选择。</Alert> : null}
      {selectedEvent ? <Alert color="moon" title="你将回到"><Text fw={700}>{selectedEvent.title || selectedEvent.topic}</Text><Text c="dimmed" size="sm">{new Date(selectedEvent.started_at ?? selectedEvent.created_at).toLocaleString('zh-CN')} · 系统不会读取此刻之后的记忆</Text></Alert> : null}
      <TextInput
        aria-label="分支名称"
        id="branch-title"
        label="分支名称"
        onChange={(event) => setTitle(event.target.value)}
        required
        value={title}
      />
      <NativeSelect
        aria-label="起点节点"
        id="origin-event"
        label="起点节点"
        onChange={(event) => setEventId(event.target.value)}
        required
        value={eventId}
      >
        <option value="">请选择</option>
        {events.data?.map((event) => (
          <option key={event.id} value={event.id}>
            {event.title || event.topic} · {event.summary || event.after_state || event.reason}
          </option>
        ))}
      </NativeSelect>
      <NativeSelect
        aria-label="模型版本"
        id="branch-model"
        label="数字人版本"
        onChange={(event) => setModelId(event.target.value)}
        value={modelId}
      >
        <option value="">使用推荐版本</option>
        {activeModels.map((model) => (
          <option key={model.id} value={model.id}>
            {model.recommended ? '推荐 · ' : ''}{modelDisplayName(model)}
          </option>
        ))}
      </NativeSelect>
      {selectedModel ? (
        <Text c="dimmed" size="sm">
          当前将使用：{modelDisplayName(selectedModel)} · {selectedModel.base_model} · 版本{' '}
          {selectedModel.id.slice(0, 8)}
        </Text>
      ) : null}
      {create.isError ? (
        <Alert color="red">创建失败，请检查节点和模型后重试</Alert>
      ) : null}
      {models.isSuccess && !hasActiveModel ? (
        <Alert color="yellow">当前没有可用数字人，请先完成训练。</Alert>
      ) : null}
      <Button
        disabled={
          create.isPending ||
          events.isLoading || models.isLoading || !selectedEvent || !hasActiveModel
        }
        leftSection={<Icon name="branch" size={16} />}
        loading={create.isPending}
        type="submit"
      >
        进入这条时间线
      </Button>
      </Stack>
    </Paper>
  )
}

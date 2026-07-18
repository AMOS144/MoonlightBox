import { useMutation, useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'

import { request } from '../../api/client'
import type { EventNode } from '../events/types'
import type { ModelVersion } from '../models/types'
import type { Branch } from './types'

export function BranchCreatePage() {
  const { projectId } = useParams()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [title, setTitle] = useState('')
  const [eventId, setEventId] = useState(searchParams.get('eventId') ?? '')
  const [modelId, setModelId] = useState('')
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
      const recommended = models.data?.find((model) => model.recommended)
      const selectedModelId = modelId || recommended?.id || ''
      return request<Branch>(`/api/projects/${projectId}/branches`, {
        method: 'POST',
        body: JSON.stringify({
          origin_event_id: eventId,
          model_version_id: selectedModelId,
          title,
          origin_time: searchParams.get('originTime') ?? event?.created_at,
          state_snapshot: {
            relationship_status: event?.after_state,
            emotions: event?.emotion_labels,
            evidence_ids: event?.evidence_ids,
          },
        }),
      })
    },
    onSuccess: (branch) => navigate(`../branches/${branch.id}`),
  })

  return (
    <form
      className="project-form"
      onSubmit={(event) => {
        event.preventDefault()
        create.mutate()
      }}
    >
      <p className="eyebrow">创建分支</p>
      <h1>这一次，我想这样说</h1>
      <label htmlFor="branch-title">分支名称</label>
      <input
        id="branch-title"
        onChange={(event) => setTitle(event.target.value)}
        required
        value={title}
      />
      <label htmlFor="origin-event">起点节点</label>
      <select
        id="origin-event"
        onChange={(event) => setEventId(event.target.value)}
        required
        value={eventId}
      >
        <option value="">请选择</option>
        {events.data?.map((event) => (
          <option key={event.id} value={event.id}>
            {event.topic} · {event.after_state}
          </option>
        ))}
      </select>
      <label htmlFor="branch-model">模型版本</label>
      <select
        id="branch-model"
        onChange={(event) => setModelId(event.target.value)}
        value={modelId}
      >
        <option value="">使用推荐版本</option>
        {models.data?.map((model) => (
          <option key={model.id} value={model.id}>
            {model.base_model}
          </option>
        ))}
      </select>
      <button className="primary-button" type="submit">
        进入这条时间线
      </button>
    </form>
  )
}

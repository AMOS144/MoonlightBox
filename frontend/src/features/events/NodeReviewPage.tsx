import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'

import { request } from '../../api/client'
import type { EventNode } from './types'

export function NodeReviewPage() {
  const { projectId } = useParams()
  const queryClient = useQueryClient()
  const events = useQuery({
    queryKey: ['events', projectId],
    queryFn: () => request<EventNode[]>(`/api/projects/${projectId}/events`),
    enabled: Boolean(projectId),
  })
  const reject = useMutation({
    mutationFn: (eventId: string) =>
      request<EventNode>(`/api/projects/${projectId}/events/${eventId}`, {
        method: 'PATCH',
        body: JSON.stringify({
          changes: { status: 'rejected' },
          reason: '人工标记为误报',
        }),
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['events', projectId] }),
  })

  return (
    <section>
      <p className="eyebrow">关键节点审核</p>
      <h1>哪些时刻改变了关系</h1>
      {events.isLoading && <p>正在读取节点……</p>}
      {events.isError && <p role="alert">节点读取失败。</p>}
      <div className="event-list">
        {events.data
          ?.filter((event) => event.status !== 'rejected')
          .map((event) => (
            <article className="event-card" key={event.id}>
              <div className="event-card__header">
                <span>{event.type}</span>
                <strong>{Math.round(event.importance * 100)}%</strong>
              </div>
              <p>
                {event.before_state} → <strong>{event.after_state}</strong>
              </p>
              <p>{event.reason}</p>
              <small>证据：{event.evidence_ids.join(' · ')}</small>
              <button onClick={() => reject.mutate(event.id)} type="button">
                标记为误报
              </button>
            </article>
          ))}
      </div>
    </section>
  )
}

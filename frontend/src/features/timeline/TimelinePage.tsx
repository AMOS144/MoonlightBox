import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'

import { request } from '../../api/client'
import type { EventNode } from '../events/types'

export function TimelinePage() {
  const { projectId } = useParams()
  const events = useQuery({
    queryKey: ['events', projectId],
    queryFn: () => request<EventNode[]>(`/api/projects/${projectId}/events`),
  })

  return (
    <section>
      <p className="eyebrow">关系时间轴</p>
      <h1>所有改变，都有迹可循</h1>
      <div className="timeline">
        {events.data?.map((event) => (
          <article className="timeline-node" key={event.id}>
            <time>{new Date(event.created_at).toLocaleDateString('zh-CN')}</time>
            <h2>{event.topic}</h2>
            <p>{event.after_state}</p>
            <Link
              className="primary-button"
              to={`../branches/new?eventId=${event.id}&originTime=${encodeURIComponent(event.created_at)}`}
            >
              从这里创建分支
            </Link>
          </article>
        ))}
      </div>
    </section>
  )
}

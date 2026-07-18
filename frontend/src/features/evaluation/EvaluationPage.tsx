import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'

import { request } from '../../api/client'
import type { ModelVersion } from '../models/types'

export function EvaluationPage() {
  const { projectId } = useParams()
  const queryClient = useQueryClient()
  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => request<ModelVersion[]>(`/api/projects/${projectId}/models`),
  })
  const recommend = useMutation({
    mutationFn: () =>
      request<ModelVersion>(`/api/projects/${projectId}/models/recommend`, {
        method: 'POST',
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['models', projectId] }),
  })

  return (
    <section>
      <p className="eyebrow">盲测与评测</p>
      <h1>哪一个更像她</h1>
      <div className="model-list">
        {models.data?.map((model) => (
          <article className="model-card" key={model.id}>
            <h2>{model.base_model}</h2>
            <p>盲测胜率：{Math.round((model.metrics.blind_win_rate ?? 0) * 100)}%</p>
            <p>风格分：{Math.round((model.metrics.style_score ?? 0) * 100)}%</p>
          </article>
        ))}
      </div>
      <button className="primary-button" onClick={() => recommend.mutate()} type="button">
        按质量门槛推荐版本
      </button>
    </section>
  )
}

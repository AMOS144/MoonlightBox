import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'

import { request } from '../../api/client'
import type { ModelVersion } from './types'

export function ModelsPage() {
  const { projectId } = useParams()
  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => request<ModelVersion[]>(`/api/projects/${projectId}/models`),
  })

  return (
    <section>
      <p className="eyebrow">人格模型</p>
      <h1>保留说话的样子</h1>
      <div className="model-list">
        {models.data?.map((model) => (
          <article className="model-card" key={model.id}>
            <h2>{model.base_model}</h2>
            <p>{model.recommended ? '当前推荐版本' : model.status}</p>
            <small>数据集：{model.dataset_hash.slice(0, 12)}</small>
          </article>
        ))}
      </div>
    </section>
  )
}

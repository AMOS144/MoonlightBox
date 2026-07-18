import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'

import { request } from '../../api/client'
import type { Branch } from './types'

export function BranchListPage() {
  const { projectId } = useParams()
  const branches = useQuery({
    queryKey: ['branches', projectId],
    queryFn: () => request<Branch[]>(`/api/projects/${projectId}/branches`),
  })

  return (
    <section>
      <p className="eyebrow">平行时间线</p>
      <h1>如果当时，我这样选择</h1>
      <div className="branch-grid">
        {branches.data?.map((branch) => (
          <Link className="branch-card" key={branch.id} to={branch.id}>
            <h2>{branch.title}</h2>
            <p>起点：{new Date(branch.origin_time).toLocaleString('zh-CN')}</p>
          </Link>
        ))}
      </div>
      {!branches.data?.length && <p>还没有分支，请先从时间轴选择一个节点。</p>}
    </section>
  )
}

import { Navigate, useParams, useSearchParams } from 'react-router-dom'

export function BranchCreatePage() {
  const { projectId = '' } = useParams()
  const [params] = useSearchParams()
  const investigation = params.get('investigation')
  // 旧创建链接收敛到起点工作台；节点背景审核发布后才能准备分支。
  return <Navigate replace to={`/projects/${projectId}/nodes${investigation ? '?investigation=' + encodeURIComponent(investigation) : ''}`} />
}

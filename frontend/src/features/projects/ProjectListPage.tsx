import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'

import { request } from '../../api/client'

export type Project = {
  id: string
  name: string
  status: string
  created_at: string
  updated_at: string
}

export function ProjectListPage() {
  const projects = useQuery({
    queryKey: ['projects'],
    queryFn: () => request<Project[]>('/api/projects'),
  })

  return (
    <main className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Moonlight Box</p>
          <h1>月光宝盒</h1>
          <p className="subtitle">从一段真实对话，回到故事改变之前。</p>
        </div>
        <Link className="primary-button" to="/projects/new">
          创建项目
        </Link>
      </header>

      {projects.isPending && <p>正在加载项目……</p>}
      {projects.isError && <p role="alert">项目加载失败，请稍后重试。</p>}
      {projects.data && projects.data.length === 0 && (
        <section className="empty-state">
          <h2>还没有时间线</h2>
          <p>导入聊天记录，创建第一个数字人项目。</p>
        </section>
      )}
      <section className="project-grid">
        {projects.data?.map((project) => (
          <Link className="project-card" key={project.id} to={`/projects/${project.id}`}>
            <span className="status-dot" />
            <h2>{project.name}</h2>
            <p>状态：{project.status}</p>
          </Link>
        ))}
      </section>
    </main>
  )
}

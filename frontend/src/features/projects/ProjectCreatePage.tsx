import { useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'

import { request } from '../../api/client'
import type { Project } from './ProjectListPage'

export function ProjectCreatePage() {
  const [name, setName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const navigate = useNavigate()

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setSubmitting(true)
    try {
      const project = await request<Project>('/api/projects', {
        method: 'POST',
        body: JSON.stringify({ name }),
      })
      await navigate(`/projects/${project.id}`)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <main className="page narrow-page">
      <p className="eyebrow">新时间线</p>
      <h1>创建项目</h1>
      <form className="project-form" onSubmit={submit}>
        <label htmlFor="project-name">项目名称</label>
        <input
          id="project-name"
          maxLength={120}
          onChange={(event) => setName(event.target.value)}
          required
          value={name}
        />
        <button className="primary-button" disabled={submitting} type="submit">
          {submitting ? '正在创建……' : '继续导入数据'}
        </button>
      </form>
    </main>
  )
}

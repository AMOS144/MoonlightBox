import { userMessage } from '../../components/feedback/messages'
import { Alert, Button, Stack, TextInput } from '@mantine/core'
import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { SetupHeader } from '../../components/SetupHeader'

import { request } from '../../api/client'
import type { Project } from './types'

export function ProjectCreatePage() {
  const [name, setName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()
  const pending = useRef<AbortController | null>(null)
  useEffect(() => () => pending.current?.abort(), [])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!name.trim() || submitting) return
    setSubmitting(true)
    setError('')
    const controller = new AbortController()
    pending.current = controller
    try {
      const project = await request<Project>('/api/projects', {
        method: 'POST',
        signal: controller.signal,
        body: JSON.stringify({ name: name.trim() }),
      })
      // 离开页面只停止前端等待，不宣称撤销后端已经创建的项目。
      if (!controller.signal.aborted) await navigate(`/projects/${project.id}/setup/import`, { replace: true })
    } catch {
      if (!controller.signal.aborted) setError('项目创建失败，请检查服务状态后重试。')
    } finally {
      if (!controller.signal.aborted) setSubmitting(false)
    }
  }

  return (
    <section className="setup-page">
      <SetupHeader step={0} title="创建项目" description="先为这份聊天资料命名，下一步选择导出的记录。" />
      <form className="setup-form" onSubmit={submit}>
        <Stack>
        <TextInput
          id="project-name"
          label="项目名称"
          maxLength={120}
          onChange={(event) => setName(event.target.value)}
          placeholder="例如：大学时期的聊天"
          description="名称仅用于区分项目，不影响人物识别。"
          required
          size="sm"
          disabled={submitting}
          value={name}
        />
        {error ? <Alert color="red" role="alert">{userMessage(error)}</Alert> : null}
        <div className="setup-actions"><Button disabled={!name.trim()} loading={submitting} size="sm" type="submit">继续导入聊天</Button></div>
        </Stack>
      </form>
    </section>
  )
}

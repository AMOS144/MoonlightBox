import { Alert, Button, Container, Paper, Stack, Text, TextInput, Title } from '@mantine/core'
import { useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'

import { request } from '../../api/client'
import type { Project } from './ProjectListPage'

export function ProjectCreatePage() {
  const [name, setName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!name.trim() || submitting) return
    setSubmitting(true)
    setError('')
    try {
      const project = await request<Project>('/api/projects', {
        method: 'POST',
        body: JSON.stringify({ name: name.trim() }),
      })
      await navigate(`/projects/${project.id}/data`, { replace: true })
    } catch {
      setError('项目创建失败，请检查服务状态后重试。')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Container component="main" py={{ base: 32, sm: 72 }} size={520}>
      <Text c="moon.4" fw={700} size="xs">新记忆档案</Text>
      <Title mt={6} order={1}>创建项目</Title>
      <Text c="dimmed" mt={8}>给这段关系起一个只有你能看见的名字。</Text>
      <Paper component="form" mt={32} onSubmit={submit} p="xl" withBorder>
        <Stack>
        <TextInput
          id="project-name"
          label="项目名称"
          maxLength={120}
          onChange={(event) => setName(event.target.value)}
          placeholder="例如：我们的聊天记录"
          required
          size="md"
          value={name}
        />
        {error ? <Alert color="red" role="alert">{error}</Alert> : null}
        <Button loading={submitting} mt="sm" type="submit">继续导入聊天</Button>
        </Stack>
      </Paper>
    </Container>
  )
}

import { useQuery } from '@tanstack/react-query'
import { Alert, Button, Group, Progress, Stack, Text } from '@mantine/core'
import { Link } from 'react-router-dom'

import { request } from '../../api/client'
import { IconLoader2 } from '@tabler/icons-react'

type ProjectJob = {
  id: string
  kind: string
  payload: Record<string, unknown>
  status: string
  progress: number
  checkpoint: Record<string, unknown> | null
  error_code: string | null
  error_message: string | null
  created_at: string
  updated_at: string
}

const visibleKinds = new Set([
  'event_analysis_v2',
  'event_analysis_v3',
  'digital_human_training_v1',
  'lightrag_world_build_v1',
])

export function ProjectTaskBar({ projectId }: { projectId: string }) {
  const jobs = useQuery({
    queryKey: ['project-jobs', projectId],
    queryFn: () =>
      request<ProjectJob[]>(`/api/jobs?project_id=${encodeURIComponent(projectId)}`),
    refetchInterval: (query) =>
      (query.state.data ?? []).some((job) =>
        ['queued', 'running'].includes(job.status),
      )
        ? 1500
        : false,
  })
  const job = [...(jobs.data ?? [])]
    .filter((candidate) =>
      visibleKinds.has(candidate.kind) &&
      ['queued', 'running'].includes(candidate.status))
    .sort((left, right) => right.updated_at.localeCompare(left.updated_at))[0]
  if (!job) return null
  const label = taskLabel(job)
  const link = taskLink(projectId, job)

  return (
    <Alert
      color="moon"
      icon={<IconLoader2 />}
      role="status"
      variant="light"
    >
      <Group justify="space-between" wrap="nowrap">
        <Stack gap={4} style={{ flex: 1 }}>
        <Text fw={700}>{label}</Text>
        <Text c="dimmed" size="sm">
          {`${Math.round(job.progress * 100)}%${iterationLabel(job)}`}
        </Text>
        <Progress value={job.progress * 100} size="xs" />
        </Stack>
        <Button component={Link} size="xs" to={link} variant="subtle">
          {linkLabel(job)}
        </Button>
      </Group>
    </Alert>
  )
}

function taskLabel(job: ProjectJob): string {
  if (job.kind === 'digital_human_training_v1') return '正在训练数字人'
  if (job.kind === 'lightrag_world_build_v1') return '正在重建人物世界'
  return '正在分析关键节点'
}

function linkLabel(job: ProjectJob): string {
  if (job.kind === 'digital_human_training_v1') return '查看训练进度'
  if (job.kind === 'lightrag_world_build_v1') return '查看人物背景'
  return '查看分析进度'
}

function taskLink(projectId: string, job: ProjectJob): string {
  if (job.kind === 'digital_human_training_v1') {
    return `/projects/${projectId}/training/${job.id}`
  }
  if (job.kind === 'lightrag_world_build_v1') {
    return `/projects/${projectId}/world`
  }
  return `/projects/${projectId}/data`
}

function iterationLabel(job: ProjectJob): string {
  const iteration = job.checkpoint?.iteration
  const total = job.checkpoint?.total_iterations
  return typeof iteration === 'number' && typeof total === 'number'
    ? ` · ${iteration}/${total}`
    : ''
}

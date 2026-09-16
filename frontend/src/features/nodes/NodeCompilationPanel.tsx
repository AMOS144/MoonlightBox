import { userMessage } from '../../components/feedback/messages'
import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Badge, Button, Group, List, Loader, Stack, Text, ThemeIcon } from '@mantine/core'
import { useNavigate } from 'react-router-dom'
import { request } from '../../api/client'
import { Icon } from '../../components/Icon'
import { useSessionState } from '../../hooks/useSessionState'

type SectionTask = { section: string; status: string; research_round: number; unresolved_questions: string[]; error_code?: string | null }
type Compilation = { job_id: string; status: string; error_code?: string; checkpoint?: { agent_run_id?: string }; section_tasks?: SectionTask[]; draft?: { id: string } }

const sectionLabels: Record<string, string> = {
  identity: '身份',
  life_context: '生活情境',
  social_world: '社会关系',
  agency: '偏好与目标',
  practices: '实践规律',
  life_course: '经历轨迹',
  relationship_with_user: '与用户关系',
}

function sectionStatusIcon(task: SectionTask) {
  if (task.status === 'completed') return <ThemeIcon color="teal" size="sm" radius="xl"><Icon name="check" size={12} /></ThemeIcon>
  if (task.status === 'failed') return <ThemeIcon color="red" size="sm" radius="xl"><Icon name="close" size={12} /></ThemeIcon>
  if (task.status === 'researching') return <Loader size={14} />
  return <ThemeIcon variant="light" color="gray" size="sm" radius="xl" />
}

function sectionStatusText(task: SectionTask) {
  if (task.status === 'completed') return '已完成'
  if (task.status === 'failed') return '本栏受阻'
  if (task.status === 'researching') return task.research_round > 1 ? `第 ${task.research_round} 轮补全` : '研究中'
  return '等待'
}

function SectionProgress({ tasks }: { tasks: SectionTask[] }) {
  if (!tasks.length) return null
  const done = tasks.filter(task => task.status === 'completed').length
  return <Stack gap={4}>
    <Text size="sm" c="dimmed">正在逐栏整理人物背景：{done}/{tasks.length} 栏完成</Text>
    <List spacing={4} size="sm" center>
      {tasks.map(task => (
        <List.Item key={task.section} icon={sectionStatusIcon(task)}>
          <Group gap="xs" wrap="nowrap">
            <Text size="sm">{sectionLabels[task.section] ?? task.section}</Text>
            <Badge size="sm" variant="light" color={task.status === 'completed' ? 'teal' : task.status === 'failed' ? 'red' : task.status === 'researching' ? 'blue' : 'gray'}>
              {sectionStatusText(task)}
            </Badge>
          </Group>
        </List.Item>
      ))}
    </List>
  </Stack>
}

export function NodeCompilationPanel({ projectId, investigationId, graphId, previewHash, autoStart = false, onReady }: {
  projectId: string; investigationId: string; graphId?: string; previewHash: string; autoStart?: boolean
  onReady?: (profileId: string) => void
}) {
  const navigate = useNavigate()
  const client = useQueryClient()
  const root = `/api/projects/${projectId}/world-agent`
  const [requestId, setRequestId] = useState(() => autoStart ? `selected-${previewHash}` : crypto.randomUUID())
  const autoStarted = useRef(false)
  const [jobId, setJobId] = useSessionState<string | null>(`node-compile:${projectId}:${previewHash}`, null)
  const latest = useQuery({ queryKey: ['node-compilation-latest', projectId, previewHash],
    queryFn: () => request<{ job_id: string } | null>(`${root}/node-compilations?preview_hash=${previewHash}`) })
  const currentId = jobId ?? latest.data?.job_id
  const job = useQuery({ queryKey: ['node-compilation', projectId, currentId],
    queryFn: () => request<Compilation>(`${root}/node-compilations/${currentId}`), enabled: Boolean(currentId),
    refetchInterval: q => ['queued', 'running', 'cancelling'].includes(q.state.data?.status ?? '') ? 2000 : false })
  const start = useMutation({ mutationFn: () => request<Compilation>(`${root}/node-compilations`, {
    method: 'POST', body: JSON.stringify({ graph_version_id: graphId, investigation_id: investigationId,
      preview_hash: previewHash, idempotency_key: requestId }),
  }), onSuccess: value => { setJobId(value.job_id); setRequestId(crypto.randomUUID()); void client.invalidateQueries({ queryKey: ['node-compilation-latest', projectId, previewHash] }) } })
  const review = useMutation({ mutationFn: () => request<{ profile_id: string }>(`${root}/node-drafts/${job.data?.draft?.id}/review`, { method: 'POST' }),
    onSuccess: value => onReady ? onReady(value.profile_id) : navigate(`/projects/${projectId}/node-profiles/${value.profile_id}`) })
  const retry = useMutation({ mutationFn: () => request(`/api/jobs/${currentId}/resume`, { method: 'POST' }),
    onSuccess: () => { void job.refetch() } })
  // 草稿就绪即自动打开审核页；每个草稿只自动跳一次，用户回到起点页后仍可手动进入。
  const [autoReviewed, setAutoReviewed] = useSessionState<string | null>(`node-compile-reviewed:${projectId}:${previewHash}`, null)
  const draftId = job.data?.draft?.id ?? null
  useEffect(() => {
    if (!draftId || autoReviewed === draftId || review.isPending) return
    setAutoReviewed(draftId)
    review.mutate()
  }, [draftId, autoReviewed, review, setAutoReviewed])
  useEffect(() => {
    if (autoStart && latest.isSuccess && !currentId && graphId && !autoStarted.current) {
      autoStarted.current = true
      start.mutate()
    }
  }, [autoStart, currentId, graphId, latest.isSuccess, start])
  const error = start.error || review.error || retry.error || job.error || latest.error
  return <Stack gap="sm">
    {error && <Alert color="red">{userMessage(error)}</Alert>}
    {!graphId && <Text size="sm">图谱尚未就绪，完成资料整理后可编译此起点背景。</Text>}
    {!currentId && !autoStart && <Button disabled={!graphId || latest.isLoading} loading={start.isPending} onClick={() => start.mutate()}>整理此刻的人物背景</Button>}
    {!currentId && autoStart && <Text size="sm">正在启动人物背景编译…</Text>}
    {job.data && <Text size="sm">{job.data.draft ? (review.isPending ? '节点背景已生成，正在打开审核页…' : '节点背景已生成，等待审核。') : ['queued', 'running'].includes(job.data.status) ? '正在整理节点背景，可以离开页面，进度会保留。' : job.data.status === 'cancelled' ? '整理已取消，可以重新开始。' : '本次整理尚未完成，可以恢复已有进度。'}</Text>}
    {job.data && ['queued', 'running'].includes(job.data.status) && <SectionProgress tasks={job.data.section_tasks ?? []} />}
    <Group>{job.data?.draft && <Button loading={review.isPending} onClick={() => review.mutate()}>审核节点背景</Button>}
      {['failed', 'interrupted'].includes(job.data?.status ?? '') && <Button loading={retry.isPending} onClick={() => retry.mutate()}>恢复编译</Button>}
      {['failed', 'cancelled'].includes(job.data?.status ?? '') && <Button variant="subtle" disabled={!graphId} loading={start.isPending} onClick={() => start.mutate()}>重新编译</Button>}</Group>
  </Stack>
}

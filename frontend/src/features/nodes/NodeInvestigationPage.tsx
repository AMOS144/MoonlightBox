import { useEffect, useRef } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Badge, Button, Card, Group, Paper, Progress, Stack, Text, Title } from '@mantine/core'
import { useParams, useSearchParams } from 'react-router-dom'

import { request } from '../../api/client'
import { userMessage } from '../../components/feedback/messages'
import { PageHeader } from '../../components/PageHeader'
import { useSessionState } from '../../hooks/useSessionState'
import { journeyKey } from '../journey/journey'
import './nodes.css'

type Boundary = {
  label: string
  kind: 'message' | 'gap'
  message_ref?: string
  side: string
}

type Candidate = {
  id: string
  revision: number
  title: string
  summary: string
  occurrence: string
  why_branch: string
  uncertainties: string[]
  status: string
  boundaries: Boundary[]
}

type Preview = {
  preview_hash: string
  candidate_revision: number
  cutoff_at: string
}

type Workspace = {
  id: string
  revision: number
  status: string
  cursor: number
  total_messages: number
  read_count?: number
  range_count?: number
  candidates: Candidate[]
  summary?: string
  error?: string
  graph_id?: string
  confirmed: { candidate_id: string; cutoff_at: string; preview_hash: string } | null
}

const statusLabels: Record<string, string> = {
  queued: '准备调查',
  running: 'Agent 正在调查',
  waiting_for_user: 'Agent 正在整理候选',
  completed: '候选已整理完成',
  failed: '调查受阻',
  interrupted: '调查已中断',
  paused: '调查已停止',
  cancelled: '调查已停止',
  cancelling: '正在结束调查',
}

function CandidateDetail({
  candidate,
  workspace,
  base,
  refresh,
}: {
  candidate: Candidate
  workspace: Workspace
  base: string
  refresh: () => void
}) {
  const choose = useMutation({
    mutationFn: async () => {
      const optionIndex = candidate.boundaries.findIndex((boundary) => boundary.kind === 'message')
      if (optionIndex < 0) throw new Error('这个候选还没有可直接使用的消息边界，请等待 Agent 完成整理。')
      const preview = await request<Preview>(`${base}/candidates/${candidate.id}/preview`, {
        method: 'POST',
        body: JSON.stringify({
          candidate_revision: candidate.revision,
          option_index: optionIndex,
          exact_time: null,
        }),
      })
      const confirmed = await request<Workspace>(`${base}/candidates/${candidate.id}/confirm`, {
        method: 'POST',
        body: JSON.stringify({ preview_hash: preview.preview_hash, continue_investigating: false }),
      })
      return confirmed
    },
    onSuccess: refresh,
  })

  return <Stack gap="md">
    <div>
      <Title order={3}>{candidate.title}</Title>
      <Text c="dimmed" size="sm" mt={4}>{candidate.occurrence || '时间由聊天记录边界确定'}</Text>
    </div>
    <Text style={{ whiteSpace: 'pre-wrap' }}>{candidate.summary}</Text>
    <Paper p="md" withBorder>
      <Text fw={600} mb={4}>为什么适合作为起点</Text>
      <Text>{candidate.why_branch}</Text>
    </Paper>
    {candidate.uncertainties.length > 0 && <Alert color="gray" title="仍不确定的部分">
      {candidate.uncertainties.map((item, index) => <Text size="sm" key={index}>{item}</Text>)}
    </Alert>}
    {workspace.confirmed?.candidate_id === candidate.id && <Alert color="teal" title="已选择为起点">
      起点时刻：{workspace.confirmed.cutoff_at}
    </Alert>}
    {!workspace.confirmed && <Button
      size="md"
      disabled={candidate.status !== 'ready' || !workspace.graph_id}
      loading={choose.isPending}
      onClick={() => choose.mutate()}
    >
      选择这个起点
    </Button>}
    {choose.error && <Alert color="red">{userMessage(choose.error)}</Alert>}
  </Stack>
}

export function NodeInvestigationPage() {
  const { projectId = '' } = useParams()
  const [searchParams, setSearchParams] = useSearchParams()
  const investigationId = searchParams.get('investigation')
  const queryClient = useQueryClient()
  const key = ['node-investigation', projectId, investigationId]
  const root = `/api/projects/${projectId}/node-investigations`
  const [selected, setSelected] = useSessionState<string | null>(`node-selection:${projectId}`, null)
  const autoStarted = useRef(false)
  const autoResumed = useRef(false)

  const workspace = useQuery<Workspace | null>({
    queryKey: key,
    queryFn: () => request(investigationId ? `${root}/${encodeURIComponent(investigationId)}` : root),
    refetchInterval: (query) => ['queued', 'running', 'cancelling', 'waiting_for_user'].includes(query.state.data?.status ?? '') ? 2000 : false,
    structuralSharing: (oldData, newData) => {
      const oldValue = oldData as Workspace | null | undefined
      const newValue = newData as Workspace | null
      return oldValue && newValue && oldValue.id === newValue.id && oldValue.revision > newValue.revision ? oldValue : newValue
    },
  })
  const data = workspace.data
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: key })
    void queryClient.invalidateQueries({ queryKey: journeyKey(projectId) })
  }
  const start = useMutation({
    mutationFn: () => request<Workspace>(root, { method: 'POST', body: JSON.stringify({ timezone: 'Asia/Shanghai' }) }),
    onSuccess: (value) => {
      setSearchParams({ investigation: value.id })
      refresh()
    },
  })
  const resume = useMutation({
    mutationFn: () => request(`${root}/${data?.id}/resume`, { method: 'POST' }),
    onSuccess: refresh,
  })

  useEffect(() => {
    if (workspace.isSuccess && !data && !autoStarted.current) {
      autoStarted.current = true
      start.mutate()
    }
  }, [data, start, workspace.isSuccess])

  useEffect(() => {
    if (data?.status === 'waiting_for_user' && !autoResumed.current) {
      autoResumed.current = true
      resume.mutate()
    }
  }, [data?.status, resume])

  const readyCandidates = (data?.status === 'completed' || data?.confirmed ? data.candidates : []).filter((candidate) =>
    candidate.status === 'ready' && candidate.boundaries.some((boundary) => boundary.kind === 'message'),
  )
  const candidate = readyCandidates.find((item) => item.id === selected)
    ?? readyCandidates.find((item) => item.id === data?.confirmed?.candidate_id)
    ?? readyCandidates[0]
  const base = `${root}/${data?.id}`
  const error = workspace.error || start.error || resume.error

  return <Stack gap="lg">
    <PageHeader title="选择起点" description="Agent 会从聊天记录里找出对关系影响最大的时刻，选一个作为分支起点。"
      action={data && <Badge variant="light">{statusLabels[data.status] ?? data.status}</Badge>} />

    {error && <Alert color="red">{userMessage(error)}</Alert>}
    {data?.error && <Alert color="red" title="调查暂未完成">
      {userMessage(data.error)} 已整理出的候选仍然保留。
      <Button ml="sm" variant="light" loading={resume.isPending} onClick={() => resume.mutate()}>重试</Button>
    </Alert>}

    {!data && <Paper p="lg" withBorder><Text>正在启动起点调查…</Text></Paper>}

    {data && <>
      {!data.confirmed && <Paper p="md" withBorder>
        <Group justify="space-between" mb="xs">
          <Text fw={600}>Agent 调查进度</Text>
          <Text size="sm" c="dimmed">{data.read_count ?? data.cursor} / {data.range_count ?? data.total_messages}</Text>
        </Group>
        <Progress value={(data.range_count ?? data.total_messages) ? (data.read_count ?? data.cursor) / (data.range_count ?? data.total_messages) * 100 : 0} />
      </Paper>}

      {readyCandidates.length === 0 ? <Paper p="xl" withBorder>
        <Title order={3}>Agent 正在整理候选</Title>
        <Text c="dimmed" mt="xs">它会自行完成检索、判断时间边界并填写候选内容。</Text>
      </Paper> : <div className="node-workbench">
        <Stack component="aside" gap="xs" className="node-candidates" aria-label="候选节点">
          {readyCandidates.map((item) => <Card
            component="button"
            type="button"
            withBorder
            p="md"
            key={item.id}
            className={`interactive-card node-candidate${candidate?.id === item.id ? ' node-candidate--selected' : ''}`}
            onClick={() => setSelected(item.id)}
          >
            <Group justify="space-between" wrap="nowrap">
              <Text fw={600} size="sm">{item.title}</Text>
              {data.confirmed?.candidate_id === item.id && <Badge size="sm" color="teal" variant="light">已选择</Badge>}
            </Group>
            <Text size="xs" c="dimmed">{item.occurrence}</Text>
          </Card>)}
        </Stack>
        <section className="node-detail" aria-label="节点详情">
          {candidate && <CandidateDetail
            key={`${candidate.id}:${candidate.revision}`}
            candidate={candidate}
            workspace={data}
            base={base}
            refresh={refresh}
          />}
        </section>
      </div>}
    </>}
  </Stack>
}

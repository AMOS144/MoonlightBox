import { useQuery } from '@tanstack/react-query'
import { Alert, Loader, Paper, Text } from '@mantine/core'
import { useParams } from 'react-router-dom'

import { request } from '../../api/client'
import { userMessage } from '../../components/feedback/messages'
import { PageHeader } from '../../components/PageHeader'
import { useSessionState } from '../../hooks/useSessionState'
import { NodeCompilationPanel } from './NodeCompilationPanel'
import { NodeProfile } from './NodeProfilePage'

type Investigation = {
  id: string
  graph_id?: string
  confirmed: { candidate_id: string; cutoff_at: string; preview_hash: string } | null
}

export function NodeBackgroundPage() {
  const { projectId = '' } = useParams()
  const investigation = useQuery<Investigation | null>({
    queryKey: ['node-investigation', projectId, null],
    queryFn: () => request(`/api/projects/${projectId}/node-investigations`),
  })
  const data = investigation.data
  const confirmed = data?.confirmed ?? null
  const [profileId, setProfileId] = useSessionState<string | null>(
    `node-bg:${projectId}:${confirmed?.preview_hash ?? 'none'}`, null)

  if (investigation.isLoading) return <Loader aria-label="正在读取起点信息" />
  if (investigation.error) return <Alert color="red">{userMessage(investigation.error)}</Alert>
  if (!data || !confirmed) {
    return <>
      <PageHeader title="起点背景" description="确认起点之后，Agent 会在这里逐栏整理并编译这个时刻的人物背景。" />
      <Paper p="md" withBorder>
        <Text fw={600} size="sm">还没有确认起点</Text>
        <Text size="sm" c="dimmed" mt={4}>确认起点后，这里会自动开始整理该时刻的人物背景。</Text>
      </Paper>
    </>
  }

  return <>
    <PageHeader title="起点背景" description={`起点：${confirmed.cutoff_at} · Agent 逐栏整理这个时刻的人物背景，审核后就能开始对话。`} />
    {!profileId && <NodeCompilationPanel
      key={confirmed.preview_hash}
      projectId={projectId}
      investigationId={data.id}
      graphId={data.graph_id}
      previewHash={confirmed.preview_hash}
      autoStart
      onReady={setProfileId}
    />}
    {profileId && <NodeProfile key={`${projectId}:${profileId}`} projectId={projectId} profileId={profileId} onRefresh={setProfileId} />}
  </>
}

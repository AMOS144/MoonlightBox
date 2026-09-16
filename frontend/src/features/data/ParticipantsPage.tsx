import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Accordion, Alert, Avatar, Button, Card, Group, Paper, Stack, Text, Title } from '@mantine/core'
import { useEffect } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { request } from '../../api/client'
import { AsyncState } from '../../components/feedback/AsyncState'
import { StageHandoff } from '../../components/feedback/StageHandoff'
import { useJourney, journeyKey } from '../journey/journey'
import { useSessionState } from '../../hooks/useSessionState'
import { ImportWizard } from './ImportWizard'
import type { EntityMergeProposal } from '../world/types'
import { SetupHeader } from '../../components/SetupHeader'
import { WorldBuildStatus } from '../projects/WorldBuildStatus'

export function ParticipantsPage() {
  const { projectId = '' } = useParams()
  const navigate = useNavigate()
  const client = useQueryClient()
  const journey = useJourney(projectId)
  const graphStatus = journey.data?.graph?.status
  // 图谱已存在或正在构建时,人物资料页直接前进到第 4 步「建立图谱」。
  // 只有别名审核这类必须在确认人物页处理的状态才留下。
  const advancing = ['building', 'profile_compilation_queued', 'compiling_profile', 'ready', 'awaiting_profile_review'].includes(graphStatus ?? '')
  useEffect(() => {
    if (advancing) navigate(`/projects/${projectId}/setup/graph`, { replace: true })
  }, [advancing, navigate, projectId])
  const [preview] = useSessionState<{ id: string } | null>(`import-preview:${projectId}`, null)
  const root = `/api/projects/${projectId}/world-profile`
  const proposals = useQuery({ queryKey: ['world-merge-proposals', projectId], queryFn: () => request<EntityMergeProposal[]>(`${root}/merge-proposals`), enabled: journey.data?.graph?.status === 'awaiting_alias_review' })
  const refresh = () => {
    void client.invalidateQueries({ queryKey: journeyKey(projectId) })
    void client.invalidateQueries({ queryKey: ['world-merge-proposals', projectId] })
    void client.invalidateQueries({ queryKey: ['world-profile-status', projectId] })
  }
  const review = useMutation({ mutationFn: ({ id, decision }: { id: string; decision: string }) => request(`${root}/merge-proposals/${id}/review?decision=${decision}`, { method: 'POST' }), onSuccess: refresh })
  const generate = useMutation({ mutationFn: () => request(`${root}/merge-proposals/generate`, { method: 'POST' }), onSuccess: refresh })
  const pending = proposals.data?.filter(p => p.decision === 'pending') ?? []
  const graphReady = ['ready', 'awaiting_profile_review'].includes(journey.data?.graph?.status ?? '')
  return <Stack className="setup-page" gap="lg">
    <AsyncState loading={journey.isLoading} error={journey.error} retry={() => void journey.refetch()} />
    {preview && journey.data && !journey.data.imports.some(i => i.preview_id === preview.id) ? <ImportWizard projectId={projectId} stage="participants" onSaved={refresh} /> : <>
      <SetupHeader step={2} title="人物资料" description="目标人物、聊天记录与资料整理进度。" />
      {journey.isSuccess && !journey.data.imports.length && <StageHandoff title="先导入聊天记录" detail="解析后在这里确认双方身份。" label="选择记录" to={`/projects/${projectId}/setup/import`} />}
      {journey.data?.participants.filter(p => p.role === 'target').map(p => <Paper key={p.id} p="lg" withBorder><Group>
        <Avatar size={48} radius="md" src={p.avatar_asset_id ? `/api/projects/${projectId}/media/${p.avatar_asset_id}` : undefined}>{p.name.slice(0, 1)}</Avatar>
        <div><Text fw={600}>{p.name}</Text><Text c="dimmed" size="sm">目标人物</Text></div>
      </Group></Paper>)}
      {journey.data?.world_build && journey.data.world_build.status !== 'succeeded' && <WorldBuildStatus key={journey.data.world_build.id} projectId={projectId} build={journey.data.world_build} />}
      {Boolean(journey.data?.imports.length) && <Paper p="md" withBorder>
        <Text>记录范围：{journey.data?.time_range.map(t => t ? new Date(t).toLocaleString('zh-CN') : '未知').join(' — ')}</Text>
        <Text size="sm" c="dimmed">已保存 {journey.data?.imports.reduce((n, i) => n + i.message_count, 0)} 条记录。追加资料不会自动替换已有分支背景。</Text>
        <Button component={Link} to={`/projects/${projectId}/setup/import`} mt="md" variant="light">追加导入</Button>
      </Paper>}
    </>}
    {journey.data?.graph?.status === 'awaiting_alias_review' && <>
      <Title order={3}>别名审核</Title>
      <AsyncState loading={proposals.isLoading} error={proposals.error || review.error || generate.error} retry={() => void proposals.refetch()} />
      <Stack gap="sm">{pending.map(p => <Card key={p.id} p="md" withBorder>
        <Text fw={600}>{p.source_entities.join('、')} → {p.target_entity}</Text>
        <Text size="sm" c="dimmed" my="xs">{p.reason}</Text>
        <Accordion>
          <Accordion.Item value="evidence">
            <Accordion.Control>查看依据</Accordion.Control>
            <Accordion.Panel><Stack gap={4}>{p.evidence.map((e, i) => <Text size="sm" key={i}>{e.quote ?? e.basis ?? '未提供文字依据'}</Text>)}</Stack></Accordion.Panel>
          </Accordion.Item>
        </Accordion>
        <Group mt="sm"><Button loading={review.isPending} onClick={() => review.mutate({ id: p.id, decision: 'approve' })}>是，合并这些名字</Button>
          <Button variant="light" disabled={review.isPending} onClick={() => review.mutate({ id: p.id, decision: 'reject' })}>不是同一个人</Button>
          <Button variant="subtle" disabled={review.isPending} onClick={() => review.mutate({ id: p.id, decision: 'defer' })}>暂时不确定</Button></Group>
      </Card>)}</Stack>
      {proposals.isSuccess && !pending.length && <Alert>没有待审核候选，资料状态仍停在别名审核。可重新检查候选，完成后系统会继续整理背景。<Button loading={generate.isPending} onClick={() => generate.mutate()} mt="sm">检查并继续整理</Button></Alert>}
    </>}
    {graphReady && <Alert color="teal">资料已整理好。</Alert>}
  </Stack>
}

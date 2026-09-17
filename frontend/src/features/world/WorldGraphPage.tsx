import { Suspense, lazy } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Group, Loader, Paper, Progress, Stack, Text } from '@mantine/core'
import { useParams } from 'react-router-dom'
import { ApiError, request } from '../../api/client'
import { SetupHeader } from '../../components/SetupHeader'
import { StageHandoff } from '../../components/feedback/StageHandoff'
import { shouldPollWorldGraphStatus } from './buildStatus'
import { useJourney } from '../journey/journey'
import { WorldBuildStatus } from '../projects/WorldBuildStatus'
import type { WorldBuildProgress, WorldGraphStatus } from './types'

// Canvas 图库体积较大，只在进入建立图谱页时按需加载。
const WorldGraphView = lazy(() => import('./WorldGraphView'))

/** 第 4 步「建立图谱」：构建中实时生长，完成后仍可在这里查看完整图谱。 */
export function WorldGraphPage() {
  const { projectId = '' } = useParams()
  const queryClient = useQueryClient()
  const graphStatus = useQuery({
    queryKey: ['world-profile-status', projectId],
    queryFn: () => request<WorldGraphStatus>(`/api/projects/${projectId}/world-profile/status`),
    enabled: Boolean(projectId),
    refetchInterval: (query) => shouldPollWorldGraphStatus(query.state.data?.status) ? 2000 : false,
  })
  const rebuild = useMutation({
    mutationFn: () => request<{ job_id: string }>(
      `/api/projects/${projectId}/world-profile/rebuild`,
      { method: 'POST' },
    ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['world-profile-status', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['world-graph', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['project-jobs', projectId] })
    },
  })

  const status = graphStatus.data?.status
  const missing = graphStatus.error instanceof ApiError && graphStatus.error.status === 404
  // 恢复一个已取消的构建时图版本可能仍是 ready，构建中标记要同时看构建任务。
  const journey = useJourney(projectId)
  const buildRunning = ['queued', 'running'].includes(journey.data?.world_build?.status ?? '')
  const building = shouldPollWorldGraphStatus(status) || buildRunning
  const failed = status === 'failed' || status === 'interrupted'

  return (
    <Stack className="setup-page" gap="lg">
      <SetupHeader step={3} title="建立图谱" description="从聊天记录构建知识图谱；构建时新节点会实时出现在下面。" />
      {graphStatus.isLoading ? (
        <Group justify="center" py={80}><Loader aria-label="正在读取图谱状态" /></Group>
      ) : null}
      {graphStatus.isError && !missing ? (
        <Alert color="red" title="图谱状态读取失败">请稍后重试。</Alert>
      ) : null}
      {missing ? (
        <>
          <Alert color="moon" title="尚未建立图谱">
            请先导入聊天记录并确认人物，然后在这里建立知识图谱。
          </Alert>
          <Button
            loading={rebuild.isPending}
            onClick={() => rebuild.mutate()}
            style={{ alignSelf: 'flex-start' }}
          >
            {rebuild.isSuccess ? '已开始构建' : '开始建立图谱'}
          </Button>
          {rebuild.isError ? <Alert color="red">暂时无法开始构建，请稍后重试。</Alert> : null}
        </>
      ) : null}
      {status === 'awaiting_alias_review' ? (
        <StageHandoff
          title="先确认人物名字"
          detail="这些名字是否指同一个人，需要在确认人物页审核；审核完成后图谱会继续整理。"
          to={`/projects/${projectId}/setup/participants`}
          label="查看待确认名字"
        />
      ) : null}
      {journey.data?.world_build && journey.data.world_build.status !== 'succeeded' ? (
        <WorldBuildStatus key={journey.data.world_build.id} projectId={projectId} build={journey.data.world_build} />
      ) : failed ? (
        <Alert color="red" title="图谱构建受阻">自动恢复未能完成构建，已有进展保留。</Alert>
      ) : null}
      {building ? <WorldGraphBuildProgress progress={graphStatus.data?.build_progress} /> : null}
      {graphStatus.isSuccess && status !== 'awaiting_alias_review' ? (
        <Suspense fallback={<Group justify="center" py={80}><Loader aria-label="正在加载图谱" /></Group>}>
          <WorldGraphView projectId={projectId} building={building} />
        </Suspense>
      ) : null}
    </Stack>
  )
}

function WorldGraphBuildProgress({ progress }: { progress: WorldBuildProgress | null | undefined }) {
  const completed = progress?.completed_questions
  const total = progress?.question_count
  const rawProgress = progress?.progress
  // 排队后会立即复用完成的图谱；尚未写入第一个 checkpoint 时，从编译阶段起点展示。
  const value = Math.round(Math.min(1, Math.max(0, rawProgress ?? 0)) * 100)
  const stage = progress?.stage
  const compiling = stage === 'compiling_profile' || stage === 'person_world_sections'
  const detail = compiling && completed !== null && completed !== undefined && total
    ? `正在整理第 ${completed} / ${total} 个背景维度`
    : stage === 'indexing_world'
      ? '正在建立候选 LightRAG 图谱'
      : stage === 'bundles_ready'
        ? '正在从聊天记录整理会话窗口'
      : progress?.status === 'queued'
        ? '人物世界任务正在等待执行'
        : '正在准备 PersonWorld 调查工作流'

  return (
    <Paper p="md" withBorder>
      <Group justify="space-between" mb="xs">
        <Text fw={600}>{compiling ? 'PersonWorld 调查进度' : '图谱构建进度'}</Text>
        <Text c="moon.4" fw={700} size="sm">{rawProgress == null ? '处理中' : `${value}%`}</Text>
      </Group>
      {rawProgress != null && <Progress aria-label="图谱构建进度" color="moon" radius="xl" size="lg" value={value} />}
      <Text c="dimmed" mt="xs" size="sm">{detail}</Text>
    </Paper>
  )
}

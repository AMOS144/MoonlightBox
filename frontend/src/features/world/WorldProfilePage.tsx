import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert,
  Badge,
  Button,
  Card,
  Group,
  Loader,
  Paper,
  Progress,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from '@mantine/core'
import { useParams } from 'react-router-dom'

import { ApiError, request } from '../../api/client'
import { SectionNav } from '../../components/SectionNav'
import { worldNav } from '../../components/sectionNavItems'
import type {
  EntityMergeProposal,
  SourcedStatement,
  SourceStatus,
  WorldBuildProgress,
  WorldGraphStatus,
  WorldProfile,
} from './types'

const statusLabels: Record<SourceStatus, string> = {
  direct: '原话',
  summarized: '历史归纳',
  inferred: '上下文推导',
  superseded: '旧阶段',
}

export function WorldProfilePage() {
  const { projectId } = useParams()
  const queryClient = useQueryClient()
  const profile = useQuery({
    queryKey: ['world-profile', projectId],
    queryFn: () => request<WorldProfile>(`/api/projects/${projectId}/world-profile`),
    enabled: Boolean(projectId),
    retry: (count, error) => !(error instanceof ApiError && error.status === 404) && count < 2,
    refetchInterval: (query) => {
      const error = query.state.error
      return error instanceof ApiError && error.status === 404 ? 2000 : false
    },
  })
  const graphStatus = useQuery({
    queryKey: ['world-profile-status', projectId],
    queryFn: () => request<WorldGraphStatus>(`/api/projects/${projectId}/world-profile/status`),
    enabled: Boolean(projectId),
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status && status !== 'ready' ? 2000 : false
    },
  })
  const rebuild = useMutation({
    mutationFn: () => request<{ job_id: string }>(
      `/api/projects/${projectId}/world-profile/rebuild`,
      { method: 'POST' },
    ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['project-jobs', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['world-profile', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['world-profile-status', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['world-merge-proposals', projectId] })
    },
  })
  const proposals = useQuery({
    queryKey: ['world-merge-proposals', projectId],
    queryFn: () => request<EntityMergeProposal[]>(`/api/projects/${projectId}/world-profile/merge-proposals`),
    enabled: Boolean(projectId),
    refetchInterval: () => graphStatus.data?.status === 'awaiting_alias_review' ? 2000 : false,
  })
  const generateProposals = useMutation({
    mutationFn: () => request<EntityMergeProposal[]>(
      `/api/projects/${projectId}/world-profile/merge-proposals/generate`,
      { method: 'POST' },
    ),
    onSuccess: (items) => {
      queryClient.setQueryData(['world-merge-proposals', projectId], items)
    },
  })
  const reviewProposal = useMutation({
    mutationFn: ({ id, decision }: { id: string; decision: 'approve' | 'reject' | 'defer' }) =>
      request<EntityMergeProposal>(
        `/api/projects/${projectId}/world-profile/merge-proposals/${id}/review?decision=${decision}`,
        { method: 'POST' },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['world-merge-proposals', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['world-profile', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['world-profile-status', projectId] })
    },
  })

  if (!projectId) return <Alert color="red">缺少项目标识。</Alert>
  if (profile.isLoading) {
    return <Group justify="center" py={80}><Loader aria-label="正在读取人物背景" /></Group>
  }
  if (profile.isError || !profile.data) {
    const missing = profile.error instanceof ApiError && profile.error.status === 404
    if (missing && graphStatus.isLoading) {
      return <Group justify="center" py={80}><Loader aria-label="正在读取人物世界状态" /></Group>
    }
    const status = graphStatus.data?.status
    if (missing && (status === 'awaiting_alias_review' || status === 'profile_compilation_queued' || status === 'compiling_profile' || status === 'building')) {
      const compiling = status === 'profile_compilation_queued' || status === 'compiling_profile'
      return (
        <Stack gap="lg">
          <SectionNav items={worldNav} label="世界" />
          <Alert color="moon" title={status === 'awaiting_alias_review' ? '请先审核人物别名' : compiling ? '别名已确认，正在编译人物背景' : '人物世界正在建立'}>
            {status === 'awaiting_alias_review'
              ? '人物背景会在所有别名合并候选完成审核后生成。'
              : compiling
                ? '系统正在复用现有 LightRAG 图谱查询并生成 PersonWorldProfile。'
                : 'LightRAG 正在读取聊天并生成别名候选，请稍候。'}
          </Alert>
          {compiling ? <WorldProfileCompileProgress progress={graphStatus.data?.build_progress} /> : null}
          <MergeProposalSection
            proposals={proposals.data ?? []}
            loading={proposals.isLoading}
            generating={generateProposals.isPending}
            reviewing={reviewProposal.isPending}
            onGenerate={() => generateProposals.mutate()}
            onReview={(id, decision) => reviewProposal.mutate({ id, decision })}
            error={proposals.isError || generateProposals.isError ? '归并提案读取或生成失败，请重试。' : null}
          />
        </Stack>
      )
    }
    return (
      <Stack gap="lg">
        <SectionNav items={worldNav} label="世界" />
        <Alert color={missing ? 'moon' : 'red'} title={missing ? '人物世界还没有建立' : '人物背景读取失败'}>
          {missing
            ? '导入聊天后系统会先通过 LightRAG 建图并生成别名候选；完成审核后才会编译人物背景。'
            : '请稍后重试。'}
        </Alert>
        {missing ? (
          <Button
            loading={rebuild.isPending}
            onClick={() => rebuild.mutate()}
            style={{ alignSelf: 'flex-start' }}
          >
            {rebuild.isSuccess ? '已开始构建' : '从现有聊天构建人物世界'}
          </Button>
        ) : null}
        {rebuild.isError ? <Alert color="red">无法开始构建，请检查 Sidecar 配置。</Alert> : null}
      </Stack>
    )
  }

  const data = profile.data
  return (
    <Stack gap="xl">
      <SectionNav items={worldNav} label="世界" />
      <Group align="flex-end" justify="space-between" wrap="wrap">
        <div>
          <Text c="moon.4" fw={700} size="xs">LIGHTRAG HISTORICAL WORLD</Text>
          <Title mt={6} order={1}>人物背景</Title>
          <Text c="dimmed" mt={8}>从完整聊天世界自动归纳。实体归并必须经过你的人工审核。</Text>
        </div>
        <Badge color="green" size="lg" variant="light">
          {data.graph.bundle_count} 个会话窗口 · {data.graph.message_count} 条消息
        </Badge>
      </Group>

      <SimpleGrid cols={{ base: 1, md: 2 }}>
        <ProfileSection
          items={[
            ...data.identity.names,
            ...data.identity.aliases,
            ...data.identity.self_descriptions,
            ...data.identity.roles,
          ]}
          title="身份与称呼"
        />
        <ProfileSection items={data.work_and_education} title="工作与教育" />
        <ProfileSection items={data.social_relationships} title="社会关系" />
        <ProfileSection items={data.places} title="个人地点" />
        <ProfileSection items={data.preferences} title="兴趣与偏好" />
        <ProfileSection items={data.recurring_activities} title="长期活动" />
        <ProfileSection
          items={[
            ...data.routine_summary.workdays,
            ...data.routine_summary.weekends,
            ...data.routine_summary.other_patterns,
          ]}
          title="生活规律"
        />
        <ProfileSection items={data.life_phases} title="生活阶段" />
        <ProfileSection
          items={[
            ...data.relationship_with_user.overview,
            ...data.relationship_with_user.changes_over_time,
          ]}
          title="与用户的关系"
        />
        <ProfileSection items={data.important_events} title="重要经历" />
      </SimpleGrid>

      {data.unresolved_candidates.length ? (
        <ProfileSection items={data.unresolved_candidates} title="尚未消歧的候选" />
      ) : null}
    </Stack>
  )
}

function WorldProfileCompileProgress({ progress }: { progress: WorldBuildProgress | null | undefined }) {
  const completed = progress?.completed_questions
  const total = progress?.question_count
  const rawProgress = progress?.progress
  // 排队后会立即复用完成的图谱；尚未写入第一个 checkpoint 时，从编译阶段起点展示。
  const value = Math.round(Math.min(0.99, Math.max(0.60, rawProgress ?? 0.60)) * 100)
  const stage = progress?.stage
  const detail = stage === 'compiling_profile' && completed !== null && completed !== undefined && total
    ? `正在整理第 ${completed} / ${total} 个背景维度`
    : stage === 'indexing_world'
      ? '正在复用已建立的 LightRAG 图谱'
      : progress?.status === 'queued'
        ? '编译任务正在等待执行'
        : '正在从 LightRAG 查询人物背景并生成档案'

  return (
    <Paper p="md" withBorder>
      <Group justify="space-between" mb="xs">
        <Text fw={600}>人物背景编译进度</Text>
        <Text c="moon.4" fw={700} size="sm">{value}%</Text>
      </Group>
      <Progress aria-label="人物背景编译进度" color="moon" radius="xl" size="lg" value={value} />
      <Text c="dimmed" mt="xs" size="sm">{detail}</Text>
    </Paper>
  )
}

function MergeProposalSection({
  proposals,
  loading,
  generating,
  reviewing,
  onGenerate,
  onReview,
  error,
}: {
  proposals: EntityMergeProposal[]
  loading: boolean
  generating: boolean
  reviewing: boolean
  onGenerate: () => void
  onReview: (id: string, decision: 'approve' | 'reject' | 'defer') => void
  error: string | null
}) {
  const pending = proposals.filter((item) => item.decision === 'pending')
  return (
    <Paper p="lg" withBorder>
      <Group justify="space-between" mb="md" wrap="wrap">
        <div>
          <Title order={3}>人物节点归并审核</Title>
          <Text c="dimmed" size="sm" mt={4}>模型只提出“可能是同一个人”的候选；批准前不会修改 LightRAG 图谱。</Text>
        </div>
        <Button loading={generating} variant="light" onClick={onGenerate}>重新生成候选</Button>
      </Group>
      {error ? <Alert color="red" mb="sm">{error}</Alert> : null}
      {loading ? <Loader aria-label="正在读取归并提案" /> : null}
      {!loading && !pending.length ? (
        <Text c="dimmed" size="sm">目前没有待审核提案。点击“重新生成候选”检查当前人物节点。</Text>
      ) : null}
      <Stack gap="sm">
        {pending.map((proposal) => (
          <Card key={proposal.id} withBorder padding="md">
            <Group justify="space-between" align="flex-start" wrap="wrap">
              <div>
                <Text fw={700}>{proposal.source_entities.join('、')} → {proposal.target_entity}</Text>
                {proposal.reason ? <Text size="sm" c="dimmed" mt={5}>{proposal.reason}</Text> : null}
                {proposal.evidence.length ? (
                  <Stack gap={4} mt={6}>
                    <Text size="xs" c="dimmed">Agent 工具返回 {proposal.evidence.length} 条依据</Text>
                    {proposal.evidence.slice(0, 4).map((item, index) => (
                      <Text key={`${item.message_id ?? item.entity ?? 'evidence'}-${index}`} size="xs" c="dimmed">
                        {item.message_id ? `消息 ${item.message_id}` : item.source_document_id ? `来源 ${item.source_document_id}` : 'LightRAG'}
                        {item.quote ? `：${item.quote}` : item.basis ? `：${item.basis}` : ''}
                      </Text>
                    ))}
                  </Stack>
                ) : null}
              </div>
              <Group gap="xs">
                <Button size="xs" color="green" loading={reviewing} onClick={() => onReview(proposal.id, 'approve')}>批准合并</Button>
                <Button size="xs" color="red" variant="light" loading={reviewing} onClick={() => onReview(proposal.id, 'reject')}>拒绝</Button>
                <Button size="xs" variant="subtle" loading={reviewing} onClick={() => onReview(proposal.id, 'defer')}>暂缓</Button>
              </Group>
            </Group>
          </Card>
        ))}
      </Stack>
    </Paper>
  )
}

function ProfileSection({ title, items }: { title: string; items: SourcedStatement[] }) {
  return (
    <Paper p="lg" withBorder>
      <Group justify="space-between" mb="md">
        <Title order={3}>{title}</Title>
        <Badge variant="light">{items.length}</Badge>
      </Group>
      <Stack gap="sm">
        {items.map((item, index) => (
          <Card key={`${item.text}-${index}`} padding="sm" withBorder>
            <Text size="sm">{item.text}</Text>
            <Group gap="xs" mt="xs">
              <Badge
                color={item.source_status === 'inferred' ? 'yellow' : item.source_status === 'direct' ? 'green' : 'gray'}
                size="xs"
                variant="light"
              >
                {statusLabels[item.source_status]}
              </Badge>
              {item.source_document_ids.length ? (
                <Text c="dimmed" size="xs">{item.source_document_ids.length} 个来源窗口</Text>
              ) : null}
              {item.temporal_status && item.temporal_status !== 'unknown' ? <Badge size="xs" variant="outline">{item.temporal_status}</Badge> : null}
              {item.referent && item.referent !== 'unknown' ? <Badge size="xs" variant="outline">{item.referent}</Badge> : null}
            </Group>
          </Card>
        ))}
        {!items.length ? <Text c="dimmed" size="sm">聊天中暂未出现，后续导入会自动更新。</Text> : null}
      </Stack>
    </Paper>
  )
}

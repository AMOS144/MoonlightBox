import { userMessage } from '../../components/feedback/messages'
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert,
  Badge,
  Button,
  Card,
  Group,
  Loader,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from '@mantine/core'
import { useEffect, useState } from 'react'
import { Link, Navigate, useParams, useSearchParams } from 'react-router-dom'
import { StageHandoff } from '../../components/feedback/StageHandoff'
import { Icon } from '../../components/Icon'
import { useJourney } from '../journey/journey'

import { ApiError, request } from '../../api/client'
import { SectionNav } from '../../components/SectionNav'
import { PageHeader } from '../../components/PageHeader'
import { worldNav } from '../../components/sectionNavItems'
import { PersonWorldRevisionPanel } from './PersonWorldRevisionPanel'
import { V3ProfileGrid } from './V3ProfileGrid'
import { profileSelectionKey } from './profileSelection'
import { shouldPollWorldGraphStatus } from './buildStatus'
import type {
  InvestigationReport,
  PhoenixAgentExecutionSummary,
  ProfileStatementSelection,
  SectionTaskAudit,
  SourcedStatement,
  SourceStatus,
  WorldGraphStatus,
  WorldProfile,
  WorldProfileDraftRead,
} from './types'

type SectionRetryResponse = {
  job_id: string
  section: string
  status: string
  retry_attempt: number
}

type BackgroundJob = {
  id: string
  status: 'queued' | 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted'
  error_message: string | null
}

const statusLabels: Record<SourceStatus, string> = {
  direct: '原话',
  summarized: '历史归纳',
  inferred: '上下文推导',
  human_corrected: '用户已纠正',
  superseded: '旧阶段',
}

export function WorldProfilePage() {
  const { projectId } = useParams()
  const [viewParams] = useSearchParams()
  const publishedView = viewParams.get('view') === 'published'
  const journey = useJourney(projectId ?? '')
  const queryClient = useQueryClient()
  const [correctionOpen, setCorrectionOpen] = useState(false)
  const [editing, setEditing] = useState(false)
  const [selections, setSelections] = useState<ProfileStatementSelection[]>([])
  const [selectionLocked, setSelectionLocked] = useState(false)
  const [sectionRetry, setSectionRetry] = useState<{ jobId: string, section: string } | null>(null)

  useEffect(() => {
    if (projectId && (viewParams.get('revision') || window.localStorage.getItem(`moonlightbox:person-world-revision:${projectId}`))) {
      setCorrectionOpen(true)
      setEditing(true)
    }
  }, [projectId, viewParams])

  const openNewCorrection = () => {
    // 返回阅读不清空选项；取消修订由会话面板负责。
    setEditing(true)
    setCorrectionOpen(true)
  }
  const reopenCorrection = openNewCorrection
  const toggleSelection = (selection: ProfileStatementSelection) => {
    if (selectionLocked) return
    setSelections((current) => current.some((item) => item.key === selection.key)
      ? current.filter((item) => item.key !== selection.key)
      : [...current, selection])
    setCorrectionOpen(true)
  }
  const selectedKeys = new Set(selections.map((item) => item.key))
  const graphStatus = useQuery({
    queryKey: ['world-profile-status', projectId],
    queryFn: () => request<WorldGraphStatus>(`/api/projects/${projectId}/world-profile/status`),
    enabled: Boolean(projectId),
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return shouldPollWorldGraphStatus(status) ? 2000 : false
    },
  })
  const profile = useQuery({
    queryKey: ['world-profile', projectId],
    queryFn: () => request<WorldProfile>(`/api/projects/${projectId}/world-profile`),
    // 候选版尚未发布时没有可读的正式 Profile，接口返回 404 是正常状态，不能把它
    // 当作“等待资料生成”而每两秒重试。先读取图状态，只有正式版本可读时才请求它。
    enabled: Boolean(projectId)
      && graphStatus.isSuccess,
    retry: (count, error) => !(error instanceof ApiError && error.status === 404) && count < 2,
  })
  const draft = useQuery({
    queryKey: ['world-profile-draft', projectId],
    queryFn: () => request<WorldProfileDraftRead>(`/api/projects/${projectId}/world-agent/draft`),
    enabled: Boolean(projectId) && graphStatus.data?.status === 'awaiting_profile_review',
    retry: false,
  })
  // 审核页不再相信领域库中的工具次数。每个栏目只保留一次 Phoenix execution id，
  // 这里读取根 Span 的汇总统计，避免下载完整 Trace 或维护第二套调用账本。
  const sectionPhoenixQueries = useQueries({
    queries: (draft.data?.section_tasks ?? []).map((task) => ({
      queryKey: [
        'phoenix-agent-execution-summary',
        task.phoenix_execution_id ?? task.phoenix_owner_id,
      ],
      queryFn: () => request<PhoenixAgentExecutionSummary>(
        task.phoenix_execution_id
          ? `/api/observability/agent-executions/${task.phoenix_execution_id}/summary`
          : `/api/observability/agent-owners/${encodeURIComponent(task.phoenix_owner_id ?? '')}/latest-summary`,
      ),
      enabled: Boolean(task.phoenix_execution_id || task.phoenix_owner_id),
      retry: 2,
      retryDelay: 500,
      staleTime: Number.POSITIVE_INFINITY,
    })),
  })
  const phoenixSummaryBySection = new Map<string, PhoenixAgentExecutionSummary>()
  ;(draft.data?.section_tasks ?? []).forEach((task, index) => {
    const summary = sectionPhoenixQueries[index]?.data
    if (summary) phoenixSummaryBySection.set(task.section, summary)
  })
  const approveDraft = useMutation({
    mutationFn: (value: WorldProfileDraftRead) => request<Record<string, unknown>>(
      `/api/projects/${projectId}/world-agent/draft/approve`,
      {
        method: 'POST',
        body: JSON.stringify({
          graph_version_id: value.graph.id,
          profile_id: value.profile.id,
        }),
      },
    ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['world-profile', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['world-profile-status', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['world-profile-draft', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['project-jobs', projectId] })
    },
  })
  const retrySection = useMutation({
    mutationFn: (section: string) => {
      const agentRunId = draft.data?.profile.agent_run_id
      if (!agentRunId) throw new Error('候选档案缺少调查运行标识，无法重试栏目。')
      return request<SectionRetryResponse>(
        `/api/projects/${projectId}/world-agent/runs/${agentRunId}/sections/${section}/retry`,
        {
          method: 'POST',
          body: JSON.stringify({ idempotency_key: newSectionRetryKey() }),
        },
      )
    },
    onSuccess: (response) => {
      setSectionRetry({ jobId: response.job_id, section: response.section })
      void queryClient.invalidateQueries({ queryKey: ['world-profile-draft', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['project-jobs', projectId] })
    },
  })
  const sectionRetryJob = useQuery({
    queryKey: ['world-profile-section-retry', sectionRetry?.jobId],
    queryFn: () => request<BackgroundJob>(`/api/jobs/${sectionRetry?.jobId}`),
    enabled: Boolean(sectionRetry?.jobId),
    refetchInterval: (query) => ['queued', 'running', 'cancelling'].includes(query.state.data?.status ?? '')
      ? 1500
      : false,
  })

  useEffect(() => {
    const job = sectionRetryJob.data
    if (!sectionRetry || !job || ['queued', 'running', 'cancelling'].includes(job.status)) return
    void queryClient.invalidateQueries({ queryKey: ['world-profile-draft', projectId] })
    void queryClient.invalidateQueries({ queryKey: ['project-jobs', projectId] })
    setSectionRetry(null)
  }, [projectId, queryClient, sectionRetry, sectionRetryJob.data])
  const recompile = useMutation({
    mutationFn: () => request<{ job_id: string }>(
      `/api/projects/${projectId}/world-agent/recompile`,
      { method: 'POST' },
    ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['world-profile-status', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['world-profile', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['project-jobs', projectId] })
    },
  })

  if (!projectId) return <Alert color="red">缺少项目标识。</Alert>
  // 保留已取得的候选内容。即使查询被手动失效并重新读取，也不能用全页 Loader
  // 替换审核页，否则用户会感觉页面闪动并被送回页面顶部。
  if (profile.isLoading || (
    graphStatus.data?.status === 'awaiting_profile_review'
    && draft.isLoading
    && !draft.data
  )) {
    return <Group justify="center" py={80}><Loader aria-label="正在读取人物背景" /></Group>
  }
  if (!publishedView && graphStatus.data?.status === 'awaiting_profile_review' && draft.data) {
    return (
      <div className={`person-world-workbench${correctionOpen ? ' person-world-workbench--agent-open' : ''}`}>
        <main className="person-world-workbench__profile">
          {journey.data?.publication && <Button component={Link} variant="light" mb="md" to={`/projects/${projectId}/world?view=published`}>查看当前已发布背景（不影响新草稿）</Button>}
          <CandidateProfileReview
            correctionOpen={correctionOpen}
            editing={editing}
            onRead={() => { setEditing(false); setCorrectionOpen(false) }}
            draft={draft.data}
            phoenixSummaryBySection={phoenixSummaryBySection}
            approving={approveDraft.isPending}
            error={approveDraft.isError
              ? '候选档案发布失败，请重试。'
              : recompile.isError
                ? '重新生成 v3 失败，请检查是否已有候选正在生成。'
              : retrySection.isError
                ? '栏目重试请求失败，请稍后重试。'
                : sectionRetryJob.data?.status === 'failed'
                  ? (sectionRetryJob.data.error_message ?? '栏目重试失败，请稍后重试。')
                  : null}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            retryingSection={sectionRetry?.section ?? null}
            onApprove={() => approveDraft.mutate(draft.data)}
            onRetrySection={(section) => retrySection.mutate(section)}
            onRegenerate={() => recompile.mutate()}
            regenerating={recompile.isPending}
            correctionLabel={selections.length || selectionLocked
              ? '打开纠正工作区'
              : '指出问题或补充事实'}
            onNewCorrection={selections.length || selectionLocked
              ? reopenCorrection
              : openNewCorrection}
            onToggle={toggleSelection}
          />
        </main>
        <PersonWorldRevisionPanel
          projectId={projectId}
          baseProfileId={draft.data.profile.id}
          opened={correctionOpen}
          selections={selections}
          onClose={() => setCorrectionOpen(false)}
          onLockChange={setSelectionLocked}
          onRestoreSelections={setSelections}
          onStartNew={() => setSelections([])}
        />
      </div>
    )
  }
  // 已发布档案与正在构建的候选图可以短暂并存。候选流程是用户此刻操作的对象，
  // 不能因为 profile API 仍正确返回上一份已发布档案，就把构建/别名审核界面遮住。
  const workflowStatus = graphStatus.data?.status
  const workflowActive = [
    'building',
    'awaiting_alias_review',
    'profile_compilation_queued',
    'compiling_profile',
    'awaiting_profile_review',
  ].includes(workflowStatus ?? '')
  if (!publishedView && workflowStatus === 'awaiting_alias_review') {
    return <StageHandoff title="先确认人物与资料" detail="这些名字是否指同一个人，需要在人物资料页确认；已有发布背景保持可读。" to={`/projects/${projectId}/setup/participants`} label="查看待确认名字" />
  }
  // 图谱构建是第 4 步「建立图谱」的职责：构建中与尚未建图的状态都转到该页。
  if (!publishedView && shouldPollWorldGraphStatus(workflowStatus)) {
    return <Navigate to={`/projects/${projectId}/setup/graph`} replace />
  }
  if (profile.isError || !profile.data) {
    const missing = profile.error instanceof ApiError && profile.error.status === 404
    if (missing && graphStatus.isLoading) {
      return <Group justify="center" py={80}><Loader aria-label="正在读取人物世界状态" /></Group>
    }
    if (missing) {
      return <Navigate to={`/projects/${projectId}/setup/graph`} replace />
    }
    return (
      <Stack className="setup-page" gap="lg">
        <Alert color="red" title="人物背景读取失败">请稍后重试。</Alert>
      </Stack>
    )
  }

  const data = profile.data
  return (
    <div className={`person-world-workbench${correctionOpen ? ' person-world-workbench--agent-open' : ''}`}>
      <Stack className="person-world-workbench__profile" gap="xl">
        <SectionNav items={worldNav} label="世界" />
        <PageHeader
          title="人物背景"
          description={editing ? '选择需要修改的内容，在侧栏说明。' : '人物理解与生活背景'}
          action={<Group>
            {editing && <Button variant="subtle" color="gray" leftSection={<Icon name="back" size={16} />} onClick={() => { setEditing(false); setCorrectionOpen(false) }}>返回阅读</Button>}
            {!correctionOpen && <Button
              variant="light"
              onClick={selections.length || selectionLocked
                ? reopenCorrection
                : openNewCorrection}
            >
              {editing ? '打开修订对话' : '修改背景'}
            </Button>}
            <Badge color="green" size="lg" variant="light">
              {data.graph.bundle_count} 个会话窗口 · {data.graph.message_count} 条消息
            </Badge>
          </Group>}
        />
        {recompile.isError ? (
          <Alert color="red">
            无法创建候选调查。若已有候选正在构建或等待审核，请先完成该流程。
          </Alert>
        ) : null}

        {selectionLocked ? (
          <Alert color="gray">
            当前纠正会话已经开始，本轮选择范围已锁定。完成或取消会话后可以重新选择。
          </Alert>
        ) : selections.length ? (
          <Alert color="moon">
            已选择 {selections.length} 条陈述。可以继续多选，然后在右侧补充你的纠正说明。
          </Alert>
        ) : null}

        {journey.data?.publication && !journey.data.branches.length && <Alert color="teal">人物背景已发布。</Alert>}
        {publishedView && workflowActive && <Button component={Link} variant="subtle" color="gray" leftSection={<Icon name="back" size={16} />} to={`/projects/${projectId}/setup/graph`}>返回图谱构建进度</Button>}
        <details><summary>调查状态与版本管理</summary><InvestigationOverview report={data.investigation_report} /><Button mt="sm" loading={recompile.isPending} variant="subtle" onClick={() => recompile.mutate()}>重新生成背景</Button></details>

        {data.profile_schema_version === 'v3' ? (
          <V3ProfileGrid editing={editing} projectId={projectId} profile={data.profile_v3 ?? {}} locked={selectionLocked} selectedKeys={selectedKeys} onToggle={toggleSelection} />
        ) : data.profile_schema_version === 'v2' ? (
          <V2ProfileGrid
            profile={data.profile_v2}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            onToggle={toggleSelection}
          />
        ) : (
        <SimpleGrid cols={{ base: 1, md: 2 }}>
          <ProfileSection
            items={[
              ...profileEntries('identity.names', '身份与称呼', data.identity.names),
              ...profileEntries('identity.aliases', '身份与称呼', data.identity.aliases),
              ...profileEntries(
                'identity.self_descriptions',
                '身份与称呼',
                data.identity.self_descriptions,
              ),
              ...profileEntries('identity.roles', '身份与称呼', data.identity.roles),
            ]}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            title="身份与称呼"
            onToggle={toggleSelection}
          />
          <ProfileSection
            items={profileEntries('work_and_education', '工作与教育', data.work_and_education)}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            title="工作与教育"
            onToggle={toggleSelection}
          />
          <ProfileSection
            items={profileEntries('social_relationships', '社会关系', data.social_relationships)}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            title="社会关系"
            onToggle={toggleSelection}
          />
          <ProfileSection
            items={profileEntries('places', '个人地点', data.places)}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            title="个人地点"
            onToggle={toggleSelection}
          />
          <ProfileSection
            items={profileEntries('preferences', '兴趣与偏好', data.preferences)}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            title="兴趣与偏好"
            onToggle={toggleSelection}
          />
          <ProfileSection
            items={profileEntries('recurring_activities', '长期活动', data.recurring_activities)}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            title="长期活动"
            onToggle={toggleSelection}
          />
          <ProfileSection
            items={[
              ...profileEntries(
                'routine_summary.workdays',
                '生活规律',
                data.routine_summary.workdays,
              ),
              ...profileEntries(
                'routine_summary.weekends',
                '生活规律',
                data.routine_summary.weekends,
              ),
              ...profileEntries(
                'routine_summary.other_patterns',
                '生活规律',
                data.routine_summary.other_patterns,
              ),
            ]}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            title="生活规律"
            onToggle={toggleSelection}
          />
          <ProfileSection
            items={profileEntries('life_phases', '生活阶段', data.life_phases)}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            title="生活阶段"
            onToggle={toggleSelection}
          />
          <ProfileSection
            items={[
              ...profileEntries(
                'relationship_with_user.overview',
                '与用户的关系',
                data.relationship_with_user.overview,
              ),
              ...profileEntries(
                'relationship_with_user.changes_over_time',
                '与用户的关系',
                data.relationship_with_user.changes_over_time,
              ),
            ]}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            title="与用户的关系"
            onToggle={toggleSelection}
          />
          <ProfileSection
            items={profileEntries('important_events', '重要经历', data.important_events)}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            title="重要经历"
            onToggle={toggleSelection}
          />
        </SimpleGrid>
        )}

        {data.profile_schema_version !== 'v2' && data.unresolved_candidates.length ? (
          <ProfileSection
            items={profileEntries(
              'unresolved_candidates',
              '尚未消歧的候选',
              data.unresolved_candidates,
            )}
            locked={selectionLocked}
            selectedKeys={selectedKeys}
            title="尚未消歧的候选"
            onToggle={toggleSelection}
          />
        ) : null}
      </Stack>
      <PersonWorldRevisionPanel
        projectId={projectId}
        baseProfileId={data.id}
        opened={correctionOpen}
        selections={selections}
        onClose={() => setCorrectionOpen(false)}
        onLockChange={setSelectionLocked}
        onRestoreSelections={setSelections}
        onStartNew={() => setSelections([])}
      />
    </div>
  )
}

function CandidateProfileReview({
  draft,
  phoenixSummaryBySection,
  approving,
  error,
  locked,
  selectedKeys,
  retryingSection,
  correctionLabel,
  correctionOpen,
  onApprove,
  onRetrySection,
  onRegenerate,
  regenerating,
  onNewCorrection,
  onToggle,
  editing,
  onRead,
}: {
  editing: boolean
  onRead: () => void
  draft: WorldProfileDraftRead
  phoenixSummaryBySection: ReadonlyMap<string, PhoenixAgentExecutionSummary>
  approving: boolean
  error: string | null
  locked: boolean
  selectedKeys: Set<string>
  retryingSection: string | null
  correctionLabel: string
  correctionOpen: boolean
  onApprove: () => void
  onRetrySection: (section: string) => void
  onRegenerate: () => void
  regenerating: boolean
  onNewCorrection: () => void
  onToggle: (selection: ProfileStatementSelection) => void
}) {
  const data = draft.profile
  const { projectId = '' } = useParams()
  const summary = data.generation_summary
  const missingSectionCount = Array.isArray(summary.missing_sections)
    ? summary.missing_sections.length
    : 0
  return (
    <Stack gap="xl">
      <SectionNav items={worldNav} label="世界" />
      <Alert color="yellow" title="待审核背景">
        确认并发布后才会用于后续分支；当前已发布版本不受影响。
      </Alert>
      <Group justify="space-between" wrap="wrap">
        <div>
          <Title order={1}>审核人物背景</Title>
          <Text c="dimmed" mt={6}>
            图版本 r{draft.graph.revision} · {data.profile_schema_version === 'v3'
              ? '新版七栏人物画像 · AI 综合理解，可对话修改'
              : `已收录 ${String(summary.accepted ?? 0)} 条 · 尚未覆盖 ${missingSectionCount} 个栏目`}
          </Text>
        </div>
        <Group>
          {editing && <Button variant="subtle" color="gray" leftSection={<Icon name="back" size={16} />} onClick={onRead}>返回阅读</Button>}
          {!correctionOpen && <Button variant="light" onClick={onNewCorrection}>{correctionLabel}</Button>}
          {data.profile_schema_version !== 'v3' ? (
            <Button loading={regenerating} onClick={onRegenerate}>重新生成 v3 画像</Button>
          ) : null}
          <Button color="green" disabled={Boolean(retryingSection)} loading={approving} onClick={onApprove}>
            确认并发布这个版本
          </Button>
        </Group>
      </Group>
      {error ? <Alert color="red">{userMessage(error)}</Alert> : null}
      {data.profile_schema_version !== 'v3' ? (
        <Alert color="yellow">这是历史画像，内容保留可读；旧版生成与栏目重试已退役，需要更新时请重新生成 v3 画像。</Alert>
      ) : null}
      <details><summary>查看调查状态、失败恢复与诊断</summary><InvestigationOverview
        report={data.investigation_report}
        taskAudits={draft.section_tasks}
        phoenixSummaryBySection={phoenixSummaryBySection}
        retryingSection={retryingSection}
        onRetrySection={data.profile_schema_version === 'v3' ? onRetrySection : undefined}
      /></details>
      {data.profile_schema_version === 'v3' ? (
        <V3ProfileGrid editing={editing} projectId={projectId} profile={data.profile_v3 ?? {}} locked={locked} selectedKeys={selectedKeys} onToggle={onToggle} />
      ) : data.profile_schema_version === 'v2' ? (
        <V2ProfileGrid
          profile={data.profile_v2}
          locked={locked}
          selectedKeys={selectedKeys}
          onToggle={onToggle}
        />
      ) : (
      <SimpleGrid cols={{ base: 1, md: 2 }}>
        <ProfileSection
          items={[
            ...profileEntries('identity.names', '身份与称呼', data.identity.names),
            ...profileEntries('identity.aliases', '身份与称呼', data.identity.aliases),
            ...profileEntries(
              'identity.self_descriptions',
              '身份与称呼',
              data.identity.self_descriptions,
            ),
            ...profileEntries('identity.roles', '身份与称呼', data.identity.roles),
          ]}
          locked={locked}
          selectedKeys={selectedKeys}
          title="身份与称呼"
          onToggle={onToggle}
        />
        <ProfileSection
          items={profileEntries('work_and_education', '工作与教育', data.work_and_education)}
          locked={locked}
          selectedKeys={selectedKeys}
          title="工作与教育"
          onToggle={onToggle}
        />
        <ProfileSection
          items={profileEntries('social_relationships', '社会关系', data.social_relationships)}
          locked={locked}
          selectedKeys={selectedKeys}
          title="社会关系"
          onToggle={onToggle}
        />
        <ProfileSection
          items={profileEntries('places', '个人地点', data.places)}
          locked={locked}
          selectedKeys={selectedKeys}
          title="个人地点"
          onToggle={onToggle}
        />
        <ProfileSection
          items={profileEntries('preferences', '兴趣与偏好', data.preferences)}
          locked={locked}
          selectedKeys={selectedKeys}
          title="兴趣与偏好"
          onToggle={onToggle}
        />
        <ProfileSection
          items={profileEntries('recurring_activities', '长期活动', data.recurring_activities)}
          locked={locked}
          selectedKeys={selectedKeys}
          title="长期活动"
          onToggle={onToggle}
        />
        <ProfileSection
          items={[
            ...profileEntries(
              'routine_summary.workdays',
              '生活规律',
              data.routine_summary.workdays,
            ),
            ...profileEntries(
              'routine_summary.weekends',
              '生活规律',
              data.routine_summary.weekends,
            ),
            ...profileEntries(
              'routine_summary.other_patterns',
              '生活规律',
              data.routine_summary.other_patterns,
            ),
          ]}
          locked={locked}
          selectedKeys={selectedKeys}
          title="生活规律"
          onToggle={onToggle}
        />
        <ProfileSection
          items={profileEntries('life_phases', '生活阶段', data.life_phases)}
          locked={locked}
          selectedKeys={selectedKeys}
          title="生活阶段"
          onToggle={onToggle}
        />
        <ProfileSection
          items={[
            ...profileEntries(
              'relationship_with_user.overview',
              '与用户的关系',
              data.relationship_with_user.overview,
            ),
            ...profileEntries(
              'relationship_with_user.changes_over_time',
              '与用户的关系',
              data.relationship_with_user.changes_over_time,
            ),
          ]}
          locked={locked}
          selectedKeys={selectedKeys}
          title="与用户的关系"
          onToggle={onToggle}
        />
        <ProfileSection
          items={profileEntries('important_events', '重要经历', data.important_events)}
          locked={locked}
          selectedKeys={selectedKeys}
          title="重要经历"
          onToggle={onToggle}
        />
      </SimpleGrid>
      )}
      {data.profile_schema_version !== 'v2' && data.unresolved_candidates.length ? (
        <ProfileSection
          items={profileEntries(
            'unresolved_candidates',
            '尚未消歧的候选',
            data.unresolved_candidates,
          )}
          locked={locked}
          selectedKeys={selectedKeys}
          title="尚未消歧的候选"
          onToggle={onToggle}
        />
      ) : null}
    </Stack>
  )
}

type V2CardDefinition = {
  title: string
  paths: Array<{ path: string[]; factPath: string }>
}

// v2 的七个栏目是人物事实的唯一展示结构。传给 Revision 的 section 与持久化 v2
// 字段完全一致，具体事实再由 claim_id 绑定；不再借用旧 Profile 的兼容字段名。
const v2Cards: V2CardDefinition[] = [
  {
    title: '身份',
    paths: [
      { path: ['identity', 'identifiers'], factPath: 'identity.identifiers' },
      { path: ['identity', 'self_descriptions'], factPath: 'identity.self_descriptions' },
      { path: ['identity', 'self_narratives'], factPath: 'identity.self_narratives' },
    ],
  },
  {
    title: '生活情境',
    paths: [
      { path: ['life_context', 'work_and_learning'], factPath: 'life_context.work_and_learning' },
      { path: ['life_context', 'home_and_care'], factPath: 'life_context.home_and_care' },
      { path: ['life_context', 'places_and_environment'], factPath: 'life_context.places_and_environment' },
      { path: ['life_context', 'functional_context'], factPath: 'life_context.functional_context' },
    ],
  },
  {
    title: '社会世界',
    paths: [{ path: ['social_world', 'ties'], factPath: 'social_world.ties' }],
  },
  {
    title: '能动性',
    paths: [
      { path: ['agency', 'preferences'], factPath: 'agency.preferences' },
      { path: ['agency', 'values_and_interpretations'], factPath: 'agency.values_and_interpretations' },
      { path: ['agency', 'goals_and_commitments'], factPath: 'agency.goals_and_commitments' },
    ],
  },
  {
    title: '实践与规律',
    paths: [
      { path: ['practices', 'recurring_activities'], factPath: 'practices.recurring_activities' },
      { path: ['practices', 'temporal_rhythms'], factPath: 'practices.temporal_rhythms' },
    ],
  },
  {
    title: '生命历程',
    paths: [
      { path: ['life_course', 'episodes'], factPath: 'life_course.episodes' },
      { path: ['life_course', 'transitions'], factPath: 'life_course.transitions' },
      { path: ['life_course', 'trajectories'], factPath: 'life_course.trajectories' },
    ],
  },
  {
    title: '与用户的关系',
    paths: [
      { path: ['relationship_with_user', 'standing'], factPath: 'relationship_with_user.standing' },
      { path: ['relationship_with_user', 'interaction_observations'], factPath: 'relationship_with_user.interaction_observations' },
      { path: ['relationship_with_user', 'interaction_patterns'], factPath: 'relationship_with_user.interaction_patterns' },
      { path: ['relationship_with_user', 'history'], factPath: 'relationship_with_user.history' },
    ],
  },
]

function V2ProfileGrid({
  profile,
  selectedKeys,
  locked,
  onToggle,
}: {
  profile: Record<string, unknown>
  selectedKeys: Set<string>
  locked: boolean
  onToggle: (selection: ProfileStatementSelection) => void
}) {
  return (
    <SimpleGrid cols={{ base: 1, md: 2 }}>
      {v2Cards.map((card) => (
        <ProfileSection
          items={card.paths.flatMap((item) => profileEntries(
            item.factPath,
            card.title,
            v2StatementsAt(profile, item.path),
          ))}
          key={card.title}
          locked={locked}
          selectedKeys={selectedKeys}
          title={card.title}
          onToggle={onToggle}
        />
      ))}
    </SimpleGrid>
  )
}

function v2StatementsAt(profile: Record<string, unknown>, path: string[]): SourcedStatement[] {
  let value: unknown = profile
  for (const part of path) {
    if (!isRecord(value)) return []
    value = value[part]
  }
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!isRecord(item) || typeof item.statement !== 'string' || !item.statement.trim()) return []
    return [{
      claim_id: typeof item.claim_id === 'string' ? item.claim_id : undefined,
      text: item.statement,
      // v2 Profile 保留直接转述与多消息归纳的区别，界面不能把后者伪装成原话。
      source_status: item.derivation === 'inferred' ? 'inferred' as const : 'direct' as const,
      source_document_ids: [],
      assertion_kind: 'unknown' as const,
      referent: 'target_person' as const,
      temporal_status: validTemporalStatus(item.temporal_status),
      source_message_ids: stringValues(item.evidence_message_ids),
      evidence_quotes: [],
    }]
  })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringValues(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

function validTemporalStatus(value: unknown): SourcedStatement['temporal_status'] {
  const values = new Set([
    'current', 'past', 'planned', 'recurring', 'one_off', 'timeless', 'unknown',
  ])
  return typeof value === 'string' && values.has(value)
    ? value as SourcedStatement['temporal_status']
    : 'unknown'
}

function ProfileSection({
  title,
  items,
  selectedKeys,
  locked,
  onToggle,
}: {
  title: string
  items: ProfileStatementSelection[]
  selectedKeys: Set<string>
  locked: boolean
  onToggle: (selection: ProfileStatementSelection) => void
}) {
  return (
    <Paper p="lg" withBorder>
      <Group justify="space-between" mb="md">
        <Title order={3}>{title}</Title>
        <Badge variant="light">{items.length}</Badge>
      </Group>
      <Stack gap="sm">
        {items.map((entry) => {
          const item = entry.statement
          const selected = selectedKeys.has(entry.key)
          return (
          <Card
            className={`profile-statement${selected ? ' profile-statement--selected' : ''}`}
            key={entry.key}
            padding={0}
            withBorder
          >
            <button
              aria-checked={selected}
              className="profile-statement__button"
              disabled={locked}
              role="checkbox"
              type="button"
              onClick={() => onToggle(entry)}
            >
              <span className="profile-statement__check" aria-hidden="true">
                {selected ? '✓' : ''}
              </span>
              <span className="profile-statement__content">
                <Text size="sm">{item.text}</Text>
            <Group gap="xs" mt="xs">
              <Badge
                color={item.source_status === 'human_corrected'
                  ? 'blue'
                  : item.source_status === 'inferred'
                    ? 'yellow'
                    : item.source_status === 'direct'
                      ? 'green'
                      : 'gray'}
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
              </span>
            </button>
          </Card>
          )
        })}
        {!items.length ? <Text c="dimmed" size="sm">聊天中暂未出现，后续导入会自动更新。</Text> : null}
      </Stack>
    </Paper>
  )
}

function InvestigationOverview({
  report,
  taskAudits = [],
  phoenixSummaryBySection = new Map(),
  retryingSection = null,
  onRetrySection,
}: {
  report: InvestigationReport
  taskAudits?: SectionTaskAudit[]
  phoenixSummaryBySection?: ReadonlyMap<string, PhoenixAgentExecutionSummary>
  retryingSection?: string | null
  onRetrySection?: (section: string) => void
}) {
  if (!report?.section_statuses?.length) return null
  const auditBySection = new Map(taskAudits.map((task) => [task.section, task]))
  return (
    <Paper p="md" withBorder>
      <Group justify="space-between" mb="xs">
        <Title order={4}>栏目调查状态</Title>
        <Text c="dimmed" size="xs">调查审计不属于人物事实</Text>
      </Group>
      <Group gap="xs">
        {report.section_statuses.map((item) => {
          const audit = auditBySection.get(item.section)
          const currentState = audit?.status ?? item.state
          const activelyRetrying = ['pending', 'researching'].includes(currentState)
          return (
          <Group gap={4} key={item.section}>
            <Badge color={investigationColor(currentState)} variant="light">
              {investigationLabel(item.section)} · {investigationStateLabel(currentState)}
            </Badge>
            {onRetrySection && ['cloud_error', 'schema_error', 'budget_exhausted'].includes(item.state) ? (
              <Button
                disabled={activelyRetrying || (Boolean(retryingSection) && retryingSection !== item.section)}
                loading={retryingSection === item.section || activelyRetrying}
                size="xs"
                variant="subtle"
                onClick={() => onRetrySection(item.section)}
              >
                重试本栏目
              </Button>
            ) : null}
          </Group>
          )
        })}
      </Group>
      {report.unresolved_questions.length ? (
        <Text c="dimmed" mt="sm" size="xs">
          待补信息：{report.unresolved_questions.slice(0, 3).join('；')}
        </Text>
      ) : null}
      {taskAudits
        .filter((task) => [
          'completed_without_evidence',
          'needs_more_research',
          'cloud_error',
          'schema_error',
          'budget_exhausted',
          'pending',
          'researching',
        ].includes(task.status))
        .map((task) => {
          const phoenix = phoenixSummaryBySection.get(task.section)
          return (
          <Card key={task.section} mt="sm" padding="sm" withBorder>
            <Group justify="space-between" wrap="wrap">
              <Text fw={600} size="sm">
                {investigationLabel(task.section)}：{investigationStateLabel(task.status)}
              </Text>
              <Text c="dimmed" size="xs">
                第 {task.research_round || 0} 轮 · {phoenix
                  ? `Phoenix：${phoenix.tool_call_count} 次调用，失败 ${phoenix.tool_error_count} 次`
                  : (task.phoenix_execution_id || task.phoenix_owner_id)
                    ? '正在读取 Phoenix 调用统计'
                    : '该历史运行未保存 Phoenix 关联键'}
                {task.retry_attempt ? ` · 已重试 ${task.retry_attempt} 次` : ''}
              </Text>
            </Group>
            {task.attempted_queries.length ? (
              <Text c="dimmed" mt={4} size="xs">
                已尝试：{task.attempted_queries
                  .map((query) => query.question)
                  .filter((question): question is string => Boolean(question))
                  .join('；')}
              </Text>
            ) : null}
            {task.unresolved_questions.length ? (
              <Text c="dimmed" mt={4} size="xs">
                仍待确认：{task.unresolved_questions.join('；')}
              </Text>
            ) : null}
            {task.error_code ? <Text c="red" mt={4} size="xs">错误码：{task.error_code}</Text> : null}
          </Card>
          )
        })}
    </Paper>
  )
}

function newSectionRetryKey() {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `person-world-section-retry-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function profileEntries(
  section: string,
  sectionLabel: string,
  statements: SourcedStatement[],
): ProfileStatementSelection[] {
  return statements.map((statement) => ({
    key: profileSelectionKey(
      section,
      statement.text,
      statement.source_message_ids ?? [],
      statement.claim_id,
    ),
    section,
    sectionLabel,
    statement,
  }))
}

function investigationLabel(section: string) {
  const labels: Record<string, string> = {
    identity: '身份',
    life_context: '生活情境',
    social_world: '社会关系',
    agency: '偏好与目标',
    practices: '实践规律',
    life_course: '经历轨迹',
    relationship_with_user: '与用户关系',
  }
  return labels[section] ?? section
}

function investigationStateLabel(state: string) {
  const labels: Record<string, string> = {
    completed: '已生成',
    completed_with_claims: '已生成',
    failed: '运行失败',
    completed_without_evidence: '无足够证据',
    needs_more_research: '需要继续调查',
    cloud_error: '云端调用失败',
    schema_error: '输出格式失败',
    budget_exhausted: '预算耗尽',
    not_started: '未开始',
    pending: '等待重试',
    researching: '正在调查',
  }
  return labels[state] ?? state
}

function investigationColor(state: string) {
  if (state === 'completed') return 'green'
  if (state === 'completed_without_evidence') return 'gray'
  if (state === 'needs_more_research') return 'yellow'
  if (state === 'pending' || state === 'researching') return 'blue'
  if (state === 'cloud_error' || state === 'schema_error') return 'red'
  if (state === 'budget_exhausted') return 'orange'
  return 'gray'
}

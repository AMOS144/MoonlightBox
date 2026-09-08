import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Badge, Button, Group, Paper, SegmentedControl, Stack, Text, Timeline as MantineTimeline, Title } from '@mantine/core'
import { useNavigate, useParams } from 'react-router-dom'

import { request } from '../../api/client'
import { Icon } from '../../components/Icon'
import { SectionNav } from '../../components/SectionNav'
import { memoryNav } from '../../components/sectionNavItems'
import { useSessionState } from '../../hooks/useSessionState'
import type { EventNode } from './types'

const EVENT_TYPE_LABELS: Record<string, string> = {
  relationship_started: '关系建立',
  intimacy_increased: '亲密升级',
  commitment: '承诺',
  boundary_change: '边界变化',
  conflict: '冲突',
  distancing: '关系疏远',
  reconciliation: '和解',
  separation: '分离',
  reconnection: '重新联系',
  cold_war: '冷战',
  long_pause: '长时间中断',
  date: '约会',
  outing: '出游',
  travel: '旅行',
  celebration: '庆祝',
  gift: '礼物',
  family_social: '见亲友',
  support_care: '照顾陪伴',
  shared_project: '共同项目',
  important_plan: '重要计划',
  life_milestone: '人生里程碑',
}

type LaneFilter = 'all' | 'relationship' | 'shared_experience'

function EventCard({ event, projectId }: { event: EventNode; projectId: string }) {
  const queryClient = useQueryClient()
  const reject = useMutation({
    mutationFn: () =>
      request<EventNode>(`/api/projects/${projectId}/events/${event.id}`, {
        method: 'PATCH',
        body: JSON.stringify({
          changes: { status: 'rejected' },
          reason: '人工标记为误报',
        }),
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['events', projectId] }),
  })
  const typeLabel = EVENT_TYPE_LABELS[event.type] ?? '其他变化'
  const titleId = `event-title-${event.id}`
  const actionLabel = `${reject.isPending ? '正在' : ''}标记${typeLabel}节点 ${event.id} 为误报`
  return (
    <Paper
      aria-label={`${typeLabel} ${event.title || typeLabel} ${event.id}`}
      className="event-card"
      component="article"
      p="lg"
      withBorder
    >
      <Title id={titleId} order={3}>{event.title || typeLabel}</Title>
      <Group gap={6} mt="sm">
        {event.title !== typeLabel ? <Badge color="gray" variant="light">{typeLabel}</Badge> : null}
        <Badge color="moon" variant="light">重要度 {Math.round(event.importance * 100)}%</Badge>
        {event.emotion_labels.slice(0, 2).map((label) => <Badge color="violet" key={label} variant="light">{label}</Badge>)}
      </Group>
      <Text className="event-card__memory" mt="md">
        {event.display_summary ||
          (event.summary_status === 'failed'
            ? '这段回忆暂时没有整理完成。'
            : '正在整理这段回忆……')}
      </Text>
      {(event.before_state || event.after_state) ? (
        <div className="event-state-change">
          <div><small>此前</small><p>{event.before_state || '未记录'}</p></div>
          <span aria-hidden="true">→</span>
          <div><small>此后</small><p>{event.after_state || '未记录'}</p></div>
        </div>
      ) : null}
      <div className="event-card__actions">
        <Button
          aria-label={actionLabel}
          className="event-card__reject-button"
          color="red"
          disabled={reject.isPending}
          leftSection={<Icon name="archive" size={15} />}
          loading={reject.isPending}
          onClick={() => reject.mutate()}
          size="xs"
          type="button"
          variant="subtle"
        >
          {reject.isPending ? '正在标记……' : '标记为误报'}
        </Button>
        {reject.isError && <Alert color="red" role="alert">标记失败，请重试。</Alert>}
      </div>
    </Paper>
  )
}

export function NodeReviewPage() {
  const { projectId } = useParams()
  const navigate = useNavigate()
  const [laneFilter, setLaneFilter] = useSessionState<LaneFilter>(
    `events:${projectId ?? ''}:lane-filter`,
    'all',
  )
  const events = useQuery({
    queryKey: ['events', projectId],
    queryFn: () => request<EventNode[]>(`/api/projects/${projectId}/events`),
    enabled: Boolean(projectId),
  })
  const activeEvents = (events.data ?? [])
    .filter((event) => !['rejected', 'superseded'].includes(event.status))
    .sort((left, right) => {
      const leftTime = left.started_at ?? left.created_at
      const rightTime = right.started_at ?? right.created_at
      return leftTime.localeCompare(rightTime) || left.id.localeCompare(right.id)
    })
  const rejectedCount = (events.data ?? []).filter(
    (event) => event.status === 'rejected',
  ).length
  const confirmTimeline = useMutation({
    mutationFn: () => {
      const runIds = new Set(
        activeEvents
          .map((event) => event.analysis_run_id)
          .filter((id): id is string => Boolean(id)),
      )
      if (runIds.size !== 1) {
        throw new Error('当前节点不属于同一次分析，请刷新后重试')
      }
      return request<{
        confirmation_id: string
        training_job_id: string
        status: string
      }>(`/api/projects/${projectId}/events/confirm-and-train`, {
        method: 'POST',
        body: JSON.stringify({
          analysis_run_id: [...runIds][0],
          event_revisions: activeEvents.map((event) => ({
            event_id: event.id,
            revision_number: event.revision_number,
          })),
        }),
      })
    },
    onSuccess: (result) => {
      navigate(`/projects/${projectId}/training/${result.training_job_id}`)
    },
  })

  return (
    <Stack gap="xl">
      <SectionNav items={memoryNav} label="回忆" />
      <div><Text c="moon.4" fw={700} size="xs">重要回忆</Text><Title mt={5} order={1}>确认关系变化与共同经历</Title><Text c="dimmed" mt={7}>沿时间顺序检查结果，排除误报后开始训练数字人。</Text></div>
      <SegmentedControl aria-label="节点通道筛选" data={[{ label: '全部', value: 'all' }, { label: '关系变化', value: 'relationship' }, { label: '共同经历', value: 'shared_experience' }]} onChange={(value) => setLaneFilter(value as LaneFilter)} value={laneFilter} />
      {events.isLoading && <Text>正在读取节点……</Text>}
      {events.isError && <Alert color="red" role="alert">节点读取失败。</Alert>}
      {events.isSuccess && activeEvents.length === 0 ? (
        <Alert color="gray">尚未发现可审核节点，请先完成数据分析。</Alert>
      ) : null}
      <MantineTimeline active={-1} bulletSize={12} color="moon" lineWidth={1}>
        {projectId &&
          activeEvents
            .filter(
              (event) =>
                laneFilter === 'all' ||
                event.source_lanes.includes(laneFilter) ||
                event.lane === laneFilter,
            )
            .map((event) => (
              <MantineTimeline.Item key={event.id} title={<Text c="dimmed" component="time" dateTime={event.started_at ?? event.created_at} size="xs">
                  {new Date(event.started_at ?? event.created_at).toLocaleDateString(
                    'zh-CN',
                  )}
                </Text>}>
                <EventCard event={event} projectId={projectId} />
              </MantineTimeline.Item>
            ))}
      </MantineTimeline>
      <Paper className="timeline-confirmation" aria-label="时间轴确认" p="md" shadow="md" withBorder>
        <div>
          <strong>有效节点 {activeEvents.length} 个</strong>
          <span>已排除节点 {rejectedCount} 个</span>
        </div>
        <Button
          disabled={activeEvents.length === 0 || confirmTimeline.isPending}
          onClick={() => confirmTimeline.mutate()}
          type="button"
        >
          {confirmTimeline.isPending
            ? '正在创建训练任务……'
            : '确认时间轴并开始训练'}
        </Button>
        {confirmTimeline.isError && (
          <p role="alert">
            {confirmTimeline.error instanceof Error
              ? confirmTimeline.error.message
              : '确认失败，请刷新后重试。'}
          </p>
        )}
      </Paper>
    </Stack>
  )
}

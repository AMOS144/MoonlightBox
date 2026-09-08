import { useQuery } from '@tanstack/react-query'
import { Alert, Badge, Button, Group, Paper, Skeleton, Stack, Text, Timeline, Title } from '@mantine/core'
import { Link, useParams } from 'react-router-dom'

import { request } from '../../api/client'
import { Icon } from '../../components/Icon'
import { SectionNav } from '../../components/SectionNav'
import { memoryNav } from '../../components/sectionNavItems'
import type { EventNode } from '../events/types'
import type { ModelVersion } from '../models/types'

export function TimelinePage() {
  const { projectId } = useParams()
  const events = useQuery({
    queryKey: ['events', projectId],
    queryFn: () => request<EventNode[]>(`/api/projects/${projectId}/events`),
    enabled: Boolean(projectId),
  })
  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => request<ModelVersion[]>(`/api/projects/${projectId}/models`),
    enabled: Boolean(projectId),
  })
  const activeModel = models.data?.some((model) => model.active)
  const visibleEvents = (events.data ?? [])
    .filter((event) => !['rejected', 'superseded'].includes(event.status))
    .sort((a, b) => (a.started_at ?? a.created_at).localeCompare(b.started_at ?? b.created_at))

  return (
    <Stack gap="xl">
      <SectionNav items={memoryNav} label="回忆" />
      <div><Text c="moon.4" fw={700} size="xs">关系时间轴</Text><Title mt={5} order={1}>重要时刻</Title><Text c="dimmed" mt={7}>按时间回看关系变化，并从任意一个已确认节点重新开始。</Text></div>
      {events.isLoading ? <Skeleton h={260} /> : null}
      {events.isError ? <Alert color="red" role="alert">时间轴读取失败，请重试。</Alert> : null}
      {events.isSuccess && visibleEvents.length === 0 ? (
        <Alert color="gray">尚未生成关键节点，请先完成数据分析。</Alert>
      ) : null}
      {models.isSuccess && !activeModel ? (
        <Alert color="yellow">数字人尚未训练完成。你可以浏览回忆，但创建平行时间线前需要先完成训练。</Alert>
      ) : null}
      <Timeline active={visibleEvents.length} bulletSize={18} color="moon" lineWidth={2}>
        {visibleEvents.map((event) => (
          <Timeline.Item key={event.id} title={new Date(event.started_at ?? event.created_at).toLocaleDateString('zh-CN')}>
            <Paper mt="xs" p="lg" withBorder>
              <Group gap={6}><Badge color="violet" variant="light">{event.type}</Badge><Badge color="gray" variant="light">重要度 {Math.round(event.importance * 100)}%</Badge></Group>
              <Title mt="sm" order={3}>{event.title || event.topic}</Title>
              <Text c="dimmed" mt={6}>{event.summary || event.after_state || event.reason}</Text>
              <Button component={Link} leftSection={<Icon name="branch" size={16} />} mt="lg" to={`../branches/new?eventId=${event.id}&originTime=${encodeURIComponent(event.started_at ?? event.created_at)}`}>从这里创建分支</Button>
            </Paper>
          </Timeline.Item>
        ))}
      </Timeline>
    </Stack>
  )
}

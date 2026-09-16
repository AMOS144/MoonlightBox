import { useQuery } from '@tanstack/react-query'
import { Alert, Button, Paper, Stack, Text, Title } from '@mantine/core'
import { Link, useParams } from 'react-router-dom'
import { request } from '../../api/client'
import { AsyncState } from '../../components/feedback/AsyncState'
import type { EventNode } from './types'

/** 旧事件仅作为历史资料展示，不再包含审核、训练或创建分支的写入口。 */
export function LegacyEventsPage() {
  const { projectId } = useParams()
  const events = useQuery({ queryKey: ['events', projectId], queryFn: () => request<EventNode[]>(`/api/projects/${projectId}/events`) })
  return <Stack><Title order={1}>旧版事件资料</Title>
    <Alert color="gray">这里保留旧系统产生的记录，不再作为新的分支起点批准。</Alert>
    <Button component={Link} to={`/projects/${projectId}/nodes`} variant="light">前往新的起点工作台</Button>
    <AsyncState loading={events.isLoading} error={events.error} retry={() => void events.refetch()} />
    {events.data?.map(e => <Paper key={e.id} p="md" withBorder><Title order={4}>{e.title || e.topic}</Title><Text size="sm" c="dimmed">{e.started_at ?? e.created_at}</Text><Text>{e.display_summary || e.summary || e.reason}</Text></Paper>)}
    {events.isSuccess && !events.data.length && <Text c="dimmed">没有旧版事件记录。</Text>}
  </Stack>
}

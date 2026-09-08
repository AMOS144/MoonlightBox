import { useQuery } from '@tanstack/react-query'
import { Alert, Badge, Button, Card, Group, Loader, Paper, Progress, SimpleGrid, Stack, Text, ThemeIcon, Title } from '@mantine/core'
import { Link, useParams } from 'react-router-dom'

import { request } from '../../api/client'
import { Icon } from '../../components/Icon'
import type { Branch } from '../branches/types'
import type { EventNode } from '../events/types'
import type { ModelVersion } from '../models/types'
import type { Project } from './ProjectListPage'

type ProjectJob = {
  id: string
  kind: string
  status: string
  progress: number
}

export function ProjectOverviewPage() {
  const { projectId } = useParams()
  const project = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => request<Project>(`/api/projects/${projectId}`),
    enabled: Boolean(projectId),
  })
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
  const branches = useQuery({
    queryKey: ['branches', projectId],
    queryFn: () => request<Branch[]>(`/api/projects/${projectId}/branches`),
    enabled: Boolean(projectId),
  })
  const jobs = useQuery({
    queryKey: ['project-jobs', projectId],
    queryFn: () => request<ProjectJob[]>(`/api/jobs?project_id=${projectId}`),
    enabled: Boolean(projectId),
  })

  if (project.isLoading) return <Group justify="center" py={80}><Loader aria-label="正在打开这段回忆" /></Group>
  if (project.isError || !project.data) {
    return <Alert color="red" role="alert" title="项目读取失败">请返回项目列表后重试。</Alert>
  }

  const activeJob = jobs.data?.find((job) => ['queued', 'running'].includes(job.status))
  const activeModel = models.data?.find((model) => model.active)
  const activeEvents = events.data?.filter((event) => !['rejected', 'superseded'].includes(event.status)) ?? []
  const recentBranches = [...(branches.data ?? [])]
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
    .slice(0, 3)
  const next = getNextAction(activeJob, activeEvents, activeModel)

  return (
    <Stack gap={36} maw={1120} mx="auto">
      <Group align="flex-end" justify="space-between" wrap="wrap">
        <div>
          <Text c="moon.4" fw={700} size="xs">私人记忆档案</Text>
          <Title mt={6} order={1}>{project.data.name}</Title>
          <Text c="dimmed" mt={8}>把聊天整理成回忆，从某个时刻重新开始。</Text>
        </div>
        <Button component={Link} leftSection={<Icon name={next.icon} size={17} />} size="md" to={next.to}>
          {next.label}
        </Button>
      </Group>

      <Paper p="xl" withBorder>
        <Group align="flex-start" wrap="nowrap">
        <ThemeIcon color="moon" radius="xl" size={48} variant="light">{next.step}</ThemeIcon>
        <div>
          <Text c="dimmed" size="xs">现在最适合做的事</Text>
          <Title mt={3} order={3}>{next.title}</Title>
          <Text c="dimmed" mt={5} size="sm">{next.description}</Text>
        </div>
        </Group>
        {activeJob ? <Progress mt="lg" value={activeJob.progress * 100} /> : null}
      </Paper>

      <SimpleGrid aria-label="项目概况" cols={{ base: 1, sm: 3 }} spacing={0}>
        {[[activeEvents.length, '段重要回忆'], [models.data?.length ?? 0, '个数字人版本'], [branches.data?.length ?? 0, '条平行时间线']].map(([value, label]) => (
          <Paper key={label} p="lg" radius={0} withBorder><Title c="moon.3" order={2}>{value}</Title><Text c="dimmed" size="sm">{label}</Text></Paper>
        ))}
      </SimpleGrid>

      <section>
        <Group justify="space-between" mb="md"><Title order={2}>最近的平行时间线</Title><Button component={Link} to="branches" variant="subtle">查看全部</Button></Group>
        {recentBranches.length ? (
          <SimpleGrid cols={{ base: 1, sm: 3 }}>
            {recentBranches.map((branch) => (
              <Card component={Link} key={branch.id} padding="lg" to={`branches/${branch.id}`} withBorder>
                <Badge color={branch.lifecycle_status === 'active' ? 'green' : 'gray'} variant="light">{branch.lifecycle_status === 'active' ? '进行中' : '只读'}</Badge>
                <Title mt="md" order={4}>{branch.title}</Title>
                <Text c="dimmed" mt={5} size="sm">从 {new Date(branch.origin_time).toLocaleDateString('zh-CN')} 开始</Text>
              </Card>
            ))}
          </SimpleGrid>
        ) : (
          <Paper p="xl" ta="center" withBorder><Title order={4}>还没有平行时间线</Title><Text c="dimmed" mt={5}>训练完成后，可以从任意一段重要回忆重新开始。</Text></Paper>
        )}
      </section>
    </Stack>
  )
}

function getNextAction(job: ProjectJob | undefined, events: EventNode[], model: ModelVersion | undefined) {
  if (job?.kind === 'digital_human_training_v1') return { step: '03', title: '数字人正在学习表达方式', description: '训练在后台继续进行，可以随时查看详细进度。', label: '查看训练进度', to: `training/${job.id}`, icon: 'activity' as const }
  if (job) return { step: '02', title: '正在整理聊天中的重要回忆', description: '系统正在分析关系变化与共同经历。', label: '查看处理进度', to: 'data', icon: 'activity' as const }
  if (model) return { step: '04', title: '选择一个想回去的时刻', description: '数字人已经准备好，从关系时间轴选择新的起点。', label: '进入关系时间轴', to: 'timeline', icon: 'timeline' as const }
  if (events.length) return { step: '02', title: '确认系统找到的重要回忆', description: '排除误报并确认时间轴后，系统会开始训练数字人。', label: '审核重要回忆', to: 'events', icon: 'nodes' as const }
  return { step: '01', title: '导入一段真实聊天', description: '选择完整的 WxEcho 导出目录，系统会保留原始记录并单独处理副本。', label: '开始导入聊天', to: 'data', icon: 'database' as const }
}

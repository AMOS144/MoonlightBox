import { Accordion, Avatar, Button, Divider, Group, Paper, Stack, Text, Title } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import { request } from '../../api/client'
import { AsyncState } from '../../components/feedback/AsyncState'
import { StageStatus } from '../../components/feedback/StageStatus'
import { JourneyTasks } from '../journey/JourneyTasks'
import { actionPath, useJourney } from '../journey/journey'
import type { Project } from './types'
import { PageHeader } from '../../components/PageHeader'
import { BranchCard } from '../branches/BranchCard'
import './projects.css'

export function ProjectOverviewPage() {
  const { projectId = '' } = useParams()
  const project = useQuery({ queryKey: ['project', projectId], queryFn: () => request<Project>(`/api/projects/${projectId}`) })
  const journey = useJourney(projectId)
  const value = journey.data
  const person = value?.participants?.find(p => p.role === 'target')
  const ready = value?.branches.find(b => b.lifecycle_status === 'active')
  const primaryPath = ready ? actionPath(projectId, 'branch', ready.id) : value ? actionPath(projectId, value.next_action.action, value.next_action.object_id) : null
  const otherBranches = value?.branches.filter(b => b.id !== ready?.id) ?? []
  // 去重操作入口，不隐藏待办的失败、问题或审批说明。
  return <Stack className="project-overview" gap="lg">
    <PageHeader title={project.data?.name ?? '项目概览'} />
    <AsyncState loading={journey.isLoading} error={journey.error || project.error} retry={() => { void journey.refetch(); void project.refetch() }} />
    {value && <>
      <Paper withBorder p="lg">
        <Group align="flex-start" wrap="nowrap">
          <Avatar size={64} radius="xl" src={person?.avatar_asset_id ? `/api/projects/${projectId}/media/${person.avatar_asset_id}` : undefined} alt={person?.name}>{person?.name?.slice(-2)}</Avatar>
          <Stack gap="sm" style={{ minWidth: 0, flex: 1 }}>
            <Title order={2}>{person?.name || '确认聊天中的人物'}</Title>
            {ready ? <>
              <Text size="sm" c="dimmed">{ready.title}</Text>
              <Text className="conversation-preview" lineClamp={3}>{ready.latest_message ? `${['self', 'user'].includes(ready.latest_message.role) ? '你' : person?.name ?? '对方'}：${ready.latest_message.text}` : '还没有新的消息'}</Text>
              <Group><Button component={Link} to={`branches/${ready.id}`}>继续聊天</Button></Group>
            </> : <>
              <Text size="sm" c="dimmed">{value.next_action.detail}</Text>
              <Group><Button component={Link} to={actionPath(projectId, value.next_action.action, value.next_action.object_id)}>{value.next_action.label}</Button></Group>
            </>}
          </Stack>
        </Group>
      </Paper>
      <Stack gap="lg">
        {value.tasks.length > 0 && <section><Title order={3} mb="sm">待处理</Title><JourneyTasks journey={value} existingPaths={[primaryPath]} /></section>}
        {otherBranches.length > 0 && <section className="overview-section">
          <Group justify="space-between" mb="sm"><Title order={3}>{ready ? '其他分支' : '我的分支'}</Title>{otherBranches.length > 5 && <Button component={Link} to="branches" variant="subtle">查看全部</Button>}</Group>
          <Stack gap="xs">{otherBranches.slice(0, 5).map(b => <BranchCard key={b.id} branch={b} to={`branches/${b.id}`} />)}</Stack>
        </section>}
      </Stack>
      <Accordion variant="separated">
        <Accordion.Item value="details">
          <Accordion.Control>资料详情</Accordion.Control>
          <Accordion.Panel><Stack gap="sm">
            <Group justify="space-between"><Text c="dimmed" size="sm">导入记录</Text><Text size="sm">{(value.imports ?? []).reduce((n, i) => n + i.message_count, 0).toLocaleString('zh-CN')} 条</Text></Group>
            <Group justify="space-between"><Text c="dimmed" size="sm">参与者</Text><Text size="sm">{(value.participants ?? []).filter(p => ['self', 'target'].includes(p.role)).map(p => p.name).join('、') || '未确认'}</Text></Group>
            <Group justify="space-between"><Text c="dimmed" size="sm">人物背景</Text><Text size="sm">{value.publication ? '已发布' : '未发布'}</Text></Group>
          </Stack></Accordion.Panel>
        </Accordion.Item>
        {value.stages.length > 0 && <Accordion.Item value="stages">
          <Accordion.Control>准备流程</Accordion.Control>
          <Accordion.Panel><Stack gap={4}>
            {value.stages.map((stage, index) => <div key={stage.key}>
              {index > 0 && <Divider mb={4} />}
              <Group justify="space-between" py={4}>
                <Button component={Link} variant="subtle" color="gray" to={actionPath(projectId, stage.action, stage.object_id)}>{stage.label}</Button>
                <StageStatus state={stage.state} />
              </Group>
            </div>)}
          </Stack></Accordion.Panel>
        </Accordion.Item>}
      </Accordion>
    </>}
  </Stack>
}

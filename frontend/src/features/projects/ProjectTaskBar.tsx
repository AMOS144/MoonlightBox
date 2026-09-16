import { Alert, Button, Group, Text } from '@mantine/core'
import { useLocation } from 'react-router-dom'
import { useJourney } from '../journey/journey'
import { StageStatus } from '../../components/feedback/StageStatus'
import { WorldBuildStatus } from './WorldBuildStatus'

export function ProjectTaskBar({ projectId }: { projectId: string }) {
  const journey = useJourney(projectId)
  const { pathname } = useLocation()
  const relative = pathname.split(`/projects/${projectId}/`)[1] ?? ''
  const key = relative.startsWith('setup/import') ? 'import' : relative.startsWith('setup/participants') ? 'participants' : relative.startsWith('world') ? 'profile' : relative.startsWith('nodes') ? 'nodes' : relative === 'branches/new' ? 'prepare' : null
  // 概览自己承载读取错误；不在页面上方重复展示同一个失败。
  // participants 和 nodes 页内已展示各自阶段状态，顶部不再重复提示。
  if (!key || key === 'participants' || key === 'nodes') return null
  if (!journey.data) return journey.isError ? <Alert color="red" mb="lg">流程状态读取失败，当前页面仍可使用。<Button variant="subtle" onClick={() => void journey.refetch()}>重试</Button></Alert> : null
  if (['profile', 'participants', 'import'].includes(key) && journey.data.world_build && journey.data.world_build.status !== 'succeeded') {
    return <WorldBuildStatus key={journey.data.world_build.id} projectId={projectId} build={journey.data.world_build} />
  }
  const stage = journey.data.stages.find(s => s.key === key)
  // 稳定状态在内容区展示；顶部只保留需要注意的执行状态。
  if (!stage || ['not_started', 'confirmed', 'completed'].includes(stage.state)) return null
  return <Alert color="gray" mb="lg" className="project-stage"><Group justify="space-between">
    <Group><Text fw={600}>{stage.label}</Text>{stage.state !== 'not_started' && <StageStatus state={stage.state} />}</Group>
  </Group>{['needs_user', 'blocked', 'paused'].includes(stage.state) && <Text size="sm" mt="xs">{stage.detail}</Text>}</Alert>
}

import { Card, Group, Stack, Text } from '@mantine/core'
import { Link } from 'react-router-dom'
import { StageStatus } from '../../components/feedback/StageStatus'
import { Icon } from '../../components/Icon'
import { actionPath, type Journey } from './journey'

export function JourneyTasks({ journey, existingPaths = [] }: { journey: Journey; existingPaths?: (string | null)[] }) {
  return <Stack gap="xs" aria-label="项目待办">{journey.tasks.map(task => {
    const path = actionPath(journey.project_id, task.action, task.object_id)
    const duplicate = existingPaths.includes(path)
    const body = <Group gap="sm" wrap="nowrap">
      <Icon name="warning" size={17} aria-hidden />
      <Stack gap={3} style={{ flex: 1, minWidth: 0 }}>
        <Text fw={600} size="sm">{task.label}</Text>
        <Text size="sm" c="dimmed" lineClamp={2}>{task.detail}</Text>
      </Stack>
      <StageStatus state={task.state} />
      {!duplicate && <Icon name="chevron" size={15} aria-hidden />}
    </Group>
    return duplicate
      ? <Card withBorder p="md" key={task.id}>{body}</Card>
      : <Card component={Link} to={path} withBorder p="md" key={task.id} className="interactive-card">{body}</Card>
  })}</Stack>
}

import { Card, Group, Stack, Text } from '@mantine/core'
import { Icon } from '../../components/Icon'
import { Link } from 'react-router-dom'
import { BranchStatus } from './BranchStatus'

/** 入口使用紧凑行布局，标题、起点与状态可同时扫读。 */
export function BranchCard({ branch, to }: {
  branch: { title: string; origin_time: string; lifecycle_status: string; latest_message?: { text: string } | null }; to: string
}) {
  return <Card component={Link} to={to} withBorder p="md" className="interactive-card">
    <Group gap="sm" wrap="nowrap">
      <Icon name="conversation" size={17} aria-hidden />
      <Stack gap={3} style={{ flex: 1, minWidth: 0 }}>
        <Text fw={600} size="sm" lineClamp={2}>{branch.title}</Text>
        {branch.latest_message && <Text size="sm" c="dimmed" lineClamp={1}>{branch.latest_message.text}</Text>}
        <Text c="dimmed" size="xs">起点 {new Date(branch.origin_time).toLocaleString('zh-CN')}</Text>
      </Stack>
      <BranchStatus status={branch.lifecycle_status} />
      <Icon name="chevron" size={15} aria-hidden />
    </Group>
  </Card>
}

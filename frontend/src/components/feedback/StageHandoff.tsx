import { Button, Group, Paper, Stack, Text } from '@mantine/core'
import { Link } from 'react-router-dom'
import { Icon } from '../Icon'

export function StageHandoff({ title, detail, to, label }: { title: string; detail?: string; to: string; label: string }) {
  return <Paper p="md" withBorder className="stage-handoff"><Group justify="space-between" align="center" gap="sm">
    <Stack gap={4} style={{ flex: '1 1 220px', minWidth: 0 }}><Text fw={600} size="sm">{title}</Text>{detail && <Text size="sm" c="dimmed" lh={1.5}>{detail}</Text>}</Stack>
    <Button component={Link} to={to} rightSection={<Icon name="next" size={16} />}>{label}</Button>
  </Group></Paper>
}

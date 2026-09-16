import { Group, Stack, Text, Title } from '@mantine/core'
import type { ReactNode } from 'react'

/** 页面只提供内容；标题层级、留白与窄屏折行统一维护。 */
export function PageHeader({ title, description, action }: {
  title: string; description?: string; action?: ReactNode
}) {
  return <Group className="page-heading" justify="space-between" align="flex-end" gap="lg">
    <Stack gap={4} style={{ minWidth: 0, flex: '1 1 220px' }}>
      <Title order={1}>{title}</Title>
      {description && <Text c="dimmed" size="sm" lh={1.5}>{description}</Text>}
    </Stack>
    {action}
  </Group>
}

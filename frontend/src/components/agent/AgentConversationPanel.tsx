import { ActionIcon, Group, Paper, Stack, Text, Title } from '@mantine/core'
import { Icon } from '../Icon'
import type { ReactNode } from 'react'

/** 共享任务外壳，不保存领域会话，不代替用户批准，也不替业务取消任务。 */
export function AgentConversationPanel({ title, subject, scope, status, onClose, children }: {
  title: string; subject: string; scope: string; status: string; onClose?: () => void; children?: ReactNode
}) {
  return <Paper p="md" withBorder component="section" aria-label={title}><Stack gap="md">
    <Group justify="space-between"><Title order={4}>{title}</Title>
      {onClose && <ActionIcon variant="subtle" onClick={onClose} aria-label="收起面板（不会取消任务）"><Icon name="close" size={18} /></ActionIcon>}
    </Group>
    <div><Text fw={600}>{subject}</Text><Text size="sm" c="dimmed">{scope}</Text><Text size="sm">{status}</Text></div>
    {children}
  </Stack></Paper>
}

import { Drawer, Group, Stack, Text, Title } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { request } from '../../api/client'
import { AsyncState } from '../../components/feedback/AsyncState'
import { V3ProfileGrid } from '../world/V3ProfileGrid'
import type { BranchPreparation } from './types'

type Plan = { date: string; blocks: { id: string; start: string; end: string; activity: string }[] }

/** 仅在用户打开资料时读取分支已绑定背景与日程，不替换聊天上下文。 */
export function ChatContextPanel({ projectId, branchId, opened, onClose }: {
  projectId: string; branchId: string; opened: boolean; onClose: () => void
}) {
  const base = `/api/projects/${projectId}/branches/${branchId}`
  const background = useQuery({ queryKey: ['branch-preparation', branchId], queryFn: () => request<BranchPreparation>(`${base}/preparation`), enabled: opened })
  const plan = useQuery({ queryKey: ['branch-day-plan-view', branchId], queryFn: () => request<Plan>(`${base}/runtime/day-plan`), enabled: opened, retry: false })
  return <Drawer opened={opened} onClose={onClose} position="right" size="xl" title="这段对话的背景">
    <Stack>
      <AsyncState loading={background.isLoading} error={background.error} retry={() => void background.refetch()} />
      {background.data?.origin_time && <Text size="sm">分支起点：{new Date(background.data.origin_time).toLocaleString('zh-CN')}</Text>}
      {background.data?.background_mode === 'latest_profile' && <Text size="xs" c="dimmed">此分支使用已绑定的最新背景，并非历史时点的独立画像。</Text>}
      <section><Title order={3}>当天日程</Title><Text size="xs" c="dimmed">计划安排，不代表这些事情已经发生。</Text>
        <AsyncState loading={plan.isLoading} error={plan.error} retry={() => void plan.refetch()} />
        {plan.data && <Stack gap="sm" mt="sm"><Text size="sm">虚拟日期 {plan.data.date}</Text>{plan.data.blocks.map(b => <Group key={b.id} align="flex-start" wrap="nowrap"><Text size="xs" c="dimmed" style={{ whiteSpace: 'nowrap' }}>{b.start}–{b.end}</Text><Text size="sm">{b.activity}</Text></Group>)}</Stack>}
      </section>
      <details><summary>当前分支绑定的人物背景</summary>
        {background.data?.background?.profile_schema_version === 'v3'
          ? <V3ProfileGrid projectId={projectId} profile={background.data.background} locked selectedKeys={new Set()} onToggle={() => {}} />
          : <Text size="sm" c="dimmed">此分支没有可展示的新版背景。</Text>}
      </details>
    </Stack>
  </Drawer>
}

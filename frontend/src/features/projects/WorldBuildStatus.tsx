import { userMessage } from '../../components/feedback/messages'
import { Button, Group, Paper, Progress, Stack, Text } from '@mantine/core'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { request } from '../../api/client'
import { journeyKey, type Journey } from '../journey/journey'

/** 显示后端恢复策略的真实状态；前端不自行计时重试。 */
export function WorldBuildStatus({ projectId, build }: { projectId: string; build: NonNullable<Journey['world_build']> }) {
  const client = useQueryClient()
  const action = useMutation({
    mutationFn: () => request(`/api/jobs/${build.id}/resume`, { method: 'POST' }),
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: journeyKey(projectId) }),
        client.invalidateQueries({ queryKey: ['project-jobs', projectId] }),
        client.invalidateQueries({ queryKey: ['world-profile-status', projectId] }),
      ])
    },
  })
  if (build.status === 'succeeded') return null
  const failed = ['failed', 'interrupted'].includes(build.status)
  const waiting = failed && build.recovery.status === 'waiting'
  const checking = waiting && build.recovery.error_code === 'lightrag_index_running'
  const exhausted = build.recovery.reason === 'retry_exhausted'
  const blocked = failed && build.recovery.status === 'terminal'
  const errorCode = build.failure?.code ?? build.recovery.error_code ?? build.recovery.reason
  const contentRejected = errorCode === 'lightrag_content_rejected'
  const title = build.status === 'cancelled' ? '图谱构建已取消'
    : build.status === 'cancelling' ? '正在取消，等待当前请求退出'
      : checking ? '资料仍在整理，稍后更新进度'
        : waiting ? '图谱构建等待自动恢复'
          : blocked ? '资料整理已暂停，等待处理' : failed ? '图谱构建需要处理' : '正在构建图谱'
  // 数量进度只代表片段整理；不能用包含准备阶段的 Job 进度冒充片段完成比例。
  const completed = Math.max(0, Math.min(build.indexed_bundles, build.bundle_count))
  const percent = build.bundle_count > 0 ? completed / build.bundle_count * 100 : 0
  return <Paper p="md" withBorder component="section" aria-label="资料整理进度"><Stack gap="xs">
    <Group justify="space-between"><Text size="sm" fw={600}>资料整理</Text><Text size="sm" c="dimmed" role="status">{title}</Text></Group>
    <Text size="sm" c="dimmed">{build.bundle_count > 0 ? `已确认 ${completed} / ${build.bundle_count} 个会话片段` : '正在准备聊天片段'}</Text>
    <Progress size="sm" radius="xl" value={percent} aria-label="会话片段整理进度" />
    {build.status === 'running' && <Text size="xs" c="dimmed">{percent === 100 ? '片段已整理，正在完成后续处理。' : '可离开此页，资料会在后台继续整理。'}</Text>}
    {waiting && build.recovery.retry_at && <Text size="sm">下次{checking ? '检查' : '恢复'}：{new Date(build.recovery.retry_at).toLocaleString('zh-CN')}</Text>}
    {failed && !checking && <Text size="sm">{userMessage(build.error_message, undefined, errorCode)}（已恢复 {build.recovery.retries ?? 0} 次）</Text>}
    {blocked && !exhausted && <Text size="sm" c="dimmed">{contentRejected ? '不会自动重复请求，也未跳过受阻资料。是否跳过该片段或调整抽取模型，需要你明确确认。' : '不会自动重复请求，也未跳过受阻资料。请处理原因后再恢复构建。'}</Text>}
    {failed && (build.failure?.document_id || build.failure?.chunk_id) && <Stack gap={2}><Text size="xs">受阻会话：{build.failure?.document_id ?? '暂未定位'}</Text>{build.failure?.chunk_id && <Text size="xs">受阻片段：{build.failure.chunk_id}</Text>}</Stack>}
    {exhausted && <Text size="sm">自动恢复次数已用尽。请处理原因后重新发起构建，已有文档仍保留。</Text>}
    <Group mt="xs">
      {(failed || build.status === 'cancelled') && !checking && <Button variant="light" disabled={action.isPending} loading={action.isPending} onClick={() => action.mutate()}>重试构建</Button>}
    </Group>
    {action.error && <Text c="red" size="sm">{userMessage(action.error)}</Text>}
  </Stack></Paper>
}

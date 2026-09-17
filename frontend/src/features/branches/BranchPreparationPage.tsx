import { userMessage } from '../../components/feedback/messages'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Paper, Stack, Text, Title } from '@mantine/core'
import { Link, useParams } from 'react-router-dom'
import { request } from '../../api/client'
import { AsyncState } from '../../components/feedback/AsyncState'
import { Icon } from '../../components/Icon'
import type { BranchPreparation } from './types'
import { V3ProfileGrid } from '../world/V3ProfileGrid'

const labels: Record<string, string> = {
  planning: '准备人物生活日程', initializing_director: '理解近期经历与未结束的交流',
  completed: '必要准备已完成',
}

export function BranchPreparationPage() {
  const { projectId, branchId } = useParams()
  const client = useQueryClient()
  const preparation = useQuery({
    queryKey: ['branch-preparation', branchId],
    queryFn: () => request<BranchPreparation>(`/api/projects/${projectId}/branches/${branchId}/preparation`),
    refetchInterval: q => q.state.data?.status === 'preparing' ? 2000 : false,
  })
  const retry = useMutation({
    mutationFn: () => request<BranchPreparation>(`/api/projects/${projectId}/branches/${branchId}/preparation/retry`, { method: 'POST' }),
    onSuccess: value => client.setQueryData(['branch-preparation', branchId], value),
  })
  const value = preparation.data
  return <Paper p="md" withBorder><Stack>
    <Title order={2}>{value?.title ?? '分支准备'}</Title>
    {value?.origin_time && <Text c="dimmed">起点：{new Date(value.origin_time).toLocaleString('zh-CN')}</Text>}
    <AsyncState loading={preparation.isLoading} error={preparation.error} retry={() => void preparation.refetch()} />
    {retry.error && value?.status !== 'ready' && <Alert color="red" title="恢复请求未成功">
      {userMessage(retry.error)}
      <Button variant="subtle" onClick={() => { retry.reset(); void preparation.refetch() }}>重新读取</Button>
    </Alert>}
    {value && <Text>{labels[value.stage] ?? '正在读取具体准备阶段'}</Text>}
    {value?.background_mode === 'latest_profile' && <Alert color="gray">这条已有分支使用已绑定的最新背景，不保证历史起点后的资料隔离。</Alert>}
    {value?.background && <details><summary>分支背景（只读）</summary><Text size="sm">显示当前分支绑定版本。</Text>
      {value.background.profile_schema_version === 'v3' ? <V3ProfileGrid projectId={projectId ?? ''} profile={value.background} locked selectedKeys={new Set()} onToggle={() => {}} /> : <Text size="sm">此分支保留原先绑定的背景，不会自动采用项目的新版本。</Text>}
    </details>}
    {value?.status === 'preparing' && <Alert color="blue">系统正在后台准备。可以离开，回来仍会显示这条分支的进展。</Alert>}
    {value?.status === 'failed' && <Alert color="red" title="准备暂时受阻"><Text>{userMessage(value.error_message ?? '准备未完成，已有进展保留。', undefined, value.error_code ?? undefined)}</Text><Button mt="md" loading={retry.isPending} onClick={() => retry.mutate()}>恢复准备</Button></Alert>}
    {value?.status === 'ready' && <Button component={Link} to={`/projects/${projectId}/branches/${branchId}`} onClick={() => { void client.invalidateQueries({ queryKey: ['branches', projectId] }) }}>进入聊天</Button>}
    <Button component={Link} to={`/projects/${projectId}/branches`} variant="subtle" color="gray" leftSection={<Icon name="back" size={16} />}>返回我的分支</Button>
  </Stack></Paper>
}

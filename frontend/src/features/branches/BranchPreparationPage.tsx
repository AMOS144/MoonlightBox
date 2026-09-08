import { useMutation, useQuery } from '@tanstack/react-query'
import { Alert, Button, Center, Paper, Progress, Stack, Text, Title } from '@mantine/core'
import { useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import { request } from '../../api/client'
import type { BranchPreparation } from './types'

const stageLabels: Record<string, string> = {
  resolving_boundary: '确认时间节点',
  freezing_messages: '加载节点前聊天',
  freezing_events: '整理共同经历',
  building_state: '还原当时关系状态',
  building_index: '建立历史记忆索引',
  validating: '校验历史边界',
  completed: '准备完成',
}

export function BranchPreparationPage() {
  const { projectId, branchId } = useParams()
  const navigate = useNavigate()
  const preparation = useQuery({
    queryKey: ['branch-preparation', branchId],
    queryFn: () =>
      request<BranchPreparation>(
        `/api/projects/${projectId}/branches/${branchId}/preparation`,
      ),
    enabled: Boolean(projectId && branchId),
    refetchInterval: (query) =>
      query.state.data?.status === 'preparing' ? 1500 : false,
  })
  const retry = useMutation({
    mutationFn: () =>
      request<BranchPreparation>(
        `/api/projects/${projectId}/branches/${branchId}/preparation/retry`,
        { method: 'POST' },
      ),
    onSuccess: (value) => {
      preparation.refetch()
      if (value.status === 'ready') {
        navigate(`/projects/${projectId}/branches/${branchId}`, {
          replace: true,
        })
      }
    },
  })

  useEffect(() => {
    if (preparation.data?.status === 'ready') {
      navigate(`/projects/${projectId}/branches/${branchId}`, { replace: true })
    }
  }, [branchId, navigate, preparation.data?.status, projectId])

  const value = preparation.data
  return (
    <Center mih="70vh">
    <Paper maw={620} p={{ base: 'lg', sm: 40 }} ta="center" w="100%" withBorder>
      <Stack>
      <Text c="moon.4" fw={700} size="xs">正在进入平行时间线</Text>
      <Title order={2}>加载这个时刻之前的记忆</Title>
      {preparation.isLoading ? <Text role="status">正在恢复准备状态……</Text> : null}
      {preparation.isError ? (
        <Alert color="red" role="alert">准备状态读取失败，请刷新后重试。</Alert>
      ) : null}
      <Text>{stageLabels[value?.stage ?? ''] ?? '正在准备基础历史'}</Text>
      <Progress size="lg" value={(value?.progress ?? 0) * 100} />
      <Text fw={700}>{Math.round((value?.progress ?? 0) * 100)}%</Text>
      {value ? (
        <Text c="dimmed" size="sm">
          已确认 {value.message_count} 条消息、{value.event_count} 个共同经历
        </Text>
      ) : null}
      {value?.status === 'failed' ? (
        <Alert color="red" title="准备失败">
          <Text mb="md">{value.error_message ?? '基础历史构建失败'}</Text>
          <Button
            loading={retry.isPending}
            onClick={() => retry.mutate()}
            type="button"
          >
            重新加载
          </Button>
        </Alert>
      ) : null}
      </Stack>
    </Paper>
    </Center>
  )
}

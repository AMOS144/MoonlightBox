import { useQuery } from '@tanstack/react-query'
import { Alert, Avatar, Badge, Button, Card, Group, SimpleGrid, Skeleton, Stack, Text, Title } from '@mantine/core'
import { Link, useParams } from 'react-router-dom'

import { request } from '../../api/client'
import type { RuntimeBranch } from './types'

export function BranchListPage() {
  const { projectId } = useParams()
  const branches = useQuery({
    queryKey: ['runtime-branches', projectId],
    queryFn: () => request<RuntimeBranch[]>(`/api/projects/${projectId}/runtime/branches`),
    enabled: Boolean(projectId),
  })

  return (
    <Stack gap="xl">
      <div><Text c="moon.4" fw={700} size="xs">平行时间线</Text><Title mt={5} order={1}>继续另一种可能</Title><Text c="dimmed" mt={7}>每条时间线都有独立起点和对话，不会修改原始聊天。</Text></div>
      {branches.isLoading ? <SimpleGrid cols={{ base: 1, md: 2 }}><Skeleton h={150} /><Skeleton h={150} /></SimpleGrid> : null}
      {branches.isError ? <Alert color="red" role="alert">时间分支读取失败，请重试。</Alert> : null}
      <SimpleGrid cols={{ base: 1, md: 2 }}>
        {branches.data?.map((branch) => (
          <Card component={Link} key={branch.id} padding="lg" to={branch.id} withBorder>
            <Group align="flex-start" wrap="nowrap"><Avatar color="violet" radius="xl">{branch.title.slice(0, 1)}</Avatar><div><Badge color={branch.lifecycle_status === 'active' ? 'green' : 'gray'} variant="light">{branch.lifecycle_status === 'active' ? '进行中' : '只读记录'}</Badge><Title mt="sm" order={3}>{branch.title}</Title><Text c="dimmed" mt={5} size="sm">从 {new Date(branch.origin_time).toLocaleString('zh-CN')} 开始</Text><Text c="dimmed" mt={3} size="xs">创建于 {new Date(branch.created_at).toLocaleDateString('zh-CN')}</Text></div></Group>
          </Card>
        ))}
      </SimpleGrid>
      {branches.isSuccess && branches.data.length === 0 ? (
        <Alert color="gray" title="还没有平行时间线"><Text mb="md">先去关系时间轴选择一个重要时刻，再从那里重新开始。</Text><Button component={Link} to="../timeline">选择一个过去的时刻</Button></Alert>
      ) : null}
    </Stack>
  )
}

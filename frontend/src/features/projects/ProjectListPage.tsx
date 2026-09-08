import { useQuery } from '@tanstack/react-query'
import { Alert, Badge, Button, Card, Container, Group, SimpleGrid, Skeleton, Stack, Text, Title } from '@mantine/core'
import { Link } from 'react-router-dom'

import { request } from '../../api/client'
import { Icon } from '../../components/Icon'

export type Project = {
  id: string
  name: string
  status: string
  created_at: string
  updated_at: string
}

export function ProjectListPage() {
  const projects = useQuery({
    queryKey: ['projects'],
    queryFn: () => request<Project[]>('/api/projects'),
  })

  return (
    <Container component="main" py={{ base: 32, sm: 64 }} size="lg">
      <Group align="flex-end" justify="space-between" mb={48}>
        <div>
          <Text c="moon.4" fw={700} size="xs" tt="uppercase">Moonlight Box</Text>
          <Title mt={6} order={1}>月光宝盒</Title>
          <Text c="dimmed" mt={8}>保存真实对话，也保存那些可以重新选择的时刻。</Text>
        </div>
        <Button component={Link} leftSection={<Icon name="plus" size={17} />} to="/projects/new">
          创建项目
        </Button>
      </Group>

      {projects.isPending && <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }}>{[0, 1, 2].map((item) => <Skeleton h={150} key={item} radius="md" />)}</SimpleGrid>}
      {projects.isError && <Alert color="red" title="项目加载失败">请检查服务状态后重试。</Alert>}
      {projects.data && projects.data.length === 0 && (
        <Card padding="xl" withBorder>
          <Stack align="center" gap="xs" py="xl">
            <Title order={3}>还没有记忆档案</Title>
            <Text c="dimmed">创建项目并导入一份 WxEcho 聊天记录。</Text>
            <Button component={Link} mt="md" to="/projects/new">创建第一个项目</Button>
          </Stack>
        </Card>
      )}
      <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }}>
        {projects.data?.map((project) => (
          <Card component={Link} key={project.id} padding="lg" to={`/projects/${project.id}`} withBorder>
            <Group justify="space-between"><Icon name="timeline" size={22} /><Badge color="gray" variant="light">{project.status}</Badge></Group>
            <Title mt="xl" order={3}>{project.name}</Title>
            <Text c="dimmed" mt={6} size="sm">更新于 {new Date(project.updated_at).toLocaleDateString('zh-CN')}</Text>
          </Card>
        ))}
      </SimpleGrid>
    </Container>
  )
}
